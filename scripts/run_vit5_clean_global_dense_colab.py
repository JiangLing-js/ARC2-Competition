#!/usr/bin/env python3
"""Run challenge-only clean global ViT-5 Dense training on Colab or locally.

Deliberately absent from this CLI: ``--solutions``.  Official test outputs
cannot enter global training, development validation, or checkpoint selection.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.arc_agi_io import load_arc_challenges
from src.hf_arc_tokenizer import load_t5_tokenizer_with_arc_tokens
from src.model_checkpointing import (
    build_model_from_checkpoint,
    load_checkpoint,
    save_vit5_hybrid_arc_full_checkpoint,
    save_vit5_hybrid_lightweight_checkpoint,
)
from src.vit5_clean_global_dense_training import (
    CleanGlobalTrainingConfig,
    global_training_protocol,
    train_clean_global_dense,
    validate_gated_same_shape_tasks,
)
from src.vit5_clean_task_split import load_frozen_split_manifest
from src.vit5_colab_baseline import resolved_environment
from src.vit5_gated_warm_start import (
    warm_start_from_completed_gated_context_progress,
)
from src.vit5_global_ttt_evaluation import path_sha256
from src.vit5_hybrid_model import (
    VIT5_INITIALIZATION_MODES,
    VIT5_INITIALIZATION_PRETRAINED_LORA,
    VIT5_INITIALIZATION_RANDOM_SCRATCH,
    build_vit5_hybrid_model,
)
from src.vit5_k4_replay import build_k4_replay_bundle
from src.vit5_dense_decoder import (
    DENSE_CLEAN_GLOBAL_CONDITIONING_MODES,
    DENSE_CONDITIONING_GATED_PAIRED,
    DENSE_CONDITIONING_SPLIT,
    DENSE_GATED_PAIRED_GATE_INITIAL_VALUE,
    DENSE_RELATIONAL_MEMORY_DISABLED,
    DENSE_RELATIONAL_MEMORY_GATE_INITIAL_VALUE,
    DENSE_RELATIONAL_MEMORY_MODES,
    DENSE_SHAPE_DECODER_MODES,
    DENSE_SHAPE_DECODER_SEPARATE_HW,
    DENSE_SHAPE_DECODER_BD_BORDER,
)
from src.vit5_per_task_full_adaptation import load_torch_payload
from src.vit5_per_task_ttt import D4_TRANSFORM_NAMES


def _parse_names(value: str) -> tuple[str, ...]:
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    if not names:
        raise argparse.ArgumentTypeError("At least one transform is required.")
    return names


def _parse_colors(value: str) -> tuple[int, ...]:
    try:
        colors = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Colors must be comma-separated integers.") from exc
    if not colors:
        raise argparse.ArgumentTypeError("At least one source color is required.")
    return colors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_dir", default="/content")
    parser.add_argument("--challenges", required=True)
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--t5_path", required=True)
    parser.add_argument(
        "--vit5_initialization",
        choices=VIT5_INITIALIZATION_MODES,
        default=VIT5_INITIALIZATION_PRETRAINED_LORA,
        help=(
            "pretrained_lora preserves the frozen ViT-5+LoRA baseline; "
            "random_scratch skips ViT-5 weights and trains the complete grid encoder."
        ),
    )
    parser.add_argument(
        "--vit5_checkpoint",
        default=None,
        help="Required only by --vit5_initialization pretrained_lora.",
    )
    parser.add_argument("--output_dir", required=True)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume_checkpoint", default=None)
    initialization.add_argument(
        "--warm_start_context_progress",
        default=None,
        help=(
            "Completed gated context-rule progress v3. Only its trainable "
            "weights are loaded into a fresh model; optimizer, scheduler, "
            "sampler, RNG and update state are reset."
        ),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--initialization_seed", type=int, default=0)
    parser.add_argument("--task_fusion_layers", type=int, default=2)
    parser.add_argument("--dense_decoder_layers", type=int, default=2)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lambda_component", type=float, default=0.25)
    parser.add_argument("--lambda_shape", type=float, default=0.2)
    parser.add_argument(
        "--dense_shape_decoder_mode",
        choices=DENSE_SHAPE_DECODER_MODES,
        default=DENSE_SHAPE_DECODER_SEPARATE_HW,
        help=(
            "separate_hw_v1 uses legacy learned H/W heads; bd_border_v1 uses "
            "a VARC-inspired 31x31 color+BD canvas and structured border decode."
        ),
    )
    parser.add_argument("--dense_recursive_steps", type=int, choices=range(1, 9), default=1)
    parser.add_argument(
        "--dense_recursive_loss_weighting",
        choices=("uniform", "hsw"),
        default="hsw",
    )
    parser.add_argument("--dense_hsw_lambda", type=float, default=0.7)
    parser.add_argument("--dense_candidate_encoder_layers", type=int, default=2)
    parser.add_argument("--dense_candidate_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--dense_candidate_dropout", type=float, default=0.0)
    parser.add_argument(
        "--dense_conditioning_mode",
        choices=DENSE_CLEAN_GLOBAL_CONDITIONING_MODES,
        default=DENSE_CONDITIONING_SPLIT,
    )
    parser.add_argument(
        "--dense_paired_gate_initial_value",
        type=float,
        default=DENSE_GATED_PAIRED_GATE_INITIAL_VALUE,
    )
    parser.add_argument(
        "--dense_relational_memory_mode",
        choices=DENSE_RELATIONAL_MEMORY_MODES,
        default=DENSE_RELATIONAL_MEMORY_DISABLED,
        help=(
            "disabled keeps exact-coordinate Paired Memory v1; "
            "set_cross_attention_v2 enables variable-length demonstration "
            "input/output relation sets and permits shape-changing train pairs."
        ),
    )
    parser.add_argument(
        "--dense_relational_gate_initial_value",
        type=float,
        default=DENSE_RELATIONAL_MEMORY_GATE_INITIAL_VALUE,
        help=(
            "Initial effective 2*sigmoid gate scale for Relational Memory v2; "
            "must lie strictly in (0,2). Ignored while the mode is disabled."
        ),
    )
    parser.add_argument("--max_optimizer_updates", type=int, default=1000)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--new_modules_lr", type=float, default=3e-4)
    parser.add_argument("--vit5_lora_lr", type=float, default=1e-4)
    parser.add_argument("--vit5_backbone_lr", type=float, default=1e-4)
    parser.add_argument("--new_modules_min_lr", type=float, default=1e-6)
    parser.add_argument("--vit5_lora_min_lr", type=float, default=1e-6)
    parser.add_argument("--vit5_backbone_min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_optimizer_updates", type=int, default=50)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--validate_every_updates", type=int, default=100)
    parser.add_argument("--save_every_updates", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--transform_names", type=_parse_names, default=D4_TRANSFORM_NAMES)
    parser.add_argument(
        "--shuffle_demonstrations", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--color_permutation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply a reversible random bijection of ARC colors 1..9 (color 0 "
            "is fixed) to every demonstration input/output, query, and target."
        ),
    )
    parser.add_argument("--max_grids", type=int, default=21)
    parser.add_argument("--grid_encoder_microbatch", type=int, default=32)
    parser.add_argument("--episode_cache_entries", type=int, default=512)
    parser.add_argument(
        "--k4_replay_batch_size",
        type=int,
        default=0,
        help=(
            "Additional streaming K=4 samples per ARC microbatch. Zero disables "
            "the experimental replay path; use roughly one replay sample per "
            "four ARC samples (for example ARC B4 + K4 B1)."
        ),
    )
    parser.add_argument("--k4_replay_loss_fraction", type=float, default=0.2)
    parser.add_argument(
        "--k4_replay_loss_mode",
        choices=("color_only", "dense_full"),
        default="color_only",
        help=(
            "color_only preserves paired color binding without teaching the "
            "ARC shape heads that every replay output is 4x4."
        ),
    )
    parser.add_argument(
        "--k4_require_initial_retention_pass",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require the fixed K4 A/B/C panel to pass at update 0 before ARC "
            "training begins. Disable only for wiring/debug smoke tests."
        ),
    )
    parser.add_argument("--k4_replay_seed", type=int, default=0)
    parser.add_argument("--k4_replay_train_rule_count", type=int, default=256)
    parser.add_argument("--k4_replay_validation_rule_count", type=int, default=64)
    parser.add_argument("--k4_replay_demonstration_count", type=int, default=3)
    parser.add_argument("--k4_replay_grid_size", type=int, default=4)
    parser.add_argument(
        "--k4_replay_source_colors", type=_parse_colors, default=(1, 2, 3, 4)
    )
    parser.add_argument(
        "--k4_replay_transform_names", type=_parse_names, default=("identity",)
    )
    parser.add_argument("--k4_replay_validation_batch_size", type=int, default=8)
    return parser


def _device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(value)


def _config(args: argparse.Namespace) -> CleanGlobalTrainingConfig:
    config = CleanGlobalTrainingConfig(
        max_optimizer_updates=args.max_optimizer_updates,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        new_modules_lr=args.new_modules_lr,
        vit5_lora_lr=args.vit5_lora_lr,
        vit5_backbone_lr=args.vit5_backbone_lr,
        new_modules_min_lr=args.new_modules_min_lr,
        vit5_lora_min_lr=args.vit5_lora_min_lr,
        vit5_backbone_min_lr=args.vit5_backbone_min_lr,
        warmup_optimizer_updates=args.warmup_optimizer_updates,
        gradient_clip_norm=args.gradient_clip_norm,
        weight_decay=args.weight_decay,
        validate_every_updates=args.validate_every_updates,
        save_every_updates=args.save_every_updates,
        seed=args.seed,
        transform_names=tuple(args.transform_names),
        shuffle_demonstrations=args.shuffle_demonstrations,
        color_permutation=args.color_permutation,
        precision=args.precision,
        max_grids=args.max_grids,
        grid_encoder_microbatch=args.grid_encoder_microbatch,
        episode_cache_entries=args.episode_cache_entries,
        lambda_component=args.lambda_component,
        lambda_shape=args.lambda_shape,
        dense_recursive_steps=args.dense_recursive_steps,
        dense_recursive_loss_weighting=args.dense_recursive_loss_weighting,
        dense_hsw_lambda=args.dense_hsw_lambda,
        dense_candidate_encoder_layers=args.dense_candidate_encoder_layers,
        dense_candidate_mlp_ratio=args.dense_candidate_mlp_ratio,
        dense_candidate_dropout=args.dense_candidate_dropout,
        dense_conditioning_mode=args.dense_conditioning_mode,
        dense_shape_decoder_mode=args.dense_shape_decoder_mode,
        dense_paired_gate_initial_value=args.dense_paired_gate_initial_value,
        dense_relational_memory_mode=args.dense_relational_memory_mode,
        dense_relational_gate_initial_value=(
            args.dense_relational_gate_initial_value
        ),
        k4_replay_batch_size=args.k4_replay_batch_size,
        k4_replay_loss_fraction=args.k4_replay_loss_fraction,
        k4_replay_loss_mode=args.k4_replay_loss_mode,
        k4_require_initial_retention_pass=args.k4_require_initial_retention_pass,
        k4_replay_seed=args.k4_replay_seed,
        k4_replay_train_rule_count=args.k4_replay_train_rule_count,
        k4_replay_validation_rule_count=args.k4_replay_validation_rule_count,
        k4_replay_demonstration_count=args.k4_replay_demonstration_count,
        k4_replay_grid_size=args.k4_replay_grid_size,
        k4_replay_source_colors=tuple(args.k4_replay_source_colors),
        k4_replay_transform_names=tuple(args.k4_replay_transform_names),
        k4_replay_validation_batch_size=args.k4_replay_validation_batch_size,
        vit5_training_mode=args.vit5_initialization,
    )
    config.validate()
    return config


def _override_paths(checkpoint: dict[str, Any], args: argparse.Namespace) -> None:
    checkpoint["model_config"]["pretrained_model_name_or_path"] = args.t5_path
    if args.vit5_checkpoint is not None:
        checkpoint["model_config"]["vit5_checkpoint_path"] = args.vit5_checkpoint
    checkpoint["model_config"]["local_files_only"] = True


def _resume_optimizer_state(checkpoint: dict[str, Any]) -> Mapping[str, Any] | None:
    if "_optimizer_state_dict" in checkpoint:
        return checkpoint["_optimizer_state_dict"]
    config = checkpoint["model_config"]
    base = Path(checkpoint.get("_checkpoint_base_dir", "."))
    state_path = Path(config["hybrid_trainable_state_path"])
    if not state_path.is_absolute():
        state_path = base / state_path
    try:
        payload = torch.load(state_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(state_path, map_location="cpu")
    return payload.get("optimizer_state_dict")


def _fresh_model(
    args: argparse.Namespace, device: torch.device
) -> tuple[Any, Any, dict[str, int], dict[str, Any]]:
    torch.manual_seed(args.initialization_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.initialization_seed)
    tokenizer, token_summary = load_t5_tokenizer_with_arc_tokens(
        args.t5_path, local_files_only=True
    )
    scratch = args.vit5_initialization == VIT5_INITIALIZATION_RANDOM_SCRATCH
    model = build_vit5_hybrid_model(
        pretrained_model_name_or_path=args.t5_path,
        tokenizer=tokenizer,
        vit5_checkpoint_path=args.vit5_checkpoint,
        decoder_mode="dense",
        max_grids=args.max_grids,
        task_fusion_layers=args.task_fusion_layers,
        grid_encoder_microbatch=args.grid_encoder_microbatch,
        freeze_vit5=not scratch,
        freeze_t5=True,
        enable_vit5_lora=not scratch,
        enable_t5_decoder_lora=False,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        added_token_ids=list(token_summary["added_token_ids"].values()),
        local_files_only=True,
        dense_decoder_layers=args.dense_decoder_layers,
        lambda_shape=args.lambda_shape,
        lambda_component=args.lambda_component,
        dense_recursive_steps=args.dense_recursive_steps,
        dense_recursive_loss_weighting=args.dense_recursive_loss_weighting,
        dense_hsw_lambda=args.dense_hsw_lambda,
        dense_candidate_encoder_layers=args.dense_candidate_encoder_layers,
        dense_candidate_mlp_ratio=args.dense_candidate_mlp_ratio,
        dense_candidate_dropout=args.dense_candidate_dropout,
        dense_conditioning_mode=args.dense_conditioning_mode,
        dense_paired_gate_initial_value=args.dense_paired_gate_initial_value,
        dense_relational_memory_mode=args.dense_relational_memory_mode,
        dense_relational_gate_initial_value=(
            args.dense_relational_gate_initial_value
        ),
        dense_shape_decoder_mode=args.dense_shape_decoder_mode,
        object_embedding_mode="none",
        vit5_initialization=args.vit5_initialization,
    ).to(device)
    token_to_id = {str(key): int(value) for key, value in token_summary["added_token_ids"].items()}
    model_config = {
        "model_type": "vit5_hybrid",
        "pretrained_model_name_or_path": str(args.t5_path),
        "tokenizer_name_or_path": str(args.t5_path),
        "vit5_checkpoint_path": (
            str(args.vit5_checkpoint) if args.vit5_checkpoint is not None else None
        ),
        "vit5_initialization": args.vit5_initialization,
        "decoder_mode": "dense",
        "max_grids": args.max_grids,
        "task_fusion_layers": args.task_fusion_layers,
        "grid_encoder_microbatch": args.grid_encoder_microbatch,
        "freeze_vit5": not scratch,
        "freeze_t5": True,
        "enable_vit5_lora": not scratch,
        "enable_t5_decoder_lora": False,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "added_token_ids": list(token_to_id.values()),
        "local_files_only": True,
        "dense_decoder_layers": args.dense_decoder_layers,
        "lambda_shape": args.lambda_shape,
        "lambda_component": args.lambda_component,
        "dense_recursive_steps": args.dense_recursive_steps,
        "dense_recursive_loss_weighting": args.dense_recursive_loss_weighting,
        "dense_hsw_lambda": args.dense_hsw_lambda,
        "dense_candidate_encoder_layers": args.dense_candidate_encoder_layers,
        "dense_candidate_mlp_ratio": args.dense_candidate_mlp_ratio,
        "dense_candidate_dropout": args.dense_candidate_dropout,
        "dense_conditioning_mode": args.dense_conditioning_mode,
        "dense_shape_decoder_mode": args.dense_shape_decoder_mode,
        "dense_paired_gate_initial_value": args.dense_paired_gate_initial_value,
        "dense_relational_memory_mode": args.dense_relational_memory_mode,
        "dense_relational_gate_initial_value": (
            args.dense_relational_gate_initial_value
        ),
        "object_embedding_mode": "none",
        "batch_schema_version": 4,
        "loss_schema_version": 2,
        "formal_baseline_eligible": True,
        "clean_global_training": True,
        "solutions_used_for_training": False,
    }
    return model, tokenizer, token_to_id, model_config


def _context_progress_path(value: str) -> Path:
    source = Path(value).expanduser().resolve()
    candidates = (
        [source]
        if source.is_file()
        else [
            source / "latest" / "context_rule_progress.pt",
            source / "context_rule_progress.pt",
        ]
    )
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise FileNotFoundError(
            "Could not find completed Gated context progress. Tried: "
            + ", ".join(str(path) for path in candidates)
        )
    return existing[0]


def _warm_start_architecture(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dense_conditioning_mode": str(args.dense_conditioning_mode),
        "dense_paired_gate_initial_value": float(
            args.dense_paired_gate_initial_value
        ),
        "task_fusion_layers": int(args.task_fusion_layers),
        "dense_decoder_layers": int(args.dense_decoder_layers),
        "lora_r": int(args.lora_r),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "lambda_component": float(args.lambda_component),
        "lambda_shape": float(args.lambda_shape),
        "object_embedding_mode": "none",
        "dense_recursive_steps": int(args.dense_recursive_steps),
        "max_grids": int(args.max_grids),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)
    config = _config(args)
    if not args.resume_checkpoint:
        if (
            args.vit5_initialization == VIT5_INITIALIZATION_PRETRAINED_LORA
            and not args.vit5_checkpoint
        ):
            raise ValueError(
                "--vit5_checkpoint is required for pretrained_lora initialization."
            )
        if args.vit5_initialization == VIT5_INITIALIZATION_RANDOM_SCRATCH:
            if args.vit5_checkpoint:
                raise ValueError(
                    "random_scratch must not receive --vit5_checkpoint."
                )
            if args.warm_start_context_progress:
                raise ValueError(
                    "random_scratch cannot warm-start a pretrained LoRA context checkpoint."
                )
        if (
            args.warm_start_context_progress
            and args.dense_shape_decoder_mode == DENSE_SHAPE_DECODER_BD_BORDER
        ):
            raise ValueError(
                "Legacy Gated context progress uses separate H/W heads and cannot "
                "warm-start bd_border_v1. Train the scratch+BD K4 qualification "
                "directly or use an explicitly matching BD checkpoint."
            )
    if (
        config.k4_replay_batch_size
        and config.k4_require_initial_retention_pass
        and not (args.resume_checkpoint or args.warm_start_context_progress)
    ):
        raise ValueError(
            "A strict K4 retention run must start from either "
            "--warm_start_context_progress or --resume_checkpoint. Use the "
            "explicit --no-k4_require_initial_retention_pass override only "
            "for a wiring/debug smoke test."
        )
    challenges = load_arc_challenges(args.challenges)
    split = load_frozen_split_manifest(
        args.split_manifest, args.challenges, available_task_ids=set(challenges)
    )
    train_ids = split["global_train_task_ids"]
    dev_ids = split["development_task_ids"]
    # The core trainer receives train demonstrations only.  Even challenge-side
    # test inputs are absent from its object graph; solutions are not a CLI/API.
    train_tasks = {
        task_id: {"train": copy.deepcopy(challenges[task_id]["train"])}
        for task_id in train_ids
    }
    dev_tasks = {
        task_id: {"train": copy.deepcopy(challenges[task_id]["train"])}
        for task_id in dev_ids
    }
    if (
        config.dense_conditioning_mode == DENSE_CONDITIONING_GATED_PAIRED
        and config.dense_relational_memory_mode
        == DENSE_RELATIONAL_MEMORY_DISABLED
    ):
        validate_gated_same_shape_tasks(train_tasks, split_name="global_train")
        validate_gated_same_shape_tasks(dev_tasks, split_name="development")
    k4_replay = build_k4_replay_bundle(config) if config.k4_replay_batch_size else None
    if k4_replay is not None:
        (output / "k4_replay_pool_manifest.json").write_text(
            json.dumps(k4_replay.pool_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    resume_state = None
    resume_optimizer = None
    warm_start_report: dict[str, Any] | None = None
    warm_start_source: Path | None = None
    if args.resume_checkpoint:
        checkpoint = load_checkpoint(args.resume_checkpoint, map_location="cpu")
        _override_paths(checkpoint, args)
        resume_state = checkpoint["training_state"]
        resume_optimizer = _resume_optimizer_state(checkpoint)
        model = build_model_from_checkpoint(checkpoint, device=device)
        tokenizer, _ = load_t5_tokenizer_with_arc_tokens(
            Path(checkpoint["_checkpoint_base_dir"]) / checkpoint["model_config"]["tokenizer_dir"],
            local_files_only=True,
        )
        token_to_id = {str(key): int(value) for key, value in checkpoint["token_to_id"].items()}
        model_config = dict(checkpoint["model_config"])
    else:
        model, tokenizer, token_to_id, model_config = _fresh_model(args, device)
        if args.warm_start_context_progress:
            if args.dense_conditioning_mode != DENSE_CONDITIONING_GATED_PAIRED:
                raise ValueError(
                    "--warm_start_context_progress requires "
                    "--dense_conditioning_mode gated_paired_associative_v1."
                )
            if (
                args.dense_relational_memory_mode
                != DENSE_RELATIONAL_MEMORY_DISABLED
            ):
                raise ValueError(
                    "Legacy Gated context progress contains exact-coordinate "
                    "Paired Memory v1 weights and cannot warm-start Relational "
                    "Memory v2. Start v2 fresh or resume a v2 checkpoint."
                )
            warm_start_source = _context_progress_path(
                args.warm_start_context_progress
            )
            if output == warm_start_source or output in warm_start_source.parents:
                raise ValueError(
                    "Clean-global output_dir must not contain the warm-start source."
                )
            source_hash_before = path_sha256(warm_start_source)
            progress = load_torch_payload(warm_start_source)
            if not isinstance(progress, Mapping):
                raise ValueError("Gated context progress payload must be a mapping.")
            if k4_replay is not None:
                source_pool_hash = str(
                    progress.get("resume_contract", {}).get(
                        "pool_manifest_sha256", "<missing>"
                    )
                )
                if source_pool_hash != k4_replay.pool_manifest_sha256:
                    raise ValueError(
                        "K4 replay pool differs from the completed Gated warm-start "
                        "pool. Use the same K4 seed/rule counts or a new explicitly "
                        "documented protocol."
                    )
            warm_start_report = warm_start_from_completed_gated_context_progress(
                model,
                progress,
                source_sha256=source_hash_before,
                expected_architecture=_warm_start_architecture(args),
            )
            if path_sha256(warm_start_source) != source_hash_before:
                raise AssertionError("Warm-start modified its source progress file.")
    effective_warm_start_report = (
        copy.deepcopy(model_config.get("gated_context_warm_start"))
        if args.resume_checkpoint
        else warm_start_report
    )
    effective_warm_start_source = (
        model_config.get("gated_context_warm_start_source")
        if args.resume_checkpoint
        else (
            str(warm_start_source) if warm_start_source is not None else None
        )
    )
    fresh_initialization_baseline_eligible = (
        bool(model_config.get("fresh_initialization_baseline_eligible", False))
        if args.resume_checkpoint
        else warm_start_report is None
    )
    model_config.update(
        {
            "split_manifest_sha256": split["split_manifest_sha256"],
            "clean_global_training_protocol": global_training_protocol(config),
            "formal_baseline_eligible": k4_replay is None,
            "fresh_initialization_baseline_eligible": (
                fresh_initialization_baseline_eligible
            ),
            "experimental_gated_arc_qualification": (
                config.dense_conditioning_mode
                == DENSE_CONDITIONING_GATED_PAIRED
            ),
            "experimental_relational_memory_v2": (
                config.dense_relational_memory_mode
                != DENSE_RELATIONAL_MEMORY_DISABLED
            ),
            "solutions_used_for_training": False,
            "synthetic_training_data_used": k4_replay is not None,
            "experimental_k4_replay": k4_replay is not None,
            "experimental_random_scratch": (
                args.vit5_initialization == VIT5_INITIALIZATION_RANDOM_SCRATCH
            ),
            "experimental_varc_bd_output": (
                args.dense_shape_decoder_mode == DENSE_SHAPE_DECODER_BD_BORDER
            ),
            "k4_replay_pool_manifest_sha256": (
                k4_replay.pool_manifest_sha256 if k4_replay is not None else None
            ),
            "initialization_mode": (
                "warm_start_from_completed_gated_context_progress_v1"
                if warm_start_report is not None
                else (
                    (
                        "resume_clean_global_k4_replay_progress_v2"
                        if k4_replay is not None
                        else "resume_clean_global_progress_v1"
                    )
                    if args.resume_checkpoint
                    else (
                        "fresh_random_scratch_full_grid_encoder_v1"
                        if args.vit5_initialization
                        == VIT5_INITIALIZATION_RANDOM_SCRATCH
                        else "fresh_pretrained_vit5_lora_v1"
                    )
                )
            ),
            "vit5_initialization": getattr(
                model, "vit5_initialization", VIT5_INITIALIZATION_PRETRAINED_LORA
            ),
            "gated_context_warm_start": effective_warm_start_report,
            "gated_context_warm_start_source": effective_warm_start_source,
        }
    )
    environment = resolved_environment()
    (output / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "config_resolved.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "split_manifest_sha256": split["split_manifest_sha256"],
                "solutions_cli_available": False,
                "samples_per_optimizer_update": (
                    config.train_batch_size + config.k4_replay_batch_size
                ) * config.gradient_accumulation_steps,
                "arc_samples_per_optimizer_update": config.train_batch_size
                * config.gradient_accumulation_steps,
                "k4_replay_samples_per_optimizer_update": (
                    config.k4_replay_batch_size
                    * config.gradient_accumulation_steps
                ),
                "initialization_mode": model_config["initialization_mode"],
                "gated_context_warm_start": effective_warm_start_report,
                "gated_context_warm_start_source": effective_warm_start_source,
                "k4_replay_contract": (
                    k4_replay.contract() if k4_replay is not None else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "split_manifest.json").write_text(
        json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("environment=" + json.dumps(environment, sort_keys=True), flush=True)
    print(
        f"global_train_tasks={len(train_ids)} development_tasks={len(dev_ids)} "
        f"train_batch_size={config.train_batch_size} "
        f"samples_per_optimizer_update={(config.train_batch_size + config.k4_replay_batch_size) * config.gradient_accumulation_steps} "
        f"dense_recursive_steps={config.dense_recursive_steps} "
        f"dense_recursive_loss_weighting={config.dense_recursive_loss_weighting} "
        f"dense_hsw_lambda={config.dense_hsw_lambda} "
        f"dense_conditioning_mode={config.dense_conditioning_mode} "
        f"dense_relational_memory_mode={config.dense_relational_memory_mode} "
        f"dense_relational_gate_initial_value={config.dense_relational_gate_initial_value} "
        f"dense_shape_decoder_mode={config.dense_shape_decoder_mode} "
        f"vit5_initialization={config.vit5_training_mode} "
        f"initialization_mode={model_config['initialization_mode']} "
        f"k4_replay_batch_size={config.k4_replay_batch_size} "
        f"k4_replay_loss_fraction={config.k4_replay_loss_fraction if k4_replay is not None else 0.0} "
        "solutions_loaded=false",
        flush=True,
    )

    latest_k4_full: dict[str, Any] | None = None
    latest_k4_record: dict[str, Any] | None = None

    def k4_full_payload(
        evaluation: Mapping[str, Any],
        *,
        optimizer_update: int,
        compact_record: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "record_type": "clean_global_k4_replay_retention_full",
            "optimizer_update": int(optimizer_update),
            "pool_manifest_sha256": (
                k4_replay.pool_manifest_sha256 if k4_replay is not None else None
            ),
            "generator_schema": (
                k4_replay.contract()["generator_schema"]
                if k4_replay is not None
                else None
            ),
            "retention_criteria": copy.deepcopy(
                compact_record.get("retention_criteria")
            ),
            "retention_passed": bool(
                compact_record.get("retention_passed", False)
            ),
            "evaluation": copy.deepcopy(dict(evaluation)),
            "solutions_loaded": False,
        }

    def k4_validation_callback(optimizer_update: int) -> dict[str, Any]:
        nonlocal latest_k4_full, latest_k4_record
        if k4_replay is None:
            raise AssertionError("K4 validation callback invoked while replay is disabled.")
        record, latest_k4_full = k4_replay.evaluate(
            model, device, optimizer_update=optimizer_update
        )
        latest_k4_record = record
        if optimizer_update == 0:
            (output / "k4_retention_initial.json").write_text(
                json.dumps(
                    k4_full_payload(
                        latest_k4_full,
                        optimizer_update=0,
                        compact_record=record,
                    ),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        print(
            "k4_retention=" + json.dumps(record, sort_keys=True),
            flush=True,
        )
        return record

    def save_callback(
        kind: str,
        saved_model: Any,
        optimizer: torch.optim.Optimizer,
        training_state: dict[str, Any],
    ) -> None:
        save_function = (
            save_vit5_hybrid_arc_full_checkpoint
            if getattr(saved_model, "vit5_initialization", None)
            == VIT5_INITIALIZATION_RANDOM_SCRATCH
            else save_vit5_hybrid_lightweight_checkpoint
        )
        save_function(
            output / kind,
            saved_model,
            tokenizer,
            token_to_id,
            model_config,
            training_state=training_state,
            optimizer=optimizer,
        )

    metrics_path = output / "metrics.jsonl"
    with metrics_path.open("a", encoding="utf-8", buffering=1) as metrics:
        summary = train_clean_global_dense(
            model,
            train_tasks,
            dev_tasks,
            split,
            device,
            config,
            resume_training_state=resume_state,
            resume_optimizer_state=resume_optimizer,
            checkpoint_callback=save_callback,
            metrics_handle=metrics,
            k4_replay_sampler=(k4_replay.sampler if k4_replay is not None else None),
            k4_replay_prepare_samples=(
                k4_replay.prepare_samples if k4_replay is not None else None
            ),
            k4_replay_validation_callback=(
                k4_validation_callback if k4_replay is not None else None
            ),
            k4_replay_contract=(
                k4_replay.contract() if k4_replay is not None else None
            ),
        )
    if k4_replay is not None:
        if latest_k4_full is None or latest_k4_record is None:
            latest_k4_record, latest_k4_full = k4_replay.evaluate(
                model,
                device,
                optimizer_update=int(summary["optimizer_updates_completed"]),
            )
        (output / "k4_retention_final.json").write_text(
            json.dumps(
                k4_full_payload(
                    latest_k4_full,
                    optimizer_update=int(summary["optimizer_updates_completed"]),
                    compact_record=latest_k4_record,
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    summary.update(
        {
            "split_manifest_sha256": split["split_manifest_sha256"],
            "latest_checkpoint": str(output / "latest"),
            "best_checkpoint": str(output / "best_development_loo"),
            "best_retention_constrained_checkpoint": (
                str(output / "best_retention_constrained_development_loo")
                if summary.get("best_retention_constrained_development_metrics")
                is not None
                else None
            ),
            "environment": environment,
            "initialization_mode": model_config["initialization_mode"],
            "gated_context_warm_start": effective_warm_start_report,
            "warm_start_source": effective_warm_start_source,
        }
    )
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("summary=" + json.dumps(summary, sort_keys=True), flush=True)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
