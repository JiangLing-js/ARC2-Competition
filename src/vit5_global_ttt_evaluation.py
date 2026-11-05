"""Leakage-safe global 0-step and per-task TTT evaluation orchestration."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .vit5_overfit_diagnostics import compare_grids
from .vit5_per_task_full_adaptation import (
    _attach_dense_cpu_values,
)
from .vit5_per_task_ttt import (
    D4_TRANSFORM_NAMES,
    PerTaskTTTConfig,
    _autocast,
    _dense_training_forward,
    augment_leave_one_out_sample,
    build_leave_one_out_samples,
    predict_task_tests,
    restore_global_base,
    snapshot_state,
    state_sha256,
    validate_transform_names,
    vit5_lora_named_parameters,
)
from .vit5_hybrid_batch import build_vit5_dense_batch
from .vit5_varc_style_ttt import (
    PROTOCOL_NAME as VARC_STYLE_CORE_PROTOCOL,
    build_varc_style_510_view_specs,
    predict_task_tests_varc_style_510,
    select_whole_grid_top2_vote,
)
from .vitarc_adapter import VitarcGridAdapter


FROZEN_EVALUATION_SCHEMA_VERSION = 3
SUPPORTED_FROZEN_EVALUATION_SCHEMA_VERSIONS = (1, 2, 3)
STANDARD_D4_PREDICTION_PROTOCOL = "standard_d4_v1"
VARC_STYLE_510_PREDICTION_PROTOCOL = "varc_inspired_raw_grid_510_v1"
PREDICTION_PROTOCOLS = (
    STANDARD_D4_PREDICTION_PROTOCOL,
    VARC_STYLE_510_PREDICTION_PROTOCOL,
)
TTT_SCOPES = ("vit5_lora_only", "lora_plus_heads", "dense_full")
TTT_AUGMENTATION_PROTOCOLS = ("dynamic_d4_v1", "varc_auxiliary_51_v1")
TTT_CHECKPOINT_POLICIES = ("best_loo", "final")


@dataclass(frozen=True)
class TTTBranchConfig:
    name: str
    scope: str
    updates: int
    lr: float = 1e-5
    min_lr: float = 1e-6
    warmup_updates: int = 20
    train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: float = 1.0
    weight_decay: float = 0.0
    validate_every_updates: int = 25
    early_stopping_patience: int = 20
    seed: int = 0
    transform_names: tuple[str, ...] = D4_TRANSFORM_NAMES
    shuffle_demonstrations: bool = True
    color_permutation: bool = False
    augmentation_protocol: str = "dynamic_d4_v1"
    epochs: int = 0
    attempts: int = 1
    warmup_epochs: int = 0
    checkpoint_policy: str = "best_loo"

    def validate(self) -> None:
        if self.scope not in TTT_SCOPES:
            raise ValueError(f"Unsupported TTT scope: {self.scope}")
        if self.updates < 0:
            raise ValueError("TTT updates must be non-negative.")
        if self.epochs < 0:
            raise ValueError("TTT epochs must be non-negative.")
        if self.updates and self.epochs:
            raise ValueError("Choose either fixed TTT updates or epoch-based TTT, not both.")
        if (self.updates or self.epochs) and (
            self.lr <= 0 or self.min_lr < 0 or self.min_lr > self.lr
        ):
            raise ValueError("Invalid TTT learning-rate range.")
        if self.attempts < 1:
            raise ValueError("TTT attempts must be at least one.")
        if self.warmup_epochs < 0 or self.warmup_epochs > self.epochs:
            raise ValueError("TTT warmup_epochs must be between zero and epochs.")
        if self.checkpoint_policy not in TTT_CHECKPOINT_POLICIES:
            raise ValueError(
                f"Unsupported TTT checkpoint policy: {self.checkpoint_policy}"
            )
        if self.train_batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("Invalid TTT batch/accumulation settings.")
        if (
            self.validate_every_updates < 1
            or self.early_stopping_patience < 1
            or self.gradient_clip_norm <= 0
        ):
            raise ValueError("Invalid TTT validation/clipping settings.")
        validate_transform_names(self.transform_names)
        if not isinstance(self.color_permutation, bool):
            raise ValueError("TTT color_permutation must be boolean.")
        if self.augmentation_protocol not in TTT_AUGMENTATION_PROTOCOLS:
            raise ValueError(
                f"Unsupported TTT augmentation protocol: {self.augmentation_protocol}"
            )
        if self.augmentation_protocol == "varc_auxiliary_51_v1" and not self.color_permutation:
            raise ValueError(
                "varc_auxiliary_51_v1 requires color_permutation=true."
            )
        if self.epochs:
            if self.augmentation_protocol != "varc_auxiliary_51_v1":
                raise ValueError(
                    "Epoch-based TTT currently requires varc_auxiliary_51_v1."
                )
            if self.gradient_accumulation_steps != 1:
                raise ValueError(
                    "Exact VARC epoch semantics require gradient_accumulation_steps=1."
                )


def varc_epoch_training_budget(
    demonstration_count: int,
    epochs: int,
    train_batch_size: int,
) -> dict[str, int]:
    """Return the exact VARC-style per-attempt training budget.

    Every epoch visits each of the 51 reversible variants for every
    leave-one-demonstration-out sample exactly once.  ``drop_last`` is false,
    so the final optimizer batch may be smaller than ``train_batch_size``.
    """
    if demonstration_count < 1:
        raise ValueError("demonstration_count must be at least one.")
    if epochs < 0:
        raise ValueError("epochs must be non-negative.")
    if train_batch_size < 1:
        raise ValueError("train_batch_size must be at least one.")
    examples_per_epoch = 51 * demonstration_count
    optimizer_updates_per_epoch = math.ceil(examples_per_epoch / train_batch_size)
    return {
        "demonstration_count": int(demonstration_count),
        "auxiliary_variants_per_demonstration": 51,
        "examples_per_epoch": int(examples_per_epoch),
        "optimizer_updates_per_epoch": int(optimizer_updates_per_epoch),
        "epochs": int(epochs),
        "optimizer_updates_per_attempt": int(optimizer_updates_per_epoch * epochs),
        "sample_presentations_per_attempt": int(examples_per_epoch * epochs),
    }


def full_nonzero_palette_mapping(rng: random.Random) -> dict[int, int]:
    """Return a target-independent ARC palette bijection with color 0 fixed.

    Defining the mapping on the complete 0..9 palette is important at test
    time: it neither consults a hidden output palette nor leaves an unseen
    predicted color without an exact inverse.
    """

    targets = list(range(1, 10))
    rng.shuffle(targets)
    return {0: 0, **dict(zip(range(1, 10), targets))}


def path_sha256(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    files = [source] if source.is_file() else sorted(
        (item for item in source.rglob("*") if item.is_file()),
        key=lambda item: str(item.relative_to(source)),
    )
    digest = hashlib.sha256()
    for item in files:
        relative = item.name if source.is_file() else str(item.relative_to(source))
        digest.update(relative.replace("\\", "/").encode("utf-8"))
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def frozen_config_sha256(config: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in config.items() if key != "frozen_config_sha256"}
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def create_frozen_evaluation_config(
    global_checkpoint: str | Path,
    split_manifest_sha256: str,
    development_results: str | Path,
    branches: Sequence[TTTBranchConfig],
    lambda_component: float,
    lambda_shape: float,
    tta: bool = False,
    precision: str = "bf16",
    tta_transform_names: Sequence[str] = D4_TRANSFORM_NAMES,
    global_training_split_manifest_sha256: str | None = None,
    external_final_challenge_set: bool = False,
    prediction_protocol: str = STANDARD_D4_PREDICTION_PROTOCOL,
    prediction_view_seed: int = 0,
    prediction_batch_size: int = 4,
    ensemble_global_zero_step: bool = False,
) -> dict[str, Any]:
    for branch in branches:
        branch.validate()
    training_split_sha = (
        str(global_training_split_manifest_sha256)
        if global_training_split_manifest_sha256 is not None
        else str(split_manifest_sha256)
    )
    if external_final_challenge_set:
        if training_split_sha == str(split_manifest_sha256):
            raise ValueError(
                "An external final challenge set must use a split manifest "
                "different from the global-training split manifest."
            )
    elif training_split_sha != str(split_manifest_sha256):
        raise ValueError(
            "A different global-training split requires "
            "external_final_challenge_set=true."
        )
    if prediction_protocol not in PREDICTION_PROTOCOLS:
        raise ValueError(f"Unsupported prediction protocol: {prediction_protocol}")
    if int(prediction_batch_size) < 1:
        raise ValueError("prediction_batch_size must be >= 1.")
    varc_style = prediction_protocol == VARC_STYLE_510_PREDICTION_PROTOCOL
    if any(branch.attempts > 1 for branch in branches) and not varc_style:
        raise ValueError("Multiple TTT attempts require the VARC-style prediction protocol.")
    config: dict[str, Any] = {
        "evaluation_schema_version": FROZEN_EVALUATION_SCHEMA_VERSION,
        "frozen_evaluation_config": True,
        "global_checkpoint": str(global_checkpoint),
        "global_checkpoint_sha256": path_sha256(global_checkpoint),
        "split_manifest_sha256": str(split_manifest_sha256),
        "global_training_split_manifest_sha256": training_split_sha,
        "external_final_challenge_set": bool(external_final_challenge_set),
        "lambda_component": float(lambda_component),
        "lambda_shape": float(lambda_shape),
        "ttt_branches": [asdict(branch) for branch in branches],
        "tta": bool(tta),
        "tta_transform_names": list(validate_transform_names(tta_transform_names)),
        "prediction_protocol": prediction_protocol,
        "prediction_view_seed": int(prediction_view_seed),
        "prediction_batch_size": int(prediction_batch_size),
        "ensemble_global_zero_step": bool(ensemble_global_zero_step),
        "prediction_view_count": 510 if varc_style else (
            len(tuple(tta_transform_names)) if tta else 1
        ),
        "candidate_selection": (
            "whole_grid_top2_vote_v1" if varc_style else "whole_grid_vote"
        ),
        "candidate_top_k": 2 if varc_style else 1,
        "ttt_attempt_ensemble": (
            "raw_candidate_whole_grid_vote_v1"
            if any(branch.attempts > 1 for branch in branches)
            else "none"
        ),
        "varc_style_exact_official_canvas_protocol": False,
        "query_target_fields_allowed_during_ttt_or_inference": False,
        "precision": str(precision),
        "checkpoint_selection": "per_branch_checkpoint_policy_v1",
        "development_results_sha256": path_sha256(development_results),
        "official_test_metrics_used_for_selection": False,
    }
    config["frozen_config_sha256"] = frozen_config_sha256(config)
    return config


def validate_frozen_evaluation_config(
    config: Mapping[str, Any],
    checkpoint_path: str | Path,
    expected_split_sha256: str,
    require_frozen: bool = True,
) -> list[TTTBranchConfig]:
    schema = int(config.get("evaluation_schema_version", 0))
    if schema not in SUPPORTED_FROZEN_EVALUATION_SCHEMA_VERSIONS:
        raise ValueError("Unsupported frozen evaluation schema.")
    if require_frozen and config.get("frozen_evaluation_config") is not True:
        raise ValueError("Final held-out evaluation requires frozen_evaluation_config=true.")
    if config.get("frozen_config_sha256") != frozen_config_sha256(config):
        raise ValueError("Frozen evaluation config SHA256 mismatch.")
    if config.get("split_manifest_sha256") != expected_split_sha256:
        raise ValueError("Frozen evaluation config references a different task split.")
    external_final = config.get("external_final_challenge_set", False)
    if not isinstance(external_final, bool):
        raise ValueError("external_final_challenge_set must be boolean.")
    training_split_sha = config.get(
        "global_training_split_manifest_sha256",
        config.get("split_manifest_sha256"),
    )
    if not isinstance(training_split_sha, str) or not training_split_sha:
        raise ValueError(
            "Frozen evaluation config is missing the global-training split SHA256."
        )
    if external_final:
        if training_split_sha == expected_split_sha256:
            raise ValueError(
                "External final evaluation must bind distinct training and "
                "evaluation split manifests."
            )
    elif training_split_sha != expected_split_sha256:
        raise ValueError(
            "Non-external evaluation must use the global-training task split."
        )
    protocol = config.get("prediction_protocol", STANDARD_D4_PREDICTION_PROTOCOL)
    if protocol not in PREDICTION_PROTOCOLS:
        raise ValueError("Frozen evaluation prediction protocol is invalid.")
    expected_selection = (
        "whole_grid_top2_vote_v1"
        if protocol == VARC_STYLE_510_PREDICTION_PROTOCOL
        else "whole_grid_vote"
    )
    if config.get("candidate_selection") != expected_selection:
        raise ValueError(
            "Frozen evaluation candidate selection does not match its "
            "prediction protocol."
        )
    if int(config.get("prediction_batch_size", 1)) < 1:
        raise ValueError("Frozen prediction_batch_size must be >= 1.")
    if not isinstance(config.get("ensemble_global_zero_step", False), bool):
        raise ValueError("ensemble_global_zero_step must be boolean.")
    if protocol == VARC_STYLE_510_PREDICTION_PROTOCOL:
        if schema < 2 or int(config.get("prediction_view_count", 0)) != 510:
            raise ValueError("VARC-style protocol requires schema v2 and 510 views.")
        if int(config.get("candidate_top_k", 0)) != 2:
            raise ValueError("VARC-style protocol requires candidate_top_k=2.")
        if config.get("varc_style_exact_official_canvas_protocol") is not False:
            raise ValueError("Raw-grid VARC-style protocol must not claim exact canvas semantics.")
        if config.get("query_target_fields_allowed_during_ttt_or_inference") is not False:
            raise ValueError("Query targets are forbidden during TTT/inference.")
    if config.get("precision", "bf16") not in {"fp32", "fp16", "bf16"}:
        raise ValueError("Frozen evaluation precision is invalid.")
    validate_transform_names(config.get("tta_transform_names", D4_TRANSFORM_NAMES))
    actual_checkpoint_sha = path_sha256(checkpoint_path)
    if config.get("global_checkpoint_sha256") != actual_checkpoint_sha:
        raise ValueError("Global checkpoint SHA256 differs from the frozen evaluation config.")
    branches = [
        TTTBranchConfig(
            **{
                **value,
                "transform_names": tuple(value.get("transform_names", D4_TRANSFORM_NAMES)),
            }
        )
        for value in config.get("ttt_branches", [])
    ]
    if len({branch.name for branch in branches}) != len(branches):
        raise ValueError("Frozen evaluation branch names must be unique.")
    for branch in branches:
        branch.validate()
        if branch.attempts > 1 and protocol != VARC_STYLE_510_PREDICTION_PROTOCOL:
            raise ValueError(
                "Multiple TTT attempts require the VARC-style prediction protocol."
            )
    expected_attempt_ensemble = (
        "raw_candidate_whole_grid_vote_v1"
        if any(branch.attempts > 1 for branch in branches)
        else "none"
    )
    if schema >= 3 and config.get("ttt_attempt_ensemble") != expected_attempt_ensemble:
        raise ValueError("Frozen TTT attempt ensemble metadata is inconsistent.")
    return branches


def resolve_ttt_scope(model: torch.nn.Module, scope: str) -> tuple[str, ...]:
    if scope not in TTT_SCOPES:
        raise ValueError(f"Unsupported TTT scope: {scope}")
    scratch = getattr(model, "vit5_initialization", None) == "random_scratch"
    if scratch:
        if scope in {"vit5_lora_only", "lora_plus_heads"}:
            raise ValueError(
                f"TTT scope {scope} requires injected ViT-5 LoRA, but this "
                "checkpoint uses random_scratch. Use dense_full or global 0-step."
            )
        # Scratch dense_full deliberately includes every ARC encoder parameter,
        # not merely the old embedding/LoRA subset.  Selection is name-based so
        # branch ordering cannot inherit a previously narrowed requires_grad set.
        selected = {
            name
            for name, _ in model.named_parameters()
            if name.startswith(("arc_encoder.", "dense_decoder."))
            and not (
                getattr(model, "object_embedding_mode", "connected_components")
                == "none"
                and name
                == "arc_encoder.grid_encoder.arc_embedding.object_embedding.weight"
            )
        }
    else:
        lora = {name for name, _ in vit5_lora_named_parameters(model)}
        if not lora:
            raise ValueError("Global checkpoint has no ViT-5 LoRA parameters.")
        if scope == "vit5_lora_only":
            selected = lora
        elif scope == "lora_plus_heads":
            selected = lora | {
                name
                for name, _ in model.named_parameters()
                if name.startswith("dense_decoder.")
            }
        else:
            # Resolve from names, not the model's current requires_grad flags.
            # A prior narrower branch must not shrink dense_full accidentally.
            dense_full_prefixes = (
                "arc_encoder.grid_encoder.arc_embedding.",
                "arc_encoder.task_fusion.",
                "arc_encoder.task_summary",
                "arc_encoder.cell_",
                "dense_decoder.",
            )
            selected = lora | {
                name
                for name, _ in model.named_parameters()
                if name.startswith(dense_full_prefixes)
                and not (
                    getattr(model, "object_embedding_mode", "connected_components")
                    == "none"
                    and name
                    == "arc_encoder.grid_encoder.arc_embedding.object_embedding.weight"
                )
            }
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in selected)
    if not selected:
        raise ValueError(f"TTT scope {scope} resolved to zero parameters.")
    forbidden = sorted(
        name
        for name in selected
        if name.startswith(("t5.", "encoder_to_t5.", "arc_token_delta."))
    )
    if forbidden:
        raise ValueError(f"TTT scope contains forbidden T5/AR parameters: {forbidden}")
    return tuple(sorted(selected))


def _scope_optimizer(
    model: torch.nn.Module, names: Sequence[str], branch: TTTBranchConfig
) -> torch.optim.AdamW:
    parameters = dict(model.named_parameters())
    selected = [parameters[name] for name in names]
    optimizer = torch.optim.AdamW(
        selected, lr=branch.lr, weight_decay=branch.weight_decay
    )
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if optimizer_ids != {id(parameters[name]) for name in names}:
        raise AssertionError("TTT optimizer parameter IDs differ from the selected scope.")
    return optimizer


def _branch_lr(
    branch: TTTBranchConfig,
    update: int,
    *,
    total_updates: int | None = None,
) -> float:
    if branch.warmup_updates and update <= branch.warmup_updates:
        return branch.lr * update / branch.warmup_updates
    resolved_total = branch.updates if total_updates is None else int(total_updates)
    decay = max(1, resolved_total - branch.warmup_updates)
    progress = min(1.0, max(0.0, (update - branch.warmup_updates) / decay))
    return branch.min_lr + 0.5 * (branch.lr - branch.min_lr) * (
        1 + math.cos(math.pi * progress)
    )


def _branch_epoch_lr(branch: TTTBranchConfig, epoch_index: int) -> float:
    """Cosine LR updated once per epoch, without VARC's off-by-one epoch."""
    if epoch_index < 0 or epoch_index >= branch.epochs:
        raise ValueError("epoch_index is outside the configured TTT epoch range.")
    if branch.warmup_epochs and epoch_index < branch.warmup_epochs:
        return branch.lr * (epoch_index + 1) / branch.warmup_epochs
    decay_epochs = max(1, branch.epochs - branch.warmup_epochs)
    decay_index = epoch_index - branch.warmup_epochs
    progress = min(1.0, max(0.0, decay_index / max(1, decay_epochs - 1)))
    return branch.min_lr + 0.5 * (branch.lr - branch.min_lr) * (
        1 + math.cos(math.pi * progress)
    )


def _validation_rank(metrics: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(metrics["exact_count"]),
        -float(metrics["target_error_cells"]),
        float(metrics["macro_pixel_accuracy"]),
        -float(metrics["validation_loss"]),
        -float(metrics["optimizer_update"]),
    )


@torch.no_grad()
def _identity_loo_validation(
    model: Any,
    samples: Sequence[Mapping[str, Any]],
    device: torch.device,
    precision: str,
    max_grids: int,
    optimizer_update: int,
) -> dict[str, Any]:
    model.eval()
    adapter = VitarcGridAdapter()
    records: list[dict[str, Any]] = []
    losses: list[float] = []
    for sample in samples:
        prepared = adapter.prepare_dttt_sample(dict(sample))
        _attach_dense_cpu_values(prepared)
        batch = build_vit5_dense_batch([prepared], device=device, max_grids=max_grids)
        with _autocast(device, precision):
            output = _dense_training_forward(model, batch)
        prediction = model.predict_dense_grid(encoder_outputs=output.encoder_cache)[0]["grid"]
        comparison = compare_grids(prediction, sample["target"])
        records.append(comparison)
        losses.append(float(output.dense_loss.detach().float().cpu()))
    return {
        "optimizer_update": optimizer_update,
        "sample_count": len(records),
        "exact_count": sum(int(record["exact"]) for record in records),
        "target_error_cells": sum(int(record["target_error_cells"]) for record in records),
        "macro_pixel_accuracy": sum(float(record["pixel_accuracy"]) for record in records)
        / len(records),
        "validation_loss": sum(losses) / len(losses),
        "official_test_output_read": False,
    }


def _trainable_state(model: torch.nn.Module, names: Sequence[str]) -> dict[str, torch.Tensor]:
    parameters = dict(model.named_parameters())
    return {name: parameters[name].detach().cpu().clone() for name in names}


def _load_trainable_state(
    model: torch.nn.Module, names: Sequence[str], state: Mapping[str, torch.Tensor]
) -> None:
    if set(names) != set(state):
        raise ValueError("Task-specific TTT state key mismatch.")
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name in names:
            parameters[name].copy_(state[name].to(parameters[name].device, parameters[name].dtype))


def adapt_task_branch(
    model: Any,
    task: Mapping[str, Any],
    task_id: str,
    tokenizer: Any,
    device: torch.device,
    precision: str,
    max_grids: int,
    base_state: Mapping[str, torch.Tensor],
    base_sha256: str,
    branch: TTTBranchConfig,
    prediction_transform_names: Sequence[str] = ("identity",),
    prediction_protocol: str = STANDARD_D4_PREDICTION_PROTOCOL,
    prediction_view_seed: int = 0,
    prediction_batch_size: int = 4,
    retain_vote_candidates: bool = False,
    ttt_attempt_index: int = 0,
    ttt_attempt_count: int = 1,
) -> dict[str, Any]:
    """Adapt once from global state, then predict every test input with one state."""
    branch.validate()
    restore_global_base(model, base_state, base_sha256)
    start_sha = state_sha256(snapshot_state(model))
    if start_sha != base_sha256:
        raise AssertionError("TTT branch did not start from the immutable global base.")
    names = resolve_ttt_scope(model, branch.scope)
    optimizer = _scope_optimizer(model, names, branch)
    loo = build_leave_one_out_samples(task, task_id)
    if not loo:
        raise ValueError(f"Task {task_id} has no train demonstrations.")
    epoch_budget = (
        varc_epoch_training_budget(
            len(loo), branch.epochs, branch.train_batch_size
        )
        if branch.epochs
        else None
    )
    requested_updates = (
        int(epoch_budget["optimizer_updates_per_attempt"])
        if epoch_budget is not None
        else int(branch.updates)
    )
    if requested_updates and not branch.epochs:
        for group in optimizer.param_groups:
            group["lr"] = _branch_lr(
                branch, 1, total_updates=requested_updates
            )
    step0 = _identity_loo_validation(model, loo, device, precision, max_grids, 0)
    best_metrics = step0
    best_state = (
        _trainable_state(model, names)
        if branch.checkpoint_policy == "best_loo"
        else None
    )
    history = [step0]
    validations_without_improvement = 0
    early_stopped = False
    started = time.perf_counter()
    updates_completed = 0
    epochs_completed = 0
    training_sample_presentations = 0
    if len(loo) >= 2 and requested_updates:
        rng = random.Random(branch.seed + int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16))
        adapter = VitarcGridAdapter()
        auxiliary_schedule: list[tuple[int, Any]] = []
        auxiliary_cursor = 0
        if branch.augmentation_protocol == "varc_auxiliary_51_v1":
            auxiliary_specs = [
                spec
                for spec in build_varc_style_510_view_specs(
                    task_id, 0, seed=branch.seed
                )
                if spec.attempt_index == 0
            ]
            if len(auxiliary_specs) != 51:
                raise AssertionError("VARC-style TTT must expose exactly 51 auxiliary tasks.")
            auxiliary_schedule = [
                (sample_index, spec)
                for sample_index in range(len(loo))
                for spec in auxiliary_specs
            ]
            rng.shuffle(auxiliary_schedule)

        def prepare_entry(sample_index: int, auxiliary: Any | None) -> dict[str, Any]:
            nonlocal training_sample_presentations
            sample = loo[sample_index]
            if auxiliary is None:
                transform = branch.transform_names[
                    rng.randrange(len(branch.transform_names))
                ]
                color_mapping = (
                    full_nonzero_palette_mapping(rng)
                    if branch.color_permutation
                    else {0: 0}
                )
            else:
                transform = auxiliary.transform
                color_mapping = auxiliary.mapping_dict()
            augmented = augment_leave_one_out_sample(
                sample,
                transform,
                rng,
                shuffle_demonstrations=branch.shuffle_demonstrations,
                color_mapping=color_mapping,
            )
            prepared = adapter.prepare_dttt_sample(augmented)
            _attach_dense_cpu_values(prepared)
            training_sample_presentations += 1
            return prepared

        def optimizer_update(
            microbatches: Sequence[Sequence[tuple[int, Any | None]]],
        ) -> None:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for entries in microbatches:
                samples = [
                    prepare_entry(sample_index, auxiliary)
                    for sample_index, auxiliary in entries
                ]
                batch = build_vit5_dense_batch(samples, device=device, max_grids=max_grids)
                with _autocast(device, precision):
                    output = _dense_training_forward(model, batch)
                    loss = output.dense_loss
                    if loss is None or not bool(torch.isfinite(loss).item()):
                        raise ValueError(f"Non-finite TTT loss for task {task_id}.")
                    scaled = loss / len(microbatches)
                scaled.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                branch.gradient_clip_norm,
            )
            optimizer.step()

        def validate_update(*, force: bool = False) -> bool:
            nonlocal best_metrics, best_state
            nonlocal validations_without_improvement, early_stopped
            if branch.checkpoint_policy != "best_loo":
                return False
            if not force and updates_completed % branch.validate_every_updates:
                return False
            if history[-1]["optimizer_update"] == updates_completed:
                return early_stopped
            metrics = _identity_loo_validation(
                model, loo, device, precision, max_grids, updates_completed
            )
            history.append(metrics)
            if _validation_rank(metrics) > _validation_rank(best_metrics):
                best_metrics = metrics
                best_state = _trainable_state(model, names)
                validations_without_improvement = 0
            else:
                validations_without_improvement += 1
            if validations_without_improvement >= branch.early_stopping_patience:
                early_stopped = True
            return early_stopped

        if branch.epochs:
            if len(auxiliary_schedule) != 51 * len(loo):
                raise AssertionError("Epoch TTT requires exactly 51 samples per demo.")
            for epoch_index in range(branch.epochs):
                epoch_schedule = list(auxiliary_schedule)
                rng.shuffle(epoch_schedule)
                epoch_lr = _branch_epoch_lr(branch, epoch_index)
                for group in optimizer.param_groups:
                    group["lr"] = epoch_lr
                completed_epoch = True
                for start in range(0, len(epoch_schedule), branch.train_batch_size):
                    entries = epoch_schedule[start : start + branch.train_batch_size]
                    optimizer_update([entries])
                    updates_completed += 1
                    if validate_update():
                        completed_epoch = False
                        break
                if completed_epoch:
                    epochs_completed += 1
                if early_stopped:
                    break
        else:
            for update in range(1, requested_updates + 1):
                microbatches: list[list[tuple[int, Any | None]]] = []
                for _ in range(branch.gradient_accumulation_steps):
                    entries: list[tuple[int, Any | None]] = []
                    for _ in range(branch.train_batch_size):
                        if auxiliary_schedule:
                            if auxiliary_cursor >= len(auxiliary_schedule):
                                rng.shuffle(auxiliary_schedule)
                                auxiliary_cursor = 0
                            entries.append(auxiliary_schedule[auxiliary_cursor])
                            auxiliary_cursor += 1
                        else:
                            entries.append((rng.randrange(len(loo)), None))
                    microbatches.append(entries)
                optimizer_update(microbatches)
                updates_completed = update
                if update < requested_updates:
                    next_lr = _branch_lr(
                        branch,
                        update + 1,
                        total_updates=requested_updates,
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = next_lr
                if validate_update():
                    break

        if branch.checkpoint_policy == "best_loo":
            validate_update(force=True)
            if best_state is None:
                raise AssertionError("best_loo TTT did not preserve a selected state.")
            _load_trainable_state(model, names, best_state)
        else:
            final_metrics = _identity_loo_validation(
                model, loo, device, precision, max_grids, updates_completed
            )
            if history[-1]["optimizer_update"] != updates_completed:
                history.append(final_metrics)
            best_metrics = final_metrics
    prediction_config = PerTaskTTTConfig(
        max_optimizer_updates=0,
        precision=precision,
        max_grids=max_grids,
    )
    if prediction_protocol == VARC_STYLE_510_PREDICTION_PROTOCOL:
        if VARC_STYLE_CORE_PROTOCOL != prediction_protocol:
            raise AssertionError("VARC-style core and frozen protocol names differ.")
        predictions = predict_task_tests_varc_style_510(
            model,
            task,
            task_id,
            tokenizer,
            device,
            precision=precision,
            max_grids=max_grids,
            seed=prediction_view_seed,
            view_batch_size=prediction_batch_size,
            return_candidates=retain_vote_candidates,
        )
    else:
        predictions = predict_task_tests(
            model,
            task,
            task_id,
            tokenizer,
            device,
            prediction_config,
            prediction_transform_names,
        )
    return {
        "branch": branch.name,
        "scope": branch.scope,
        "augmentation_protocol": branch.augmentation_protocol,
        "checkpoint_policy": branch.checkpoint_policy,
        "color_permutation": branch.color_permutation,
        "requested_updates": requested_updates,
        "requested_epochs": branch.epochs,
        "updates_completed": updates_completed,
        "epochs_completed": epochs_completed,
        "epoch_training_budget": epoch_budget,
        "ttt_attempt_index": int(ttt_attempt_index),
        "ttt_attempt_count": int(ttt_attempt_count),
        "best_optimizer_update": int(best_metrics["optimizer_update"]),
        "fell_back_to_step0": int(best_metrics["optimizer_update"]) == 0,
        "early_stopped": early_stopped,
        "stopping_update": updates_completed,
        "step0_identity_loo": step0,
        "best_identity_loo": best_metrics,
        "validation_history": history,
        "global_start_state_sha256": start_sha,
        "trainable_parameter_count": sum(
            dict(model.named_parameters())[name].numel() for name in names
        ),
        "trainable_parameter_names": list(names),
        "test_predictions": predictions,
        "test_input_count": len(task.get("test", [])),
        "ttt_training_sample_presentations": training_sample_presentations,
        "approximate_auxiliary_epochs": (
            training_sample_presentations / (51 * len(loo))
            if branch.augmentation_protocol == "varc_auxiliary_51_v1"
            else None
        ),
        "prediction_protocol": prediction_protocol,
        "single_adaptation_state_shared_by_all_test_inputs": True,
        "official_test_output_read": False,
        "elapsed_seconds": time.perf_counter() - started,
    }


@torch.no_grad()
def global_zero_step_predictions(
    model: Any,
    task: Mapping[str, Any],
    task_id: str,
    tokenizer: Any,
    device: torch.device,
    precision: str,
    max_grids: int,
    base_state: Mapping[str, torch.Tensor],
    base_sha256: str,
    prediction_transform_names: Sequence[str] = ("identity",),
    prediction_protocol: str = STANDARD_D4_PREDICTION_PROTOCOL,
    prediction_view_seed: int = 0,
    prediction_batch_size: int = 4,
) -> dict[str, Any]:
    restore_global_base(model, base_state, base_sha256)
    config = PerTaskTTTConfig(max_optimizer_updates=0, precision=precision, max_grids=max_grids)
    if prediction_protocol == VARC_STYLE_510_PREDICTION_PROTOCOL:
        predictions = predict_task_tests_varc_style_510(
            model,
            task,
            task_id,
            tokenizer,
            device,
            precision=precision,
            max_grids=max_grids,
            seed=prediction_view_seed,
            view_batch_size=prediction_batch_size,
        )
    else:
        predictions = predict_task_tests(
            model, task, task_id, tokenizer, device, config, prediction_transform_names
        )
    return {
        "branch": "global_0_step",
        "scope": "none",
        "global_start_state_sha256": state_sha256(snapshot_state(model)),
        "test_predictions": predictions,
        "test_input_count": len(task.get("test", [])),
        "official_test_output_read": False,
        "prediction_protocol": prediction_protocol,
    }


def _merge_varc_ttt_attempts(
    base_branch: TTTBranchConfig,
    attempt_results: Sequence[Mapping[str, Any]],
    base_sha256: str,
) -> dict[str, Any]:
    """Merge every raw 510-view candidate across independent TTT attempts."""
    if len(attempt_results) < 2:
        raise ValueError("At least two TTT attempts are required for merged voting.")
    query_count = int(attempt_results[0]["test_input_count"])
    if any(int(result["test_input_count"]) != query_count for result in attempt_results):
        raise ValueError("TTT attempts disagree on the number of test inputs.")
    merged_predictions: list[dict[str, Any]] = []
    for test_index in range(query_count):
        combined_candidates: list[dict[str, Any]] = []
        manifest_hashes: list[str] = []
        per_attempt_counts: list[int] = []
        task_id: str | None = None
        for attempt_index, result in enumerate(attempt_results):
            prediction = result["test_predictions"][test_index]
            raw = prediction.pop("_vote_candidates", None)
            if not isinstance(raw, list) or not raw:
                raise ValueError("A TTT attempt did not retain its raw vote candidates.")
            per_attempt_counts.append(len(raw))
            manifest_hashes.append(str(prediction["view_manifest_sha256"]))
            task_id = str(prediction["task_id"])
            for candidate in raw:
                mutable = dict(candidate)
                original_view_index = int(mutable["view_index"])
                mutable["within_attempt_view_index"] = original_view_index
                mutable["ttt_attempt_index"] = attempt_index
                mutable["view_index"] = attempt_index * 510 + original_view_index
                mutable["is_reference_view"] = bool(
                    attempt_index == 0 and mutable.get("is_reference_view", False)
                )
                combined_candidates.append(mutable)
        voted = select_whole_grid_top2_vote(combined_candidates)
        voted.update(
            {
                "task_id": task_id,
                "test_index": int(test_index),
                "ttt_attempt_count": len(attempt_results),
                "candidates_per_attempt": per_attempt_counts,
                "attempt_view_manifest_sha256": manifest_hashes,
                "cross_attempt_vote": "raw_candidate_whole_grid_vote_v1",
                "auxiliary_view_count_per_attempt": 51,
                "inference_views_per_attempt": 510,
            }
        )
        merged_predictions.append(voted)
    return {
        "branch": f"{base_branch.name}_merged_{len(attempt_results)}attempts",
        "scope": base_branch.scope,
        "augmentation_protocol": base_branch.augmentation_protocol,
        "checkpoint_policy": base_branch.checkpoint_policy,
        "ttt_attempt_count": len(attempt_results),
        "source_attempt_branches": [str(result["branch"]) for result in attempt_results],
        "requested_epochs_per_attempt": base_branch.epochs,
        "requested_updates_per_attempt": [
            int(result["requested_updates"]) for result in attempt_results
        ],
        "updates_completed_all_attempts": sum(
            int(result["updates_completed"]) for result in attempt_results
        ),
        "ttt_training_sample_presentations_all_attempts": sum(
            int(result["ttt_training_sample_presentations"])
            for result in attempt_results
        ),
        "global_start_state_sha256": base_sha256,
        "test_predictions": merged_predictions,
        "test_input_count": query_count,
        "prediction_protocol": VARC_STYLE_510_PREDICTION_PROTOCOL,
        "single_adaptation_state_shared_by_all_test_inputs": True,
        "official_test_output_read": False,
    }


def evaluate_tasks_from_global_checkpoint(
    model: Any,
    tasks: Mapping[str, Mapping[str, Any]],
    tokenizer: Any,
    device: torch.device,
    precision: str,
    max_grids: int,
    branches: Sequence[TTTBranchConfig],
    tta: bool = False,
    tta_transform_names: Sequence[str] = D4_TRANSFORM_NAMES,
    prediction_protocol: str = STANDARD_D4_PREDICTION_PROTOCOL,
    prediction_view_seed: int = 0,
    prediction_batch_size: int = 4,
    ensemble_global_zero_step: bool = False,
) -> list[dict[str, Any]]:
    if prediction_protocol not in PREDICTION_PROTOCOLS:
        raise ValueError(f"Unsupported prediction protocol: {prediction_protocol}")
    if prediction_batch_size < 1:
        raise ValueError("prediction_batch_size must be >= 1.")
    base_state = snapshot_state(model)
    base_sha = state_sha256(base_state)
    results: list[dict[str, Any]] = []
    prediction_transforms = (
        validate_transform_names(tta_transform_names) if tta else ("identity",)
    )
    for task_id in sorted(tasks):
        task = tasks[task_id]
        zero_step_protocol = (
            prediction_protocol
            if ensemble_global_zero_step
            else STANDARD_D4_PREDICTION_PROTOCOL
        )
        branch_results = [
            global_zero_step_predictions(
                model, task, task_id, tokenizer, device, precision, max_grids,
                base_state, base_sha, prediction_transforms,
                zero_step_protocol, prediction_view_seed, prediction_batch_size,
            )
        ]
        for branch in branches:
            if branch.attempts > 1 and prediction_protocol != VARC_STYLE_510_PREDICTION_PROTOCOL:
                raise ValueError(
                    "Multiple TTT attempts require the VARC-style prediction protocol."
                )
            attempt_results: list[dict[str, Any]] = []
            for attempt_index in range(branch.attempts):
                attempt_branch = replace(
                    branch,
                    name=(
                        f"{branch.name}_attempt_{attempt_index}"
                        if branch.attempts > 1
                        else branch.name
                    ),
                    seed=branch.seed + attempt_index,
                    attempts=1,
                )
                attempt_results.append(
                    adapt_task_branch(
                        model, task, task_id, tokenizer, device, precision, max_grids,
                        base_state, base_sha, attempt_branch, prediction_transforms,
                        prediction_protocol, prediction_view_seed, prediction_batch_size,
                        retain_vote_candidates=branch.attempts > 1,
                        ttt_attempt_index=attempt_index,
                        ttt_attempt_count=branch.attempts,
                    )
                )
            branch_results.extend(attempt_results)
            if branch.attempts > 1:
                branch_results.append(
                    _merge_varc_ttt_attempts(branch, attempt_results, base_sha)
                )
        if any(result["global_start_state_sha256"] != base_sha for result in branch_results):
            raise AssertionError(f"Task {task_id} branch isolation audit failed.")
        results.append(
            {
                "task_id": task_id,
                "train_pair_count": len(task.get("train", [])),
                "test_input_count": len(task.get("test", [])),
                "global_base_state_sha256": base_sha,
                "branches": branch_results,
            }
        )
    restore_global_base(model, base_state, base_sha)
    return results


def score_fixed_predictions(
    task_results: Sequence[Mapping[str, Any]], solutions: Mapping[str, Any]
) -> dict[str, Any]:
    """Score only already-fixed predictions; this function has no model argument."""
    branch_records: dict[str, list[dict[str, Any]]] = {}
    solved_pass1: dict[str, int] = {}
    solved_pass2: dict[str, int] = {}
    for task in task_results:
        task_id = str(task["task_id"])
        targets = solutions[task_id]
        for branch in task["branches"]:
            name = str(branch["branch"])
            records = branch_records.setdefault(name, [])
            task_exact_pass1: list[bool] = []
            task_exact_pass2: list[bool] = []
            for test_index, prediction in enumerate(branch["test_predictions"]):
                grid = prediction.get("final_prediction", prediction.get("prediction"))
                if grid is None:
                    raise ValueError("Fixed prediction record has no final grid.")
                comparison = compare_grids(grid, targets[test_index])
                alternatives: list[Any] = [grid]
                for top in prediction.get("top_predictions", []):
                    candidate_grid = top.get("prediction") if isinstance(top, Mapping) else top
                    if candidate_grid is not None:
                        alternatives.append(candidate_grid)
                second = prediction.get("second_prediction")
                if second is not None:
                    alternatives.append(second)
                distinct: list[Any] = []
                seen: set[tuple[tuple[int, ...], ...]] = set()
                for candidate_grid in alternatives:
                    key = tuple(tuple(int(value) for value in row) for row in candidate_grid)
                    if key not in seen:
                        seen.add(key)
                        distinct.append(candidate_grid)
                    if len(distinct) == 2:
                        break
                pass2 = any(
                    bool(compare_grids(candidate_grid, targets[test_index])["exact"])
                    for candidate_grid in distinct
                )
                records.append(
                    {
                        "task_id": task_id,
                        "test_index": test_index,
                        **comparison,
                        "pass_at_1_exact": bool(comparison["exact"]),
                        "pass_at_2_exact": bool(pass2),
                        "top_k_distinct_candidate_count": len(distinct),
                    }
                )
                task_exact_pass1.append(bool(comparison["exact"]))
                task_exact_pass2.append(bool(pass2))
            solved_pass1[name] = solved_pass1.get(name, 0) + int(
                bool(task_exact_pass1) and all(task_exact_pass1)
            )
            solved_pass2[name] = solved_pass2.get(name, 0) + int(
                bool(task_exact_pass2) and all(task_exact_pass2)
            )
    summary: dict[str, Any] = {}
    for name, records in branch_records.items():
        sample_pass1 = sum(int(record["pass_at_1_exact"]) for record in records)
        sample_pass2 = sum(int(record["pass_at_2_exact"]) for record in records)
        summary[name] = {
            "sample_count": len(records),
            # Backward-compatible aliases continue to mean pass@1.
            "sample_exact": sample_pass1,
            "task_solved": solved_pass1[name],
            "sample_exact_pass_at_1": sample_pass1,
            "sample_exact_pass_at_2": sample_pass2,
            "sample_pass_at_1_fraction": sample_pass1 / len(records),
            "sample_pass_at_2_fraction": sample_pass2 / len(records),
            "task_solved_pass_at_1": solved_pass1[name],
            "task_solved_pass_at_2": solved_pass2[name],
            "task_pass_at_1_fraction": solved_pass1[name] / len(task_results),
            "task_pass_at_2_fraction": solved_pass2[name] / len(task_results),
            "task_count": len(task_results),
            "macro_pixel_accuracy": sum(float(record["pixel_accuracy"]) for record in records)
            / len(records),
            "records": records,
        }
    return summary
