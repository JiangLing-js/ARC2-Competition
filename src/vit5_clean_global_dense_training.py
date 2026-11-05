"""Challenge-only clean global Dense training for ViT-5 ARC.

The trainer consumes only task.train demonstrations from global-train tasks.
Development tasks are used only through deterministic identity leave-one-out
validation.  It has deliberately no solutions/test-target interface.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch

from .vit5_clean_task_split import split_manifest_sha256, validate_split_manifest
from .vit5_hybrid_batch import (
    TARGET_COMPONENT_MODE,
    VIT5_HYBRID_BATCH_SCHEMA_VERSION,
    build_vit5_dense_batch,
)
from .vit5_dense_decoder import (
    COMPONENT_LOSS_TYPE,
    DENSE_CLEAN_GLOBAL_CONDITIONING_MODES,
    DENSE_CONDITIONING_GATED_PAIRED,
    DENSE_CONDITIONING_SPLIT,
    DENSE_GATED_PAIRED_GATE_INITIAL_VALUE,
    DENSE_RELATIONAL_MEMORY_DISABLED,
    DENSE_RELATIONAL_MEMORY_GATE_INITIAL_VALUE,
    DENSE_RELATIONAL_MEMORY_MODES,
    DENSE_SHAPE_DECODER_BD_BORDER,
    DENSE_SHAPE_DECODER_MODES,
    DENSE_SHAPE_DECODER_SEPARATE_HW,
    VIT5_DENSE_LOSS_SCHEMA_VERSION,
    decode_dense_output,
)
from .vit5_overfit_diagnostics import compare_grids
from .vit5_per_task_full_adaptation import (
    AugmentationTrace,
    _attach_dense_cpu_values,
    aggregate_target_component_stats,
    resolve_dense_full_trainable_scope,
)
from .vit5_per_task_ttt import (
    D4_TRANSFORM_NAMES,
    _autocast,
    _dense_training_forward,
    augment_leave_one_out_sample,
    build_leave_one_out_samples,
    validate_transform_names,
)
from .vitarc_adapter import VitarcGridAdapter


GLOBAL_TRAINING_PROGRESS_FORMAT = "vit5_clean_global_dense_progress_v1"
GLOBAL_TRAINING_PROTOCOL = "challenge_only_global_train_development_identity_loo_v1"
GLOBAL_K4_REPLAY_PROGRESS_FORMAT = "vit5_clean_global_dense_k4_replay_progress_v2"
GLOBAL_K4_REPLAY_PROTOCOL = "arc_train_plus_streaming_k4_replay_development_identity_loo_v1"


@dataclass(frozen=True)
class CleanGlobalTrainingConfig:
    max_optimizer_updates: int = 1000
    train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    new_modules_lr: float = 3e-4
    vit5_lora_lr: float = 1e-4
    vit5_backbone_lr: float = 1e-4
    new_modules_min_lr: float = 1e-6
    vit5_lora_min_lr: float = 1e-6
    vit5_backbone_min_lr: float = 1e-6
    warmup_optimizer_updates: int = 50
    gradient_clip_norm: float = 1.0
    weight_decay: float = 0.0
    validate_every_updates: int = 100
    save_every_updates: int = 50
    seed: int = 0
    transform_names: tuple[str, ...] = D4_TRANSFORM_NAMES
    shuffle_demonstrations: bool = True
    color_permutation: bool = False
    precision: str = "bf16"
    max_grids: int = 21
    grid_encoder_microbatch: int | None = 32
    episode_cache_entries: int = 512
    lambda_component: float = 0.25
    lambda_shape: float = 0.2
    dense_recursive_steps: int = 1
    dense_recursive_loss_weighting: str = "hsw"
    dense_hsw_lambda: float = 0.7
    dense_candidate_encoder_layers: int = 2
    dense_candidate_mlp_ratio: float = 4.0
    dense_candidate_dropout: float = 0.0
    dense_conditioning_mode: str = DENSE_CONDITIONING_SPLIT
    dense_shape_decoder_mode: str = DENSE_SHAPE_DECODER_SEPARATE_HW
    dense_paired_gate_initial_value: float = (
        DENSE_GATED_PAIRED_GATE_INITIAL_VALUE
    )
    dense_relational_memory_mode: str = DENSE_RELATIONAL_MEMORY_DISABLED
    dense_relational_gate_initial_value: float = (
        DENSE_RELATIONAL_MEMORY_GATE_INITIAL_VALUE
    )
    k4_replay_batch_size: int = 0
    k4_replay_loss_fraction: float = 0.2
    k4_replay_loss_mode: str = "color_only"
    k4_require_initial_retention_pass: bool = True
    k4_replay_seed: int = 0
    k4_replay_train_rule_count: int = 256
    k4_replay_validation_rule_count: int = 64
    k4_replay_demonstration_count: int = 3
    k4_replay_grid_size: int = 4
    k4_replay_source_colors: tuple[int, ...] = (1, 2, 3, 4)
    k4_replay_transform_names: tuple[str, ...] = ("identity",)
    k4_replay_validation_batch_size: int = 8
    vit5_training_mode: str = "pretrained_lora"
    object_embedding_mode: str = "none"
    batch_schema_version: int = VIT5_HYBRID_BATCH_SCHEMA_VERSION
    loss_schema_version: int = VIT5_DENSE_LOSS_SCHEMA_VERSION
    target_component_mode: str = TARGET_COMPONENT_MODE
    component_loss_type: str = COMPONENT_LOSS_TYPE

    def validate(self) -> None:
        positive = (
            "max_optimizer_updates",
            "train_batch_size",
            "gradient_accumulation_steps",
            "validate_every_updates",
            "save_every_updates",
            "max_grids",
        )
        if any(int(getattr(self, name)) < 1 for name in positive):
            raise ValueError(f"Clean global positive fields must be >=1: {positive}")
        if self.warmup_optimizer_updates < 0:
            raise ValueError("warmup_optimizer_updates must be non-negative.")
        if self.gradient_clip_norm <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid gradient clipping/weight decay.")
        for peak, minimum in (
            (self.new_modules_lr, self.new_modules_min_lr),
            (self.vit5_lora_lr, self.vit5_lora_min_lr),
            (self.vit5_backbone_lr, self.vit5_backbone_min_lr),
        ):
            if peak <= 0 or minimum < 0 or minimum > peak:
                raise ValueError("Invalid peak/min learning-rate pair.")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16 or bf16.")
        if self.episode_cache_entries < 0:
            raise ValueError("episode_cache_entries must be non-negative.")
        if not isinstance(self.color_permutation, bool):
            raise ValueError("color_permutation must be boolean.")
        if self.object_embedding_mode != "none":
            raise ValueError("The first clean global protocol requires object_embedding_mode=none.")
        if self.vit5_training_mode not in {"pretrained_lora", "random_scratch"}:
            raise ValueError(
                "vit5_training_mode must be pretrained_lora or random_scratch."
            )
        if self.dense_shape_decoder_mode not in DENSE_SHAPE_DECODER_MODES:
            raise ValueError(
                "dense_shape_decoder_mode must be one of "
                f"{DENSE_SHAPE_DECODER_MODES}."
            )
        if (
            self.dense_shape_decoder_mode == DENSE_SHAPE_DECODER_BD_BORDER
            and self.dense_recursive_steps != 1
        ):
            raise ValueError("bd_border_v1 requires dense_recursive_steps=1.")
        if not 1 <= self.dense_recursive_steps <= 8:
            raise ValueError("dense_recursive_steps must be in [1,8].")
        if self.dense_recursive_loss_weighting not in {"uniform", "hsw"}:
            raise ValueError("dense_recursive_loss_weighting must be uniform or hsw.")
        if not 0.0 < self.dense_hsw_lambda <= 1.0:
            raise ValueError("dense_hsw_lambda must be in (0,1].")
        if self.dense_candidate_encoder_layers < 1:
            raise ValueError("dense_candidate_encoder_layers must be >=1.")
        if self.dense_candidate_mlp_ratio <= 0 or not 0.0 <= self.dense_candidate_dropout < 1.0:
            raise ValueError("Invalid Dense candidate encoder MLP ratio/dropout.")
        if self.dense_conditioning_mode not in DENSE_CLEAN_GLOBAL_CONDITIONING_MODES:
            raise ValueError(
                "dense_conditioning_mode must be one of "
                f"{DENSE_CLEAN_GLOBAL_CONDITIONING_MODES}."
            )
        if (
            self.dense_conditioning_mode == DENSE_CONDITIONING_GATED_PAIRED
            and self.dense_relational_memory_mode
            == DENSE_RELATIONAL_MEMORY_DISABLED
        ):
            if self.dense_recursive_steps != 1:
                raise ValueError(
                    "Experimental Gated paired clean-global training requires "
                    "dense_recursive_steps=1."
                )
            if not 0.0 < float(self.dense_paired_gate_initial_value) < 2.0:
                raise ValueError(
                    "Experimental Gated paired initial effective scale must be "
                    "strictly in (0,2)."
                )
        if self.dense_relational_memory_mode not in DENSE_RELATIONAL_MEMORY_MODES:
            raise ValueError(
                "dense_relational_memory_mode must be one of "
                f"{DENSE_RELATIONAL_MEMORY_MODES}."
            )
        if (
            self.dense_relational_memory_mode != DENSE_RELATIONAL_MEMORY_DISABLED
            and self.dense_conditioning_mode != DENSE_CONDITIONING_GATED_PAIRED
        ):
            raise ValueError(
                "Relational Memory v2 currently requires "
                "dense_conditioning_mode=gated_paired_associative_v1."
            )
        if (
            self.dense_relational_memory_mode
            != DENSE_RELATIONAL_MEMORY_DISABLED
            and not 0.0 < float(self.dense_relational_gate_initial_value) < 2.0
        ):
            raise ValueError(
                "Relational Memory v2 initial effective scale must be strictly "
                "in (0,2)."
            )
        if self.k4_replay_batch_size < 0:
            raise ValueError("k4_replay_batch_size must be non-negative.")
        if self.k4_replay_batch_size:
            if self.dense_conditioning_mode != DENSE_CONDITIONING_GATED_PAIRED:
                raise ValueError(
                    "K4 replay requires dense_conditioning_mode="
                    "gated_paired_associative_v1."
                )
            if not 0.0 < float(self.k4_replay_loss_fraction) < 1.0:
                raise ValueError(
                    "k4_replay_loss_fraction must lie strictly in (0,1) when "
                    "replay is enabled."
                )
            if self.k4_replay_loss_mode not in {"color_only", "dense_full"}:
                raise ValueError(
                    "k4_replay_loss_mode must be color_only or dense_full."
                )
            if self.k4_replay_seed < 0:
                raise ValueError("k4_replay_seed must be non-negative.")
            if self.k4_replay_train_rule_count < 2:
                raise ValueError("k4_replay_train_rule_count must be >=2.")
            if self.k4_replay_validation_rule_count < 1:
                raise ValueError("k4_replay_validation_rule_count must be >=1.")
            if self.k4_replay_demonstration_count < 1:
                raise ValueError("k4_replay_demonstration_count must be >=1.")
            if 2 * self.k4_replay_demonstration_count + 1 > self.max_grids:
                raise ValueError("K4 replay episodes exceed max_grids.")
            if not 1 <= self.k4_replay_grid_size <= 8:
                raise ValueError("k4_replay_grid_size must be in [1,8].")
            colors = tuple(int(value) for value in self.k4_replay_source_colors)
            if not colors or len(colors) != len(set(colors)):
                raise ValueError("k4_replay_source_colors must be non-empty and unique.")
            if len(colors) != 4:
                raise ValueError(
                    "The K4 replay v1 protocol requires exactly four source colors."
                )
            if any(value < 0 or value > 9 for value in colors):
                raise ValueError("k4_replay_source_colors must lie in [0,9].")
            if len(colors) > self.k4_replay_grid_size**2:
                raise ValueError("K4 replay grid cannot expose every source color.")
            if self.k4_replay_validation_batch_size < 1:
                raise ValueError("k4_replay_validation_batch_size must be >=1.")
            validate_transform_names(self.k4_replay_transform_names)
        if self.batch_schema_version != 4 or self.loss_schema_version != 2:
            raise ValueError("Clean global training requires batch schema v4/loss schema v2.")
        if self.target_component_mode != TARGET_COMPONENT_MODE:
            raise ValueError("Unexpected target component mode.")
        if self.component_loss_type != COMPONENT_LOSS_TYPE:
            raise ValueError("Unexpected component loss type.")
        validate_transform_names(self.transform_names)


def _grid_shape(grid: Sequence[Sequence[int]]) -> tuple[int, int]:
    height = len(grid)
    width = len(grid[0]) if height else 0
    return height, width


def validate_gated_same_shape_tasks(
    tasks: Mapping[str, Mapping[str, Any]], *, split_name: str
) -> None:
    """Fail fast when the v1 exact-coordinate paired reader is inapplicable.

    Gated paired v1 binds a demonstration input cell to the output cell at the
    same coordinate.  It therefore has no defined transport for a train pair
    whose input and output rectangles differ.  The pilot must use a manifest
    that selects only same-shape tasks instead of silently dropping pairs.
    """

    incompatible: list[str] = []
    for task_id in sorted(tasks):
        for pair_index, pair in enumerate(tasks[task_id].get("train", ())):
            input_shape = _grid_shape(pair["input"])
            output_shape = _grid_shape(pair["output"])
            if input_shape != output_shape:
                incompatible.append(
                    f"{task_id}:train[{pair_index}] {input_shape}->{output_shape}"
                )
    if incompatible:
        preview = ", ".join(incompatible[:8])
        suffix = "" if len(incompatible) <= 8 else f" (+{len(incompatible) - 8} more)"
        raise ValueError(
            "gated_paired_associative_v1 clean-global qualification requires "
            f"same-shape train pairs in {split_name}; incompatible={preview}{suffix}. "
            "Create an explicit same-shape split manifest instead of filtering "
            "inside the trainer."
        )


def global_config_sha256(config: CleanGlobalTrainingConfig) -> str:
    payload = asdict(config)
    # This option was added after the original clean-global checkpoints.  A
    # disabled option must remain resume-compatible with those checkpoints;
    # enabling it is deliberately hash-visible because it changes the sampled
    # training stream.
    if not config.color_permutation:
        payload.pop("color_permutation", None)
    # The legacy H/W decoder predates an explicit mode field.  Keep established
    # resume hashes stable while making the new BD contract hash-visible.
    if config.dense_shape_decoder_mode == DENSE_SHAPE_DECODER_SEPARATE_HW:
        payload.pop("dense_shape_decoder_mode", None)
    # Keep old pretrained+LoRA resume hashes byte-for-byte compatible.  The
    # scratch-only LR pair becomes contractual only for the scratch protocol.
    if config.vit5_training_mode == "pretrained_lora":
        payload.pop("vit5_training_mode", None)
        payload.pop("vit5_backbone_lr", None)
        payload.pop("vit5_backbone_min_lr", None)
    # Preserve the established resume hash for the original production modes.
    # The Gated-only field is contractual only when that experimental path is
    # selected; adding it must not invalidate unrelated clean-global resumes.
    if (
        config.dense_conditioning_mode != DENSE_CONDITIONING_GATED_PAIRED
        or config.dense_relational_memory_mode
        != DENSE_RELATIONAL_MEMORY_DISABLED
    ):
        payload.pop("dense_paired_gate_initial_value", None)
    if config.dense_relational_memory_mode == DENSE_RELATIONAL_MEMORY_DISABLED:
        payload.pop("dense_relational_memory_mode", None)
        payload.pop("dense_relational_gate_initial_value", None)
    if not config.k4_replay_batch_size:
        for name in tuple(payload):
            if name.startswith("k4_replay_"):
                payload.pop(name)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def k4_replay_samples_per_batch(config: CleanGlobalTrainingConfig) -> int:
    return int(config.k4_replay_batch_size)


def global_training_protocol(config: CleanGlobalTrainingConfig) -> str:
    return GLOBAL_K4_REPLAY_PROTOCOL if config.k4_replay_batch_size else GLOBAL_TRAINING_PROTOCOL


def global_progress_format(config: CleanGlobalTrainingConfig) -> str:
    return (
        GLOBAL_K4_REPLAY_PROGRESS_FORMAT
        if config.k4_replay_batch_size
        else GLOBAL_TRAINING_PROGRESS_FORMAT
    )


def global_resume_contract(
    config: CleanGlobalTrainingConfig,
    split_manifest: Mapping[str, Any],
    k4_replay_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    contract = {
        "protocol": global_training_protocol(config),
        "config_sha256": global_config_sha256(config),
        "split_manifest_sha256": split_manifest_sha256(split_manifest),
        "batch_schema_version": config.batch_schema_version,
        "loss_schema_version": config.loss_schema_version,
    }
    if config.k4_replay_batch_size:
        if not k4_replay_contract:
            raise ValueError("K4 replay requires a non-empty replay contract.")
        contract["k4_replay_contract"] = copy.deepcopy(dict(k4_replay_contract))
    elif k4_replay_contract is not None:
        raise ValueError("A K4 replay contract was supplied while replay is disabled.")
    return contract


def validate_global_resume(
    training_state: Mapping[str, Any],
    config: CleanGlobalTrainingConfig,
    split_manifest: Mapping[str, Any],
    k4_replay_contract: Mapping[str, Any] | None = None,
) -> None:
    if training_state.get("format") != global_progress_format(config):
        raise ValueError("Clean global resume checkpoint format mismatch.")
    saved = training_state.get("resume_contract")
    current = global_resume_contract(config, split_manifest, k4_replay_contract)
    if saved != current:
        raise ValueError(
            "Clean global resume configuration/split mismatch. Use a new output directory.\n"
            f"saved={saved}\ncurrent={current}"
        )


def _source_demo_indices(sample: Mapping[str, Any]) -> list[int]:
    heldout = int(sample["heldout_index"])
    count = len(sample["context_pairs"]) + 1
    return [index for index in range(count) if index != heldout]


class GlobalEpisodeSampler:
    """Sample independent task/LOO/D4/color/demo-order tuples with an LRU cache.

    Color augmentation is a bijection of the complete non-background ARC
    palette ``1..9`` and always fixes color 0.  Sampling the complete palette,
    rather than only colors observed in an episode, keeps the distribution
    independent of any held-out/test target and makes every mapping exactly
    reversible.
    """

    def __init__(
        self,
        tasks: Mapping[str, Mapping[str, Any]],
        seed: int,
        transform_names: Sequence[str],
        shuffle_demonstrations: bool,
        cache_entries: int = 512,
        color_permutation: bool = False,
    ) -> None:
        self.tasks = {str(key): value for key, value in tasks.items()}
        self.task_ids = sorted(
            task_id for task_id, task in self.tasks.items() if len(task.get("train", [])) >= 2
        )
        if not self.task_ids:
            raise ValueError("Global training requires tasks with at least two train pairs.")
        self.ineligible_task_ids = sorted(set(self.tasks) - set(self.task_ids))
        self.loo = {
            task_id: build_leave_one_out_samples(self.tasks[task_id], task_id)
            for task_id in self.task_ids
        }
        self.rng = random.Random(int(seed))
        self.transform_names = validate_transform_names(transform_names)
        self.shuffle_demonstrations = bool(shuffle_demonstrations)
        self.color_permutation = bool(color_permutation)
        self.cache_entries = int(cache_entries)
        self.cache: OrderedDict[
            tuple[str, int, str, tuple[int, ...]], dict[str, Any]
        ] = OrderedDict()
        self.adapter = VitarcGridAdapter()

    def state_dict(self) -> dict[str, Any]:
        return {"rng_state": self.rng.getstate()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.rng.setstate(state["rng_state"])

    def _prepared_episode(
        self,
        task_id: str,
        heldout_index: int,
        transform_name: str,
        color_mapping: Mapping[int, int],
    ) -> dict[str, Any]:
        mapping_key = tuple(int(color_mapping.get(color, color)) for color in range(10))
        key = (task_id, heldout_index, transform_name, mapping_key)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache.move_to_end(key)
            return cached
        sample = self.loo[task_id][heldout_index]
        augmented = augment_leave_one_out_sample(
            sample,
            transform_name,
            random.Random(0),
            shuffle_demonstrations=False,
            color_mapping=color_mapping,
        )
        prepared = self.adapter.prepare_dttt_sample(augmented)
        _attach_dense_cpu_values(prepared)
        if self.cache_entries:
            self.cache[key] = prepared
            self.cache.move_to_end(key)
            while len(self.cache) > self.cache_entries:
                self.cache.popitem(last=False)
        return prepared

    def sample_one(self) -> tuple[dict[str, Any], dict[str, Any]]:
        task_id = self.task_ids[self.rng.randrange(len(self.task_ids))]
        heldout = self.rng.randrange(len(self.loo[task_id]))
        transform = self.transform_names[self.rng.randrange(len(self.transform_names))]
        if self.color_permutation:
            shuffled = list(range(1, 10))
            self.rng.shuffle(shuffled)
            color_mapping = {0: 0, **dict(zip(range(1, 10), shuffled))}
        else:
            # Do not consume RNG state on the legacy path: old checkpoints must
            # resume the exact same sample stream as before this option existed.
            color_mapping = {0: 0}
        cached = self._prepared_episode(
            task_id, heldout, transform, color_mapping
        )
        contexts = list(cached["context_pairs"])
        order = list(range(len(contexts)))
        if self.shuffle_demonstrations:
            self.rng.shuffle(order)
        source = _source_demo_indices(
            {"heldout_index": heldout, "context_pairs": contexts}
        )
        prepared = {
            **cached,
            "context_pairs": [contexts[index] for index in order],
            "order_indices": [source[index] for index in order],
        }
        mapping_items = [
            [color, int(color_mapping.get(color, color))] for color in range(10)
        ]
        trace_transform = transform
        if self.color_permutation:
            encoded_mapping = ",".join(
                f"{source}>{target}" for source, target in mapping_items
            )
            # AugmentationTrace v1 canonicalizes only the transform string, so
            # encode the exact mapping there as well as exposing structured
            # fields below.  This preserves legacy trace hashes when disabled.
            trace_transform = f"{transform}|fixed0_full_palette:{encoded_mapping}"
        trace = {
            "task_id": task_id,
            "held_out_index": heldout,
            "transform": trace_transform,
            "geometry_transform": transform,
            "color_mapping": mapping_items,
            "demonstration_order": prepared["order_indices"],
        }
        return prepared, trace

    def sample_batch(self, batch_size: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        values = [self.sample_one() for _ in range(int(batch_size))]
        return [value[0] for value in values], [value[1] for value in values]


def global_learning_rates(
    config: CleanGlobalTrainingConfig, optimizer_update: int
) -> dict[str, float]:
    def value(peak: float, minimum: float) -> float:
        if config.warmup_optimizer_updates and optimizer_update <= config.warmup_optimizer_updates:
            return peak * optimizer_update / config.warmup_optimizer_updates
        decay_steps = max(1, config.max_optimizer_updates - config.warmup_optimizer_updates)
        progress = min(
            1.0,
            max(0.0, (optimizer_update - config.warmup_optimizer_updates) / decay_steps),
        )
        return minimum + 0.5 * (peak - minimum) * (1 + math.cos(math.pi * progress))

    return {
        "new_modules": value(config.new_modules_lr, config.new_modules_min_lr),
        "vit5_lora": value(config.vit5_lora_lr, config.vit5_lora_min_lr),
        "vit5_backbone": value(
            config.vit5_backbone_lr, config.vit5_backbone_min_lr
        ),
    }


def configure_global_optimizer(
    model: torch.nn.Module, config: CleanGlobalTrainingConfig
) -> tuple[torch.optim.AdamW, dict[str, Any]]:
    scratch = config.vit5_training_mode == "random_scratch"
    model_mode = getattr(model, "vit5_initialization", None)
    if model_mode is not None and model_mode != config.vit5_training_mode:
        raise ValueError(
            "Model/config ViT-5 initialization mismatch: "
            f"model={model_mode!r}, config={config.vit5_training_mode!r}."
        )
    if scratch:
        if getattr(model, "decoder_mode", None) != "dense":
            raise ValueError("random_scratch global training requires decoder_mode=dense.")
        if getattr(model, "freeze_vit5", True):
            raise ValueError("random_scratch global training requires a full grid encoder.")
        names = tuple(
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        )
        forbidden = [
            name
            for name in names
            if name.startswith(("t5.", "encoder_to_t5.", "arc_token_delta."))
            or ".lora_" in name
        ]
        if forbidden:
            raise ValueError(
                "random_scratch Dense scope contains forbidden T5/AR/LoRA "
                f"parameters: {forbidden}"
            )
        scratch_allowed = set(names)
        startup_audit = {
            "total_optimizer_parameters": sum(
                parameter.numel()
                for name, parameter in model.named_parameters()
                if name in scratch_allowed
            )
        }
    else:
        names, startup_audit = resolve_dense_full_trainable_scope(model)
    allowed = set(names)
    grouped: dict[str, list[tuple[str, torch.nn.Parameter]]] = {
        "new_modules": [],
        "vit5_lora": [],
        "vit5_backbone": [],
    }
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in allowed)
        if name in allowed:
            if (
                scratch
                and name.startswith("arc_encoder.grid_encoder.")
                and not name.startswith(
                    "arc_encoder.grid_encoder.arc_embedding."
                )
            ):
                group = "vit5_backbone"
            else:
                group = "vit5_lora" if ".lora_" in name else "new_modules"
            grouped[group].append((name, parameter))
    if set(name for values in grouped.values() for name, _ in values) != allowed:
        raise AssertionError("Global optimizer scope differs from Dense global scope.")
    required_groups = (
        ("new_modules", "vit5_backbone")
        if scratch
        else ("new_modules", "vit5_lora")
    )
    empty = [group for group in required_groups if not grouped[group]]
    if empty:
        raise ValueError(f"Required global optimizer groups are empty: {empty}")
    forbidden_nonempty = ["vit5_lora"] if scratch else ["vit5_backbone"]
    if any(grouped[group] for group in forbidden_nonempty):
        raise AssertionError(
            f"Unexpected optimizer group for initialization mode: {forbidden_nonempty}"
        )
    rates = global_learning_rates(config, 1)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [parameter for _, parameter in grouped[group]],
                "lr": rates[group],
                "weight_decay": config.weight_decay,
                "group_name": group,
            }
            for group in required_groups
        ]
    )
    audit = {
        "groups": {
            group: {
                "parameter_names": [name for name, _ in values],
                "parameter_count": sum(parameter.numel() for _, parameter in values),
                "parameter_tensor_count": len(values),
            }
            for group, values in grouped.items()
        },
        "total_optimizer_parameters": sum(
            parameter.numel() for values in grouped.values() for _, parameter in values
        ),
        "overlap_count": 0,
        "startup_scope_parameter_count": startup_audit["total_optimizer_parameters"],
    }
    return optimizer, audit


def _set_lrs(optimizer: torch.optim.Optimizer, values: Mapping[str, float]) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(values[str(group["group_name"])])


@torch.no_grad()
def evaluate_development_identity_loo(
    model: Any,
    development_tasks: Mapping[str, Mapping[str, Any]],
    device: torch.device,
    config: CleanGlobalTrainingConfig,
    optimizer_update: int,
) -> dict[str, Any]:
    """Validate without task.test and without any solutions mapping."""
    model.eval()
    adapter = VitarcGridAdapter()
    records: list[dict[str, Any]] = []
    losses: list[float] = []
    for task_id in sorted(development_tasks):
        for sample in build_leave_one_out_samples(development_tasks[task_id], task_id):
            prepared = adapter.prepare_dttt_sample(copy.deepcopy(sample))
            _attach_dense_cpu_values(prepared)
            batch = build_vit5_dense_batch([prepared], device=device, max_grids=config.max_grids)
            with _autocast(device, config.precision):
                output = _dense_training_forward(model, batch)
            if output.dense_loss is None or not bool(torch.isfinite(output.dense_loss).item()):
                raise ValueError("Development identity LOO produced non-finite loss.")
            dense_output = output.dense_output
            if dense_output is None or dense_output.round_outputs is None:
                raise RuntimeError("Development Dense output is missing recursive rounds.")
            prediction = decode_dense_output(
                dense_output.color_logits,
                dense_output.height_logits,
                dense_output.width_logits,
                joint_shape_logits=dense_output.joint_shape_logits,
            )[0]
            target = sample["target"]
            comparison = compare_grids(prediction["grid"], target)
            round_records: list[dict[str, Any]] = []
            for round_index, round_output in enumerate(dense_output.round_outputs):
                round_prediction = decode_dense_output(
                    round_output.color_logits,
                    round_output.height_logits,
                    round_output.width_logits,
                    joint_shape_logits=round_output.joint_shape_logits,
                )[0]
                round_comparison = compare_grids(round_prediction["grid"], target)
                round_records.append(
                    {
                        "round_index": round_index + 1,
                        "exact": bool(round_comparison["exact"]),
                        "pixel_accuracy": float(round_comparison["pixel_accuracy"]),
                        "target_error_cells": int(
                            round_comparison["target_error_cells"]
                        ),
                        "missing_cells": int(round_comparison["missing_cells"]),
                        "extra_cells": int(round_comparison["extra_cells"]),
                        "predicted_height": int(round_prediction["predicted_height"]),
                        "predicted_width": int(round_prediction["predicted_width"]),
                    }
                )
            records.append(
                {
                    "task_id": task_id,
                    "heldout_index": int(sample["heldout_index"]),
                    "exact": bool(comparison["exact"]),
                    "target_error_cells": int(comparison["target_error_cells"]),
                    "pixel_accuracy": float(comparison["pixel_accuracy"]),
                    "rounds": round_records,
                }
            )
            losses.append(float(output.dense_loss.detach().float().cpu()))
    if not records:
        raise ValueError("Development split has no identity LOO samples.")
    round_summary = [
        {
            "round_index": round_index + 1,
            "exact_count": sum(
                int(record["rounds"][round_index]["exact"]) for record in records
            ),
            "macro_pixel_accuracy": sum(
                float(record["rounds"][round_index]["pixel_accuracy"])
                for record in records
            )
            / len(records),
            "target_error_cells": sum(
                int(record["rounds"][round_index]["target_error_cells"])
                for record in records
            ),
        }
        for round_index in range(config.dense_recursive_steps)
    ]
    return {
        "record_type": "clean_global_development_identity_loo",
        "optimizer_update": int(optimizer_update),
        "task_count": len(development_tasks),
        "sample_count": len(records),
        "exact_count": sum(int(record["exact"]) for record in records),
        "target_error_cells": sum(int(record["target_error_cells"]) for record in records),
        "macro_pixel_accuracy": sum(record["pixel_accuracy"] for record in records)
        / len(records),
        "validation_loss": sum(losses) / len(losses),
        "dense_round_summary": round_summary,
        "test_inputs_read": False,
        "solutions_loaded": False,
        "records": records,
    }


def global_validation_rank(metrics: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(metrics["exact_count"]),
        -float(metrics["target_error_cells"]),
        float(metrics["macro_pixel_accuracy"]),
        -float(metrics["validation_loss"]),
        -float(metrics["optimizer_update"]),
    )


def train_clean_global_dense(
    model: Any,
    global_train_tasks: Mapping[str, Mapping[str, Any]],
    development_tasks: Mapping[str, Mapping[str, Any]],
    split_manifest: Mapping[str, Any],
    device: torch.device,
    config: CleanGlobalTrainingConfig,
    resume_training_state: Mapping[str, Any] | None = None,
    resume_optimizer_state: Mapping[str, Any] | None = None,
    checkpoint_callback: Any | None = None,
    metrics_handle: Any | None = None,
    k4_replay_sampler: Any | None = None,
    k4_replay_prepare_samples: Any | None = None,
    k4_replay_validation_callback: Any | None = None,
    k4_replay_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config.validate()
    validate_split_manifest(split_manifest)
    if (
        config.dense_conditioning_mode == DENSE_CONDITIONING_GATED_PAIRED
        and config.dense_relational_memory_mode
        == DENSE_RELATIONAL_MEMORY_DISABLED
    ):
        validate_gated_same_shape_tasks(
            global_train_tasks, split_name="global_train"
        )
        validate_gated_same_shape_tasks(
            development_tasks, split_name="development"
        )
    replay_per_batch = k4_replay_samples_per_batch(config)
    arc_per_batch = config.train_batch_size
    replay_hooks = (
        k4_replay_sampler,
        k4_replay_prepare_samples,
        k4_replay_validation_callback,
        k4_replay_contract,
    )
    if replay_per_batch:
        if any(value is None for value in replay_hooks):
            raise ValueError(
                "K4 replay is enabled but its sampler, preparation callback, "
                "validation callback, or contract is missing."
            )
    elif any(value is not None for value in replay_hooks):
        raise ValueError("K4 replay hooks were supplied while replay is disabled.")
    optimizer, optimizer_audit = configure_global_optimizer(model, config)
    sampler = GlobalEpisodeSampler(
        global_train_tasks,
        config.seed,
        config.transform_names,
        config.shuffle_demonstrations,
        config.episode_cache_entries,
        config.color_permutation,
    )
    trace = AugmentationTrace()
    best_metrics: dict[str, Any] | None = None
    best_retention_constrained_metrics: dict[str, Any] | None = None
    best_retention_constrained_k4_metrics: dict[str, Any] | None = None
    start_update = 1
    updates_completed = 0
    elapsed_before = 0.0
    last_train: dict[str, Any] | None = None
    last_validation: dict[str, Any] | None = None
    initial_k4_retention: dict[str, Any] | None = None
    last_k4_retention: dict[str, Any] | None = None
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.reset_peak_memory_stats(device)
    if resume_training_state is not None:
        validate_global_resume(
            resume_training_state,
            config,
            split_manifest,
            k4_replay_contract,
        )
        if resume_optimizer_state is None:
            raise ValueError("Global resume requires optimizer state.")
        optimizer.load_state_dict(resume_optimizer_state)
        sampler.load_state_dict(resume_training_state["sampler_state"])
        if replay_per_batch:
            if "k4_replay_sampler_state" not in resume_training_state:
                raise ValueError("K4 replay resume is missing replay sampler state.")
            k4_replay_sampler.load_state_dict(
                resume_training_state["k4_replay_sampler_state"]
            )
        trace = AugmentationTrace.from_state_dict(resume_training_state["augmentation_trace"])
        torch.set_rng_state(resume_training_state["torch_rng_state"])
        if device.type == "cuda" and resume_training_state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(resume_training_state["cuda_rng_state"])
        best_metrics = resume_training_state.get("best_development_metrics")
        best_retention_constrained_metrics = resume_training_state.get(
            "best_retention_constrained_development_metrics"
        )
        best_retention_constrained_k4_metrics = resume_training_state.get(
            "best_retention_constrained_k4_metrics"
        )
        updates_completed = int(resume_training_state["optimizer_update"])
        start_update = updates_completed + 1
        elapsed_before = float(resume_training_state.get("elapsed_seconds", 0.0))
        last_train = resume_training_state.get("last_train_metrics")
        last_validation = resume_training_state.get("last_development_metrics")
        initial_k4_retention = resume_training_state.get(
            "initial_k4_retention_metrics"
        )
        last_k4_retention = resume_training_state.get("last_k4_retention_metrics")
        if (
            replay_per_batch
            and config.k4_require_initial_retention_pass
            and not bool(
                (initial_k4_retention or {}).get("retention_passed", False)
            )
        ):
            raise ValueError(
                "The resumed K4 retention experiment did not pass its update-0 "
                "A/B/C qualification. Start from a qualified Gated checkpoint "
                "or explicitly disable the diagnostic guard."
            )
    elif replay_per_batch:
        last_k4_retention = dict(k4_replay_validation_callback(0))
        if int(last_k4_retention.get("optimizer_update", -1)) != 0:
            raise ValueError("Initial K4 retention callback must report optimizer_update=0.")
        initial_k4_retention = copy.deepcopy(last_k4_retention)
        if (
            config.k4_require_initial_retention_pass
            and not bool(last_k4_retention.get("retention_passed", False))
        ):
            raise ValueError(
                "K4 replay requires a qualified update-0 Gated model, but the "
                "fixed A/B/C retention panel failed. Refusing to spend the ARC "
                "training budget on an invalid retention experiment."
            )
        if metrics_handle is not None:
            metrics_handle.write(json.dumps(last_k4_retention, sort_keys=True) + "\n")
            metrics_handle.flush()
    started = time.perf_counter()
    for update in range(start_update, config.max_optimizer_updates + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sums = {name: 0.0 for name in (
            "dense_loss", "raw_cell_loss", "component_balanced_loss",
            "color_loss", "height_loss", "width_loss", "border_loss",
        )}
        replay_sums = {name: 0.0 for name in sums}
        optimized_mixed_loss_sum = 0.0
        round_sums = [
            {
                "dense_loss": 0.0,
                "normal_shape_pixel_accuracy": 0.0,
                "oracle_shape_pixel_accuracy": 0.0,
                "grid_exact_match": 0.0,
                "height_accuracy": 0.0,
                "width_accuracy": 0.0,
            }
            for _ in range(config.dense_recursive_steps)
        ]
        transition_sums = [
            {
                "incorrect_to_correct_rate": 0.0,
                "correct_to_wrong_regression_rate": 0.0,
                "net_corrected_cells": 0.0,
            }
            for _ in range(max(0, config.dense_recursive_steps - 1))
        ]
        round_loss_weights: list[float] | None = None
        component_stats: list[Mapping[str, Any]] = []
        sampled_task_ids: list[str] = []
        prepare_seconds = 0.0
        replay_prepare_seconds = 0.0
        arc_loss_weight = (
            1.0 - config.k4_replay_loss_fraction if replay_per_batch else 1.0
        )
        replay_loss_weight = (
            config.k4_replay_loss_fraction if replay_per_batch else 0.0
        )
        for _ in range(config.gradient_accumulation_steps):
            prepare_started = time.perf_counter()
            samples, trace_entries = sampler.sample_batch(arc_per_batch)
            for entry in trace_entries:
                trace.update(entry)
                sampled_task_ids.append(str(entry["task_id"]))
            if len(samples) != config.train_batch_size:
                raise AssertionError("ARC microbatch has the wrong size.")
            batch = build_vit5_dense_batch(samples, device=device, max_grids=config.max_grids)
            prepare_seconds += time.perf_counter() - prepare_started
            component_stats.extend(batch["hybrid_batch_info"]["target_component_stats"])
            with _autocast(device, config.precision):
                output = _dense_training_forward(model, batch)
                loss = output.dense_loss
                if loss is None or not bool(torch.isfinite(loss).item()):
                    raise ValueError(f"Non-finite clean global loss at update {update}.")
                scaled = (
                    arc_loss_weight * loss / config.gradient_accumulation_steps
                )
            scaled.backward()
            optimized_mixed_loss_sum += arc_loss_weight * float(
                loss.detach().float().cpu()
            )
            for name in sums:
                value = getattr(output, name)
                if value is not None:
                    sums[name] += float(value.detach().float().cpu())
            dense_output = output.dense_output
            if dense_output is None or dense_output.round_outputs is None:
                raise RuntimeError("Clean global Dense output is missing recursive rounds.")
            if len(dense_output.round_outputs) != config.dense_recursive_steps:
                raise RuntimeError("Dense recursive round count differs from resolved config.")
            round_loss_weights = dense_output.round_loss_weights.detach().float().cpu().tolist()
            recursive = dense_output.recursive_metrics
            if recursive is None:
                raise RuntimeError("Supervised Dense output is missing recursive metrics.")
            for round_index, round_output in enumerate(dense_output.round_outputs):
                round_sums[round_index]["dense_loss"] += float(
                    round_output.dense_loss.detach().float().cpu()
                )
                round_sums[round_index]["normal_shape_pixel_accuracy"] += float(
                    recursive["round_normal_shape_pixel_accuracy"][round_index]
                    .detach().float().cpu()
                )
                round_sums[round_index]["oracle_shape_pixel_accuracy"] += float(
                    recursive["round_oracle_shape_pixel_accuracy"][round_index]
                    .detach().float().cpu()
                )
                for output_name, metric_name in (
                    ("grid_exact_match", "round_grid_exact_match"),
                    ("height_accuracy", "round_height_accuracy"),
                    ("width_accuracy", "round_width_accuracy"),
                ):
                    round_sums[round_index][output_name] += float(
                        recursive[metric_name][round_index].detach().float().cpu()
                    )
            for transition_index, transition in enumerate(recursive["transitions"]):
                for name in transition_sums[transition_index]:
                    transition_sums[transition_index][name] += float(
                        transition[name].detach().float().cpu()
                    )
            if replay_per_batch:
                replay_prepare_started = time.perf_counter()
                raw_replay, _ = k4_replay_sampler.sample_batch(replay_per_batch)
                prepared_replay = list(k4_replay_prepare_samples(raw_replay))
                if len(prepared_replay) != replay_per_batch:
                    raise ValueError(
                        "K4 replay preparation changed the configured sample count."
                    )
                replay_batch = build_vit5_dense_batch(
                    prepared_replay, device=device, max_grids=config.max_grids
                )
                replay_prepare_seconds += (
                    time.perf_counter() - replay_prepare_started
                )
                with _autocast(device, config.precision):
                    replay_output = _dense_training_forward(model, replay_batch)
                    replay_loss = (
                        replay_output.color_loss
                        if config.k4_replay_loss_mode == "color_only"
                        else replay_output.dense_loss
                    )
                    if replay_loss is None or not bool(
                        torch.isfinite(replay_loss).item()
                    ):
                        raise ValueError(
                            f"Non-finite K4 replay loss at update {update}."
                        )
                    scaled_replay = (
                        replay_loss_weight
                        * replay_loss
                        / config.gradient_accumulation_steps
                    )
                scaled_replay.backward()
                optimized_mixed_loss_sum += replay_loss_weight * float(
                    replay_loss.detach().float().cpu()
                )
                for name in replay_sums:
                    value = getattr(replay_output, name)
                    if value is not None:
                        replay_sums[name] += float(value.detach().float().cpu())
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        pre_clip = float(
            torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip_norm)
            .detach().float().cpu()
        )
        if not math.isfinite(pre_clip):
            raise ValueError(f"Non-finite clean global gradient norm at update {update}.")
        used_lrs = {str(group["group_name"]): float(group["lr"]) for group in optimizer.param_groups}
        optimizer.step()
        updates_completed = update
        if update < config.max_optimizer_updates:
            _set_lrs(optimizer, global_learning_rates(config, update + 1))
        denominator = config.gradient_accumulation_steps
        last_train = {
            "record_type": "clean_global_train_update",
            "optimizer_update": update,
            "train_batch_size": config.train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "samples_per_optimizer_update": (
                arc_per_batch + replay_per_batch
            ) * config.gradient_accumulation_steps,
            "arc_samples_per_optimizer_update": arc_per_batch
            * config.gradient_accumulation_steps,
            "k4_replay_samples_per_optimizer_update": replay_per_batch
            * config.gradient_accumulation_steps,
            **{name: value / denominator for name, value in sums.items()},
            "optimized_mixed_loss": optimized_mixed_loss_sum / denominator,
            "arc_loss_weight": arc_loss_weight,
            "k4_replay_loss_weight": replay_loss_weight,
            "k4_replay_loss_mode": (
                config.k4_replay_loss_mode if replay_per_batch else None
            ),
            "k4_replay_optimization_loss": (
                replay_sums[
                    "color_loss"
                    if config.k4_replay_loss_mode == "color_only"
                    else "dense_loss"
                ]
                / denominator
                if replay_per_batch
                else None
            ),
            "k4_replay_losses": {
                name: value / denominator for name, value in replay_sums.items()
            } if replay_per_batch else None,
            "lambda_component": config.lambda_component,
            "weighted_component_contribution": config.lambda_component
            * sums["component_balanced_loss"] / denominator,
            "shape_decoder_mode": config.dense_shape_decoder_mode,
            "shape_loss": (
                sums["border_loss"]
                if config.dense_shape_decoder_mode == DENSE_SHAPE_DECODER_BD_BORDER
                else sums["height_loss"] + sums["width_loss"]
            ) / denominator,
            "weighted_shape_contribution": config.lambda_shape
            * (
                sums["border_loss"]
                if config.dense_shape_decoder_mode == DENSE_SHAPE_DECODER_BD_BORDER
                else sums["height_loss"] + sums["width_loss"]
            ) / denominator,
            "pre_clip_grad_norm": pre_clip,
            "learning_rates": used_lrs,
            "batch_prepare_seconds": prepare_seconds,
            "k4_replay_batch_prepare_seconds": replay_prepare_seconds,
            "sampled_task_count": len(sampled_task_ids),
            "unique_sampled_task_count": len(set(sampled_task_ids)),
            "sampled_task_ids": sampled_task_ids,
            "arc_sample_count": arc_per_batch
            * config.gradient_accumulation_steps,
            "k4_replay_sample_count": replay_per_batch
            * config.gradient_accumulation_steps,
            "total_sample_count": (
                arc_per_batch + replay_per_batch
            ) * config.gradient_accumulation_steps,
            "dense_recursive_steps": config.dense_recursive_steps,
            "dense_recursive_loss_weighting": config.dense_recursive_loss_weighting,
            "dense_round_loss_weights": round_loss_weights,
            "dense_round_metrics": [
                {
                    "round_index": round_index + 1,
                    **{
                        name: value / denominator
                        for name, value in values.items()
                    },
                }
                for round_index, values in enumerate(round_sums)
            ],
            "dense_round_transitions": [
                {
                    "from_round": transition_index + 1,
                    "to_round": transition_index + 2,
                    **{
                        name: value / denominator
                        for name, value in values.items()
                    },
                }
                for transition_index, values in enumerate(transition_sums)
            ],
            **aggregate_target_component_stats(component_stats),
        }
        if metrics_handle is not None:
            metrics_handle.write(json.dumps(last_train, sort_keys=True) + "\n")
        is_validation = update % config.validate_every_updates == 0 or update == config.max_optimizer_updates
        is_best = False
        if is_validation:
            last_validation = evaluate_development_identity_loo(
                model, development_tasks, device, config, update
            )
            if replay_per_batch:
                last_k4_retention = dict(k4_replay_validation_callback(update))
                if int(last_k4_retention.get("optimizer_update", -1)) != update:
                    raise ValueError(
                        "K4 retention callback reported the wrong optimizer update."
                    )
            if metrics_handle is not None:
                metrics_handle.write(json.dumps(last_validation, sort_keys=True) + "\n")
                if last_k4_retention is not None:
                    metrics_handle.write(
                        json.dumps(last_k4_retention, sort_keys=True) + "\n"
                    )
                metrics_handle.flush()
            if best_metrics is None or global_validation_rank(last_validation) > global_validation_rank(best_metrics):
                best_metrics = last_validation
                is_best = True
            is_retention_qualified = bool(
                last_k4_retention is not None
                and last_k4_retention.get("retention_passed", False)
            )
            is_best_retention_constrained = bool(
                is_retention_qualified
                and (
                    best_retention_constrained_metrics is None
                    or global_validation_rank(last_validation)
                    > global_validation_rank(best_retention_constrained_metrics)
                )
            )
            if is_best_retention_constrained:
                best_retention_constrained_metrics = copy.deepcopy(last_validation)
                best_retention_constrained_k4_metrics = copy.deepcopy(
                    last_k4_retention
                )
        else:
            is_best_retention_constrained = False
        should_save = (
            update % config.save_every_updates == 0
            or update == config.max_optimizer_updates
            or is_best
            or is_best_retention_constrained
        )
        if should_save and checkpoint_callback is not None:
            elapsed = elapsed_before + time.perf_counter() - started
            state = {
                "format": global_progress_format(config),
                "resume_contract": global_resume_contract(
                    config, split_manifest, k4_replay_contract
                ),
                "optimizer_update": update,
                "best_development_metrics": best_metrics,
                "best_retention_constrained_development_metrics": (
                    best_retention_constrained_metrics
                ),
                "best_retention_constrained_k4_metrics": (
                    best_retention_constrained_k4_metrics
                ),
                "last_development_metrics": last_validation,
                "initial_k4_retention_metrics": initial_k4_retention,
                "last_k4_retention_metrics": last_k4_retention,
                "last_train_metrics": last_train,
                "sampler_state": sampler.state_dict(),
                "k4_replay_sampler_state": (
                    k4_replay_sampler.state_dict() if replay_per_batch else None
                ),
                "augmentation_trace": trace.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                "elapsed_seconds": elapsed,
                "total_training_samples": update
                * (arc_per_batch + replay_per_batch)
                * config.gradient_accumulation_steps,
                "total_arc_training_samples": update
                * arc_per_batch
                * config.gradient_accumulation_steps,
                "total_k4_replay_samples": update
                * replay_per_batch
                * config.gradient_accumulation_steps,
                "total_forward_backward_batches": update
                * config.gradient_accumulation_steps
                * (2 if replay_per_batch else 1),
                "arc_forward_backward_batches": update
                * config.gradient_accumulation_steps,
                "k4_replay_forward_backward_batches": update
                * config.gradient_accumulation_steps
                * int(bool(replay_per_batch)),
                "solutions_loaded": False,
            }
            checkpoint_callback("latest", model, optimizer, state)
            if is_best:
                checkpoint_callback("best_development_loo", model, optimizer, state)
            if is_best_retention_constrained:
                checkpoint_callback(
                    "best_retention_constrained_development_loo",
                    model,
                    optimizer,
                    state,
                )
    k4_replay_audit = k4_replay_sampler.audit() if replay_per_batch else None
    if replay_per_batch:
        expected_replay_samples = (
            updates_completed
            * replay_per_batch
            * config.gradient_accumulation_steps
        )
        if int(k4_replay_audit["generated_sample_count"]) != expected_replay_samples:
            raise AssertionError("K4 replay sample count does not match the budget.")
        if int(k4_replay_audit["unique_query_count"]) != expected_replay_samples:
            raise AssertionError("K4 replay unexpectedly reused a query grid.")
        if int(k4_replay_audit["fixed_evaluation_overlap_count"]) != 0:
            raise AssertionError("K4 replay overlapped its fixed retention panel.")
    peak_cuda_allocated_bytes = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )
    peak_cuda_reserved_bytes = (
        int(torch.cuda.max_memory_reserved(device))
        if device.type == "cuda"
        else None
    )
    return {
        "protocol": global_training_protocol(config),
        "optimizer_updates_completed": updates_completed,
        "best_development_metrics": best_metrics,
        "best_retention_constrained_development_metrics": (
            best_retention_constrained_metrics
        ),
        "best_retention_constrained_k4_metrics": (
            best_retention_constrained_k4_metrics
        ),
        "last_development_metrics": last_validation,
        "initial_k4_retention_metrics": initial_k4_retention,
        "last_k4_retention_metrics": last_k4_retention,
        "k4_replay_enabled": bool(replay_per_batch),
        "k4_replay_loss_fraction": (
            config.k4_replay_loss_fraction if replay_per_batch else 0.0
        ),
        "k4_replay_loss_mode": (
            config.k4_replay_loss_mode if replay_per_batch else None
        ),
        "initial_k4_retention_passed": (
            bool((initial_k4_retention or {}).get("retention_passed", False))
            if replay_per_batch
            else None
        ),
        "final_k4_retention_passed": (
            bool((last_k4_retention or {}).get("retention_passed", False))
            if replay_per_batch
            else None
        ),
        "last_train_metrics": last_train,
        "optimizer_audit": optimizer_audit,
        "augmentation_trace_sha256": trace.digest,
        "augmentation_trace_sample_count": trace.sample_count,
        "total_arc_training_samples": updates_completed
        * arc_per_batch
        * config.gradient_accumulation_steps,
        "total_k4_replay_samples": updates_completed
        * replay_per_batch
        * config.gradient_accumulation_steps,
        "total_training_samples": updates_completed
        * (arc_per_batch + replay_per_batch)
        * config.gradient_accumulation_steps,
        "total_forward_backward_batches": updates_completed
        * config.gradient_accumulation_steps
        * (2 if replay_per_batch else 1),
        "arc_forward_backward_batches": updates_completed
        * config.gradient_accumulation_steps,
        "k4_replay_forward_backward_batches": updates_completed
        * config.gradient_accumulation_steps
        * int(bool(replay_per_batch)),
        "k4_replay_audit": k4_replay_audit,
        "eligible_global_train_tasks": len(sampler.task_ids),
        "ineligible_global_train_task_ids": sampler.ineligible_task_ids,
        "solutions_loaded": False,
        "test_inputs_used_for_training": False,
        "peak_cuda_allocated_bytes": peak_cuda_allocated_bytes,
        "peak_cuda_reserved_bytes": peak_cuda_reserved_bytes,
        "peak_cuda_allocated_gib": (
            peak_cuda_allocated_bytes / float(1024**3)
            if peak_cuda_allocated_bytes is not None
            else None
        ),
        "peak_cuda_reserved_gib": (
            peak_cuda_reserved_bytes / float(1024**3)
            if peak_cuda_reserved_bytes is not None
            else None
        ),
        "elapsed_seconds": elapsed_before + time.perf_counter() - started,
    }
