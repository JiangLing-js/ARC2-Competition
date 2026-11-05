"""Strict Dense per-task TTT for the ARC ViT-5 hybrid model.

The adaptation path in this module deliberately accepts challenge tasks only.
Official test solutions are consumed by :func:`attach_final_diagnostics` after
all predictions have been fixed.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from .arc_supervised_dataset import prepare_and_format_samples
from .vit5_hybrid_batch import (
    build_vit5_hybrid_batch,
    encoder_kwargs_from_vit5_hybrid_batch,
)
from .vit5_overfit_diagnostics import (
    compare_grids,
    component_color_diagnostics,
    grid_shape,
)


METHOD_LABEL = "per-task specialization after 10-task joint adaptation"
SCHEMA_VERSION = 1
D4_TRANSFORM_NAMES = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "horizontal_flip",
    "vertical_flip",
    "main_diagonal_transpose",
    "anti_diagonal_transpose",
)
INVERSE_D4 = {
    "identity": "identity",
    "rot90": "rot270",
    "rot180": "rot180",
    "rot270": "rot90",
    "horizontal_flip": "horizontal_flip",
    "vertical_flip": "vertical_flip",
    "main_diagonal_transpose": "main_diagonal_transpose",
    "anti_diagonal_transpose": "anti_diagonal_transpose",
}


@dataclass(frozen=True)
class PerTaskTTTConfig:
    max_optimizer_updates: int = 2000
    lr: float = 1e-5
    min_lr: float = 1e-6
    warmup_updates: int = 50
    gradient_accumulation_steps: int = 4
    gradient_clip_norm: float = 1.0
    weight_decay: float = 0.0
    validate_every_updates: int = 25
    save_every_updates: int = 25
    early_stopping_patience: int = 20
    seed: int = 0
    transform_names: tuple[str, ...] = D4_TRANSFORM_NAMES
    shuffle_demonstrations: bool = True
    color_permutation: bool = False
    tta: bool = False
    tta_transform_names: tuple[str, ...] = D4_TRANSFORM_NAMES
    fallback_to_base: bool = True
    precision: str = "bf16"
    max_grids: int = 21

    def validate(self) -> None:
        if self.max_optimizer_updates < 0:
            raise ValueError("max_optimizer_updates must be non-negative.")
        if not 0 < self.lr or not 0 <= self.min_lr <= self.lr:
            raise ValueError("Require 0 <= min_lr <= lr and lr > 0.")
        if self.warmup_updates < 0:
            raise ValueError("warmup_updates must be non-negative.")
        for name in (
            "gradient_accumulation_steps",
            "validate_every_updates",
            "save_every_updates",
            "early_stopping_patience",
            "max_grids",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1.")
        if self.gradient_clip_norm <= 0 or self.weight_decay < 0:
            raise ValueError("gradient_clip_norm must be > 0 and weight_decay >= 0.")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16 or bf16.")
        validate_transform_names(self.transform_names)
        validate_transform_names(self.tta_transform_names)


def validate_grid(grid: Sequence[Sequence[int]]) -> list[list[int]]:
    normalized = [[int(value) for value in row] for row in grid]
    if not normalized or not normalized[0]:
        raise ValueError("ARC grids must be non-empty.")
    width = len(normalized[0])
    if any(len(row) != width for row in normalized):
        raise ValueError("ARC grids must be rectangular.")
    if any(value < 0 or value > 9 for row in normalized for value in row):
        raise ValueError("ARC colors must be in [0,9].")
    if len(normalized) > 30 or width > 30:
        raise ValueError("ARC grids must fit inside 30x30.")
    return normalized


def validate_transform_names(names: Iterable[str]) -> tuple[str, ...]:
    resolved = tuple(str(name) for name in names)
    unknown = sorted(set(resolved) - set(D4_TRANSFORM_NAMES))
    if unknown:
        raise ValueError(f"Unknown D4 transforms: {unknown}")
    if not resolved:
        raise ValueError("At least one transform is required.")
    return resolved


def transform_grid(grid: Sequence[Sequence[int]], name: str) -> list[list[int]]:
    """Apply one exact D4 transform, including rectangular grids."""
    values = validate_grid(grid)
    validate_transform_names((name,))
    if name == "identity":
        return copy.deepcopy(values)
    if name == "rot90":
        return [list(row) for row in zip(*values[::-1])]
    if name == "rot180":
        return [row[::-1] for row in values[::-1]]
    if name == "rot270":
        return [list(row) for row in zip(*values)][::-1]
    if name == "horizontal_flip":
        return [row[::-1] for row in values]
    if name == "vertical_flip":
        return values[::-1]
    if name == "main_diagonal_transpose":
        return [list(row) for row in zip(*values)]
    # Reflection across the anti-diagonal: (r,c) -> (W-1-c,H-1-r).
    height, width = len(values), len(values[0])
    output = [[0 for _ in range(height)] for _ in range(width)]
    for row in range(height):
        for col in range(width):
            output[width - 1 - col][height - 1 - row] = values[row][col]
    return output


def inverse_transform_grid(grid: Sequence[Sequence[int]], name: str) -> list[list[int]]:
    return transform_grid(grid, INVERSE_D4[name])


def task_nonzero_colors(task: Mapping[str, Any]) -> tuple[int, ...]:
    colors: set[int] = set()
    for pair in task.get("train", []):
        for key in ("input", "output"):
            colors.update(value for row in validate_grid(pair[key]) for value in row if value)
    for item in task.get("test", []):
        colors.update(value for row in validate_grid(item["input"]) for value in row if value)
    return tuple(sorted(colors))


def random_color_mapping(colors: Sequence[int], rng: random.Random) -> dict[int, int]:
    source = sorted({int(value) for value in colors if int(value) != 0})
    target = list(source)
    rng.shuffle(target)
    return {0: 0, **dict(zip(source, target))}


def apply_color_mapping(
    grid: Sequence[Sequence[int]], mapping: Mapping[int, int]
) -> list[list[int]]:
    values = validate_grid(grid)
    return [[int(mapping.get(value, value)) for value in row] for row in values]


def inverse_color_mapping(mapping: Mapping[int, int]) -> dict[int, int]:
    inverse = {int(value): int(key) for key, value in mapping.items()}
    if len(inverse) != len(mapping):
        raise ValueError("Color mapping must be one-to-one.")
    return inverse


def build_leave_one_out_samples(task: Mapping[str, Any], task_id: str) -> list[dict[str, Any]]:
    """Build identity LOO samples using train pairs only."""
    train_pairs = task.get("train", [])
    samples: list[dict[str, Any]] = []
    for heldout_index, heldout in enumerate(train_pairs):
        contexts = [
            {"input": copy.deepcopy(pair["input"]), "output": copy.deepcopy(pair["output"])}
            for index, pair in enumerate(train_pairs)
            if index != heldout_index
        ]
        samples.append(
            {
                "context_pairs": contexts,
                "query": copy.deepcopy(heldout["input"]),
                "target": copy.deepcopy(heldout["output"]),
                "heldout_index": heldout_index,
                "transform": "identity",
                "source": "per_task_ttt_leave_one_out_train_only",
                "task_id": task_id,
            }
        )
    return samples


def augment_leave_one_out_sample(
    sample: Mapping[str, Any],
    transform_name: str,
    rng: random.Random,
    shuffle_demonstrations: bool = True,
    color_mapping: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    """Synchronously transform every grid while retaining input/output pairs."""
    contexts = copy.deepcopy(sample["context_pairs"])
    if shuffle_demonstrations:
        rng.shuffle(contexts)

    def apply(grid: Sequence[Sequence[int]]) -> list[list[int]]:
        transformed = transform_grid(grid, transform_name)
        return apply_color_mapping(transformed, color_mapping or {})

    return {
        **{key: copy.deepcopy(value) for key, value in sample.items() if key not in {
            "context_pairs", "query", "target", "transform"
        }},
        "context_pairs": [
            {"input": apply(pair["input"]), "output": apply(pair["output"])}
            for pair in contexts
        ],
        "query": apply(sample["query"]),
        "target": apply(sample["target"]),
        "transform": transform_name,
        "color_mapping": dict(color_mapping or {0: 0}),
        "source": "per_task_ttt_dynamic_augmented_leave_one_out",
    }


def build_prediction_sample(
    task: Mapping[str, Any], task_id: str, test_index: int, transform_name: str = "identity"
) -> dict[str, Any]:
    """Build an encoder-only prediction sample without reading a test output.

    The formatter requires a target field, so a constant dummy target is used.
    The returned batch is consumed only through encoder kwargs.
    """
    item = task["test"][test_index]
    if "input" not in item:
        raise ValueError(f"Task {task_id} test {test_index} has no input.")
    contexts = [
        {
            "input": transform_grid(pair["input"], transform_name),
            "output": transform_grid(pair["output"], transform_name),
        }
        for pair in task.get("train", [])
    ]
    return {
        "context_pairs": contexts,
        "query": transform_grid(item["input"], transform_name),
        "target": [[0]],
        "heldout_index": None,
        "transform": transform_name,
        "source": "per_task_ttt_prediction_input_only",
        "task_id": task_id,
        "test_index": test_index,
        "prediction_only": True,
    }


def vit5_lora_named_parameters(model: Any) -> list[tuple[str, torch.nn.Parameter]]:
    expected_modules = set(getattr(model, "vit5_lora_targets", ()))
    if not expected_modules:
        if getattr(model, "vit5_initialization", None) == "random_scratch":
            raise ValueError(
                "random_scratch checkpoints do not contain ViT-5 LoRA parameters; "
                "use scratch dense_full TTT or a 0-step evaluation."
            )
        raise ValueError("Checkpoint model has no injected ViT-5 LoRA targets.")
    selected: list[tuple[str, torch.nn.Parameter]] = []
    seen_modules: set[str] = set()
    for name, parameter in model.named_parameters():
        if name.endswith(".lora_A.weight") or name.endswith(".lora_B.weight"):
            module_name = name.rsplit(".lora_", 1)[0]
            if module_name in expected_modules:
                selected.append((name, parameter))
                seen_modules.add(module_name)
    if seen_modules != expected_modules:
        raise ValueError(
            "ViT-5 LoRA parameter audit failed: "
            f"missing_modules={sorted(expected_modules - seen_modules)}, "
            f"unexpected_modules={sorted(seen_modules - expected_modules)}"
        )
    if len(selected) != 2 * len(expected_modules):
        raise ValueError("Each ViT-5 LoRA target must expose exactly A and B weights.")
    return selected


def configure_lora_only_training(model: Any) -> list[tuple[str, torch.nn.Parameter]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected = vit5_lora_named_parameters(model)
    for _, parameter in selected:
        parameter.requires_grad_(True)
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if [name for name, _ in trainable] != [name for name, _ in selected]:
        raise AssertionError("Trainable parameters are not exactly the ViT-5 LoRA parameters.")
    return selected


def create_lora_optimizer(
    model: Any, lr: float, weight_decay: float
) -> tuple[torch.optim.AdamW, list[tuple[str, torch.nn.Parameter]]]:
    selected = configure_lora_only_training(model)
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in selected], lr=lr, weight_decay=weight_decay
    )
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    selected_ids = {id(parameter) for _, parameter in selected}
    if optimizer_ids != selected_ids:
        raise AssertionError("Optimizer parameter IDs must exactly equal ViT-5 LoRA IDs.")
    return optimizer, selected


def snapshot_state(model: Any) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def state_sha256(
    state: Mapping[str, torch.Tensor], include_lora: bool = True
) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        is_lora = ".lora_A." in name or ".lora_B." in name
        if not include_lora and is_lora:
            continue
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def lora_state(model: Any) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in vit5_lora_named_parameters(model)
    }


def load_lora_state(model: Any, state: Mapping[str, torch.Tensor]) -> None:
    parameters = dict(vit5_lora_named_parameters(model))
    if set(parameters) != set(state):
        raise ValueError("LoRA state key mismatch.")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name].device, parameters[name].dtype))


def restore_global_base(
    model: Any, base_state: Mapping[str, torch.Tensor], expected_sha256: str
) -> None:
    model.load_state_dict(base_state, strict=True)
    current = snapshot_state(model)
    actual = state_sha256(current)
    if actual != expected_sha256:
        raise AssertionError(f"Global base restore hash mismatch: {actual} != {expected_sha256}")


def scheduled_lr(config: PerTaskTTTConfig, optimizer_update: int) -> float:
    """Linear warmup followed by cosine decay; update is one-indexed."""
    if optimizer_update < 1:
        return 0.0
    if config.warmup_updates and optimizer_update <= config.warmup_updates:
        return config.lr * optimizer_update / config.warmup_updates
    decay_updates = max(1, config.max_optimizer_updates - config.warmup_updates)
    progress = min(1.0, max(0.0, (optimizer_update - config.warmup_updates) / decay_updates))
    return config.min_lr + 0.5 * (config.lr - config.min_lr) * (1.0 + math.cos(math.pi * progress))


def validation_rank(metrics: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(metrics["exact_count"]),
        -float(metrics.get("target_error_cells", metrics["wrong_cells"])),
        float(metrics["pixel_accuracy"]),
        float(metrics["shape_exact_count"]),
        -float(metrics["validation_loss"]),
        -float(metrics["optimizer_update"]),
    )


def is_better_validation(candidate: Mapping[str, Any], best: Mapping[str, Any]) -> bool:
    return validation_rank(candidate) > validation_rank(best)


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _batch_from_samples(
    samples: list[dict[str, Any]], tokenizer: Any, device: torch.device, max_grids: int
) -> dict[str, Any]:
    prepared, formatted = prepare_and_format_samples(samples)
    return build_vit5_hybrid_batch(
        prepared, formatted, tokenizer=tokenizer, device=device, max_grids=max_grids
    )


def _dense_training_forward(model: Any, batch: Mapping[str, Any]) -> Any:
    return model(
        dense_target_ids=batch["dense_target_ids"].long(),
        dense_target_mask=batch["dense_target_mask"].bool(),
        dense_target_component_ids=batch["dense_target_component_ids"].long(),
        target_height=batch["target_height"].long(),
        target_width=batch["target_width"].long(),
        **encoder_kwargs_from_vit5_hybrid_batch(dict(batch)),
    )


@torch.no_grad()
def evaluate_leave_one_out(
    model: Any,
    samples: Sequence[dict[str, Any]],
    tokenizer: Any,
    device: torch.device,
    config: PerTaskTTTConfig,
    optimizer_update: int,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    exact_count = wrong_cells = target_error_cells = correct_cells = target_cells = shape_exact_count = 0
    records: list[dict[str, Any]] = []
    for sample in samples:
        batch = _batch_from_samples([dict(sample)], tokenizer, device, config.max_grids)
        with _autocast(device, config.precision):
            output = _dense_training_forward(model, batch)
        if output.dense_loss is None or not torch.isfinite(output.dense_loss):
            raise ValueError("LOO validation produced a non-finite loss.")
        dense = output.dense_output
        assert dense is not None
        decoded = model.predict_dense_grid(encoder_outputs=output.encoder_cache)[0]
        target = validate_grid(sample["target"])
        comparison = compare_grids(decoded["grid"], target)
        target_h, target_w = len(target), len(target[0])
        shape_exact = decoded["predicted_height"] == target_h and decoded["predicted_width"] == target_w
        losses.append(float(output.dense_loss.detach().float().cpu()))
        exact_count += int(comparison["exact"])
        wrong_cells += int(comparison["wrong_cells"])
        target_error_cells += int(comparison["target_error_cells"])
        correct_cells += int(comparison["correct_cells"])
        target_cells += int(comparison["target_cells"])
        shape_exact_count += int(shape_exact)
        records.append({
            "heldout_index": int(sample["heldout_index"]),
            "prediction": decoded["grid"],
            "target": target,
            "exact": bool(comparison["exact"]),
            "pixel_accuracy": float(comparison["pixel_accuracy"]),
            "wrong_cells": int(comparison["wrong_cells"]),
            "target_error_cells": int(comparison["target_error_cells"]),
            "shape_exact": bool(shape_exact),
        })
    if not losses:
        raise ValueError("LOO validation requires at least one sample.")
    return {
        "record_type": "leave_one_out_validation",
        "optimizer_update": int(optimizer_update),
        "sample_count": len(samples),
        "exact_count": exact_count,
        "wrong_cells": wrong_cells,
        "target_error_cells": target_error_cells,
        "pixel_accuracy": correct_cells / target_cells,
        "shape_exact_count": shape_exact_count,
        "validation_loss": sum(losses) / len(losses),
        "records": records,
    }


@torch.no_grad()
def predict_task_tests(
    model: Any,
    task: Mapping[str, Any],
    task_id: str,
    tokenizer: Any,
    device: torch.device,
    config: PerTaskTTTConfig,
    transform_names: Sequence[str] = ("identity",),
) -> list[dict[str, Any]]:
    model.eval()
    all_predictions: list[dict[str, Any]] = []
    for test_index in range(len(task.get("test", []))):
        candidates: list[dict[str, Any]] = []
        for transform_index, transform_name in enumerate(transform_names):
            sample = build_prediction_sample(task, task_id, test_index, transform_name)
            batch = _batch_from_samples([sample], tokenizer, device, config.max_grids)
            with _autocast(device, config.precision):
                result = model.predict_dense_grid(
                    include_raw_logits=True,
                    **encoder_kwargs_from_vit5_hybrid_batch(batch)
                )[0]
            color_logits = result.pop("color_logits")
            result.pop("height_logits", None)
            result.pop("width_logits", None)
            # BD checkpoints expose a 31x31x11 tensor whose last class is a
            # boundary marker, not an ARC color.  Prediction artifacts must
            # contain only legal 0..9 grids, including this diagnostic canvas.
            if tuple(color_logits.shape) == (31, 31, 11):
                full_color_logits = color_logits[:30, :30, :10]
            elif tuple(color_logits.shape) == (30, 30, 10):
                full_color_logits = color_logits
            else:
                raise ValueError(
                    "Unexpected Dense raw color-logit shape during prediction: "
                    f"{tuple(color_logits.shape)}."
                )
            full_color_grid = (
                full_color_logits.argmax(dim=-1).detach().cpu().tolist()
            )
            if any(
                not 0 <= int(value) <= 9
                for row in full_color_grid
                for value in row
            ):
                raise ValueError("Dense prediction artifact contains a non-ARC color.")
            restored_grid = inverse_transform_grid(result["grid"], transform_name)
            restored_full_color_grid = inverse_transform_grid(full_color_grid, transform_name)
            candidate = {
                "transform": transform_name,
                "transform_index": transform_index,
                "prediction": restored_grid,
                "predicted_height": len(restored_grid),
                "predicted_width": len(restored_grid[0]),
                "height_confidence": float(result["height_confidence"]),
                "width_confidence": float(result["width_confidence"]),
                "mean_cell_confidence": float(result["mean_cell_confidence"]),
                "full_color_grid": restored_full_color_grid,
            }
            if "shape_decoder_mode" in result:
                candidate["shape_decoder_mode"] = str(result["shape_decoder_mode"])
            if "joint_shape_confidence" in result:
                candidate["joint_shape_confidence"] = float(
                    result["joint_shape_confidence"]
                )
            candidates.append(candidate)
        all_predictions.append(select_whole_grid_vote(candidates))
    return all_predictions


def _grid_key(grid: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in grid)


def select_whole_grid_vote(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("TTA voting requires candidates.")
    groups: dict[tuple[tuple[int, ...], ...], list[dict[str, Any]]] = {}
    for candidate in candidates:
        groups.setdefault(_grid_key(candidate["prediction"]), []).append(candidate)

    def group_rank(item: tuple[tuple[tuple[int, ...], ...], list[dict[str, Any]]]):
        _, members = item
        confidence = sum(float(member["mean_cell_confidence"]) for member in members) / len(members)
        identity = any(member["transform"] == "identity" for member in members)
        first_index = min(int(member["transform_index"]) for member in members)
        return len(members), confidence, int(identity), -first_index

    winning_key, members = max(groups.items(), key=group_rank)
    representative = min(
        members,
        key=lambda item: (item["transform"] != "identity", int(item["transform_index"])),
    )
    selected = {
        "final_prediction": [list(row) for row in winning_key],
        "predicted_height": len(winning_key),
        "predicted_width": len(winning_key[0]),
        "vote_count": len(members),
        "candidate_count": len(candidates),
        "mean_cell_confidence": sum(float(item["mean_cell_confidence"]) for item in members) / len(members),
        "height_confidence": float(representative["height_confidence"]),
        "width_confidence": float(representative["width_confidence"]),
        "winning_transform": representative["transform"],
        "full_color_grid": representative["full_color_grid"],
        "candidates": list(candidates),
    }
    if "shape_decoder_mode" in representative:
        selected["shape_decoder_mode"] = representative["shape_decoder_mode"]
    if "joint_shape_confidence" in representative:
        selected["joint_shape_confidence"] = representative[
            "joint_shape_confidence"
        ]
    return selected


def _sample_dynamic_augmentation(
    loo_samples: Sequence[dict[str, Any]],
    task_colors: Sequence[int],
    rng: random.Random,
    config: PerTaskTTTConfig,
) -> dict[str, Any]:
    sample = loo_samples[rng.randrange(len(loo_samples))]
    transform_name = config.transform_names[rng.randrange(len(config.transform_names))]
    mapping = random_color_mapping(task_colors, rng) if config.color_permutation else {0: 0}
    return augment_leave_one_out_sample(
        sample,
        transform_name,
        rng,
        shuffle_demonstrations=config.shuffle_demonstrations,
        color_mapping=mapping,
    )


def adapt_one_task(
    model: Any,
    task: Mapping[str, Any],
    task_id: str,
    tokenizer: Any,
    device: torch.device,
    config: PerTaskTTTConfig,
    base_state: Mapping[str, torch.Tensor],
    base_sha256: str,
    metrics_handle: Any | None = None,
    progress_callback: Any | None = None,
    save_callback: Any | None = None,
) -> dict[str, Any]:
    """Adapt one task from the immutable global base and return fixed predictions."""
    config.validate()
    started = time.perf_counter()
    restore_global_base(model, base_state, base_sha256)
    base_lora = lora_state(model)
    non_lora_sha = state_sha256(snapshot_state(model), include_lora=False)
    loo_samples = build_leave_one_out_samples(task, task_id)
    task_seed = config.seed + int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16)
    rng = random.Random(task_seed)
    torch.manual_seed(task_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(task_seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Base test outputs are recorded before adaptation and never scored here.
    base_predictions = predict_task_tests(
        model, task, task_id, tokenizer, device, config, ("identity",)
    )
    if len(loo_samples) < 2:
        return {
            "task_id": task_id,
            "train_pair_count": len(loo_samples),
            "test_input_count": len(task.get("test", [])),
            "task_seed": task_seed,
            "skip_reason": "fewer_than_two_train_pairs",
            "lora_trainable_parameter_count": sum(value.numel() for value in base_lora.values()),
            "step0_leave_one_out_metrics": None,
            "validation_history": [],
            "best_optimizer_update": 0,
            "best_lr": 0.0,
            "stopping_update": 0,
            "early_stopped": False,
            "fell_back_to_step0": True,
            "base_test_predictions": base_predictions,
            "ttt_identity_predictions": base_predictions,
            "tta_predictions": base_predictions,
            "final_predictions": [item["final_prediction"] for item in base_predictions],
            "elapsed_seconds": time.perf_counter() - started,
            "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            "base_lora_sha256": state_sha256(base_lora),
            "best_lora_state": base_lora,
        }

    step0 = evaluate_leave_one_out(
        model, loo_samples, tokenizer, device, config, optimizer_update=0
    )
    history = [step0]
    best_metrics = step0
    best_state = base_lora
    best_lr = 0.0
    validations_without_improvement = 0
    early_stopped = False
    stopping_update = 0
    optimizer, selected = create_lora_optimizer(model, config.lr, config.weight_decay)
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if optimizer_ids != {id(parameter) for _, parameter in selected}:
        raise AssertionError("Per-task optimizer audit failed.")
    colors = task_nonzero_colors(task)

    for update in range(1, config.max_optimizer_updates + 1):
        stop_now = False
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(config.gradient_accumulation_steps):
            augmented = _sample_dynamic_augmentation(loo_samples, colors, rng, config)
            batch = _batch_from_samples([augmented], tokenizer, device, config.max_grids)
            with _autocast(device, config.precision):
                output = _dense_training_forward(model, batch)
                loss = output.dense_loss
                if loss is None or not torch.isfinite(loss):
                    raise ValueError(f"Task {task_id} produced non-finite Dense loss at update {update}.")
                scaled_loss = loss / config.gradient_accumulation_steps
            scaled_loss.backward()
            accumulated_loss += float(loss.detach().float().cpu())
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            [parameter for _, parameter in selected], config.gradient_clip_norm
        ).detach().cpu())
        current_lr = scheduled_lr(config, update)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.step()
        stopping_update = update
        training_record = {
            "record_type": "train_update",
            "task_id": task_id,
            "optimizer_update": update,
            "learning_rate": current_lr,
            "dense_loss": accumulated_loss / config.gradient_accumulation_steps,
            "pre_clip_grad_norm": grad_norm,
        }
        if metrics_handle is not None:
            metrics_handle.write(json.dumps(training_record, sort_keys=True) + "\n")

        if update % config.validate_every_updates == 0 or update == config.max_optimizer_updates:
            validation = evaluate_leave_one_out(
                model, loo_samples, tokenizer, device, config, optimizer_update=update
            )
            history.append(validation)
            if metrics_handle is not None:
                metrics_handle.write(json.dumps({**validation, "task_id": task_id}, sort_keys=True) + "\n")
                metrics_handle.flush()
            if is_better_validation(validation, best_metrics):
                best_metrics = validation
                best_state = lora_state(model)
                best_lr = current_lr
                validations_without_improvement = 0
            else:
                validations_without_improvement += 1
            if progress_callback is not None:
                progress_callback(task_id, update, training_record, validation)
            if validations_without_improvement >= config.early_stopping_patience:
                early_stopped = True
                stop_now = True

        if update % config.save_every_updates == 0:
            if metrics_handle is not None:
                metrics_handle.flush()
            if save_callback is not None:
                save_callback(
                    task_id,
                    update,
                    lora_state(model),
                    best_state,
                    best_metrics,
                )
        if stop_now:
            break

    load_lora_state(model, best_state)
    identity_predictions = predict_task_tests(
        model, task, task_id, tokenizer, device, config, ("identity",)
    )
    tta_predictions = (
        predict_task_tests(
            model, task, task_id, tokenizer, device, config, config.tta_transform_names
        )
        if config.tta
        else identity_predictions
    )
    if state_sha256(snapshot_state(model), include_lora=False) != non_lora_sha:
        raise AssertionError(f"Task {task_id} modified a non-LoRA parameter.")
    fell_back = int(best_metrics["optimizer_update"]) == 0
    if fell_back and not config.fallback_to_base:
        # Strict mode still retains step 0 as required; this flag is recorded for
        # compatibility but cannot disable the no-regression safety rule.
        pass
    elapsed = time.perf_counter() - started
    return {
        "task_id": task_id,
        "train_pair_count": len(loo_samples),
        "test_input_count": len(task.get("test", [])),
        "task_seed": task_seed,
        "skip_reason": None,
        "lora_trainable_parameter_count": sum(parameter.numel() for _, parameter in selected),
        "step0_leave_one_out_metrics": step0,
        "validation_history": history,
        "best_optimizer_update": int(best_metrics["optimizer_update"]),
        "best_lr": best_lr,
        "best_leave_one_out_metrics": best_metrics,
        "stopping_update": stopping_update,
        "early_stopped": early_stopped,
        "fell_back_to_step0": fell_back,
        "base_test_predictions": base_predictions,
        "ttt_identity_predictions": identity_predictions,
        "tta_predictions": tta_predictions,
        "final_predictions": [item["final_prediction"] for item in tta_predictions],
        "elapsed_seconds": elapsed,
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "base_lora_sha256": state_sha256(base_lora),
        "best_lora_sha256": state_sha256(best_state),
        "best_lora_state": best_state,
    }


def _stage_metrics(
    task_results: Sequence[Mapping[str, Any]],
    solutions: Mapping[str, Sequence[list[list[int]]]],
    prediction_key: str,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    solved_tasks = 0
    for task_result in task_results:
        task_id = str(task_result["task_id"])
        if task_id not in solutions:
            continue
        predictions = task_result[prediction_key]
        targets = solutions[task_id]
        if len(predictions) != len(targets):
            raise ValueError(f"Solution count mismatch for task {task_id}.")
        task_exact = True
        for prediction_item, target in zip(predictions, targets):
            predicted = prediction_item["final_prediction"] if isinstance(prediction_item, Mapping) else prediction_item
            comparison = compare_grids(predicted, target)
            target_h, target_w = grid_shape(target) or (0, 0)
            pred_h, pred_w = grid_shape(predicted) or (0, 0)
            oracle_source = prediction_item.get("full_color_grid") if isinstance(prediction_item, Mapping) else None
            oracle_grid = (
                [row[:target_w] for row in oracle_source[:target_h]]
                if oracle_source is not None
                else predicted
            )
            oracle_comparison = compare_grids(oracle_grid, target)
            component_metrics = component_color_diagnostics(predicted, target)
            task_exact = task_exact and bool(comparison["exact"])
            records.append({
                **comparison,
                "height_exact": pred_h == target_h,
                "width_exact": pred_w == target_w,
                "shape_exact": (pred_h, pred_w) == (target_h, target_w),
                "oracle_shape_pixel_accuracy": oracle_comparison["pixel_accuracy"],
                **component_metrics,
            })
        solved_tasks += int(task_exact)
    total_target = sum(int(item["target_cells"]) for item in records)
    total_correct = sum(int(item["correct_cells"]) for item in records)
    return {
        "sample_exact": sum(int(item["exact"]) for item in records),
        "sample_count": len(records),
        "macro_pixel_accuracy": sum(float(item["pixel_accuracy"]) for item in records) / len(records) if records else None,
        "micro_pixel_accuracy": total_correct / total_target if total_target else None,
        "oracle_shape_pixel_accuracy": sum(float(item["oracle_shape_pixel_accuracy"]) for item in records) / len(records) if records else None,
        "macro_color_precision": sum(float(item["macro_color_precision"]) for item in records) / len(records) if records else None,
        "macro_color_recall": sum(float(item["macro_color_recall"]) for item in records) / len(records) if records else None,
        "macro_color_iou": sum(float(item["macro_color_iou"]) for item in records) / len(records) if records else None,
        "component_recall_mean": sum(float(item["component_recall_mean"]) for item in records) / len(records) if records else None,
        "fully_recovered_components": sum(int(item["fully_recovered_components"]) for item in records),
        "fully_missing_components": sum(int(item["fully_missing_components"]) for item in records),
        "height_accuracy": sum(int(item["height_exact"]) for item in records) / len(records) if records else None,
        "width_accuracy": sum(int(item["width_exact"]) for item in records) / len(records) if records else None,
        "both_shape_accuracy": sum(int(item["shape_exact"]) for item in records) / len(records) if records else None,
        "wrong_cells_total": sum(int(item["wrong_cells"]) for item in records),
        "target_error_cells_total": sum(int(item["target_error_cells"]) for item in records),
        "missing_cells_total": sum(int(item["missing_cells"]) for item in records),
        "extra_cells_total": sum(int(item["extra_cells"]) for item in records),
        "solved_tasks": solved_tasks,
        "task_count": len({result["task_id"] for result in task_results if result["task_id"] in solutions}),
    }


def attach_final_diagnostics(
    task_results: list[dict[str, Any]],
    solutions: Mapping[str, Sequence[list[list[int]]]],
) -> dict[str, Any]:
    """Post-hoc scoring called only after all predictions are fixed."""
    base = _stage_metrics(task_results, solutions, "base_test_predictions")
    identity = _stage_metrics(task_results, solutions, "ttt_identity_predictions")
    final = _stage_metrics(task_results, solutions, "tta_predictions")
    improved = maintained = damaged = 0
    for result in task_results:
        task_id = result["task_id"]
        if task_id not in solutions:
            continue
        targets = solutions[task_id]
        base_solved = all(
            compare_grids(item["final_prediction"], target)["exact"]
            for item, target in zip(result["base_test_predictions"], targets)
        )
        final_solved = all(
            compare_grids(item["final_prediction"], target)["exact"]
            for item, target in zip(result["tta_predictions"], targets)
        )
        improved += int(final_solved and not base_solved)
        damaged += int(base_solved and not final_solved)
        maintained += int(base_solved == final_solved)
        for index, target in enumerate(targets):
            for key in ("base_test_predictions", "ttt_identity_predictions", "tta_predictions"):
                result[key][index]["official_diagnostics"] = compare_grids(
                    result[key][index]["final_prediction"], target
                )
                result[key][index]["official_diagnostics"].update(
                    component_color_diagnostics(
                        result[key][index]["final_prediction"], target
                    )
                )
    return {
        "base": base,
        "ttt_identity": identity,
        "ttt_tta": final,
        "improved_task_count": improved,
        "maintained_task_count": maintained,
        "damaged_task_count": damaged,
        "fallback_to_step0_task_count": sum(bool(result["fell_back_to_step0"]) for result in task_results),
        "best_update_distribution": [int(result["best_optimizer_update"]) for result in task_results],
        "per_task_elapsed_seconds": {result["task_id"]: result["elapsed_seconds"] for result in task_results},
    }


def serializable_task_result(result: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "best_lora_state"}


def save_run_outputs(
    output_dir: str | Path,
    task_results: list[dict[str, Any]],
    config: PerTaskTTTConfig,
    environment: Mapping[str, Any],
    base_checkpoint: str,
    base_sha256: str,
    diagnostics: Mapping[str, Any] | None,
) -> dict[str, Any]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    serializable = [serializable_task_result(result) for result in task_results]
    predictions = {
        result["task_id"]: result["final_predictions"] for result in serializable
    }
    (target / "per_task_ttt_predictions.json").write_text(
        json.dumps({"method_label": METHOD_LABEL, "tasks": serializable, "predictions": predictions}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    csv_fields = (
        "task_id", "test_index", "predicted_height", "predicted_width",
        "mean_cell_confidence", "height_confidence", "width_confidence",
        "best_optimizer_update", "fell_back_to_step0", "official_exact",
        "official_pixel_accuracy",
    )
    with (target / "per_task_ttt_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for result in serializable:
            for index, item in enumerate(result["tta_predictions"]):
                official = item.get("official_diagnostics", {})
                writer.writerow({
                    "task_id": result["task_id"],
                    "test_index": index,
                    "predicted_height": item["predicted_height"],
                    "predicted_width": item["predicted_width"],
                    "mean_cell_confidence": item["mean_cell_confidence"],
                    "height_confidence": item["height_confidence"],
                    "width_confidence": item["width_confidence"],
                    "best_optimizer_update": result["best_optimizer_update"],
                    "fell_back_to_step0": result["fell_back_to_step0"],
                    "official_exact": official.get("exact"),
                    "official_pixel_accuracy": official.get("pixel_accuracy"),
                })
    summary = {
        "schema_version": SCHEMA_VERSION,
        "method_label": METHOD_LABEL,
        "strict_unseen_task_ttt": False,
        "base_checkpoint": base_checkpoint,
        "base_state_sha256": base_sha256,
        "task_count": len(task_results),
        "test_sample_count": sum(result["test_input_count"] for result in task_results),
        "diagnostics": diagnostics,
    }
    (target / "per_task_ttt_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (target / "config_resolved.json").write_text(
        json.dumps({**asdict(config), "method_label": METHOD_LABEL}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (target / "environment.json").write_text(
        json.dumps(dict(environment), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
