"""Checkpoint helpers for project-local ARC ViTARC/TTT models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .hf_arc_tokenizer import load_t5_tokenizer_with_arc_tokens
from .pretrained_t5_ope import add_lora_to_t5_if_available, build_pretrained_t5_ope_model
from .tiny_t5_ope import build_tiny_t5_ope_model
from .tiny_t5_rpe_encoder import build_tiny_t5_rpe_encoder_model
from .vit5_dense_decoder import (
    DENSE_CANDIDATE_FUSION_TYPE,
    DENSE_CANDIDATE_GATE_INITIAL_VALUE,
    DENSE_CANDIDATE_GATE_PARAMETERIZATION,
    DENSE_CONDITIONING_FUSION_TYPE,
    DENSE_CONDITIONING_FORCED_PAIRED,
    DENSE_CONDITIONING_GATED_PAIRED,
    DENSE_CONDITIONING_LEGACY,
    DENSE_CONDITIONING_SCHEMA_VERSION,
    DENSE_CONDITIONING_SAME_COORDINATE,
    DENSE_CONDITIONING_SPLIT,
    DENSE_FORCED_PAIRED_FUSION_TYPE,
    DENSE_FORCED_PAIRED_NO_CONTEXT_POLICY,
    DENSE_FORCED_PAIRED_PAIRING,
    DENSE_FORCED_PAIRED_SCHEMA_VERSION,
    DENSE_FORCED_PAIRED_SOURCE,
    DENSE_GATED_PAIRED_FUSION_TYPE,
    DENSE_GATED_PAIRED_GATE_INPUT,
    DENSE_GATED_PAIRED_GATE_PARAMETERIZATION,
    DENSE_GATED_PAIRED_INJECTION_POSITION,
    DENSE_GATED_PAIRED_SCHEMA_VERSION,
    DENSE_GATED_RELATIONAL_SCHEMA_VERSION,
    DENSE_RELATIONAL_MEMORY_DISABLED,
    DENSE_RELATIONAL_MEMORY_FUSION_TYPE,
    DENSE_RELATIONAL_MEMORY_GATE_INPUT,
    DENSE_RELATIONAL_MEMORY_GATE_PARAMETERIZATION,
    DENSE_RELATIONAL_MEMORY_GEOMETRY,
    DENSE_RELATIONAL_MEMORY_INJECTION_POSITION,
    DENSE_RELATIONAL_MEMORY_MODES,
    DENSE_RELATIONAL_MEMORY_PAIRING,
    DENSE_RELATIONAL_MEMORY_SCHEMA_VERSION,
    DENSE_RELATIONAL_MEMORY_SET_CROSS_ATTENTION_V2,
    DENSE_RELATIONAL_MEMORY_SOURCE,
    DENSE_SAME_COORDINATE_FUSION_TYPE,
    DENSE_SAME_COORDINATE_GATE_INITIAL_VALUE,
    DENSE_SAME_COORDINATE_GATE_PARAMETERIZATION,
    DENSE_SAME_COORDINATE_SCHEMA_VERSION,
    DENSE_SUPPORTED_CONDITIONING_MODES,
    DENSE_PAIRED_CONDITIONING_MODES,
    DENSE_RECURSION_SCHEMA_VERSION,
    DENSE_BD_BORDER_SCHEMA_VERSION,
    DENSE_BD_CANVAS_SIZE,
    DENSE_BD_CLASS_ID,
    DENSE_BD_NUM_CLASSES,
    DENSE_SHAPE_DECODER_BD_BORDER,
    DENSE_SHAPE_DECODER_MODES,
    DENSE_SHAPE_DECODER_SEPARATE_HW,
)
from .vit5_hybrid_model import build_vit5_hybrid_model
from .vit5_hybrid_batch import VIT5_HYBRID_BATCH_SCHEMA_VERSION


def _load_local_torch_payload(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> Any:
    """Load a trusted project-local payload across PyTorch weights_only defaults."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch versions predating the weights_only argument.
        return torch.load(path, map_location=map_location)


def _lightweight_training_state_metadata(
    training_state: dict | None,
) -> dict[str, Any]:
    """Return JSON-only provenance; the complete state lives in the .pt payload."""
    state = training_state or {}
    summary: dict[str, Any] = {
        "stored_in": "hybrid_trainable_state.pt",
        "binary_training_state": True,
    }
    for key in ("format", "optimizer_update", "global_step", "epoch"):
        value = state.get(key)
        if value is None or isinstance(value, (str, int, float, bool)):
            if value is not None:
                summary[key] = value
    return summary


def parameter_dtype_summary(model: Any) -> dict[str, int]:
    """Count parameters by dtype for checkpoint metadata."""
    summary: dict[str, int] = {}
    for parameter in model.parameters():
        key = str(parameter.dtype)
        summary[key] = summary.get(key, 0) + parameter.numel()
    return summary


def save_checkpoint(
    path: str | Path,
    model: Any,
    token_to_id: dict[str, int],
    model_config: dict,
    training_state: dict | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
) -> None:
    """Save model state, fixed vocab, config, and training metadata."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    saved_config = dict(model_config)
    if saved_config.get("model_type") == "vit5_hybrid":
        saved_config["vocabulary_info"] = dict(model.vocabulary_info)
        saved_config["original_model_vocab_size"] = int(model.original_model_vocab_size)
        saved_config["tokenizer_length"] = int(model.tokenizer_length)
        saved_config["resolved_vocab_size"] = int(model.resolved_vocab_size)
        saved_config["is_test_fixture"] = bool(model.is_test_fixture)
        saved_config["decoder_mode"] = model.decoder_mode
        saved_config["dense_decoder_config"] = model.dense_decoder_config
        saved_config["dense_shape_decoder_mode"] = getattr(
            model, "dense_shape_decoder_mode", DENSE_SHAPE_DECODER_SEPARATE_HW
        )
        saved_config["dense_conditioning_mode"] = (
            model.dense_decoder.conditioning_mode
            if model.dense_decoder is not None
            else None
        )
        saved_config["dense_conditioning_schema_version"] = (
            int(model.dense_decoder_config["conditioning_schema_version"])
            if model.dense_decoder_config is not None
            else None
        )
        relational_mode = (model.dense_decoder_config or {}).get(
            "relational_memory_mode", DENSE_RELATIONAL_MEMORY_DISABLED
        )
        if relational_mode != DENSE_RELATIONAL_MEMORY_DISABLED:
            saved_config["dense_relational_memory_mode"] = relational_mode
            saved_config["dense_relational_gate_initial_value"] = float(
                model.dense_decoder_config[
                    "relational_memory_gate_initial_effective_scale"
                ]
            )
        else:
            saved_config.pop("dense_relational_memory_mode", None)
            saved_config.pop("dense_relational_gate_initial_value", None)
        saved_config["hybrid_training_strategy"] = model.hybrid_training_strategy
        saved_config["lambda_ar"] = model.lambda_ar
        saved_config["lambda_dense"] = model.lambda_dense
        saved_config["lambda_shape"] = model.lambda_shape
        saved_config["lambda_component"] = model.lambda_component
        saved_config["batch_schema_version"] = VIT5_HYBRID_BATCH_SCHEMA_VERSION
        saved_config["loss_schema_version"] = model.loss_schema_version
        saved_config["target_component_mode"] = model.target_component_mode
        saved_config["component_loss_type"] = model.component_loss_type
        saved_config["object_embedding_mode"] = model.object_embedding_mode
        saved_config["vit5_initialization"] = getattr(
            model, "vit5_initialization", "pretrained_lora"
        )
        saved_config["vit5_random_initialization_report"] = (
            model.vit5_random_initialization_report.to_dict()
            if getattr(model, "vit5_random_initialization_report", None) is not None
            else None
        )
        saved_config["max_objects"] = int(model.max_objects)
        saved_config["object_connectivity"] = 4
        saved_config["object_background_color"] = 0
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "token_to_id": dict(token_to_id),
            "model_config": saved_config,
            "training_state": training_state or {},
            "parameter_dtype_summary": parameter_dtype_summary(model),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        },
        checkpoint_path,
    )


def _get_lm_head(model: Any) -> Any | None:
    if hasattr(model, "_get_lm_head"):
        return model._get_lm_head()
    t5 = getattr(model, "t5", model)
    if hasattr(t5, "lm_head"):
        return t5.lm_head
    base_model = getattr(t5, "base_model", None)
    nested_model = getattr(base_model, "model", None)
    return getattr(nested_model, "lm_head", None)


def _embedding_rows(module: Any, token_ids: list[int]) -> torch.Tensor | None:
    if module is None or not hasattr(module, "weight"):
        return None
    valid_ids = [token_id for token_id in token_ids if 0 <= token_id < module.weight.shape[0]]
    if not valid_ids:
        return None
    return module.weight.detach().cpu()[valid_ids].clone()


def save_arc_token_embeddings(
    path: str | Path,
    model: Any,
    added_token_ids: list[int],
) -> None:
    """Save only the added ARC/project token embedding rows."""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    token_ids = [int(token_id) for token_id in added_token_ids]
    t5 = getattr(model, "t5", model)
    shared = t5.get_input_embeddings()
    encoder_embeddings = getattr(t5.get_encoder(), "embed_tokens", None)
    decoder_embeddings = getattr(t5.get_decoder(), "embed_tokens", None)
    lm_head = _get_lm_head(model)
    payload = {
        "added_token_ids": token_ids,
        "shared": _embedding_rows(shared, token_ids),
        "encoder": _embedding_rows(encoder_embeddings, token_ids),
        "decoder": _embedding_rows(decoder_embeddings, token_ids),
        "lm_head": _embedding_rows(lm_head, token_ids),
    }
    torch.save(payload, target_path)


def load_arc_token_embeddings(
    path: str | Path,
    model: Any,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load added-token embedding rows into a rebuilt pretrained model."""
    payload = torch.load(Path(path), map_location=map_location)
    token_ids = [int(token_id) for token_id in payload.get("added_token_ids", [])]
    t5 = getattr(model, "t5", model)

    def _copy_rows(module: Any, rows: torch.Tensor | None) -> bool:
        if module is None or rows is None or not hasattr(module, "weight"):
            return False
        with torch.no_grad():
            valid_pairs = [
                (row_index, token_id)
                for row_index, token_id in enumerate(token_ids)
                if 0 <= token_id < module.weight.shape[0]
            ]
            if not valid_pairs:
                return False
            source_rows = torch.stack([rows[row_index] for row_index, _ in valid_pairs]).to(
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
            target_ids = torch.tensor(
                [token_id for _, token_id in valid_pairs],
                dtype=torch.long,
                device=module.weight.device,
            )
            module.weight.index_copy_(0, target_ids, source_rows)
        return True

    loaded = {
        "shared": _copy_rows(t5.get_input_embeddings(), payload.get("shared")),
        "encoder": _copy_rows(getattr(t5.get_encoder(), "embed_tokens", None), payload.get("encoder")),
        "decoder": _copy_rows(getattr(t5.get_decoder(), "embed_tokens", None), payload.get("decoder")),
        "lm_head": _copy_rows(_get_lm_head(model), payload.get("lm_head")),
        "added_token_ids": token_ids,
    }
    return loaded


def save_lora_only_checkpoint(
    output_dir: str | Path,
    model: Any,
    tokenizer: Any,
    token_to_id: dict[str, int],
    model_config: dict,
    training_state: dict | None = None,
) -> dict[str, str]:
    """Save a lightweight pretrained_ope checkpoint with LoRA and ARC token rows."""
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    lora_dir = target_dir / "lora_adapter"
    tokenizer_dir = target_dir / "tokenizer"
    embeddings_path = target_dir / "arc_token_embeddings.pt"
    meta_path = target_dir / "checkpoint_meta.json"

    model.t5.save_pretrained(lora_dir)
    tokenizer.save_pretrained(tokenizer_dir)
    added_token_ids = [int(token_id) for token_id in model_config.get("added_token_ids", [])]
    save_arc_token_embeddings(embeddings_path, model, added_token_ids)

    checkpoint_format = "pretrained_ope_lora_only"
    saved_config = dict(model_config)
    saved_config.update(
        {
            "checkpoint_format": checkpoint_format,
            "lora_adapter_dir": str(lora_dir),
            "tokenizer_dir": str(tokenizer_dir),
            "arc_token_embeddings_path": str(embeddings_path),
            "lora_enabled": True,
        }
    )
    meta = {
        "checkpoint_format": checkpoint_format,
        "token_to_id": dict(token_to_id),
        "model_config": saved_config,
        "training_state": training_state or {},
        "parameter_dtype_summary": parameter_dtype_summary(model),
    }
    import json

    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    return {
        "lora_adapter_dir": str(lora_dir),
        "tokenizer_dir": str(tokenizer_dir),
        "arc_token_embeddings_path": str(embeddings_path),
        "checkpoint_meta": str(meta_path),
    }


VIT5_HYBRID_LIGHTWEIGHT_FORMAT = "vit5_hybrid_lightweight_v1"
VIT5_HYBRID_ARC_FULL_FORMAT = "vit5_hybrid_arc_full_v1"
VIT5_HYBRID_OBJECT_EMBEDDING_KEY = (
    "arc_encoder.grid_encoder.arc_embedding.object_embedding.weight"
)
VIT5_HYBRID_TRAINABLE_PREFIXES = (
    "arc_token_delta.",
    "arc_encoder.grid_encoder.arc_embedding.",
    "arc_encoder.task_fusion.",
    "arc_encoder.cell_refine_norm.",
    "arc_encoder.cell_to_grid_attention.",
    "arc_encoder.cell_refine_gamma",
    "encoder_to_t5.",
    "dense_decoder.",
)
VIT5_HYBRID_ARC_FULL_PREFIXES = (
    "arc_token_delta.",
    "arc_encoder.",
    "encoder_to_t5.",
    "dense_decoder.",
)


def _vit5_hybrid_lightweight_state(model: Any) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    selected: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        is_new_module = name.startswith(VIT5_HYBRID_TRAINABLE_PREFIXES)
        is_lora = ".lora_A." in name or ".lora_B." in name
        if is_new_module or is_lora:
            selected[name] = value.detach().cpu().clone()
    if not selected:
        raise ValueError("vit5_hybrid lightweight state selection produced no tensors.")
    return selected


def _vit5_hybrid_arc_full_state(model: Any) -> dict[str, torch.Tensor]:
    """Return every ARC-side tensor, including the complete grid backbone."""
    selected = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name.startswith(VIT5_HYBRID_ARC_FULL_PREFIXES)
    }
    grid_keys = [
        name for name in selected if name.startswith("arc_encoder.grid_encoder.")
    ]
    if not grid_keys or not selected:
        raise ValueError("ARC-full checkpoint selection produced no grid encoder state.")
    return selected


def save_vit5_hybrid_lightweight_checkpoint(
    output_dir: str | Path,
    model: Any,
    tokenizer: Any,
    token_to_id: dict[str, int],
    model_config: dict,
    training_state: dict | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
) -> dict[str, str]:
    """Save frozen-base autoregressive hybrid trainable state and ARC deltas."""
    if not bool(getattr(model, "freeze_vit5", False)):
        raise ValueError("Lightweight vit5_hybrid checkpoint requires freeze_vit5=True.")
    if not bool(getattr(model, "freeze_t5", False)):
        raise ValueError("Lightweight vit5_hybrid checkpoint requires freeze_t5=True.")
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = target_dir / "tokenizer"
    state_path = target_dir / "hybrid_trainable_state.pt"
    meta_path = target_dir / "checkpoint_meta.json"
    tokenizer.save_pretrained(tokenizer_dir)
    state_tmp_path = state_path.with_name(state_path.name + ".tmp")
    torch.save(
        {
            "format": VIT5_HYBRID_LIGHTWEIGHT_FORMAT,
            "state_dict": _vit5_hybrid_lightweight_state(model),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            # RNG states and sampler state contain tensors/tuples and therefore
            # must never be serialized through checkpoint_meta.json.
            "training_state": training_state or {},
        },
        state_tmp_path,
    )
    state_tmp_path.replace(state_path)
    saved_config = dict(model_config)
    saved_config.update(
        {
            "checkpoint_format": VIT5_HYBRID_LIGHTWEIGHT_FORMAT,
            "tokenizer_dir": "tokenizer",
            "hybrid_trainable_state_path": "hybrid_trainable_state.pt",
            "vit5_checkpoint_path": str(Path(saved_config["vit5_checkpoint_path"]).resolve()),
            "vocabulary_info": dict(model.vocabulary_info),
            "original_model_vocab_size": int(model.original_model_vocab_size),
            "tokenizer_length": int(model.tokenizer_length),
            "resolved_vocab_size": int(model.resolved_vocab_size),
            "is_test_fixture": bool(model.is_test_fixture),
            "decoder_mode": model.decoder_mode,
            "dense_decoder_config": model.dense_decoder_config,
            "dense_shape_decoder_mode": getattr(
                model, "dense_shape_decoder_mode", DENSE_SHAPE_DECODER_SEPARATE_HW
            ),
            "dense_conditioning_mode": (
                model.dense_decoder.conditioning_mode
                if model.dense_decoder is not None
                else None
            ),
            "dense_conditioning_schema_version": (
                int(model.dense_decoder_config["conditioning_schema_version"])
                if model.dense_decoder_config is not None
                else None
            ),
            "hybrid_training_strategy": model.hybrid_training_strategy,
            "lambda_ar": model.lambda_ar,
            "lambda_dense": model.lambda_dense,
            "lambda_shape": model.lambda_shape,
            "lambda_component": model.lambda_component,
            "batch_schema_version": VIT5_HYBRID_BATCH_SCHEMA_VERSION,
            "loss_schema_version": model.loss_schema_version,
            "target_component_mode": model.target_component_mode,
            "component_loss_type": model.component_loss_type,
            "object_embedding_mode": model.object_embedding_mode,
            "vit5_initialization": getattr(
                model, "vit5_initialization", "pretrained_lora"
            ),
            "vit5_random_initialization_report": (
                model.vit5_random_initialization_report.to_dict()
                if getattr(model, "vit5_random_initialization_report", None) is not None
                else None
            ),
            "max_objects": int(model.max_objects),
            "object_connectivity": 4,
            "object_background_color": 0,
        }
    )
    relational_mode = (model.dense_decoder_config or {}).get(
        "relational_memory_mode", DENSE_RELATIONAL_MEMORY_DISABLED
    )
    if relational_mode != DENSE_RELATIONAL_MEMORY_DISABLED:
        saved_config["dense_relational_memory_mode"] = relational_mode
        saved_config["dense_relational_gate_initial_value"] = float(
            model.dense_decoder_config[
                "relational_memory_gate_initial_effective_scale"
            ]
        )
    else:
        saved_config.pop("dense_relational_memory_mode", None)
        saved_config.pop("dense_relational_gate_initial_value", None)
    meta = {
        "checkpoint_format": VIT5_HYBRID_LIGHTWEIGHT_FORMAT,
        "token_to_id": dict(token_to_id),
        "model_config": saved_config,
        "training_state": _lightweight_training_state_metadata(training_state),
        "parameter_dtype_summary": parameter_dtype_summary(model),
    }
    import json

    meta_tmp_path = meta_path.with_name(meta_path.name + ".tmp")
    with meta_tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    meta_tmp_path.replace(meta_path)
    return {
        "tokenizer_dir": str(tokenizer_dir),
        "hybrid_trainable_state_path": str(state_path),
        "checkpoint_meta": str(meta_path),
    }


def save_vit5_hybrid_arc_full_checkpoint(
    output_dir: str | Path,
    model: Any,
    tokenizer: Any,
    token_to_id: dict[str, int],
    model_config: dict,
    training_state: dict | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
) -> dict[str, str]:
    """Save a portable scratch checkpoint with the complete ARC backbone.

    The frozen T5 payload remains referenced by ``pretrained_model_name_or_path``;
    every ARC-side parameter and buffer, including all twelve grid-transformer
    blocks, is serialized and checked as an exact key set during restoration.
    """
    if getattr(model, "vit5_initialization", None) != "random_scratch":
        raise ValueError("ARC-full checkpoint requires random_scratch initialization.")
    if bool(getattr(model, "freeze_vit5", True)):
        raise ValueError("ARC-full checkpoint requires freeze_vit5=False.")
    if getattr(model, "vit5_lora_targets", None):
        raise ValueError("ARC-full scratch checkpoints must not contain ViT-5 LoRA.")

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = target_dir / "tokenizer"
    state_path = target_dir / "hybrid_arc_full_state.pt"
    meta_path = target_dir / "checkpoint_meta.json"
    tokenizer.save_pretrained(tokenizer_dir)

    state_tmp_path = state_path.with_name(state_path.name + ".tmp")
    torch.save(
        {
            "format": VIT5_HYBRID_ARC_FULL_FORMAT,
            "state_dict": _vit5_hybrid_arc_full_state(model),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "training_state": training_state or {},
        },
        state_tmp_path,
    )
    state_tmp_path.replace(state_path)

    saved_config = dict(model_config)
    saved_config.update(
        {
            "checkpoint_format": VIT5_HYBRID_ARC_FULL_FORMAT,
            "tokenizer_dir": "tokenizer",
            "hybrid_arc_full_state_path": "hybrid_arc_full_state.pt",
            "vit5_checkpoint_path": None,
            "vit5_initialization": "random_scratch",
            "vit5_random_initialization_report": (
                model.vit5_random_initialization_report.to_dict()
                if model.vit5_random_initialization_report is not None
                else None
            ),
            "vocabulary_info": dict(model.vocabulary_info),
            "original_model_vocab_size": int(model.original_model_vocab_size),
            "tokenizer_length": int(model.tokenizer_length),
            "resolved_vocab_size": int(model.resolved_vocab_size),
            "is_test_fixture": bool(model.is_test_fixture),
            "decoder_mode": model.decoder_mode,
            "dense_decoder_config": model.dense_decoder_config,
            "dense_shape_decoder_mode": getattr(
                model, "dense_shape_decoder_mode", DENSE_SHAPE_DECODER_SEPARATE_HW
            ),
            "dense_conditioning_mode": (
                model.dense_decoder.conditioning_mode
                if model.dense_decoder is not None
                else None
            ),
            "dense_conditioning_schema_version": (
                int(model.dense_decoder_config["conditioning_schema_version"])
                if model.dense_decoder_config is not None
                else None
            ),
            "hybrid_training_strategy": model.hybrid_training_strategy,
            "lambda_ar": model.lambda_ar,
            "lambda_dense": model.lambda_dense,
            "lambda_shape": model.lambda_shape,
            "lambda_component": model.lambda_component,
            "batch_schema_version": VIT5_HYBRID_BATCH_SCHEMA_VERSION,
            "loss_schema_version": model.loss_schema_version,
            "target_component_mode": model.target_component_mode,
            "component_loss_type": model.component_loss_type,
            "object_embedding_mode": model.object_embedding_mode,
            "max_objects": int(model.max_objects),
            "object_connectivity": 4,
            "object_background_color": 0,
        }
    )
    relational_mode = (model.dense_decoder_config or {}).get(
        "relational_memory_mode", DENSE_RELATIONAL_MEMORY_DISABLED
    )
    if relational_mode != DENSE_RELATIONAL_MEMORY_DISABLED:
        saved_config["dense_relational_memory_mode"] = relational_mode
        saved_config["dense_relational_gate_initial_value"] = float(
            model.dense_decoder_config[
                "relational_memory_gate_initial_effective_scale"
            ]
        )
    else:
        saved_config.pop("dense_relational_memory_mode", None)
        saved_config.pop("dense_relational_gate_initial_value", None)
    meta = {
        "checkpoint_format": VIT5_HYBRID_ARC_FULL_FORMAT,
        "token_to_id": dict(token_to_id),
        "model_config": saved_config,
        "training_state": _lightweight_training_state_metadata(training_state),
        "parameter_dtype_summary": parameter_dtype_summary(model),
    }
    import json

    meta_tmp_path = meta_path.with_name(meta_path.name + ".tmp")
    with meta_tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    meta_tmp_path.replace(meta_path)
    return {
        "tokenizer_dir": str(tokenizer_dir),
        "hybrid_arc_full_state_path": str(state_path),
        "checkpoint_meta": str(meta_path),
    }


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Load and validate a project-local torch checkpoint."""
    checkpoint_path = Path(path)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "checkpoint_meta.json"
    if checkpoint_path.suffix.lower() == ".json":
        import json

        with checkpoint_path.open("r", encoding="utf-8") as handle:
            checkpoint = json.load(handle)
        checkpoint["_checkpoint_base_dir"] = str(checkpoint_path.parent.resolve())
        if checkpoint.get("checkpoint_format") in {
            VIT5_HYBRID_LIGHTWEIGHT_FORMAT,
            VIT5_HYBRID_ARC_FULL_FORMAT,
        }:
            path_key = (
                "hybrid_arc_full_state_path"
                if checkpoint.get("checkpoint_format") == VIT5_HYBRID_ARC_FULL_FORMAT
                else "hybrid_trainable_state_path"
            )
            state_value = checkpoint.get("model_config", {}).get(path_key)
            if not state_value:
                raise ValueError(
                    "Directory vit5_hybrid metadata is missing its state payload path."
                )
            state_path = Path(state_value)
            if not state_path.is_absolute():
                state_path = checkpoint_path.parent / state_path
            payload = _load_local_torch_payload(state_path, map_location)
            if "training_state" in payload:
                checkpoint["training_state"] = payload["training_state"]
            optimizer_state = payload.get("optimizer_state_dict")
            scheduler_state = payload.get("scheduler_state_dict")
            # Match full-checkpoint callers while retaining the underscored
            # aliases used by the clean-global launcher added earlier.
            checkpoint["optimizer_state_dict"] = optimizer_state
            checkpoint["scheduler_state_dict"] = scheduler_state
            checkpoint["_optimizer_state_dict"] = optimizer_state
            checkpoint["_scheduler_state_dict"] = scheduler_state
    else:
        checkpoint = _load_local_torch_payload(checkpoint_path, map_location)
    required = ("token_to_id", "model_config", "training_state")
    for key in required:
        if key not in checkpoint:
            raise KeyError(f"Checkpoint is missing required key: {key}")
    lightweight_formats = {
        "pretrained_ope_lora_only",
        VIT5_HYBRID_LIGHTWEIGHT_FORMAT,
        VIT5_HYBRID_ARC_FULL_FORMAT,
    }
    if checkpoint.get("checkpoint_format") not in lightweight_formats and "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint is missing required key: model_state_dict")
    return checkpoint


def build_model_from_checkpoint(
    checkpoint: dict,
    device: str | torch.device = "cpu",
):
    """Rebuild a supported model from checkpoint config and load weights."""
    config = dict(checkpoint["model_config"])
    model_type = config.get("model_type")
    if model_type not in {"ope", "rpe_encoder", "pretrained_ope", "vit5_hybrid"}:
        raise ValueError(f"Unknown model_type in checkpoint: {model_type}")
    if model_type == "vit5_hybrid":
        base_dir = Path(checkpoint.get("_checkpoint_base_dir", "."))
        dense_config = config.get("dense_decoder_config") or {}
        top_level_shape_decoder = config.get("dense_shape_decoder_mode")
        nested_shape_decoder = dense_config.get("shape_decoder_mode")
        if (
            top_level_shape_decoder is not None
            and nested_shape_decoder is not None
            and top_level_shape_decoder != nested_shape_decoder
        ):
            raise ValueError(
                "Dense shape decoder checkpoint config conflict: "
                f"dense_shape_decoder_mode={top_level_shape_decoder!r}, "
                "dense_decoder_config.shape_decoder_mode="
                f"{nested_shape_decoder!r}."
            )
        shape_decoder_mode = (
            nested_shape_decoder
            or top_level_shape_decoder
            or DENSE_SHAPE_DECODER_SEPARATE_HW
        )
        if shape_decoder_mode not in DENSE_SHAPE_DECODER_MODES:
            raise ValueError(
                f"Unsupported Dense shape decoder mode: {shape_decoder_mode!r}."
            )
        if shape_decoder_mode == DENSE_SHAPE_DECODER_BD_BORDER:
            if (
                top_level_shape_decoder != DENSE_SHAPE_DECODER_BD_BORDER
                or nested_shape_decoder != DENSE_SHAPE_DECODER_BD_BORDER
            ):
                raise ValueError(
                    "BD checkpoints must declare matching top-level and nested "
                    "shape decoder modes."
                )
            expected_bd_contract = {
                "bd_border_schema_version": DENSE_BD_BORDER_SCHEMA_VERSION,
                "output_canvas_size": DENSE_BD_CANVAS_SIZE,
                "output_class_count": DENSE_BD_NUM_CLASSES,
                "bd_class_id": DENSE_BD_CLASS_ID,
                "bd_border_geometry": "right_down_l_h_plus_w_plus_one",
                "bd_external_policy": "ignore",
                "bd_shape_decode": (
                    "joint_separately_normalized_non_bd_interior_plus_"
                    "bd_l_border_argmax"
                ),
                "bd_loss_normalization": "per_sample_color_and_border_means",
                "recursive_steps": 1,
            }
            bd_differences = {
                name: {"expected": expected, "actual": dense_config.get(name)}
                for name, expected in expected_bd_contract.items()
                if dense_config.get(name) != expected
            }
            if bd_differences:
                raise ValueError(
                    "BD shape decoder checkpoint contract mismatch: "
                    f"{bd_differences}."
                )
        top_level_conditioning = config.get("dense_conditioning_mode")
        nested_conditioning = dense_config.get("conditioning_mode")
        if (
            top_level_conditioning is not None
            and nested_conditioning is not None
            and top_level_conditioning != nested_conditioning
        ):
            raise ValueError(
                "Dense conditioning checkpoint config conflict: "
                f"dense_conditioning_mode={top_level_conditioning!r}, "
                "dense_decoder_config.conditioning_mode="
                f"{nested_conditioning!r}."
            )
        top_level_conditioning_schema = config.get(
            "dense_conditioning_schema_version"
        )
        nested_conditioning_schema = dense_config.get(
            "conditioning_schema_version"
        )
        # Absence is an explicit legacy inference rule for checkpoints written
        # before the split-query architecture existed.  Fresh constructors use
        # split_query_context_v1 and always persist the field.
        conditioning_mode = (
            nested_conditioning
            or top_level_conditioning
            or DENSE_CONDITIONING_LEGACY
        )
        if conditioning_mode not in DENSE_SUPPORTED_CONDITIONING_MODES:
            raise ValueError(
                f"Unsupported Dense conditioning mode: {conditioning_mode!r}."
            )
        top_level_relational_memory = config.get("dense_relational_memory_mode")
        nested_relational_memory = dense_config.get("relational_memory_mode")
        if (
            top_level_relational_memory is not None
            and nested_relational_memory is not None
            and top_level_relational_memory != nested_relational_memory
        ):
            raise ValueError(
                "Dense relational-memory checkpoint config conflict: "
                "dense_relational_memory_mode="
                f"{top_level_relational_memory!r}, "
                "dense_decoder_config.relational_memory_mode="
                f"{nested_relational_memory!r}."
            )
        relational_memory_mode = (
            nested_relational_memory
            or top_level_relational_memory
            or DENSE_RELATIONAL_MEMORY_DISABLED
        )
        if relational_memory_mode not in DENSE_RELATIONAL_MEMORY_MODES:
            raise ValueError(
                "Unsupported Dense relational-memory mode: "
                f"{relational_memory_mode!r}."
            )
        relational_memory_enabled = (
            relational_memory_mode
            == DENSE_RELATIONAL_MEMORY_SET_CROSS_ATTENTION_V2
        )
        if relational_memory_enabled:
            if conditioning_mode != DENSE_CONDITIONING_GATED_PAIRED:
                raise ValueError(
                    "set_cross_attention_v2 requires "
                    "gated_paired_associative_v1 conditioning."
                )
            if (
                top_level_relational_memory != relational_memory_mode
                or nested_relational_memory != relational_memory_mode
            ):
                raise ValueError(
                    "Relational Memory v2 checkpoints must declare matching "
                    "top-level and nested relational-memory mode metadata."
                )
        conditioning_schema_by_mode = {
            DENSE_CONDITIONING_SPLIT: DENSE_CONDITIONING_SCHEMA_VERSION,
            DENSE_CONDITIONING_SAME_COORDINATE: (
                DENSE_SAME_COORDINATE_SCHEMA_VERSION
            ),
            DENSE_CONDITIONING_FORCED_PAIRED: DENSE_FORCED_PAIRED_SCHEMA_VERSION,
            DENSE_CONDITIONING_GATED_PAIRED: (
                DENSE_GATED_RELATIONAL_SCHEMA_VERSION
                if relational_memory_enabled
                else DENSE_GATED_PAIRED_SCHEMA_VERSION
            ),
        }
        if conditioning_mode in conditioning_schema_by_mode:
            if conditioning_mode in DENSE_PAIRED_CONDITIONING_MODES and (
                top_level_conditioning != conditioning_mode
                or nested_conditioning != conditioning_mode
                or top_level_conditioning_schema is None
                or nested_conditioning_schema is None
            ):
                raise ValueError(
                    "Paired-associative Dense checkpoints must declare matching "
                    "top-level and nested conditioning mode/schema metadata."
                )
            if (
                conditioning_mode in DENSE_PAIRED_CONDITIONING_MODES
                and int(top_level_conditioning_schema)
                != int(nested_conditioning_schema)
            ):
                raise ValueError(
                    "Dense conditioning checkpoint schema conflict: "
                    "dense_conditioning_schema_version="
                    f"{top_level_conditioning_schema!r}, "
                    "dense_decoder_config.conditioning_schema_version="
                    f"{nested_conditioning_schema!r}."
                )
            conditioning_schema = int(
                dense_config.get(
                    "conditioning_schema_version",
                    config.get("dense_conditioning_schema_version", -1),
                )
            )
            expected_schema = int(conditioning_schema_by_mode[conditioning_mode])
            if conditioning_schema != expected_schema:
                raise ValueError(
                    "Unsupported or missing Dense conditioning schema version. "
                    f"Expected {conditioning_mode} schema v{expected_schema}; "
                    "legacy weights may not be resumed silently."
                )
            same_coordinate_gate_initial_value = (
                dense_config.get("same_coordinate_gate_initial_value")
                if conditioning_mode
                in {
                    DENSE_CONDITIONING_SAME_COORDINATE,
                    *DENSE_PAIRED_CONDITIONING_MODES,
                }
                else None
            )
            if conditioning_mode in {
                DENSE_CONDITIONING_SAME_COORDINATE,
                *DENSE_PAIRED_CONDITIONING_MODES,
            }:
                if not isinstance(same_coordinate_gate_initial_value, (int, float)):
                    raise ValueError(
                        "Dense conditioning contract mismatch: missing finite "
                        "same_coordinate_gate_initial_value."
                    )
                if (
                    not torch.isfinite(
                        torch.tensor(float(same_coordinate_gate_initial_value))
                    )
                    or float(same_coordinate_gate_initial_value) == 0.0
                ):
                    raise ValueError(
                        "Dense conditioning contract mismatch: "
                        "same_coordinate_gate_initial_value must be finite and non-zero."
                    )
            if conditioning_mode == DENSE_CONDITIONING_FORCED_PAIRED:
                expected_conditioning_contract = {
                    "conditioning_fusion_type": DENSE_FORCED_PAIRED_FUSION_TYPE,
                    "base_query_parameterization": "shared",
                    "query_injection_position": "post_mlp_every_layer",
                    "query_injection_is_final_residual": True,
                    "context_memory_mode": "disabled_for_color_path",
                    "query_branch_scale_initial_value": float(
                        same_coordinate_gate_initial_value
                    ),
                    "context_branch_scale_initial_value": None,
                    "same_coordinate_source": (
                        "pre_fusion_query_cell_memory_scattered_by_flat_indices"
                    ),
                    "same_coordinate_padding_policy": "masked_no_injection",
                    "same_coordinate_gate_parameterization": (
                        DENSE_SAME_COORDINATE_GATE_PARAMETERIZATION
                    ),
                    "same_coordinate_gate_initial_value": float(
                        same_coordinate_gate_initial_value
                    ),
                    "paired_memory_source": DENSE_FORCED_PAIRED_SOURCE,
                    "paired_memory_pairing": DENSE_FORCED_PAIRED_PAIRING,
                    "paired_memory_coordinate_policy": (
                        "exact_input_output_coordinate_set_required"
                    ),
                    "paired_memory_no_context_policy": (
                        DENSE_FORCED_PAIRED_NO_CONTEXT_POLICY
                    ),
                    "paired_memory_gate": "fixed_one_no_gate",
                    "paired_memory_persistent_injection": True,
                    "generic_context_color_path": False,
                }
            elif conditioning_mode == DENSE_CONDITIONING_GATED_PAIRED:
                if relational_memory_enabled:
                    relational_gate_initial_value = dense_config.get(
                        "relational_memory_gate_initial_effective_scale"
                    )
                    if not isinstance(
                        relational_gate_initial_value, (int, float)
                    ):
                        raise ValueError(
                            "Relational Memory v2 checkpoint is missing a numeric "
                            "relational_memory_gate_initial_effective_scale."
                        )
                    relational_gate_initial_value = float(
                        relational_gate_initial_value
                    )
                    if (
                        not torch.isfinite(
                            torch.tensor(relational_gate_initial_value)
                        )
                        or not 0.0 < relational_gate_initial_value < 2.0
                    ):
                        raise ValueError(
                            "Relational Memory v2 initial effective scale must be "
                            "finite and in (0,2)."
                        )
                    top_level_relational_gate = config.get(
                        "dense_relational_gate_initial_value"
                    )
                    if not isinstance(top_level_relational_gate, (int, float)):
                        raise ValueError(
                            "Relational Memory v2 checkpoints must declare a "
                            "numeric top-level dense_relational_gate_initial_value."
                        )
                    if float(top_level_relational_gate) != (
                        relational_gate_initial_value
                    ):
                        raise ValueError(
                            "Dense relational-memory gate checkpoint config "
                            "conflict: dense_relational_gate_initial_value="
                            f"{top_level_relational_gate!r}, "
                            "dense_decoder_config."
                            "relational_memory_gate_initial_effective_scale="
                            f"{relational_gate_initial_value!r}."
                        )
                    expected_conditioning_contract = {
                        "conditioning_fusion_type": (
                            DENSE_RELATIONAL_MEMORY_FUSION_TYPE
                        ),
                        "base_query_parameterization": "shared",
                        "query_injection_position": "post_mlp_every_layer",
                        "query_injection_is_final_residual": True,
                        "context_memory_mode": "packed_without_query_cells",
                        "query_branch_scale_initial_value": float(
                            same_coordinate_gate_initial_value
                        ),
                        "context_branch_scale_initial_value": 1.0,
                        "same_coordinate_source": (
                            "pre_fusion_query_cell_memory_scattered_by_flat_indices"
                        ),
                        "same_coordinate_padding_policy": "masked_no_injection",
                        "same_coordinate_gate_parameterization": (
                            DENSE_SAME_COORDINATE_GATE_PARAMETERIZATION
                        ),
                        "same_coordinate_gate_initial_value": float(
                            same_coordinate_gate_initial_value
                        ),
                        "generic_context_color_path": True,
                        "paired_memory_source": None,
                        "paired_memory_pairing": None,
                        "paired_memory_coordinate_policy": None,
                        "paired_memory_no_context_policy": None,
                        "paired_memory_gate": None,
                        "paired_memory_persistent_injection": None,
                        "relational_memory_mode": (
                            DENSE_RELATIONAL_MEMORY_SET_CROSS_ATTENTION_V2
                        ),
                        "relational_memory_schema_version": (
                            DENSE_RELATIONAL_MEMORY_SCHEMA_VERSION
                        ),
                        "relational_memory_source": (
                            DENSE_RELATIONAL_MEMORY_SOURCE
                        ),
                        "relational_memory_pairing": (
                            DENSE_RELATIONAL_MEMORY_PAIRING
                        ),
                        "relational_memory_geometry": (
                            DENSE_RELATIONAL_MEMORY_GEOMETRY
                        ),
                        "relational_memory_relation_tokens": (
                            "one_per_valid_demonstration_output_cell"
                        ),
                        "relational_memory_example_isolation": True,
                        "relational_memory_input_output_lengths_may_differ": True,
                        "relational_memory_no_context_policy": (
                            "exact_zero_residual"
                        ),
                        "relational_memory_gate": (
                            DENSE_RELATIONAL_MEMORY_GATE_PARAMETERIZATION
                        ),
                        "relational_memory_gate_initial_value": (
                            relational_gate_initial_value
                        ),
                        "relational_memory_gate_initial_effective_scale": (
                            relational_gate_initial_value
                        ),
                        "relational_memory_gate_input": (
                            DENSE_RELATIONAL_MEMORY_GATE_INPUT
                        ),
                        "relational_memory_gate_range": "open_0_2",
                        "relational_memory_injection_position": (
                            DENSE_RELATIONAL_MEMORY_INJECTION_POSITION
                        ),
                        "relational_memory_all_output_positions": True,
                        "relational_memory_persistent_injection": True,
                    }
                else:
                    paired_gate_initial_value = dense_config.get(
                        "paired_memory_gate_initial_effective_scale"
                    )
                    if not isinstance(paired_gate_initial_value, (int, float)):
                        raise ValueError(
                            "Gated paired Dense checkpoint is missing a numeric "
                            "paired_memory_gate_initial_effective_scale."
                        )
                    paired_gate_initial_value = float(paired_gate_initial_value)
                    if (
                        not torch.isfinite(torch.tensor(paired_gate_initial_value))
                        or not 0.0 < paired_gate_initial_value < 2.0
                    ):
                        raise ValueError(
                            "Gated paired initial effective scale must be finite "
                            "and in (0,2)."
                        )
                    expected_conditioning_contract = {
                        "conditioning_fusion_type": DENSE_GATED_PAIRED_FUSION_TYPE,
                        "base_query_parameterization": "shared",
                        "query_injection_position": "post_mlp_every_layer",
                        "query_injection_is_final_residual": True,
                        "context_memory_mode": "packed_without_query_cells",
                        "query_branch_scale_initial_value": float(
                            same_coordinate_gate_initial_value
                        ),
                        "context_branch_scale_initial_value": 1.0,
                        "same_coordinate_source": (
                            "pre_fusion_query_cell_memory_scattered_by_flat_indices"
                        ),
                        "same_coordinate_padding_policy": "masked_no_injection",
                        "same_coordinate_gate_parameterization": (
                            DENSE_SAME_COORDINATE_GATE_PARAMETERIZATION
                        ),
                        "same_coordinate_gate_initial_value": float(
                            same_coordinate_gate_initial_value
                        ),
                        "paired_memory_source": DENSE_FORCED_PAIRED_SOURCE,
                        "paired_memory_pairing": DENSE_FORCED_PAIRED_PAIRING,
                        "paired_memory_coordinate_policy": (
                            "exact_input_output_coordinate_set_required"
                        ),
                        "paired_memory_no_context_policy": (
                            DENSE_FORCED_PAIRED_NO_CONTEXT_POLICY
                        ),
                        "paired_memory_gate": (
                            DENSE_GATED_PAIRED_GATE_PARAMETERIZATION
                        ),
                        "paired_memory_gate_initial_value": (
                            paired_gate_initial_value
                        ),
                        "paired_memory_gate_initial_effective_scale": (
                            paired_gate_initial_value
                        ),
                        "paired_memory_gate_input": DENSE_GATED_PAIRED_GATE_INPUT,
                        "paired_memory_gate_range": "open_0_2",
                        "paired_memory_injection_position": (
                            DENSE_GATED_PAIRED_INJECTION_POSITION
                        ),
                        "paired_memory_persistent_injection": True,
                        "generic_context_color_path": True,
                    }
            else:
                expected_conditioning_contract = {
                    "conditioning_fusion_type": (
                        DENSE_SAME_COORDINATE_FUSION_TYPE
                        if conditioning_mode == DENSE_CONDITIONING_SAME_COORDINATE
                        else DENSE_CONDITIONING_FUSION_TYPE
                    ),
                    "base_query_parameterization": "shared",
                    "query_injection_position": "post_mlp_every_layer",
                    "query_injection_is_final_residual": True,
                    "context_memory_mode": "packed_without_query_cells",
                    "query_branch_scale_initial_value": (
                        float(same_coordinate_gate_initial_value)
                        if conditioning_mode == DENSE_CONDITIONING_SAME_COORDINATE
                        else 1.0
                    ),
                    "context_branch_scale_initial_value": 1.0,
                }
            if conditioning_mode == DENSE_CONDITIONING_SAME_COORDINATE:
                expected_conditioning_contract.update(
                    {
                        "same_coordinate_source": (
                            "query_cell_memory_scattered_by_flat_indices"
                        ),
                        "same_coordinate_padding_policy": "masked_no_injection",
                        "same_coordinate_gate_parameterization": (
                            DENSE_SAME_COORDINATE_GATE_PARAMETERIZATION
                        ),
                        "same_coordinate_gate_initial_value": float(
                            same_coordinate_gate_initial_value
                        ),
                    }
                )
            for key, expected in expected_conditioning_contract.items():
                if dense_config.get(key) != expected:
                    raise ValueError(
                        "Dense conditioning contract mismatch: "
                        f"dense_decoder_config.{key}={dense_config.get(key)!r}; "
                        f"expected {expected!r}."
                    )
        recursive_aliases = {
            "dense_recursive_steps": "recursive_steps",
            "dense_recursive_loss_weighting": "recursive_loss_weighting",
            "dense_hsw_lambda": "hsw_lambda",
            "dense_candidate_encoder_layers": "candidate_encoder_layers",
            "dense_candidate_mlp_ratio": "candidate_mlp_ratio",
            "dense_candidate_dropout": "candidate_dropout",
        }
        for top_level, nested in recursive_aliases.items():
            if top_level in config and nested in dense_config:
                if config[top_level] != dense_config[nested]:
                    raise ValueError(
                        "Dense recursive checkpoint config conflict: "
                        f"{top_level}={config[top_level]!r}, "
                        f"dense_decoder_config.{nested}={dense_config[nested]!r}."
                    )
        recursive_steps = int(dense_config.get("recursive_steps", config.get(
            "dense_recursive_steps", 1
        )))
        if recursive_steps > 1:
            if int(dense_config.get("recursion_schema_version", -1)) != int(
                DENSE_RECURSION_SCHEMA_VERSION
            ):
                raise ValueError(
                    "Unsupported or missing Dense recursion schema version. "
                    f"Expected gated-residual schema v{DENSE_RECURSION_SCHEMA_VERSION}; "
                    "legacy recursive weights may not be resumed silently."
                )
            if dense_config.get("candidate_detach") is not True:
                raise ValueError("Recursive Dense checkpoint must declare candidate_detach=true.")
            expected_recursion_contract = {
                "candidate_fusion_type": DENSE_CANDIDATE_FUSION_TYPE,
                "candidate_gate_parameterization": (
                    DENSE_CANDIDATE_GATE_PARAMETERIZATION
                ),
                "candidate_gate_initial_value": DENSE_CANDIDATE_GATE_INITIAL_VALUE,
                "candidate_shape_gate_parameterization": "per_channel_tanh",
                "candidate_shape_gate_initial_value": 0.0,
            }
            for key, expected in expected_recursion_contract.items():
                if dense_config.get(key) != expected:
                    raise ValueError(
                        "Recursive Dense checkpoint fusion contract mismatch: "
                        f"dense_decoder_config.{key}={dense_config.get(key)!r}; "
                        f"expected {expected!r}."
                    )

        def resolve_artifact(value: str | None) -> str | None:
            if not value:
                return None
            path = Path(value)
            return str(path if path.is_absolute() else base_dir / path)

        tokenizer_path = resolve_artifact(config.get("tokenizer_dir")) or config.get(
            "tokenizer_name_or_path"
        )
        if not tokenizer_path:
            raise ValueError("vit5_hybrid checkpoint is missing tokenizer_dir/tokenizer_name_or_path.")
        tokenizer, _ = load_t5_tokenizer_with_arc_tokens(
            tokenizer_path,
            local_files_only=config.get("local_files_only", False),
        )
        model = build_vit5_hybrid_model(
            pretrained_model_name_or_path=config["pretrained_model_name_or_path"],
            tokenizer=tokenizer,
            vit5_checkpoint_path=config["vit5_checkpoint_path"],
            decoder_mode=config.get("decoder_mode", "autoregressive"),
            max_h=config.get("max_grid_height", 30),
            max_w=config.get("max_grid_width", 30),
            max_grids=config.get("max_grids", 21),
            max_examples=config.get("max_examples", 11),
            max_objects=config.get("max_objects", 900),
            task_fusion_layers=config.get("task_fusion_layers", 2),
            grid_encoder_microbatch=config.get("grid_encoder_microbatch"),
            freeze_vit5=config.get("freeze_vit5", True),
            freeze_t5=config.get("freeze_t5", True),
            enable_vit5_lora=config.get("enable_vit5_lora", False),
            enable_t5_decoder_lora=config.get("enable_t5_decoder_lora", False),
            lora_r=config.get("lora_r", 8),
            lora_alpha=config.get("lora_alpha", 16),
            lora_dropout=config.get("lora_dropout", 0.05),
            added_token_ids=config.get("added_token_ids"),
            gradient_checkpointing=config.get("gradient_checkpointing", False),
            local_files_only=config.get("local_files_only", False),
            is_test_fixture=config.get("is_test_fixture", False),
            dense_hidden_size=config.get(
                "dense_hidden_size",
                (config.get("dense_decoder_config") or {}).get("hidden_size", 384),
            ),
            dense_num_heads=config.get(
                "dense_num_heads",
                (config.get("dense_decoder_config") or {}).get("num_heads", 6),
            ),
            dense_decoder_layers=config.get(
                "dense_decoder_layers",
                (config.get("dense_decoder_config") or {}).get("num_layers", 2),
            ),
            dense_mlp_ratio=config.get(
                "dense_mlp_ratio",
                (config.get("dense_decoder_config") or {}).get("mlp_ratio", 4.0),
            ),
            dense_dropout=config.get(
                "dense_dropout",
                (config.get("dense_decoder_config") or {}).get("dropout", 0.0),
            ),
            lambda_shape=config.get(
                "lambda_shape",
                (config.get("dense_decoder_config") or {}).get("lambda_shape", 0.2),
            ),
            lambda_component=config.get(
                "lambda_component",
                (config.get("dense_decoder_config") or {}).get("lambda_component", 0.0),
            ),
            dense_recursive_steps=config.get(
                "dense_recursive_steps",
                (config.get("dense_decoder_config") or {}).get("recursive_steps", 1),
            ),
            dense_recursive_loss_weighting=config.get(
                "dense_recursive_loss_weighting",
                (config.get("dense_decoder_config") or {}).get(
                    "recursive_loss_weighting", "hsw"
                ),
            ),
            dense_hsw_lambda=config.get(
                "dense_hsw_lambda",
                (config.get("dense_decoder_config") or {}).get("hsw_lambda", 0.7),
            ),
            dense_candidate_encoder_layers=config.get(
                "dense_candidate_encoder_layers",
                (config.get("dense_decoder_config") or {}).get(
                    "candidate_encoder_layers", 2
                ),
            ),
            dense_candidate_mlp_ratio=config.get(
                "dense_candidate_mlp_ratio",
                (config.get("dense_decoder_config") or {}).get(
                    "candidate_mlp_ratio", 4.0
                ),
            ),
            dense_candidate_dropout=config.get(
                "dense_candidate_dropout",
                (config.get("dense_decoder_config") or {}).get(
                    "candidate_dropout", 0.0
                ),
            ),
            dense_conditioning_mode=conditioning_mode,
            # Legacy and split-query metadata intentionally serialize the
            # direct-coordinate-only field as ``null``.  Do not coerce that
            # sentinel with ``float(None)`` while reconstructing an unrelated
            # decoder mode; it is contractual only for same-coordinate v2 and
            # Forced paired v3.
            dense_same_coordinate_gate_initial_value=(
                float(same_coordinate_gate_initial_value)
                if conditioning_mode
                in {
                    DENSE_CONDITIONING_SAME_COORDINATE,
                    *DENSE_PAIRED_CONDITIONING_MODES,
                }
                else DENSE_SAME_COORDINATE_GATE_INITIAL_VALUE
            ),
            dense_paired_gate_initial_value=(
                float(
                    dense_config.get(
                        "paired_memory_gate_initial_effective_scale", 1.0
                    )
                )
                if (
                    conditioning_mode == DENSE_CONDITIONING_GATED_PAIRED
                    and not relational_memory_enabled
                )
                else 1.0
            ),
            dense_relational_memory_mode=relational_memory_mode,
            dense_relational_gate_initial_value=(
                float(
                    dense_config.get(
                        "relational_memory_gate_initial_effective_scale", 1.0
                    )
                )
                if relational_memory_enabled
                else 1.0
            ),
            dense_shape_decoder_mode=shape_decoder_mode,
            lambda_ar=config.get("lambda_ar", 1.0),
            lambda_dense=config.get("lambda_dense", 1.0),
            hybrid_training_strategy=config.get("hybrid_training_strategy", "joint"),
            object_embedding_mode=config.get(
                "object_embedding_mode", "connected_components"
            ),
            vit5_initialization=config.get(
                "vit5_initialization", "pretrained_lora"
            ),
        )
        saved_vocabulary_info = config.get("vocabulary_info")
        if saved_vocabulary_info is not None and saved_vocabulary_info != model.vocabulary_info:
            raise ValueError(
                "vit5_hybrid vocabulary metadata differs on reload: "
                f"saved={saved_vocabulary_info}, runtime={model.vocabulary_info}"
            )
        if checkpoint.get("checkpoint_format") == VIT5_HYBRID_LIGHTWEIGHT_FORMAT:
            state_path = resolve_artifact(config.get("hybrid_trainable_state_path"))
            if not state_path:
                raise ValueError("Lightweight vit5_hybrid checkpoint is missing trainable state path.")
            payload = _load_local_torch_payload(state_path, "cpu")
            if payload.get("format") != VIT5_HYBRID_LIGHTWEIGHT_FORMAT:
                raise ValueError("Hybrid trainable state format mismatch.")
            expected_state = _vit5_hybrid_lightweight_state(model)
            saved_state = payload.get("state_dict", {})
            missing_state = set(expected_state) - set(saved_state)
            unexpected_state = set(saved_state) - set(expected_state)
            approved_legacy_missing = (
                {VIT5_HYBRID_OBJECT_EMBEDDING_KEY}
                if int(config.get("batch_schema_version", 2)) < 3
                else set()
            )
            if unexpected_state or not missing_state.issubset(approved_legacy_missing):
                raise ValueError(
                    "Hybrid trainable state key mismatch: "
                    f"missing={sorted(missing_state)}, "
                    f"unexpected={sorted(unexpected_state)}"
                )
            incompatible = model.load_state_dict(saved_state, strict=False)
            if incompatible.unexpected_keys:
                raise ValueError(
                    f"Unexpected lightweight hybrid state keys: {incompatible.unexpected_keys}"
                )
            # ``load_state_dict(strict=False)`` reports every intentionally
            # omitted frozen/base tensor as missing for a lightweight state.
            # Completeness of the trainable subset was already checked above
            # against ``expected_state``; only unexpected keys are meaningful
            # in the global incompatible-key result.
        elif checkpoint.get("checkpoint_format") == VIT5_HYBRID_ARC_FULL_FORMAT:
            if config.get("vit5_initialization") != "random_scratch":
                raise ValueError(
                    "ARC-full checkpoint must declare vit5_initialization=random_scratch."
                )
            if config.get("vit5_checkpoint_path") is not None:
                raise ValueError(
                    "ARC-full random_scratch checkpoint must not reference ViT-5 weights."
                )
            state_path = resolve_artifact(config.get("hybrid_arc_full_state_path"))
            if not state_path:
                raise ValueError("ARC-full vit5_hybrid checkpoint is missing state path.")
            payload = _load_local_torch_payload(state_path, "cpu")
            if payload.get("format") != VIT5_HYBRID_ARC_FULL_FORMAT:
                raise ValueError("ARC-full hybrid state format mismatch.")
            expected_state = _vit5_hybrid_arc_full_state(model)
            saved_state = payload.get("state_dict", {})
            missing_state = set(expected_state) - set(saved_state)
            unexpected_state = set(saved_state) - set(expected_state)
            if missing_state or unexpected_state:
                raise ValueError(
                    "ARC-full state key mismatch: "
                    f"missing={sorted(missing_state)}, "
                    f"unexpected={sorted(unexpected_state)}"
                )
            incompatible = model.load_state_dict(saved_state, strict=False)
            if incompatible.unexpected_keys:
                raise ValueError(
                    f"Unexpected ARC-full state keys: {incompatible.unexpected_keys}"
                )
        else:
            saved_state = checkpoint["model_state_dict"]
            if (
                int(config.get("batch_schema_version", 2)) < 3
                and VIT5_HYBRID_OBJECT_EMBEDDING_KEY not in saved_state
            ):
                incompatible = model.load_state_dict(saved_state, strict=False)
                missing = set(incompatible.missing_keys)
                if missing != {VIT5_HYBRID_OBJECT_EMBEDDING_KEY} or incompatible.unexpected_keys:
                    raise ValueError(
                        "Legacy full hybrid checkpoint state mismatch: "
                        f"missing={sorted(missing)}, "
                        f"unexpected={sorted(incompatible.unexpected_keys)}"
                    )
            else:
                model.load_state_dict(saved_state)
        if config.get("object_embedding_mode") == "none":
            object_weight = (
                model.arc_encoder.grid_encoder.arc_embedding.object_embedding.weight
            )
            if bool(torch.count_nonzero(object_weight).item()):
                raise ValueError(
                    "object_embedding_mode=none checkpoint contains non-zero object rows."
                )
            object_weight.requires_grad_(False)
        model.to(torch.device(device))
        return model
    if model_type == "pretrained_ope":
        tokenizer_path = config.get("tokenizer_dir") or config.get("tokenizer_name_or_path")
        if not tokenizer_path:
            raise ValueError("pretrained_ope checkpoint is missing tokenizer_dir/tokenizer_name_or_path.")
        tokenizer, _ = load_t5_tokenizer_with_arc_tokens(
            tokenizer_path,
            local_files_only=config.get("local_files_only", False),
        )
        model = build_pretrained_t5_ope_model(
            pretrained_model_name_or_path=config["pretrained_model_name_or_path"],
            tokenizer=tokenizer,
            row_vocab_size=config["row_vocab_size"],
            col_vocab_size=config["col_vocab_size"],
            object_vocab_size=config["object_vocab_size"],
            dropout_rate=config.get("dropout_rate", 0.0),
            pemixer_mode=config.get("pemixer_mode", "learnable_scalars"),
            ope_mode=config.get("ope_mode", "sum"),
            freeze_t5=config.get("freeze_t5", False),
            train_new_token_embeddings=config.get("train_new_token_embeddings", False),
            added_token_ids=config.get("added_token_ids"),
            disable_t5_relative_position_bias=config.get("disable_t5_relative_position_bias", False),
            gradient_checkpointing=config.get("gradient_checkpointing", False),
            local_files_only=config.get("local_files_only", False),
        )
        if checkpoint.get("checkpoint_format") == "pretrained_ope_lora_only":
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ImportError("peft is required to load LoRA-only pretrained_ope checkpoints.") from exc
            adapter_dir = config.get("lora_adapter_dir")
            if not adapter_dir:
                raise ValueError("LoRA-only checkpoint is missing lora_adapter_dir.")
            model.t5 = PeftModel.from_pretrained(model.t5, adapter_dir)
            if config.get("train_new_token_embeddings", False):
                model.enable_new_token_embedding_training()
            embeddings_path = config.get("arc_token_embeddings_path")
            if embeddings_path:
                load_arc_token_embeddings(embeddings_path, model)
        elif config.get("lora_enabled", False):
            model, _ = add_lora_to_t5_if_available(
                model,
                r=config.get("lora_r", 8),
                alpha=config.get("lora_alpha", 16),
                dropout=config.get("lora_dropout", 0.05),
                target_modules=config.get("lora_target_modules", ["q", "k", "v", "o"]),
            )
        if checkpoint.get("checkpoint_format") != "pretrained_ope_lora_only":
            model.load_state_dict(checkpoint["model_state_dict"])
        model.to(torch.device(device))
        return model
    common = {
        "vocab_size": config["vocab_size"],
        "pad_token_id": config["pad_token_id"],
        "eos_token_id": config["eos_token_id"],
        "row_vocab_size": config["row_vocab_size"],
        "col_vocab_size": config["col_vocab_size"],
        "d_model": config.get("d_model", 64),
        "d_ff": config.get("d_ff", 128),
        "num_heads": config.get("num_heads", 4),
        "dropout_rate": config.get("dropout_rate", 0.0),
        "pemixer_mode": config.get("pemixer_mode", "learnable_scalars"),
    }
    if model_type == "ope":
        model = build_tiny_t5_ope_model(
            **common,
            object_vocab_size=config["object_vocab_size"],
            num_layers=config.get("num_layers", config.get("encoder_layers", 1)),
            num_decoder_layers=config.get("decoder_layers", config.get("num_decoder_layers", 1)),
            ope_mode=config.get("ope_mode", "sum"),
        )
    elif model_type == "rpe_encoder":
        model = build_tiny_t5_rpe_encoder_model(
            **common,
            object_vocab_size=config.get("object_vocab_size"),
            encoder_layers=config.get("encoder_layers", 1),
            decoder_layers=config.get("decoder_layers", 1),
            use_ope=config.get("use_ope", True),
            ope_mode=config.get("ope_mode", "sum"),
            metric=config.get("metric", "manhattan"),
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(torch.device(device))
    return model
