"""Autoregressive ViT-5/T5 hybrid model for ARC."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .hf_arc_tokenizer import get_arc_special_tokens
from .vit5_arc_encoder import (
    HierarchicalViT5ArcEncoder,
    VIT5_SMALL_DIM,
    ViT5ArcEncoderOutput,
)
from .vit5_hybrid_batch import ARC_MAX_OBJECTS
from .vit5_checkpoint_adapter import (
    ViT5RandomInitializationReport,
    ViT5TransferReport,
    initialize_vit5_small_random_scratch,
    transfer_vit5_small_checkpoint,
)
from .vit5_dense_decoder import (
    COMPONENT_LOSS_TYPE,
    DENSE_CONDITIONING_FORCED_PAIRED,
    DENSE_CONDITIONING_SPLIT,
    DENSE_GATED_PAIRED_GATE_INITIAL_VALUE,
    DENSE_PAIRED_CONDITIONING_MODES,
    DENSE_RELATIONAL_MEMORY_DISABLED,
    DENSE_RELATIONAL_MEMORY_SET_CROSS_ATTENTION_V2,
    DENSE_SHAPE_DECODER_SEPARATE_HW,
    TARGET_COMPONENT_MODE,
    VIT5_DENSE_LOSS_SCHEMA_VERSION,
    DenseDecoderOutput,
    DenseGridQueryDecoder,
    decode_dense_output,
)
from .vit5_layers import RMSNorm


VIT5_INITIALIZATION_PRETRAINED_LORA = "pretrained_lora"
VIT5_INITIALIZATION_RANDOM_SCRATCH = "random_scratch"
VIT5_INITIALIZATION_MODES = (
    VIT5_INITIALIZATION_PRETRAINED_LORA,
    VIT5_INITIALIZATION_RANDOM_SCRATCH,
)


@dataclass
class Vit5HybridEncoderCache:
    """Encode-once memory shared by teacher forcing and incremental generation."""

    last_hidden_state: torch.Tensor
    attention_mask: torch.Tensor
    arc_last_hidden_state: torch.Tensor
    memory_layout: list[dict[str, Any]]
    arc_encoder_output: ViT5ArcEncoderOutput | None = None


@dataclass
class Vit5HybridOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor | None
    past_key_values: Any | None
    encoder_cache: Vit5HybridEncoderCache
    t5_output: Any | None
    dense_output: DenseDecoderOutput | None = None
    ar_loss: torch.Tensor | None = None
    raw_cell_loss: torch.Tensor | None = None
    component_balanced_loss: torch.Tensor | None = None
    color_loss: torch.Tensor | None = None
    height_loss: torch.Tensor | None = None
    width_loss: torch.Tensor | None = None
    border_loss: torch.Tensor | None = None
    shape_loss: torch.Tensor | None = None
    dense_loss: torch.Tensor | None = None
    total_loss: torch.Tensor | None = None
    active_head: str | None = None
    loss_contributions: dict[str, torch.Tensor] | None = None


class LoRALinear(nn.Module):
    """Small dependency-free LoRA wrapper for an existing Linear layer."""

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRALinear can only wrap nn.Linear modules.")
        if r < 1:
            raise ValueError("LoRA rank must be positive.")
        self.base_layer = base_layer
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = float(alpha) / float(r)
        self.lora_dropout = nn.Dropout(dropout)
        self.lora_A = nn.Linear(base_layer.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base_layer.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

    @property
    def weight(self) -> torch.nn.Parameter:
        return self.base_layer.weight

    @property
    def bias(self) -> torch.nn.Parameter | None:
        return self.base_layer.bias

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base_layer(inputs) + self.lora_B(
            self.lora_A(self.lora_dropout(inputs))
        ) * self.scaling


class ArcTokenDeltaEmbedding(nn.Module):
    """Small tied input/output delta table for non-contiguous ARC token ids."""

    def __init__(self, token_ids: list[int], d_model: int, vocab_size: int) -> None:
        super().__init__()
        resolved_ids = list(dict.fromkeys(int(token_id) for token_id in token_ids))
        if not resolved_ids:
            raise ValueError("ARC token delta requires at least one token id.")
        if min(resolved_ids) < 0 or max(resolved_ids) >= vocab_size:
            raise ValueError(
                f"ARC token ids must be inside model vocab [0,{vocab_size}), got {resolved_ids}."
            )
        self.delta_weight = nn.Parameter(torch.zeros(len(resolved_ids), d_model))
        self.register_buffer("token_ids", torch.tensor(resolved_ids, dtype=torch.long))
        id_to_row = torch.full((vocab_size,), -1, dtype=torch.long)
        id_to_row[self.token_ids] = torch.arange(len(resolved_ids), dtype=torch.long)
        self.register_buffer("id_to_row", id_to_row)
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)

    @property
    def num_arc_tokens(self) -> int:
        return int(self.token_ids.numel())

    def _rows_for_ids(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.numel() and (input_ids.min() < 0 or input_ids.max() >= self.vocab_size):
            raise ValueError("Decoder token id is outside resolved model vocabulary.")
        rows = self.id_to_row[input_ids]
        return rows, rows.ge(0)

    def embed(self, input_ids: torch.Tensor, base_embedding: nn.Module) -> torch.Tensor:
        base = base_embedding(input_ids)
        rows, mask = self._rows_for_ids(input_ids)
        safe_rows = rows.clamp_min(0)
        delta = F.embedding(safe_rows, self.delta_weight)
        return base + delta * mask.unsqueeze(-1).to(delta.dtype)

    def add_to_logits(
        self,
        base_logits: torch.Tensor,
        decoder_hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        if base_logits.shape[-1] != self.vocab_size:
            raise ValueError(
                f"Base logits vocab={base_logits.shape[-1]} != resolved vocab={self.vocab_size}."
            )
        arc_delta_logits = decoder_hidden_state @ self.delta_weight.transpose(0, 1)
        output = base_logits.clone()
        output.index_add_(-1, self.token_ids, arc_delta_logits)
        return output


def _unused_token_id_ranges(tokenizer: Any, resolved_vocab_size: int) -> list[list[int]]:
    if hasattr(tokenizer, "get_vocab"):
        used = {
            int(token_id)
            for token_id in tokenizer.get_vocab().values()
            if 0 <= int(token_id) < resolved_vocab_size
        }
    else:
        used = set(range(min(len(tokenizer), resolved_vocab_size)))
    unused = [token_id for token_id in range(resolved_vocab_size) if token_id not in used]
    if not unused:
        return []
    ranges: list[list[int]] = []
    start = previous = unused[0]
    for token_id in unused[1:]:
        if token_id != previous + 1:
            ranges.append([start, previous])
            start = token_id
        previous = token_id
    ranges.append([start, previous])
    return ranges


def _get_submodule_parent(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parent_name, _, child_name = module_name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    return parent, child_name


def inject_lora_modules(
    root: nn.Module,
    module_names: list[str],
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
) -> list[str]:
    """Replace an audited list of Linear modules with LoRA wrappers."""
    if not module_names:
        raise ValueError("LoRA target audit matched zero modules.")
    injected: list[str] = []
    for name in module_names:
        parent, child_name = _get_submodule_parent(root, name)
        current = getattr(parent, child_name)
        if isinstance(current, LoRALinear):
            raise ValueError(f"Module {name} already has LoRA injected.")
        if not isinstance(current, nn.Linear):
            raise TypeError(f"LoRA target {name} is {type(current).__name__}, expected Linear.")
        setattr(parent, child_name, LoRALinear(current, r=r, alpha=alpha, dropout=dropout))
        injected.append(name)
    return injected


def resolve_vit5_lora_targets(model: nn.Module) -> list[str]:
    pattern = re.compile(r"^arc_encoder\.grid_encoder\.blocks\.\d+\.attn\.(qkv|proj)$")
    return sorted(
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and pattern.match(name)
    )


def resolve_t5_decoder_lora_targets(model: nn.Module) -> list[str]:
    allowed_suffixes = {"q", "k", "v", "o"}
    targets: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not name.startswith("t5.decoder."):
            continue
        if ".SelfAttention." not in name and ".EncDecAttention." not in name:
            continue
        if name.rsplit(".", 1)[-1] in allowed_suffixes:
            targets.append(name)
    targets = sorted(targets)
    forbidden = [name for name in targets if name.startswith("t5.encoder.")]
    if forbidden:
        raise ValueError(f"T5 encoder LoRA targets are forbidden: {forbidden}")
    return targets


class Vit5HybridForArc(nn.Module):
    """Hierarchical discrete ARC encoder with a pretrained T5 decoder."""

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        tokenizer: Any,
        vit5_checkpoint_path: str | None,
        decoder_mode: str = "autoregressive",
        max_h: int = 30,
        max_w: int = 30,
        max_grids: int = 21,
        max_examples: int = 11,
        max_objects: int = ARC_MAX_OBJECTS,
        task_fusion_layers: int = 2,
        grid_encoder_microbatch: int | None = None,
        freeze_vit5: bool = True,
        freeze_t5: bool = True,
        enable_vit5_lora: bool = False,
        enable_t5_decoder_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        added_token_ids: list[int] | None = None,
        gradient_checkpointing: bool = False,
        local_files_only: bool = False,
        use_sdpa: bool = True,
        is_test_fixture: bool = False,
        dense_hidden_size: int = VIT5_SMALL_DIM,
        dense_num_heads: int = 6,
        dense_decoder_layers: int = 2,
        dense_mlp_ratio: float = 4.0,
        dense_dropout: float = 0.0,
        lambda_shape: float = 0.2,
        lambda_component: float = 0.0,
        dense_recursive_steps: int = 1,
        dense_recursive_loss_weighting: str = "hsw",
        dense_hsw_lambda: float = 0.7,
        dense_candidate_encoder_layers: int = 2,
        dense_candidate_mlp_ratio: float = 4.0,
        dense_candidate_dropout: float = 0.0,
        dense_conditioning_mode: str = DENSE_CONDITIONING_SPLIT,
        dense_same_coordinate_gate_initial_value: float = 1.0,
        dense_paired_gate_initial_value: float = (
            DENSE_GATED_PAIRED_GATE_INITIAL_VALUE
        ),
        dense_relational_memory_mode: str = DENSE_RELATIONAL_MEMORY_DISABLED,
        dense_relational_gate_initial_value: float = (
            DENSE_GATED_PAIRED_GATE_INITIAL_VALUE
        ),
        dense_shape_decoder_mode: str = DENSE_SHAPE_DECODER_SEPARATE_HW,
        lambda_ar: float = 1.0,
        lambda_dense: float = 1.0,
        hybrid_training_strategy: str = "joint",
        object_embedding_mode: str = "connected_components",
        vit5_initialization: str = VIT5_INITIALIZATION_PRETRAINED_LORA,
    ) -> None:
        super().__init__()
        if decoder_mode not in {"autoregressive", "dense", "hybrid"}:
            raise ValueError("vit5_hybrid supports autoregressive, dense or hybrid mode.")
        if hybrid_training_strategy not in {"joint", "alternating"}:
            raise ValueError("hybrid_training_strategy must be joint or alternating.")
        if decoder_mode == "dense" and enable_t5_decoder_lora:
            raise ValueError("T5 decoder LoRA is not used or allowed in dense mode.")
        if vit5_initialization not in VIT5_INITIALIZATION_MODES:
            raise ValueError(
                f"vit5_initialization must be one of {VIT5_INITIALIZATION_MODES}."
            )
        if vit5_initialization == VIT5_INITIALIZATION_PRETRAINED_LORA:
            if not vit5_checkpoint_path:
                raise ValueError(
                    "pretrained_lora initialization requires vit5_checkpoint_path."
                )
        else:
            if vit5_checkpoint_path:
                raise ValueError(
                    "random_scratch initialization must not receive a ViT-5 checkpoint."
                )
            if freeze_vit5:
                raise ValueError("random_scratch requires freeze_vit5=False.")
            if enable_vit5_lora:
                raise ValueError(
                    "random_scratch trains the full grid encoder and forbids ViT-5 LoRA."
                )
        if object_embedding_mode not in {"none", "connected_components"}:
            raise ValueError(
                "object_embedding_mode must be none or connected_components."
            )
        if lambda_ar < 0 or lambda_dense < 0 or lambda_shape < 0 or lambda_component < 0:
            raise ValueError("Hybrid loss weights must be non-negative.")
        if dense_hidden_size != VIT5_SMALL_DIM:
            raise ValueError("Phase 2A dense hidden size must equal ViT-5 Small dim 384.")
        if dense_conditioning_mode in DENSE_PAIRED_CONDITIONING_MODES:
            paired_mode_name = str(dense_conditioning_mode)
            if decoder_mode != "dense":
                raise ValueError(
                    f"{paired_mode_name} is diagnostic-only and "
                    "requires decoder_mode='dense'."
                )
            if dense_recursive_steps != 1:
                raise ValueError(
                    f"{paired_mode_name} requires dense_recursive_steps=1."
                )
            if object_embedding_mode != "none":
                raise ValueError(
                    f"{paired_mode_name} requires object_embedding_mode='none'."
                )
        from transformers import T5ForConditionalGeneration
        from .model_parameter_audit import validate_vit5_hybrid_runtime

        self.runtime_versions = validate_vit5_hybrid_runtime()
        self.decoder_mode = decoder_mode
        self.hybrid_training_strategy = hybrid_training_strategy
        self.lambda_ar = float(lambda_ar)
        self.lambda_dense = float(lambda_dense)
        self.lambda_shape = float(lambda_shape)
        self.lambda_component = float(lambda_component)
        self.dense_shape_decoder_mode = str(dense_shape_decoder_mode)
        self.dense_relational_memory_mode = str(dense_relational_memory_mode)
        self.loss_schema_version = VIT5_DENSE_LOSS_SCHEMA_VERSION
        self.target_component_mode = TARGET_COMPONENT_MODE
        self.component_loss_type = COMPONENT_LOSS_TYPE
        self.object_embedding_mode = str(object_embedding_mode)
        self.pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        source_path = Path(pretrained_model_name_or_path)
        self.t5_resolved_path = (
            str(source_path.resolve()) if source_path.exists() else str(pretrained_model_name_or_path)
        )
        self.is_test_fixture = bool(is_test_fixture)
        self.vit5_initialization = str(vit5_initialization)
        self.vit5_checkpoint_path = (
            str(vit5_checkpoint_path) if vit5_checkpoint_path is not None else None
        )
        self.freeze_vit5 = bool(freeze_vit5 or enable_vit5_lora)
        self.freeze_t5 = bool(freeze_t5 or enable_t5_decoder_lora or decoder_mode == "dense")
        self.max_objects = int(max_objects)
        self.t5 = T5ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=local_files_only,
        )
        self.t5_config_class = type(self.t5.config).__name__
        self.original_model_vocab_size = int(self.t5.config.vocab_size)
        self.tokenizer_length = int(len(tokenizer))
        self.resolved_vocab_size = max(self.original_model_vocab_size, self.tokenizer_length)
        if self.tokenizer_length > self.original_model_vocab_size:
            try:
                self.t5.resize_token_embeddings(self.resolved_vocab_size, mean_resizing=False)
            except TypeError:
                self.t5.resize_token_embeddings(self.resolved_vocab_size)
            with torch.no_grad():
                self.t5.get_input_embeddings().weight[
                    self.original_model_vocab_size : self.resolved_vocab_size
                ].zero_()
                if (
                    hasattr(self.t5, "lm_head")
                    and self.t5.lm_head.weight.data_ptr()
                    != self.t5.get_input_embeddings().weight.data_ptr()
                ):
                    self.t5.lm_head.weight[
                        self.original_model_vocab_size : self.resolved_vocab_size
                    ].zero_()
        if int(self.t5.config.vocab_size) != self.resolved_vocab_size:
            raise ValueError(
                f"Resolved T5 vocab mismatch: config={self.t5.config.vocab_size}, "
                f"resolved={self.resolved_vocab_size}."
            )
        self.tokenizer = tokenizer
        self.added_token_ids = self._resolve_added_token_ids(added_token_ids)
        if any(
            token_id < 0 or token_id >= self.resolved_vocab_size
            for token_id in self.added_token_ids
        ):
            raise ValueError(
                f"ARC token ids {self.added_token_ids} exceed resolved vocab "
                f"{self.resolved_vocab_size}."
            )
        self.arc_token_delta = ArcTokenDeltaEmbedding(
            self.added_token_ids,
            d_model=int(self.t5.config.d_model),
            vocab_size=self.resolved_vocab_size,
        )
        self.vocabulary_info = {
            "original_model_vocab_size": self.original_model_vocab_size,
            "tokenizer_length": self.tokenizer_length,
            "resolved_model_vocab_size": self.resolved_vocab_size,
            "arc_token_ids": list(self.added_token_ids),
            "unused_vocab_row_ranges": _unused_token_id_ranges(
                tokenizer, self.resolved_vocab_size
            ),
        }
        if gradient_checkpointing:
            self.t5.gradient_checkpointing_enable()
            self.t5.config.use_cache = False

        self.arc_encoder = HierarchicalViT5ArcEncoder(
            max_h=max_h,
            max_w=max_w,
            max_grids=max_grids,
            max_examples=max_examples,
            task_fusion_layers=task_fusion_layers,
            grid_encoder_microbatch=grid_encoder_microbatch,
            use_sdpa=use_sdpa,
            max_objects=max_objects,
            enable_forced_paired_diagnostic=(
                dense_conditioning_mode in DENSE_PAIRED_CONDITIONING_MODES
                and dense_relational_memory_mode
                == DENSE_RELATIONAL_MEMORY_DISABLED
            ),
            enable_relational_memory_v2=(
                dense_relational_memory_mode
                == DENSE_RELATIONAL_MEMORY_SET_CROSS_ATTENTION_V2
            ),
        )
        self.vit5_transfer_report: ViT5TransferReport | None = None
        self.vit5_random_initialization_report: (
            ViT5RandomInitializationReport | None
        ) = None
        if self.vit5_initialization == VIT5_INITIALIZATION_PRETRAINED_LORA:
            self.vit5_transfer_report = transfer_vit5_small_checkpoint(
                self.arc_encoder.grid_encoder,
                vit5_checkpoint_path,
            )
        else:
            self.vit5_random_initialization_report = (
                initialize_vit5_small_random_scratch(
                    self.arc_encoder.grid_encoder
                )
            )
        self.encoder_to_t5 = nn.Sequential(
            RMSNorm(VIT5_SMALL_DIM, eps=1e-6),
            nn.Linear(VIT5_SMALL_DIM, int(self.t5.config.d_model), bias=False),
        )
        nn.init.xavier_uniform_(self.encoder_to_t5[1].weight)
        self.dense_decoder = (
            DenseGridQueryDecoder(
                hidden_size=dense_hidden_size,
                num_heads=dense_num_heads,
                num_layers=dense_decoder_layers,
                mlp_ratio=dense_mlp_ratio,
                dropout=dense_dropout,
                lambda_shape=lambda_shape,
                lambda_component=lambda_component,
                recursive_steps=dense_recursive_steps,
                recursive_loss_weighting=dense_recursive_loss_weighting,
                hsw_lambda=dense_hsw_lambda,
                candidate_encoder_layers=dense_candidate_encoder_layers,
                candidate_mlp_ratio=dense_candidate_mlp_ratio,
                candidate_dropout=dense_candidate_dropout,
                conditioning_mode=dense_conditioning_mode,
                same_coordinate_gate_initial_value=(
                    dense_same_coordinate_gate_initial_value
                ),
                paired_gate_initial_value=dense_paired_gate_initial_value,
                relational_memory_mode=dense_relational_memory_mode,
                relational_gate_initial_value=(
                    dense_relational_gate_initial_value
                ),
                shape_decoder_mode=dense_shape_decoder_mode,
            )
            if decoder_mode in {"dense", "hybrid"}
            else None
        )
        self.dense_decoder_config = (
            self.dense_decoder.config_dict() if self.dense_decoder is not None else None
        )
        self.encode_invocation_count = 0
        self.vit5_lora_targets: list[str] = []
        self.t5_decoder_lora_targets: list[str] = []

        self._apply_freezing()
        if enable_vit5_lora:
            targets = resolve_vit5_lora_targets(self)
            self.vit5_lora_targets = inject_lora_modules(
                self, targets, r=lora_r, alpha=lora_alpha, dropout=lora_dropout
            )
            print(
                f"vit5 LoRA targets ({len(self.vit5_lora_targets)}): "
                + ", ".join(self.vit5_lora_targets)
            )
        if enable_t5_decoder_lora:
            targets = resolve_t5_decoder_lora_targets(self)
            if any(name.startswith("t5.encoder.") for name in targets):
                raise ValueError("T5 encoder LoRA target audit failed.")
            self.t5_decoder_lora_targets = inject_lora_modules(
                self, targets, r=lora_r, alpha=lora_alpha, dropout=lora_dropout
            )
            print(
                f"t5 decoder LoRA targets ({len(self.t5_decoder_lora_targets)}): "
                + ", ".join(self.t5_decoder_lora_targets)
            )

    def _resolve_added_token_ids(self, added_token_ids: list[int] | None) -> list[int]:
        if added_token_ids is not None:
            return list(dict.fromkeys(int(value) for value in added_token_ids))
        result: list[int] = []
        for token in [*get_arc_special_tokens(), "<s>"]:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id is None or token_id == self.tokenizer.unk_token_id:
                continue
            result.append(int(token_id))
        return list(dict.fromkeys(result))

    def _get_lm_head(self) -> nn.Module:
        return self.t5.lm_head

    def _lm_head_is_tied(self) -> bool:
        return self.t5.lm_head.weight.data_ptr() == self.t5.get_input_embeddings().weight.data_ptr()

    def _apply_freezing(self) -> None:
        for parameter in self.t5.encoder.parameters():
            parameter.requires_grad = False
        if self.freeze_t5:
            for parameter in self.t5.parameters():
                parameter.requires_grad = False
        # Shared input embeddings and the tied/untied LM head are always frozen.
        # ARC adaptation is represented only by ``arc_token_delta``.
        for parameter in self.t5.get_input_embeddings().parameters():
            parameter.requires_grad = False
        for parameter in self.t5.lm_head.parameters():
            parameter.requires_grad = False
        if self.freeze_vit5:
            for name, parameter in self.arc_encoder.grid_encoder.named_parameters():
                if not name.startswith("arc_embedding."):
                    parameter.requires_grad = False
        # New ARC modules are always trainable.
        for parameter in self.arc_encoder.grid_encoder.arc_embedding.parameters():
            parameter.requires_grad = True
        object_table = self.arc_encoder.grid_encoder.arc_embedding.object_embedding
        if self.object_embedding_mode == "none":
            with torch.no_grad():
                object_table.weight.zero_()
            object_table.weight.requires_grad = False
        for parameter in self.arc_encoder.task_fusion.parameters():
            parameter.requires_grad = True
        for parameter in self.arc_encoder.cell_refine_norm.parameters():
            parameter.requires_grad = True
        for parameter in self.arc_encoder.cell_to_grid_attention.parameters():
            parameter.requires_grad = True
        self.arc_encoder.cell_refine_gamma.requires_grad = True
        for parameter in self.encoder_to_t5.parameters():
            parameter.requires_grad = self.decoder_mode in {"autoregressive", "hybrid"}
        self.arc_token_delta.delta_weight.requires_grad = self.decoder_mode in {
            "autoregressive",
            "hybrid",
        }
        if self.dense_decoder is not None:
            for parameter in self.dense_decoder.parameters():
                parameter.requires_grad = True

    def enable_new_token_embedding_training(self) -> dict[str, Any]:
        if self.decoder_mode not in {"autoregressive", "hybrid"}:
            raise ValueError("ARC token deltas are not trained in dense decoder mode.")
        self.arc_token_delta.delta_weight.requires_grad = True
        return {
            "enabled": True,
            "added_token_ids": list(self.added_token_ids),
            "delta_shape": tuple(self.arc_token_delta.delta_weight.shape),
            "full_shared_embedding_trainable": False,
        }

    def get_input_embeddings(self) -> nn.Module:
        return self.t5.get_input_embeddings()

    def embed_decoder_tokens(self, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        return self.arc_token_delta.embed(decoder_input_ids, self.t5.get_input_embeddings())

    def encode_arc(
        self,
        project_to_t5: bool | None = None,
        **encoder_inputs: torch.Tensor,
    ) -> Vit5HybridEncoderCache:
        self.encode_invocation_count += 1
        arc_output = self.arc_encoder(**encoder_inputs)
        if project_to_t5 is None:
            project_to_t5 = self.decoder_mode in {"autoregressive", "hybrid"}
        decoder_memory = (
            self.encoder_to_t5(arc_output.last_hidden_state)
            if project_to_t5
            else arc_output.last_hidden_state
        )
        return Vit5HybridEncoderCache(
            last_hidden_state=decoder_memory,
            attention_mask=arc_output.attention_mask.long(),
            arc_last_hidden_state=arc_output.last_hidden_state,
            memory_layout=arc_output.memory_layout,
            arc_encoder_output=arc_output,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        encoder_outputs: Vit5HybridEncoderCache | None = None,
        past_key_values: Any | None = None,
        use_cache: bool | None = None,
        dense_target_ids: torch.Tensor | None = None,
        dense_target_mask: torch.Tensor | None = None,
        dense_target_component_ids: torch.Tensor | None = None,
        target_height: torch.Tensor | None = None,
        target_width: torch.Tensor | None = None,
        active_head: str | None = None,
        **encoder_inputs: Any,
    ) -> Vit5HybridOutput:
        del input_ids, attention_mask
        if self.decoder_mode == "autoregressive":
            resolved_head = "autoregressive"
        elif self.decoder_mode == "dense":
            resolved_head = "dense"
        else:
            resolved_head = active_head or "both"
            if resolved_head not in {"autoregressive", "dense", "both"}:
                raise ValueError("Hybrid active_head must be autoregressive, dense or both.")
        if encoder_outputs is None:
            required_prefix = "encoder_"
            explicit_encoder_inputs = {
                key: value
                for key, value in encoder_inputs.items()
                if key.startswith(required_prefix)
            }
            if not explicit_encoder_inputs:
                raise ValueError("Explicit encoder_* ARC tensors are required.")
            encoder_outputs = self.encode_arc(
                project_to_t5=resolved_head in {"autoregressive", "both"},
                **explicit_encoder_inputs,
            )

        dense_output: DenseDecoderOutput | None = None
        if resolved_head in {"dense", "both"}:
            if self.dense_decoder is None or encoder_outputs.arc_encoder_output is None:
                raise RuntimeError("Dense mode requires the ARC encoder output and dense decoder.")
            arc_output = encoder_outputs.arc_encoder_output
            paired_mode = (
                self.dense_decoder.conditioning_mode
                in DENSE_PAIRED_CONDITIONING_MODES
            )
            relational_v2 = (
                getattr(
                    self.dense_decoder,
                    "relational_memory_mode",
                    DENSE_RELATIONAL_MEMORY_DISABLED,
                )
                == DENSE_RELATIONAL_MEMORY_SET_CROSS_ATTENTION_V2
            )
            if paired_mode:
                # Both paired modes use raw pre-fusion query cells for Q and
                # same-coordinate injection.  Forced also isolates shape from
                # demonstrations; Gated restores the normal fused query summary
                # so the context branch can support future shape reasoning.
                query_grid_summary = (
                    arc_output.forced_query_grid_summary
                    if self.dense_decoder.conditioning_mode
                    == DENSE_CONDITIONING_FORCED_PAIRED
                    else arc_output.query_grid_summary
                )
                query_cell_memory = arc_output.forced_query_cell_memory
                query_cell_mask = arc_output.forced_query_cell_mask
                query_cell_flat_indices = arc_output.forced_query_cell_flat_indices
                missing_raw_query = (
                    query_grid_summary is None
                    or query_cell_memory is None
                    or query_cell_mask is None
                    or query_cell_flat_indices is None
                )
                missing_relation = relational_v2 and any(
                    value is None
                    for value in (
                        arc_output.relational_demo_input_memory,
                        arc_output.relational_demo_input_mask,
                        arc_output.relational_demo_input_example_ids,
                        arc_output.relational_demo_input_geometry,
                        arc_output.relational_demo_output_memory,
                        arc_output.relational_demo_output_mask,
                        arc_output.relational_demo_output_example_ids,
                        arc_output.relational_demo_output_geometry,
                    )
                )
                missing_paired = (not relational_v2) and (
                    arc_output.paired_demo_input_memory is None
                    or arc_output.paired_demo_output_memory is None
                    or arc_output.paired_demo_mask is None
                )
                if missing_raw_query or missing_relation or missing_paired:
                    raise RuntimeError(
                        "ARC encoder did not expose the requested paired/relational "
                        "conditioning tensors."
                    )
            else:
                query_grid_summary = arc_output.query_grid_summary
                query_cell_memory = arc_output.query_cell_memory
                query_cell_mask = arc_output.query_cell_mask
                query_cell_flat_indices = arc_output.query_cell_flat_indices
            dense_output = self.dense_decoder(
                encoder_memory=arc_output.last_hidden_state,
                encoder_attention_mask=arc_output.attention_mask,
                context_attention_mask=arc_output.context_attention_mask,
                memory_layout=arc_output.memory_layout,
                task_summary=arc_output.task_summary,
                query_grid_summary=query_grid_summary,
                query_cell_memory=query_cell_memory,
                query_cell_mask=query_cell_mask,
                query_cell_flat_indices=query_cell_flat_indices,
                paired_demo_input_memory=(
                    arc_output.paired_demo_input_memory
                    if paired_mode and not relational_v2
                    else None
                ),
                paired_demo_output_memory=(
                    arc_output.paired_demo_output_memory
                    if paired_mode and not relational_v2
                    else None
                ),
                paired_demo_mask=(
                    arc_output.paired_demo_mask
                    if paired_mode and not relational_v2
                    else None
                ),
                relational_demo_input_memory=(
                    arc_output.relational_demo_input_memory if relational_v2 else None
                ),
                relational_demo_input_mask=(
                    arc_output.relational_demo_input_mask if relational_v2 else None
                ),
                relational_demo_input_example_ids=(
                    arc_output.relational_demo_input_example_ids
                    if relational_v2
                    else None
                ),
                relational_demo_input_geometry=(
                    arc_output.relational_demo_input_geometry
                    if relational_v2
                    else None
                ),
                relational_demo_output_memory=(
                    arc_output.relational_demo_output_memory if relational_v2 else None
                ),
                relational_demo_output_mask=(
                    arc_output.relational_demo_output_mask if relational_v2 else None
                ),
                relational_demo_output_example_ids=(
                    arc_output.relational_demo_output_example_ids
                    if relational_v2
                    else None
                ),
                relational_demo_output_geometry=(
                    arc_output.relational_demo_output_geometry
                    if relational_v2
                    else None
                ),
                dense_target_ids=dense_target_ids,
                dense_target_mask=dense_target_mask,
                dense_target_component_ids=dense_target_component_ids,
                target_height=target_height,
                target_width=target_width,
            )
        logits: torch.Tensor | None = None
        ar_loss: torch.Tensor | None = None
        t5_output: Any | None = None
        if resolved_head in {"autoregressive", "both"}:
            from transformers.modeling_outputs import BaseModelOutput

            if decoder_input_ids is None:
                if labels is None:
                    raise ValueError("decoder_input_ids or labels are required for T5 decoding.")
                decoder_input_ids = self.t5._shift_right(labels)
            decoder_inputs_embeds = self.embed_decoder_tokens(decoder_input_ids.long())
            t5_output = self.t5(
                encoder_outputs=BaseModelOutput(last_hidden_state=encoder_outputs.last_hidden_state),
                attention_mask=encoder_outputs.attention_mask,
                labels=labels,
                decoder_inputs_embeds=decoder_inputs_embeds,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_hidden_states=True,
                return_dict=True,
            )
            decoder_hidden = t5_output.decoder_hidden_states[-1]
            if bool(getattr(self.t5.config, "tie_word_embeddings", False)):
                decoder_hidden = decoder_hidden * (float(self.t5.config.d_model) ** -0.5)
            logits = self.arc_token_delta.add_to_logits(t5_output.logits, decoder_hidden)
            if labels is not None:
                ar_loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    ignore_index=-100,
                )

        raw_cell_loss = dense_output.raw_cell_loss if dense_output is not None else None
        component_balanced_loss = (
            dense_output.component_balanced_loss if dense_output is not None else None
        )
        color_loss = dense_output.color_loss if dense_output is not None else None
        height_loss = dense_output.height_loss if dense_output is not None else None
        width_loss = dense_output.width_loss if dense_output is not None else None
        border_loss = dense_output.border_loss if dense_output is not None else None
        shape_loss = (
            border_loss
            if border_loss is not None
            else (
                height_loss + width_loss
                if height_loss is not None and width_loss is not None
                else None
            )
        )
        dense_loss = dense_output.dense_loss if dense_output is not None else None
        contributions: dict[str, torch.Tensor] = {}
        if ar_loss is not None:
            contributions["weighted_ar"] = self.lambda_ar * ar_loss
        if dense_loss is not None:
            contributions["weighted_dense_branch"] = self.lambda_dense * dense_loss
        if raw_cell_loss is not None:
            contributions["weighted_raw_cell"] = self.lambda_dense * raw_cell_loss
        if component_balanced_loss is not None:
            contributions["weighted_component"] = (
                self.lambda_dense * self.lambda_component * component_balanced_loss
            )
        if shape_loss is not None:
            contributions["weighted_shape"] = (
                self.lambda_dense * self.lambda_shape * shape_loss
            )
        total_terms = [
            contributions[name]
            for name in ("weighted_ar", "weighted_dense_branch")
            if name in contributions
        ]
        total_loss = sum(total_terms) if total_terms else None
        if self.decoder_mode == "autoregressive":
            total_loss = ar_loss
        elif self.decoder_mode == "dense":
            total_loss = dense_loss
        return Vit5HybridOutput(
            loss=total_loss,
            logits=logits,
            past_key_values=t5_output.past_key_values if t5_output is not None else None,
            encoder_cache=encoder_outputs,
            t5_output=t5_output,
            dense_output=dense_output,
            ar_loss=ar_loss,
            raw_cell_loss=raw_cell_loss,
            component_balanced_loss=component_balanced_loss,
            color_loss=color_loss,
            height_loss=height_loss,
            width_loss=width_loss,
            border_loss=border_loss,
            shape_loss=shape_loss,
            dense_loss=dense_loss,
            total_loss=total_loss,
            active_head=resolved_head,
            loss_contributions=contributions,
        )

    @torch.no_grad()
    def predict_dense_grid(
        self,
        encoder_outputs: Vit5HybridEncoderCache | None = None,
        include_raw_logits: bool = False,
        include_recursive_rounds: bool = False,
        **encoder_inputs: torch.Tensor,
    ) -> list[dict[str, Any]]:
        if self.decoder_mode not in {"dense", "hybrid"}:
            raise ValueError("predict_dense_grid requires dense or hybrid decoder mode.")
        self.eval()
        output = self(
            encoder_outputs=encoder_outputs, active_head="dense", **encoder_inputs
        )
        if output.dense_output is None:
            raise RuntimeError("Dense forward did not return dense output.")
        decoded = decode_dense_output(
            output.dense_output.color_logits,
            output.dense_output.height_logits,
            output.dense_output.width_logits,
            joint_shape_logits=output.dense_output.joint_shape_logits,
            include_raw_logits=include_raw_logits,
        )
        if include_recursive_rounds:
            if output.dense_output.round_outputs is None:
                raise RuntimeError("Dense recursive round outputs are unavailable.")
            round_decoded = [
                decode_dense_output(
                    round_output.color_logits,
                    round_output.height_logits,
                    round_output.width_logits,
                    joint_shape_logits=round_output.joint_shape_logits,
                )
                for round_output in output.dense_output.round_outputs
            ]
            for batch_index, item in enumerate(decoded):
                item["recursive_rounds"] = [
                    round_items[batch_index] for round_items in round_decoded
                ]
        return decoded

    @torch.no_grad()
    def grammar_constrained_generate(
        self,
        token_to_id: dict[str, int],
        pad_token_id: int,
        max_steps: int = 1124,
        grid_rows: int = 33,
        content_cols: int = 32,
        encoder_outputs: Vit5HybridEncoderCache | None = None,
        **encoder_inputs: torch.Tensor,
    ) -> dict[str, Any]:
        """Encode once, then run cached grammar-constrained T5 decoding."""
        from .vitarc_grammar_decoder import VitarcGrammarState, mask_logits_to_allowed_ids

        if self.decoder_mode not in {"autoregressive", "hybrid"}:
            raise ValueError("Grammar generation requires autoregressive or hybrid mode.")
        self.eval()
        grammar_tokens = [
            "<s>",
            "</s>",
            *[f"<arc_{index}>" for index in range(10)],
            "<arc_pad>",
            "<arc_nl>",
            "<arc_endxgrid>",
            "<arc_endygrid>",
            "<arc_endxygrid>",
        ]
        for token in grammar_tokens:
            if token not in token_to_id:
                raise ValueError(f"Required grammar token {token!r} is missing from vocab.")
            token_id = int(token_to_id[token])
            tokenizer_id = int(self.tokenizer.convert_tokens_to_ids(token))
            if token_id != tokenizer_id or not 0 <= token_id < self.resolved_vocab_size:
                raise ValueError(
                    f"Grammar token {token!r} has unsafe id {token_id}; "
                    f"tokenizer id={tokenizer_id}, resolved vocab={self.resolved_vocab_size}."
                )
        cache = encoder_outputs or self.encode_arc(**encoder_inputs)
        state = VitarcGrammarState(grid_rows=grid_rows, content_cols=content_cols)
        id_to_token = {token_id: token for token, token_id in token_to_id.items()}
        next_input = torch.tensor(
            [[pad_token_id]], dtype=torch.long, device=cache.last_hidden_state.device
        )
        past = None
        generated_ids: list[int] = []
        generated_tokens: list[str] = []
        for _ in range(max_steps):
            output = self(
                encoder_outputs=cache,
                decoder_input_ids=next_input,
                past_key_values=past,
                use_cache=True,
                active_head="autoregressive",
            )
            logits = output.logits[:, -1]
            allowed = state.allowed_token_ids(token_to_id)
            masked = mask_logits_to_allowed_ids(logits, allowed)
            next_id = int(masked.argmax(dim=-1).item())
            next_token = id_to_token[next_id]
            state.update(next_token)
            generated_ids.append(next_id)
            generated_tokens.append(next_token)
            past = output.past_key_values
            next_input = torch.tensor(
                [[next_id]], dtype=torch.long, device=cache.last_hidden_state.device
            )
            if state.is_done:
                break
        if not state.is_done:
            raise ValueError(f"Grammar decoding did not reach DONE: {state.summary()}")
        return {
            "generated_ids": generated_ids,
            "generated_tokens": generated_tokens,
            "grammar_summary": state.summary(),
            "steps": len(generated_ids),
            "encoder_invocations": self.encode_invocation_count,
            "encoder_cache": cache,
        }

    @torch.no_grad()
    def predict_hybrid_candidates(
        self,
        token_to_id: dict[str, int],
        pad_token_id: int,
        include_raw_dense_logits: bool = False,
        **encoder_inputs: torch.Tensor,
    ) -> dict[str, Any]:
        """Explicit debug-only dual candidate call; it never fuses or ranks heads."""
        if self.decoder_mode != "hybrid":
            raise ValueError("Dual candidate analysis requires decoder_mode='hybrid'.")
        cache = self.encode_arc(project_to_t5=True, **encoder_inputs)
        autoregressive = self.grammar_constrained_generate(
            token_to_id=token_to_id,
            pad_token_id=pad_token_id,
            encoder_outputs=cache,
        )
        dense = self.predict_dense_grid(
            encoder_outputs=cache,
            include_raw_logits=include_raw_dense_logits,
        )
        return {
            "autoregressive": autoregressive,
            "dense": dense,
            "encoder_invocations": 1,
        }

    def training_components_summary(self) -> dict[str, Any]:
        trainable_by_prefix: dict[str, int] = {}
        trainable_names: list[str] = []
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                trainable_names.append(name)
                prefix = name.split(".", 1)[0]
                trainable_by_prefix[prefix] = trainable_by_prefix.get(prefix, 0) + parameter.numel()
        return {
            "total_params": sum(parameter.numel() for parameter in self.parameters()),
            "trainable_params": sum(
                parameter.numel() for parameter in self.parameters() if parameter.requires_grad
            ),
            "lora_trainable_params": sum(
                parameter.numel()
                for name, parameter in self.named_parameters()
                if parameter.requires_grad and "lora_" in name
            ),
            "trainable_by_prefix": trainable_by_prefix,
            "trainable_parameter_names": trainable_names,
            "vit5_lora_targets": list(self.vit5_lora_targets),
            "t5_decoder_lora_targets": list(self.t5_decoder_lora_targets),
            "vit5_initialization": self.vit5_initialization,
            "vit5_checkpoint_used": self.vit5_transfer_report is not None,
            "vit5_random_initialization_report": (
                self.vit5_random_initialization_report.to_dict()
                if self.vit5_random_initialization_report is not None
                else None
            ),
            "added_token_ids": list(self.added_token_ids),
            "new_token_embedding_trainable_rows": (
                len(self.added_token_ids)
                if self.arc_token_delta.delta_weight.requires_grad
                else 0
            ),
            "arc_token_delta_params": self.arc_token_delta.delta_weight.numel(),
            "vocabulary_info": dict(self.vocabulary_info),
            "is_test_fixture": self.is_test_fixture,
            "decoder_mode": self.decoder_mode,
            "dense_decoder_config": self.dense_decoder_config,
            "object_embedding": {
                "enabled": self.object_embedding_mode == "connected_components",
                "mode": self.object_embedding_mode,
                "max_objects": self.max_objects,
                "background_color": 0,
                "connectivity": 4,
                "parameter_count": self.arc_encoder.grid_encoder.arc_embedding.object_embedding.weight.numel(),
            },
            "hybrid_training_strategy": self.hybrid_training_strategy,
            "lambda_ar": self.lambda_ar,
            "lambda_dense": self.lambda_dense,
            "lambda_shape": self.lambda_shape,
            "lambda_component": self.lambda_component,
            "loss_schema_version": self.loss_schema_version,
            "target_component_mode": self.target_component_mode,
            "component_loss_type": self.component_loss_type,
            "dense_decoder_params": (
                sum(parameter.numel() for parameter in self.dense_decoder.parameters())
                if self.dense_decoder is not None
                else 0
            ),
        }


def build_vit5_hybrid_model(**kwargs: Any) -> Vit5HybridForArc:
    return Vit5HybridForArc(**kwargs)
