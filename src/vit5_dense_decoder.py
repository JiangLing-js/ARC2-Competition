"""Dense ARC query decoder with separate-H/W and BD-border output modes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .vit5_layers import RMSNorm, initialize_vit5_module

VIT5_DENSE_LOSS_SCHEMA_VERSION = 2
DENSE_SHAPE_DECODER_SEPARATE_HW = "separate_hw_v1"
DENSE_SHAPE_DECODER_BD_BORDER = "bd_border_v1"
DENSE_SHAPE_DECODER_MODES = (
    DENSE_SHAPE_DECODER_SEPARATE_HW,
    DENSE_SHAPE_DECODER_BD_BORDER,
)
DENSE_BD_BORDER_SCHEMA_VERSION = 1
DENSE_BD_CLASS_ID = 10
DENSE_BD_CANVAS_SIZE = 31
DENSE_BD_NUM_CLASSES = 11
DENSE_RECURSION_SCHEMA_VERSION = 2
DENSE_CANDIDATE_MAX_ROUNDS = 8
DENSE_CANDIDATE_FUSION_TYPE = "separate_cross_attention_gated_residual"
DENSE_CANDIDATE_GATE_PARAMETERIZATION = "per_layer_scalar_tanh"
DENSE_CANDIDATE_GATE_INITIAL_VALUE = 0.0
TARGET_COMPONENT_MODE = "all_color_4_connected"
COMPONENT_LOSS_TYPE = "component_balanced_mean_ce"
RECURSIVE_LOSS_WEIGHTINGS = ("uniform", "hsw")
DENSE_CONDITIONING_SCHEMA_VERSION = 1
DENSE_SAME_COORDINATE_SCHEMA_VERSION = 2
DENSE_FORCED_PAIRED_SCHEMA_VERSION = 3
DENSE_GATED_PAIRED_SCHEMA_VERSION = 4
DENSE_GATED_RELATIONAL_SCHEMA_VERSION = 5
DENSE_RELATIONAL_MEMORY_SCHEMA_VERSION = 2
DENSE_CONDITIONING_LEGACY = "legacy_packed_v1"
DENSE_CONDITIONING_SPLIT = "split_query_context_v1"
DENSE_CONDITIONING_SAME_COORDINATE = "same_coordinate_bridge_v1"
DENSE_CONDITIONING_FORCED_PAIRED = "forced_paired_associative_v1"
DENSE_CONDITIONING_GATED_PAIRED = "gated_paired_associative_v1"
# Production trainers deliberately retain the original three modes.  The
# paired-associative modes are exposed only by the dedicated context-rule
# diagnostic until their ARC-general relation semantics are validated.
DENSE_CONDITIONING_MODES = (
    DENSE_CONDITIONING_LEGACY,
    DENSE_CONDITIONING_SPLIT,
    DENSE_CONDITIONING_SAME_COORDINATE,
)
# Paired-associative conditioning is still narrower than a general ARC
# relation mechanism.  Gated has nevertheless passed its dedicated synthetic
# qualification and is allowed in an explicitly labelled production pilot.
# Forced remains a diagnostic control and must never be accepted by a normal
# trainer CLI.
DENSE_EXPERIMENTAL_PRODUCTION_CONDITIONING_MODES = (
    DENSE_CONDITIONING_GATED_PAIRED,
)
DENSE_CLEAN_GLOBAL_CONDITIONING_MODES = (
    *DENSE_CONDITIONING_MODES,
    *DENSE_EXPERIMENTAL_PRODUCTION_CONDITIONING_MODES,
)
DENSE_SUPPORTED_CONDITIONING_MODES = (
    *DENSE_CONDITIONING_MODES,
    DENSE_CONDITIONING_FORCED_PAIRED,
    DENSE_CONDITIONING_GATED_PAIRED,
)
DENSE_CONDITIONING_FUSION_TYPE = (
    "separate_normalized_context_and_query_cross_attention"
)
DENSE_QUERY_INJECTION_POSITION = "post_mlp_every_layer"
DENSE_CONTEXT_MEMORY_MODE = "packed_without_query_cells"
DENSE_QUERY_BRANCH_SCALE_INITIAL_VALUE = 1.0
DENSE_CONTEXT_BRANCH_SCALE_INITIAL_VALUE = 1.0
DENSE_SAME_COORDINATE_FUSION_TYPE = (
    "separate_normalized_context_and_same_coordinate_residual"
)
DENSE_SAME_COORDINATE_GATE_PARAMETERIZATION = "per_layer_per_channel_direct"
DENSE_SAME_COORDINATE_GATE_INITIAL_VALUE = 1.0
DENSE_FORCED_PAIRED_FUSION_TYPE = (
    "forced_pre_fusion_paired_associative_read"
)
DENSE_FORCED_PAIRED_PAIRING = "same_example_same_flat_coordinate"
DENSE_FORCED_PAIRED_SOURCE = "pre_task_fusion_independent_grid_cells"
DENSE_FORCED_PAIRED_NO_CONTEXT_POLICY = "exact_zero_residual"
DENSE_GATED_PAIRED_FUSION_TYPE = (
    "separate_generic_context_and_gated_pre_fusion_paired_associative_read"
)
DENSE_GATED_PAIRED_GATE_PARAMETERIZATION = (
    "per_layer_per_cell_scalar_sigmoid_x2"
)
DENSE_GATED_PAIRED_GATE_INITIAL_VALUE = 1.0
DENSE_GATED_PAIRED_GATE_INPUT = (
    "normalized_post_query_hidden_and_paired_residual"
)
DENSE_GATED_PAIRED_INJECTION_POSITION = (
    "post_query_every_layer_final_residual"
)
DENSE_RELATIONAL_MEMORY_DISABLED = "disabled"
DENSE_RELATIONAL_MEMORY_SET_CROSS_ATTENTION_V2 = "set_cross_attention_v2"
DENSE_RELATIONAL_MEMORY_MODES = (
    DENSE_RELATIONAL_MEMORY_DISABLED,
    DENSE_RELATIONAL_MEMORY_SET_CROSS_ATTENTION_V2,
)
DENSE_RELATIONAL_MEMORY_FUSION_TYPE = (
    "generic_context_same_coordinate_query_and_gated_relation_set_cross_attention"
)
DENSE_RELATIONAL_MEMORY_SOURCE = "pre_task_fusion_independent_demo_cell_sets"
DENSE_RELATIONAL_MEMORY_PAIRING = "same_example_output_cells_query_input_set"
DENSE_RELATIONAL_MEMORY_GEOMETRY = (
    "row_over_max_grid_height_minus_1_col_over_max_grid_width_minus_1_"
    "grid_height_over_30_grid_width_over_30"
)
DENSE_RELATIONAL_MEMORY_GATE_PARAMETERIZATION = (
    "per_layer_per_output_cell_scalar_sigmoid_x2"
)
DENSE_RELATIONAL_MEMORY_GATE_INITIAL_VALUE = 1.0
DENSE_RELATIONAL_MEMORY_GATE_INPUT = (
    "normalized_post_query_hidden_and_relational_cross_attention_residual"
)
DENSE_RELATIONAL_MEMORY_INJECTION_POSITION = (
    "post_same_coordinate_query_every_layer_all_output_positions"
)
DENSE_PAIRED_CONDITIONING_MODES = (
    DENSE_CONDITIONING_FORCED_PAIRED,
    DENSE_CONDITIONING_GATED_PAIRED,
)


@dataclass
class DenseDecoderOutput:
    color_logits: torch.Tensor
    height_logits: torch.Tensor
    width_logits: torch.Tensor
    raw_cell_loss: torch.Tensor | None = None
    component_balanced_loss: torch.Tensor | None = None
    color_loss: torch.Tensor | None = None
    height_loss: torch.Tensor | None = None
    width_loss: torch.Tensor | None = None
    border_loss: torch.Tensor | None = None
    dense_loss: torch.Tensor | None = None
    metrics: dict[str, torch.Tensor] | None = None
    query_hidden_state: torch.Tensor | None = None
    round_outputs: tuple["DenseDecoderOutput", ...] | None = None
    round_loss_weights: torch.Tensor | None = None
    recursive_metrics: dict[str, Any] | None = None
    joint_shape_logits: torch.Tensor | None = None


@dataclass(frozen=True)
class DenseCandidateState:
    """Detached, constraint-projected candidate passed to the next round."""

    color_ids: torch.Tensor
    height: torch.Tensor
    width: torch.Tensor
    cell_mask: torch.Tensor


@dataclass
class DenseCandidateEncoderOutput:
    memory: torch.Tensor
    attention_mask: torch.Tensor
    summary: torch.Tensor
    cell_mask: torch.Tensor


def recursive_supervision_weights(
    recursive_steps: int,
    weighting: str = "hsw",
    hsw_lambda: float = 0.7,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return normalized uniform or exponentially decayed supervision weights."""
    if recursive_steps < 1:
        raise ValueError("recursive_steps must be >= 1.")
    if weighting not in RECURSIVE_LOSS_WEIGHTINGS:
        raise ValueError(
            f"recursive loss weighting must be one of {RECURSIVE_LOSS_WEIGHTINGS}."
        )
    if not 0.0 < hsw_lambda <= 1.0:
        raise ValueError("hsw_lambda must be in (0,1].")
    if weighting == "uniform":
        values = torch.ones(recursive_steps, device=device, dtype=dtype)
    else:
        exponent = torch.arange(recursive_steps, device=device, dtype=dtype)
        values = torch.as_tensor(hsw_lambda, device=device, dtype=dtype).pow(exponent)
    return values / values.sum()


@torch.no_grad()
def project_dense_candidate(
    color_logits: torch.Tensor,
    height_logits: torch.Tensor,
    width_logits: torch.Tensor,
) -> DenseCandidateState:
    """Greedily project Dense logits to a legal, detached ARC candidate state."""
    if color_logits.ndim != 4 or color_logits.shape[1:] != (30, 30, 10):
        raise ValueError("candidate color_logits must have shape [B,30,30,10].")
    batch_size = color_logits.shape[0]
    if height_logits.shape != (batch_size, 30) or width_logits.shape != (
        batch_size,
        30,
    ):
        raise ValueError("candidate shape logits must both have shape [B,30].")
    color_ids = color_logits.detach().argmax(dim=-1).long()
    height = (height_logits.detach().argmax(dim=-1) + 1).long()
    width = (width_logits.detach().argmax(dim=-1) + 1).long()
    rows = torch.arange(30, device=color_ids.device).view(1, 30, 1)
    cols = torch.arange(30, device=color_ids.device).view(1, 1, 30)
    cell_mask = (rows < height.view(-1, 1, 1)) & (
        cols < width.view(-1, 1, 1)
    )
    if color_ids.requires_grad or height.requires_grad or width.requires_grad:
        raise AssertionError("Projected Dense candidates must be detached.")
    return DenseCandidateState(color_ids, height, width, cell_mask)


class _CandidateEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_norm = RMSNorm(hidden_size)
        self.self_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.mlp_norm = RMSNorm(hidden_size)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )
        self.apply(initialize_vit5_module)

    def forward(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        normalized = self.self_norm(hidden)
        attended, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~attention_mask.bool(),
            need_weights=False,
        )
        hidden = hidden + attended
        hidden = hidden + self.mlp(self.mlp_norm(hidden))
        return hidden * attention_mask.unsqueeze(-1).to(hidden.dtype)


class DenseCandidateEncoder(nn.Module):
    """Encode a detached H/W/color candidate into refinement memory."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        max_rounds: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_size: int = 30,
        num_colors: int = 10,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("Candidate encoder requires at least one layer.")
        if max_rounds < 2:
            raise ValueError("Candidate encoder is only used for recursive_steps >= 2.")
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.max_rounds = int(max_rounds)
        self.max_size = int(max_size)
        self.candidate_summary_token = nn.Parameter(
            torch.zeros(1, 1, hidden_size)
        )
        self.color_embedding = nn.Embedding(num_colors, hidden_size)
        self.row_embedding = nn.Embedding(max_size, hidden_size)
        self.col_embedding = nn.Embedding(max_size, hidden_size)
        self.height_embedding = nn.Embedding(max_size, hidden_size)
        self.width_embedding = nn.Embedding(max_size, hidden_size)
        self.round_embedding = nn.Embedding(max_rounds, hidden_size)
        self.layers = nn.ModuleList(
            [
                _CandidateEncoderLayer(
                    hidden_size, num_heads, mlp_ratio, dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size)
        self.apply(initialize_vit5_module)
        nn.init.trunc_normal_(self.candidate_summary_token, std=0.02)

    def forward(
        self, candidate: DenseCandidateState, round_index: int
    ) -> DenseCandidateEncoderOutput:
        if not 1 <= int(round_index) < self.max_rounds:
            raise ValueError(
                f"candidate round_index must be in [1,{self.max_rounds - 1}]."
            )
        color_ids = candidate.color_ids
        if color_ids.ndim != 3 or color_ids.shape[1:] != (
            self.max_size,
            self.max_size,
        ):
            raise ValueError("candidate color_ids must have shape [B,30,30].")
        if candidate.cell_mask.shape != color_ids.shape:
            raise ValueError("candidate cell_mask must match candidate color_ids.")
        batch_size = color_ids.shape[0]
        if candidate.height.shape != (batch_size,) or candidate.width.shape != (
            batch_size,
        ):
            raise ValueError("candidate height and width must have shape [B].")
        if bool(((color_ids < 0) | (color_ids >= self.color_embedding.num_embeddings)).any().item()):
            raise ValueError("candidate colors must be legal ARC color IDs.")
        if bool(
            (
                (candidate.height < 1)
                | (candidate.height > self.max_size)
                | (candidate.width < 1)
                | (candidate.width > self.max_size)
            ).any().item()
        ):
            raise ValueError("candidate height/width must be in [1,max_size].")
        expected_rows = torch.arange(self.max_size, device=color_ids.device).view(
            1, self.max_size, 1
        )
        expected_cols = torch.arange(self.max_size, device=color_ids.device).view(
            1, 1, self.max_size
        )
        expected_mask = (expected_rows < candidate.height[:, None, None]) & (
            expected_cols < candidate.width[:, None, None]
        )
        if not torch.equal(candidate.cell_mask.bool(), expected_mask):
            raise ValueError("candidate cell_mask must exactly match its predicted H/W.")
        round_ids = torch.full(
            (batch_size,), int(round_index), dtype=torch.long, device=color_ids.device
        )
        round_embedding = self.round_embedding(round_ids)
        flat_mask = candidate.cell_mask.flatten(1).bool()
        valid_counts = flat_mask.sum(dim=1)
        if bool((valid_counts == 0).any().item()):
            raise ValueError("Every projected Dense candidate must have valid cells.")
        max_valid_cells = int(valid_counts.max().item())
        packed_indices = torch.zeros(
            batch_size,
            max_valid_cells,
            dtype=torch.long,
            device=color_ids.device,
        )
        packed_mask = torch.arange(
            max_valid_cells, device=color_ids.device
        ).unsqueeze(0) < valid_counts.unsqueeze(1)
        flat_positions = torch.arange(
            self.max_size * self.max_size, device=color_ids.device
        )
        for batch_index in range(batch_size):
            selected = flat_positions[flat_mask[batch_index]]
            packed_indices[batch_index, : selected.numel()] = selected
        flat_colors = color_ids.flatten(1)
        packed_colors = flat_colors.gather(1, packed_indices)
        flat_rows = torch.arange(self.max_size, device=color_ids.device).view(
            self.max_size, 1
        ).expand(-1, self.max_size).flatten()
        flat_cols = torch.arange(self.max_size, device=color_ids.device).view(
            1, self.max_size
        ).expand(self.max_size, -1).flatten()
        packed_rows = flat_rows[packed_indices]
        packed_cols = flat_cols[packed_indices]
        cells = (
            self.color_embedding(packed_colors)
            + self.row_embedding(packed_rows)
            + self.col_embedding(packed_cols)
            + round_embedding[:, None, :]
        )
        summary = self.candidate_summary_token.expand(batch_size, -1, -1)
        summary = summary + (
            self.height_embedding(candidate.height.long() - 1)
            + self.width_embedding(candidate.width.long() - 1)
            + round_embedding
        ).unsqueeze(1)
        hidden = torch.cat((summary, cells), dim=1)
        attention_mask = torch.cat(
            (
                torch.ones(
                    batch_size, 1, dtype=torch.bool, device=color_ids.device
                ),
                packed_mask,
            ),
            dim=1,
        )
        for layer in self.layers:
            hidden = layer(hidden, attention_mask)
        hidden = self.norm(hidden) * attention_mask.unsqueeze(-1).to(hidden.dtype)
        return DenseCandidateEncoderOutput(
            memory=hidden,
            attention_mask=attention_mask,
            summary=hidden[:, 0],
            cell_mask=candidate.cell_mask,
        )


class _PairedAssociativeRead(nn.Module):
    """Read paired demonstration values with raw query cells as addresses.

    This module is deliberately shared by all Dense layers.  It is the only
    demonstration-dependent color path in the Forced diagnostic: query cells
    attend to independently encoded demonstration-input cells (keys) and read
    their exactly paired demonstration-output cells (values).  Rows without
    demonstrations are skipped instead of passing an all-masked sequence to
    ``MultiheadAttention``; their residual is therefore exactly zero.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query_norm = RMSNorm(hidden_size)
        self.key_norm = RMSNorm(hidden_size)
        self.value_norm = RMSNorm(hidden_size)
        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
            bias=False,
        )
        self.apply(initialize_vit5_module)

    def forward(
        self,
        query_memory: torch.Tensor,
        query_mask: torch.Tensor,
        paired_input_memory: torch.Tensor,
        paired_output_memory: torch.Tensor,
        paired_mask: torch.Tensor,
    ) -> torch.Tensor:
        if query_memory.ndim != 3:
            raise ValueError("query_memory must have shape [B,Q,H].")
        if query_mask.shape != query_memory.shape[:2]:
            raise ValueError("query_mask must have shape [B,Q].")
        if paired_input_memory.ndim != 3:
            raise ValueError("paired demonstration memory must have shape [B,R,H].")
        if paired_output_memory.shape != paired_input_memory.shape:
            raise ValueError("paired input/output memories must have identical shape.")
        if paired_mask.shape != paired_input_memory.shape[:2]:
            raise ValueError("paired_mask must have shape [B,R].")
        if paired_input_memory.shape[0] != query_memory.shape[0]:
            raise ValueError("query and paired memories must have the same batch size.")
        if paired_input_memory.shape[-1] != query_memory.shape[-1]:
            raise ValueError("query and paired memories must have the same hidden size.")
        query_mask = query_mask.bool()
        paired_mask = paired_mask.bool()
        if bool((query_mask.sum(dim=1) < 1).any().item()):
            raise ValueError(
                "Every paired-associative sample needs at least one query cell."
            )

        # Keep the destination in the residual-stream dtype.  Autocast may
        # execute MHA in BF16 while the independently encoded cells remain
        # FP32; assignment performs an autograd-preserving boundary cast.
        result = torch.zeros_like(query_memory)
        nonempty_rows = paired_mask.any(dim=1)
        if bool(nonempty_rows.any().item()):
            selected_query = self.query_norm(query_memory[nonempty_rows])
            selected_keys = self.key_norm(paired_input_memory[nonempty_rows])
            selected_values = self.value_norm(paired_output_memory[nonempty_rows])
            selected_mask = paired_mask[nonempty_rows]
            attended, _ = self.attention(
                selected_query,
                selected_keys,
                selected_values,
                key_padding_mask=~selected_mask,
                need_weights=False,
            )
            attended = attended * query_mask[nonempty_rows].unsqueeze(-1).to(
                attended.dtype
            )
            result[nonempty_rows] = attended.to(result.dtype)
        return result


class _RelationalSetMemoryEncoder(nn.Module):
    """Bind variable-length demonstration input/output sets by example.

    Every valid demonstration-output cell is an independent query over all
    demonstration-input cells carrying the same example ID.  The resulting
    relation token therefore contains output content/geometry *and* a learned
    read of its paired input set, without requiring the two grids to share a
    shape, token count, or coordinate set.  Examples are processed separately
    before their relation tokens are flattened back into one memory per task.

    Geometry is supplied explicitly as ``(relative_row, relative_col,
    height/30, width/30)``.  Masked payloads are replaced with exact zeros
    before any projection, and fully empty batches bypass attention entirely.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.input_content_norm = RMSNorm(hidden_size)
        self.output_content_norm = RMSNorm(hidden_size)
        self.geometry_projection = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.role_embedding = nn.Embedding(2, hidden_size)
        self.output_query_norm = RMSNorm(hidden_size)
        self.input_memory_norm = RMSNorm(hidden_size)
        self.input_set_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
            bias=False,
        )
        self.relation_mlp_norm = RMSNorm(hidden_size)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.relation_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )
        self.output_norm = RMSNorm(hidden_size)
        self.apply(initialize_vit5_module)

    def _validate_set(
        self,
        *,
        name: str,
        memory: torch.Tensor,
        mask: torch.Tensor,
        example_ids: torch.Tensor,
        geometry: torch.Tensor,
        batch_size: int,
    ) -> None:
        if memory.ndim != 3 or memory.shape[0] != batch_size:
            raise ValueError(f"{name}_memory must have shape [B,N,H].")
        if memory.shape[-1] != self.hidden_size:
            raise ValueError(f"{name}_memory must use the Dense hidden size.")
        if mask.shape != memory.shape[:2]:
            raise ValueError(f"{name}_mask must have shape [B,N].")
        if example_ids.shape != memory.shape[:2]:
            raise ValueError(f"{name}_example_ids must have shape [B,N].")
        if geometry.shape != (*memory.shape[:2], 4):
            raise ValueError(f"{name}_geometry must have shape [B,N,4].")
        valid = mask.bool()
        if bool((example_ids[valid] < 0).any().item()):
            raise ValueError(f"Valid {name} cells require non-negative example IDs.")
        if bool((example_ids[~valid] != -1).any().item()):
            raise ValueError(f"Padded {name} cells must use example ID -1.")
        valid_geometry = geometry[valid]
        if valid_geometry.numel():
            if not bool(torch.isfinite(valid_geometry).all().item()):
                raise ValueError(f"Valid {name} geometry must be finite.")
            if bool(
                ((valid_geometry < 0.0) | (valid_geometry > 1.0)).any().item()
            ):
                raise ValueError(f"Valid {name} geometry must lie in [0,1].")

    def forward(
        self,
        input_memory: torch.Tensor,
        input_mask: torch.Tensor,
        input_example_ids: torch.Tensor,
        input_geometry: torch.Tensor,
        output_memory: torch.Tensor,
        output_mask: torch.Tensor,
        output_example_ids: torch.Tensor,
        output_geometry: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = input_memory.shape[0] if input_memory.ndim >= 1 else -1
        self._validate_set(
            name="relational_demo_input",
            memory=input_memory,
            mask=input_mask,
            example_ids=input_example_ids,
            geometry=input_geometry,
            batch_size=batch_size,
        )
        self._validate_set(
            name="relational_demo_output",
            memory=output_memory,
            mask=output_mask,
            example_ids=output_example_ids,
            geometry=output_geometry,
            batch_size=batch_size,
        )
        input_mask = input_mask.bool()
        output_mask = output_mask.bool()
        for batch_index in range(batch_size):
            input_examples = set(
                input_example_ids[batch_index, input_mask[batch_index]]
                .detach()
                .cpu()
                .tolist()
            )
            output_examples = set(
                output_example_ids[batch_index, output_mask[batch_index]]
                .detach()
                .cpu()
                .tolist()
            )
            if input_examples != output_examples:
                raise ValueError(
                    "Relational memory requires one non-empty input and output "
                    "set for every example; "
                    f"batch={batch_index}, input_examples={sorted(input_examples)}, "
                    f"output_examples={sorted(output_examples)}."
                )

        # One physical padded output slot is allowed for a fully empty C
        # condition.  It must stay exactly zero and must not reach MHA.
        relation_memory = torch.zeros_like(output_memory)
        if not bool(output_mask.any().item()):
            return relation_memory, output_mask

        safe_input_memory = torch.where(
            input_mask.unsqueeze(-1), input_memory, torch.zeros_like(input_memory)
        )
        safe_output_memory = torch.where(
            output_mask.unsqueeze(-1), output_memory, torch.zeros_like(output_memory)
        )
        safe_input_geometry = torch.where(
            input_mask.unsqueeze(-1),
            input_geometry,
            torch.zeros_like(input_geometry),
        ).to(dtype=input_memory.dtype)
        safe_output_geometry = torch.where(
            output_mask.unsqueeze(-1),
            output_geometry,
            torch.zeros_like(output_geometry),
        ).to(dtype=output_memory.dtype)
        input_role = self.role_embedding.weight[0].to(dtype=input_memory.dtype)
        output_role = self.role_embedding.weight[1].to(dtype=output_memory.dtype)
        input_tokens = (
            self.input_content_norm(safe_input_memory)
            + self.geometry_projection(safe_input_geometry).to(input_memory.dtype)
            + input_role
        )
        output_tokens = (
            self.output_content_norm(safe_output_memory)
            + self.geometry_projection(safe_output_geometry).to(output_memory.dtype)
            + output_role
        )

        # ARC has only a handful of demonstrations.  A short per-example loop
        # avoids materializing a large [B*D,N,H] copy while enforcing strict
        # example isolation and allowing independent input/output lengths.
        for batch_index in range(batch_size):
            examples = torch.unique(
                output_example_ids[batch_index, output_mask[batch_index]],
                sorted=True,
            )
            for example_id in examples:
                selected_input = input_mask[batch_index] & input_example_ids[
                    batch_index
                ].eq(example_id)
                selected_output = output_mask[batch_index] & output_example_ids[
                    batch_index
                ].eq(example_id)
                input_set = input_tokens[batch_index, selected_input].unsqueeze(0)
                output_set = output_tokens[batch_index, selected_output].unsqueeze(0)
                attended, _ = self.input_set_attention(
                    self.output_query_norm(output_set),
                    self.input_memory_norm(input_set),
                    self.input_memory_norm(input_set),
                    need_weights=False,
                )
                joint = output_set + attended
                joint = joint + self.relation_mlp(self.relation_mlp_norm(joint))
                encoded = self.output_norm(joint).squeeze(0)
                relation_memory[batch_index, selected_output] = encoded.to(
                    relation_memory.dtype
                )
        relation_memory = relation_memory * output_mask.unsqueeze(-1).to(
            relation_memory.dtype
        )
        return relation_memory, output_mask


class _DenseDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        conditioning_mode: str,
        same_coordinate_gate_initial_value: float,
        paired_gate_initial_value: float,
        relational_memory_mode: str,
        relational_gate_initial_value: float,
    ) -> None:
        super().__init__()
        if conditioning_mode not in DENSE_SUPPORTED_CONDITIONING_MODES:
            raise ValueError(
                "conditioning_mode must be one of "
                f"{DENSE_SUPPORTED_CONDITIONING_MODES}."
            )
        self.conditioning_mode = str(conditioning_mode)
        if relational_memory_mode not in DENSE_RELATIONAL_MEMORY_MODES:
            raise ValueError(
                "relational_memory_mode must be one of "
                f"{DENSE_RELATIONAL_MEMORY_MODES}."
            )
        self.relational_memory_mode = str(relational_memory_mode)
        self.relational_memory_enabled = (
            self.relational_memory_mode
            == DENSE_RELATIONAL_MEMORY_SET_CROSS_ATTENTION_V2
        )
        self.self_norm = RMSNorm(hidden_size)
        self.self_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        if self.conditioning_mode == DENSE_CONDITIONING_LEGACY:
            # Preserve exact names and shapes for explicit legacy checkpoint
            # reconstruction.  New checkpoints never silently select this path.
            self.cross_norm = RMSNorm(hidden_size)
            self.cross_attention = nn.MultiheadAttention(
                hidden_size, num_heads, dropout=dropout, batch_first=True
            )
        elif self.conditioning_mode != DENSE_CONDITIONING_FORCED_PAIRED:
            self.context_query_norm = RMSNorm(hidden_size)
            self.context_memory_norm = RMSNorm(hidden_size)
            self.context_cross_attention = nn.MultiheadAttention(
                hidden_size, num_heads, dropout=dropout, batch_first=True
            )
            self.context_output_norm = RMSNorm(hidden_size)
            self.context_branch_scale = nn.Parameter(
                torch.full(
                    (hidden_size,), DENSE_CONTEXT_BRANCH_SCALE_INITIAL_VALUE
                )
            )
        if self.conditioning_mode != DENSE_CONDITIONING_LEGACY:
            self.query_memory_norm = RMSNorm(hidden_size)
            if self.conditioning_mode == DENSE_CONDITIONING_SPLIT:
                self.query_query_norm = RMSNorm(hidden_size)
                self.query_cross_attention = nn.MultiheadAttention(
                    hidden_size, num_heads, dropout=dropout, batch_first=True
                )
            else:
                # The ARC encoder already exposes exact query-cell row/column
                # coordinates.  This bridge preserves that correspondence
                # instead of asking a global attention softmax to rediscover it.
                self.same_coordinate_projection = nn.Linear(
                    hidden_size, hidden_size, bias=False
                )
            self.query_output_norm = RMSNorm(hidden_size)
            if self.conditioning_mode in DENSE_PAIRED_CONDITIONING_MODES:
                if not self.relational_memory_enabled:
                    self.paired_output_norm = RMSNorm(hidden_size)
                if (
                    self.conditioning_mode == DENSE_CONDITIONING_GATED_PAIRED
                    and not self.relational_memory_enabled
                ):
                    # A scalar gate is predicted independently for every output
                    # cell.  It can decide whether the current task/position
                    # should trust the associative read, but cannot rewrite the
                    # value representation channel by channel.
                    self.paired_gate_hidden_norm = RMSNorm(hidden_size)
                    self.paired_gate_value_norm = RMSNorm(hidden_size)
                    self.paired_gate_projection = nn.Linear(
                        hidden_size * 2, 1, bias=True
                    )
                    self.paired_gate_initial_value = float(
                        paired_gate_initial_value
                    )
            if self.relational_memory_enabled:
                self.relational_query_norm = RMSNorm(hidden_size)
                self.relational_memory_norm = RMSNorm(hidden_size)
                self.relational_cross_attention = nn.MultiheadAttention(
                    hidden_size,
                    num_heads,
                    dropout=dropout,
                    batch_first=True,
                    bias=False,
                )
                self.relational_output_norm = RMSNorm(hidden_size)
                self.relational_gate_hidden_norm = RMSNorm(hidden_size)
                self.relational_gate_value_norm = RMSNorm(hidden_size)
                self.relational_gate_projection = nn.Linear(
                    hidden_size * 2, 1, bias=True
                )
                self.relational_gate_initial_value = float(
                    relational_gate_initial_value
                )
            # Query content is essential, unlike optional recursive candidates;
            # it must influence the first forward and all branch weights from
            # the first backward, so this scale is deliberately non-zero.
            initial_query_scale = (
                same_coordinate_gate_initial_value
                if self.conditioning_mode in (
                    DENSE_CONDITIONING_SAME_COORDINATE,
                    *DENSE_PAIRED_CONDITIONING_MODES,
                )
                else DENSE_QUERY_BRANCH_SCALE_INITIAL_VALUE
            )
            self.query_branch_scale = nn.Parameter(
                torch.full((hidden_size,), initial_query_scale)
            )
        self.mlp_norm = RMSNorm(hidden_size)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )
        self.apply(initialize_vit5_module)
        self.reset_paired_gate_initialization()
        self.reset_relational_gate_initialization()

    def reset_paired_gate_initialization(self) -> None:
        """Restore the contractual initial effective Gated scale."""
        if (
            self.conditioning_mode != DENSE_CONDITIONING_GATED_PAIRED
            or self.relational_memory_enabled
        ):
            return
        # scale = 2*sigmoid(bias), so bias=log(scale/(2-scale)).
        scale = float(self.paired_gate_initial_value)
        bias = math.log(scale / (2.0 - scale))
        nn.init.zeros_(self.paired_gate_projection.weight)
        nn.init.constant_(self.paired_gate_projection.bias, bias)

    def reset_relational_gate_initialization(self) -> None:
        """Restore the contractual initial Relational v2 effective scale."""
        if not self.relational_memory_enabled:
            return
        scale = float(self.relational_gate_initial_value)
        bias = math.log(scale / (2.0 - scale))
        nn.init.zeros_(self.relational_gate_projection.weight)
        nn.init.constant_(self.relational_gate_projection.bias, bias)

    def forward(
        self,
        queries: torch.Tensor,
        encoder_memory: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        query_cell_memory: torch.Tensor | None = None,
        query_cell_mask: torch.Tensor | None = None,
        query_cell_flat_indices: torch.Tensor | None = None,
        paired_query_read: torch.Tensor | None = None,
        relational_memory: torch.Tensor | None = None,
        relational_memory_mask: torch.Tensor | None = None,
        defer_query_injection: bool = False,
    ) -> torch.Tensor:
        normalized = self.self_norm(queries)
        attended, _ = self.self_attention(
            normalized, normalized, normalized, need_weights=False
        )
        queries = queries + attended
        if self.conditioning_mode == DENSE_CONDITIONING_LEGACY:
            normalized = self.cross_norm(queries)
            attended, _ = self.cross_attention(
                normalized,
                encoder_memory,
                encoder_memory,
                key_padding_mask=~encoder_attention_mask.bool(),
                need_weights=False,
            )
            queries = queries + attended
            return queries + self.mlp(self.mlp_norm(queries))

        if query_cell_memory is None or query_cell_mask is None:
            raise ValueError(
                f"{self.conditioning_mode} requires query_cell_memory and "
                "query_cell_mask."
            )
        if self.conditioning_mode != DENSE_CONDITIONING_FORCED_PAIRED:
            normalized_context = self.context_memory_norm(encoder_memory)
            context_attended, _ = self.context_cross_attention(
                self.context_query_norm(queries),
                normalized_context,
                normalized_context,
                key_padding_mask=~encoder_attention_mask.bool(),
                need_weights=False,
            )
            queries = queries + self.context_branch_scale * self.context_output_norm(
                context_attended
            )
        elif paired_query_read is None:
            raise ValueError(
                "forced_paired_associative_v1 requires paired_query_read."
            )
        queries = queries + self.mlp(self.mlp_norm(queries))

        if defer_query_injection:
            return queries

        output = self.inject_query_cells(
            queries,
            query_cell_memory,
            query_cell_mask,
            query_cell_flat_indices,
            paired_query_read=paired_query_read,
        )
        return self.inject_relational_memory(
            output,
            relational_memory,
            relational_memory_mask,
        )

    def inject_query_cells(
        self,
        queries: torch.Tensor,
        query_cell_memory: torch.Tensor | None,
        query_cell_mask: torch.Tensor | None,
        query_cell_flat_indices: torch.Tensor | None = None,
        paired_query_read: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the dedicated query-cell branch as the final layer residual."""
        if self.conditioning_mode == DENSE_CONDITIONING_LEGACY:
            return queries
        if query_cell_memory is None or query_cell_mask is None:
            raise ValueError(
                f"{self.conditioning_mode} requires query_cell_memory and "
                "query_cell_mask."
            )

        # Persistent query injection is deliberately last in every layer.  The
        # next layer may mix it, but that layer injects the source cells again;
        # the final color head therefore always receives a fresh query-specific
        # residual instead of only a repeatedly diluted early signal.
        normalized_query_memory = self.query_memory_norm(query_cell_memory)
        if self.conditioning_mode == DENSE_CONDITIONING_SPLIT:
            query_attended, _ = self.query_cross_attention(
                self.query_query_norm(queries),
                normalized_query_memory,
                normalized_query_memory,
                key_padding_mask=~query_cell_mask.bool(),
                need_weights=False,
            )
            query_residual = query_attended
        else:
            if query_cell_flat_indices is None:
                raise ValueError(
                    "same_coordinate_bridge_v1 requires query_cell_flat_indices."
                )
            if query_cell_flat_indices.shape != query_cell_mask.shape:
                raise ValueError(
                    "query_cell_flat_indices must have shape [B,Q]."
                )
            valid = query_cell_mask.bool()
            valid_indices = query_cell_flat_indices[valid]
            if bool(((valid_indices < 0) | (valid_indices >= queries.shape[1])).any()):
                raise ValueError("Valid query-cell flat indices are out of range.")
            if bool((query_cell_flat_indices[~valid] != -1).any()):
                raise ValueError("Padded query cells must use flat index -1.")
            projected = self.same_coordinate_projection(normalized_query_memory)
            query_residual = torch.zeros_like(queries)
            safe_indices = query_cell_flat_indices.clamp_min(0).unsqueeze(-1).expand(
                -1, -1, queries.shape[-1]
            )
            # ``scatter_add_`` is not autocast-aware and requires an exact dtype
            # match between its destination and source.  Under CUDA BF16
            # autocast the projection is commonly BF16 while the residual stream
            # can remain FP32 (for example after RMSNorm's FP32 accumulation).
            # Align at the residual boundary explicitly; this cast remains in the
            # autograd graph and therefore preserves gradients to the projection
            # and query-cell memory.
            scatter_source = (
                projected * valid.unsqueeze(-1).to(projected.dtype)
            ).to(dtype=query_residual.dtype)
            query_residual.scatter_add_(
                1,
                safe_indices,
                scatter_source,
            )
        output = queries + self.query_branch_scale * self.query_output_norm(
            query_residual
        )
        if (
            self.conditioning_mode in DENSE_PAIRED_CONDITIONING_MODES
            and not self.relational_memory_enabled
        ):
            if paired_query_read is None:
                raise ValueError(
                    f"{self.conditioning_mode} requires paired_query_read."
                )
            if paired_query_read.shape != query_cell_memory.shape:
                raise ValueError(
                    "paired_query_read must match query_cell_memory [B,Q,H]."
                )
            pair_residual = torch.zeros_like(queries)
            pair_source = (
                paired_query_read
                * query_cell_mask.bool().unsqueeze(-1).to(paired_query_read.dtype)
            ).to(pair_residual.dtype)
            pair_residual.scatter_add_(1, safe_indices, pair_source)
            normalized_pair = self.paired_output_norm(pair_residual)
            if self.conditioning_mode == DENSE_CONDITIONING_FORCED_PAIRED:
                # Forced has no learned gate: the associative read is injected
                # at full scale after every decoder layer.
                output = output + normalized_pair
            else:
                gate_input = torch.cat(
                    (
                        self.paired_gate_hidden_norm(output),
                        self.paired_gate_value_norm(pair_residual),
                    ),
                    dim=-1,
                )
                gate_scale = 2.0 * torch.sigmoid(
                    self.paired_gate_projection(gate_input)
                )
                output = output + gate_scale.to(normalized_pair.dtype) * normalized_pair
        return output

    def inject_relational_memory(
        self,
        queries: torch.Tensor,
        relational_memory: torch.Tensor | None,
        relational_memory_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Let every output position read relation tokens through a gated path."""
        if not self.relational_memory_enabled:
            return queries
        if relational_memory is None or relational_memory_mask is None:
            raise ValueError(
                "set_cross_attention_v2 requires relational memory and mask."
            )
        if relational_memory.ndim != 3:
            raise ValueError("relational_memory must have shape [B,R,H].")
        if relational_memory.shape[0] != queries.shape[0] or (
            relational_memory.shape[-1] != queries.shape[-1]
        ):
            raise ValueError(
                "relational_memory must match the output batch and hidden size."
            )
        if relational_memory_mask.shape != relational_memory.shape[:2]:
            raise ValueError("relational_memory_mask must have shape [B,R].")

        relation_delta = torch.zeros_like(queries)
        valid_rows = relational_memory_mask.bool().any(dim=1)
        if bool(valid_rows.any().item()):
            attended, _ = self.relational_cross_attention(
                self.relational_query_norm(queries[valid_rows]),
                self.relational_memory_norm(relational_memory[valid_rows]),
                self.relational_memory_norm(relational_memory[valid_rows]),
                key_padding_mask=~relational_memory_mask[valid_rows].bool(),
                need_weights=False,
            )
            relation_delta[valid_rows] = attended.to(relation_delta.dtype)
        normalized_relation = self.relational_output_norm(relation_delta)
        gate_input = torch.cat(
            (
                self.relational_gate_hidden_norm(queries),
                self.relational_gate_value_norm(relation_delta),
            ),
            dim=-1,
        )
        gate_scale = 2.0 * torch.sigmoid(
            self.relational_gate_projection(gate_input)
        )
        return queries + gate_scale.to(normalized_relation.dtype) * normalized_relation


class _CandidateCrossAttentionResidual(nn.Module):
    """Candidate-only cross-attention behind a zero-initialized residual gate.

    Candidate keys never share a softmax with the task-memory keys.  At
    initialization ``tanh(gate) == 0``, so adding this module is an exact
    no-op and every recursive round starts from the proposal decoder's output.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.cross_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate = nn.Parameter(
            torch.tensor(DENSE_CANDIDATE_GATE_INITIAL_VALUE, dtype=torch.float32)
        )
        self.apply(initialize_vit5_module)

    def forward(
        self,
        queries: torch.Tensor,
        candidate_memory: torch.Tensor,
        candidate_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if candidate_memory.ndim != 3:
            raise ValueError("candidate_memory must have shape [B,S,H].")
        if candidate_attention_mask.shape != candidate_memory.shape[:2]:
            raise ValueError(
                "candidate_attention_mask must match candidate_memory [B,S]."
            )
        normalized = self.norm(queries)
        candidate_delta, _ = self.cross_attention(
            normalized,
            candidate_memory,
            candidate_memory,
            key_padding_mask=~candidate_attention_mask.bool(),
            need_weights=False,
        )
        return queries + torch.tanh(self.gate).to(queries.dtype) * candidate_delta


def _validate_dense_targets(
    target_ids: torch.Tensor,
    target_mask: torch.Tensor,
    target_height: torch.Tensor,
    target_width: torch.Tensor,
    max_size: int | tuple[int, int],
) -> None:
    if isinstance(max_size, tuple):
        max_height, max_width = (int(value) for value in max_size)
    else:
        max_height = max_width = int(max_size)
    if target_ids.shape != target_mask.shape or target_ids.shape[1:] != (
        max_height,
        max_width,
    ):
        raise ValueError(
            "Dense targets must both have "
            f"[B,{max_height},{max_width}] shape."
        )
    if target_height.shape != (target_ids.shape[0],) or target_width.shape != (
        target_ids.shape[0],
    ):
        raise ValueError("target_height and target_width must have shape [B].")
    if torch.any((target_ids[target_mask] < 0) | (target_ids[target_mask] > 9)).item():
        raise ValueError("Valid dense target colors must be in [0,9].")
    if torch.any((target_height < 1) | (target_height > max_height)).item():
        raise ValueError(f"target_height must be in [1,{max_height}].")
    if torch.any((target_width < 1) | (target_width > max_width)).item():
        raise ValueError(f"target_width must be in [1,{max_width}].")
    valid_counts = target_mask.flatten(1).sum(dim=1)
    if torch.any(valid_counts == 0).item():
        raise ValueError("Every dense target sample must contain at least one valid cell.")
    expected_counts = target_height.long() * target_width.long()
    if not torch.equal(valid_counts.long(), expected_counts):
        raise ValueError(
            "dense_target_mask cell counts must equal target_height * target_width."
        )


def build_bd_border_targets(
    target_ids: torch.Tensor,
    target_mask: torch.Tensor,
    target_height: torch.Tensor,
    target_width: torch.Tensor,
    *,
    max_size: int | tuple[int, int] = 30,
    bd_class_id: int = DENSE_BD_CLASS_ID,
) -> dict[str, torch.Tensor]:
    """Build the supervised 31x31 color-plus-BD canvas.

    Real colors occupy ``[0:H, 0:W]``.  BD is supervised on the right/down
    L-shaped boundary: ``[0:H, W]`` plus ``[H, 0:W+1]``.  Every remaining
    canvas position is deliberately ignored rather than treated as a
    background class.
    """
    _validate_dense_targets(
        target_ids,
        target_mask,
        target_height,
        target_width,
        max_size,
    )
    if isinstance(max_size, tuple):
        max_height, max_width = (int(value) for value in max_size)
    else:
        max_height = max_width = int(max_size)
    canvas_height = max_height + 1
    canvas_width = max_width + 1
    batch_size = target_ids.shape[0]
    bd_target_ids = torch.zeros(
        batch_size,
        canvas_height,
        canvas_width,
        dtype=torch.long,
        device=target_ids.device,
    )
    bd_target_ids[:, :max_height, :max_width] = target_ids.long()
    rows = torch.arange(canvas_height, device=target_ids.device).view(
        1, canvas_height, 1
    )
    cols = torch.arange(canvas_width, device=target_ids.device).view(
        1, 1, canvas_width
    )
    heights = target_height.long().view(-1, 1, 1)
    widths = target_width.long().view(-1, 1, 1)
    color_mask = (rows < heights) & (cols < widths)
    right_border = (rows < heights) & (cols == widths)
    bottom_border = (rows == heights) & (cols <= widths)
    border_mask = right_border | bottom_border
    bd_target_ids[border_mask] = int(bd_class_id)
    supervised_mask = color_mask | border_mask
    if not torch.equal(
        color_mask[:, :max_height, :max_width], target_mask.bool()
    ):
        raise AssertionError("BD color mask must reproduce dense_target_mask exactly.")
    if bool((color_mask & border_mask).any().item()):
        raise AssertionError("BD color and border supervision must be disjoint.")
    expected_border_counts = target_height.long() + target_width.long() + 1
    if not torch.equal(border_mask.flatten(1).sum(dim=1), expected_border_counts):
        raise AssertionError("BD L-border must contain H + W + 1 cells.")
    return {
        "target_ids": bd_target_ids,
        "supervised_mask": supervised_mask,
        "color_mask": color_mask,
        "border_mask": border_mask,
    }


def bd_joint_shape_logits(
    logits: torch.Tensor,
    *,
    max_size: int = 30,
    bd_class_id: int = DENSE_BD_CLASS_ID,
) -> torch.Tensor:
    """Score legal rectangles from both non-BD interior and BD L-border evidence."""
    expected = (max_size + 1, max_size + 1, bd_class_id + 1)
    if logits.ndim != 4 or logits.shape[1:] != expected:
        raise ValueError(
            "BD logits must have shape "
            f"[B,{max_size + 1},{max_size + 1},{bd_class_id + 1}]."
        )
    canvas_size = max_size + 1
    device = logits.device
    heights = torch.arange(1, max_size + 1, device=device).view(
        max_size, 1, 1, 1
    )
    widths = torch.arange(1, max_size + 1, device=device).view(
        1, max_size, 1, 1
    )
    rows = torch.arange(canvas_size, device=device).view(1, 1, canvas_size, 1)
    cols = torch.arange(canvas_size, device=device).view(1, 1, 1, canvas_size)
    border_mask = ((rows < heights) & (cols == widths)) | (
        (rows == heights) & (cols <= widths)
    )
    interior_mask = (rows < heights) & (cols < widths)
    # Shape scoring remains FP32 under mixed precision.  The two regions are
    # normalized independently so a large interior cannot swamp its boundary
    # (or vice versa).
    log_probabilities = logits.float().log_softmax(dim=-1)
    border_weights = border_mask.to(dtype=log_probabilities.dtype)
    interior_weights = interior_mask.to(dtype=log_probabilities.dtype)
    border_counts = border_weights.sum(dim=(-1, -2)).clamp_min(1)
    interior_counts = interior_weights.sum(dim=(-1, -2)).clamp_min(1)
    bd_log_probability = log_probabilities[..., bd_class_id]
    non_bd_log_probability = torch.logsumexp(
        log_probabilities[..., :bd_class_id], dim=-1
    )
    border_score = torch.einsum(
        "brc,hwrc->bhw", bd_log_probability, border_weights
    ) / border_counts.unsqueeze(0)
    interior_score = torch.einsum(
        "brc,hwrc->bhw", non_bd_log_probability, interior_weights
    ) / interior_counts.unsqueeze(0)
    return interior_score + border_score


def _joint_shape_predictions(joint_shape_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if joint_shape_logits.ndim != 3 or joint_shape_logits.shape[1:] != (30, 30):
        raise ValueError("joint_shape_logits must have shape [B,30,30].")
    flat = joint_shape_logits.flatten(1).argmax(dim=-1)
    return flat.div(30, rounding_mode="floor") + 1, flat.remainder(30) + 1


def validate_target_component_ids(
    component_ids: torch.Tensor,
    target_mask: torch.Tensor,
) -> None:
    if component_ids.shape != target_mask.shape:
        raise ValueError("dense_target_component_ids must match dense_target_mask shape.")
    if component_ids.dtype != torch.long:
        raise ValueError("dense_target_component_ids must use torch.long dtype.")
    if bool((component_ids[target_mask] <= 0).any().item()):
        raise ValueError("Every valid target cell must have component_id > 0.")
    if bool((component_ids[~target_mask] != 0).any().item()):
        raise ValueError("Every padded target cell must have component_id == 0.")


def component_balanced_mean_ce(
    per_cell_ce: torch.Tensor,
    target_mask: torch.Tensor,
    component_ids: torch.Tensor,
) -> torch.Tensor:
    """Mean over samples, then components, then cells in each component."""
    if per_cell_ce.shape != target_mask.shape:
        raise ValueError("per_cell_ce must match target_mask shape.")
    validate_target_component_ids(component_ids, target_mask)
    sample_losses: list[torch.Tensor] = []
    for batch_index in range(per_cell_ce.shape[0]):
        valid_ids = component_ids[batch_index][target_mask[batch_index]]
        component_losses: list[torch.Tensor] = []
        for component_id in torch.unique(valid_ids, sorted=True):
            component_mask = target_mask[batch_index] & component_ids[batch_index].eq(
                component_id
            )
            if not bool(component_mask.any().item()):
                raise AssertionError("A target component ID resolved to an empty mask.")
            component_losses.append(per_cell_ce[batch_index][component_mask].mean())
        if not component_losses:
            raise ValueError("Every target sample must contain at least one component.")
        sample_losses.append(torch.stack(component_losses).mean())
    return torch.stack(sample_losses).mean()


def dense_decoder_metrics(
    color_logits: torch.Tensor,
    height_logits: torch.Tensor,
    width_logits: torch.Tensor,
    target_ids: torch.Tensor,
    target_mask: torch.Tensor,
    target_height: torch.Tensor,
    target_width: torch.Tensor,
    joint_shape_logits: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if (
        joint_shape_logits is None
        and color_logits.ndim == 4
        and color_logits.shape[1:] == (31, 31, 11)
    ):
        joint_shape_logits = bd_joint_shape_logits(color_logits)
    if joint_shape_logits is None:
        predicted_height = height_logits.argmax(dim=-1) + 1
        predicted_width = width_logits.argmax(dim=-1) + 1
        predicted_colors = color_logits.argmax(dim=-1)
        scored_color_logits = color_logits
    else:
        predicted_height, predicted_width = _joint_shape_predictions(
            joint_shape_logits
        )
        predicted_colors = color_logits[:, :30, :30, :10].argmax(dim=-1)
        scored_color_logits = color_logits[:, :30, :30, :]
    valid_correct = (predicted_colors == target_ids) & target_mask
    per_sample_pixel = valid_correct.flatten(1).sum(dim=1).float() / target_mask.flatten(1).sum(
        dim=1
    ).float()
    height_correct = predicted_height.eq(target_height)
    width_correct = predicted_width.eq(target_width)
    exact_colors = valid_correct.flatten(1).sum(dim=1).eq(target_mask.flatten(1).sum(dim=1))
    exact = height_correct & width_correct & exact_colors
    cell_probabilities = scored_color_logits.float().softmax(dim=-1)
    if joint_shape_logits is None:
        cell_confidence = cell_probabilities.amax(dim=-1)
    else:
        selected_colors = predicted_colors.unsqueeze(-1)
        cell_confidence = cell_probabilities.gather(-1, selected_colors).squeeze(-1)
    mean_cell_confidence = (
        (cell_confidence * target_mask).flatten(1).sum(dim=1)
        / target_mask.flatten(1).sum(dim=1)
    )
    return {
        "height_accuracy": height_correct.float().mean(),
        "width_accuracy": width_correct.float().mean(),
        "both_shape_accuracy": (height_correct & width_correct).float().mean(),
        "valid_cell_pixel_accuracy": per_sample_pixel.mean(),
        "grid_exact_match": exact.float().mean(),
        "predicted_height": predicted_height,
        "predicted_width": predicted_width,
        "target_height": target_height,
        "target_width": target_width,
        "mean_cell_confidence": mean_cell_confidence,
    }


def dense_recursive_metrics(
    round_outputs: Sequence[DenseDecoderOutput],
    target_ids: torch.Tensor,
    target_mask: torch.Tensor,
    target_height: torch.Tensor,
    target_width: torch.Tensor,
) -> dict[str, Any]:
    """Measure per-round progress and correction/regression transitions."""
    if not round_outputs:
        raise ValueError("Recursive metrics require at least one round output.")
    if target_ids.shape != target_mask.shape:
        raise ValueError("Recursive metric target IDs and mask must match.")
    round_oracle_pixel: list[torch.Tensor] = []
    round_normal_pixel: list[torch.Tensor] = []
    round_exact: list[torch.Tensor] = []
    round_height: list[torch.Tensor] = []
    round_width: list[torch.Tensor] = []
    round_wrong: list[torch.Tensor] = []
    round_missing: list[torch.Tensor] = []
    round_extra: list[torch.Tensor] = []
    correct_masks: list[torch.Tensor] = []
    for output in round_outputs:
        if output.metrics is None:
            raise ValueError("Every supervised recursive round requires metrics.")
        round_oracle_pixel.append(output.metrics["valid_cell_pixel_accuracy"])
        round_exact.append(output.metrics["grid_exact_match"])
        round_height.append(output.metrics["height_accuracy"])
        round_width.append(output.metrics["width_accuracy"])
        if output.joint_shape_logits is None:
            predicted_colors = output.color_logits.argmax(dim=-1)
            predicted_height = output.height_logits.argmax(dim=-1) + 1
            predicted_width = output.width_logits.argmax(dim=-1) + 1
        else:
            predicted_colors = output.color_logits[:, :30, :30, :10].argmax(dim=-1)
            predicted_height, predicted_width = _joint_shape_predictions(
                output.joint_shape_logits
            )
        rows = torch.arange(target_ids.shape[1], device=target_ids.device).view(
            1, -1, 1
        )
        cols = torch.arange(target_ids.shape[2], device=target_ids.device).view(
            1, 1, -1
        )
        predicted_mask = (rows < predicted_height[:, None, None]) & (
            cols < predicted_width[:, None, None]
        )
        overlap = target_mask & predicted_mask
        correct = overlap & predicted_colors.eq(target_ids)
        wrong = overlap & ~predicted_colors.eq(target_ids)
        missing = target_mask & ~predicted_mask
        extra = predicted_mask & ~target_mask
        union = target_mask | predicted_mask
        per_sample_normal_pixel = correct.flatten(1).sum(dim=1).float() / union.flatten(
            1
        ).sum(dim=1).clamp_min(1).float()
        round_normal_pixel.append(per_sample_normal_pixel.mean())
        round_wrong.append(wrong.sum())
        round_missing.append(missing.sum())
        round_extra.append(extra.sum())
        # A target cell outside the predicted shape is incorrect for recursive
        # correction/regression accounting; this is not oracle-shape scoring.
        correct_masks.append(correct)
    transitions: list[dict[str, torch.Tensor]] = []
    for previous, current in zip(correct_masks, correct_masks[1:]):
        previous_correct = previous[target_mask]
        current_correct = current[target_mask]
        corrected = (~previous_correct & current_correct).sum()
        regressed = (previous_correct & ~current_correct).sum()
        previous_wrong = (~previous_correct).sum()
        previous_right = previous_correct.sum()
        transitions.append(
            {
                "incorrect_to_correct_cells": corrected,
                "correct_to_wrong_cells": regressed,
                "incorrect_to_correct_rate": corrected.float()
                / previous_wrong.clamp_min(1).float(),
                "correct_to_wrong_regression_rate": regressed.float()
                / previous_right.clamp_min(1).float(),
                "net_corrected_cells": corrected - regressed,
            }
        )
    return {
        "round_normal_shape_pixel_accuracy": torch.stack(round_normal_pixel),
        "round_oracle_shape_pixel_accuracy": torch.stack(round_oracle_pixel),
        "round_grid_exact_match": torch.stack(round_exact),
        "round_height_accuracy": torch.stack(round_height),
        "round_width_accuracy": torch.stack(round_width),
        "round_wrong_cells": torch.stack(round_wrong),
        "round_missing_cells": torch.stack(round_missing),
        "round_extra_cells": torch.stack(round_extra),
        "transitions": transitions,
    }


def decode_dense_output(
    color_logits: torch.Tensor,
    height_logits: torch.Tensor,
    width_logits: torch.Tensor,
    include_raw_logits: bool = False,
    joint_shape_logits: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    """Decode a batch using only dense head predictions."""
    separate_hw = color_logits.ndim == 4 and color_logits.shape[1:] == (30, 30, 10)
    bd_border = color_logits.ndim == 4 and color_logits.shape[1:] == (31, 31, 11)
    if not (separate_hw or bd_border):
        raise ValueError(
            "color_logits must have shape [B,30,30,10] or [B,31,31,11]."
        )
    batch_size = color_logits.shape[0]
    if height_logits.shape != (batch_size, 30) or width_logits.shape != (batch_size, 30):
        raise ValueError("height_logits and width_logits must have shape [B,30].")
    if bd_border:
        if joint_shape_logits is None:
            joint_shape_logits = bd_joint_shape_logits(color_logits)
        if joint_shape_logits.shape != (batch_size, 30, 30):
            raise ValueError("joint_shape_logits must have shape [B,30,30].")
        joint_probabilities = joint_shape_logits.float().flatten(1).softmax(dim=-1)
        joint_probabilities = joint_probabilities.reshape(batch_size, 30, 30)
        heights, widths = _joint_shape_predictions(joint_shape_logits)
        height_probabilities = joint_probabilities.sum(dim=2)
        width_probabilities = joint_probabilities.sum(dim=1)
        full_probabilities = color_logits.float().softmax(dim=-1)
        colors = color_logits[..., :10].argmax(dim=-1)
        color_probabilities = full_probabilities[..., :10]
    else:
        if joint_shape_logits is not None:
            raise ValueError(
                "joint_shape_logits is only legal for [B,31,31,11] BD output."
            )
        joint_probabilities = None
        height_probabilities = height_logits.float().softmax(dim=-1)
        width_probabilities = width_logits.float().softmax(dim=-1)
        color_probabilities = color_logits.float().softmax(dim=-1)
        heights = height_probabilities.argmax(dim=-1) + 1
        widths = width_probabilities.argmax(dim=-1) + 1
        colors = color_probabilities.argmax(dim=-1)
    decoded: list[dict[str, Any]] = []
    for index in range(batch_size):
        height = int(heights[index].item())
        width = int(widths[index].item())
        if not 1 <= height <= 30 or not 1 <= width <= 30:
            raise ValueError(f"Predicted dense shape {(height, width)} is outside [1,30].")
        grid = colors[index, :height, :width].cpu().tolist()
        if not grid or not grid[0] or any(
            not 0 <= int(value) <= 9 for row in grid for value in row
        ):
            raise ValueError("Dense decoder produced an invalid ARC grid.")
        cell_confidence = color_probabilities[index, :height, :width].amax(dim=-1)
        item: dict[str, Any] = {
            "grid": grid,
            "predicted_height": height,
            "predicted_width": width,
            "height_confidence": float(height_probabilities[index, height - 1].item()),
            "width_confidence": float(width_probabilities[index, width - 1].item()),
            "mean_cell_confidence": float(cell_confidence.mean().item()),
        }
        if joint_probabilities is not None:
            item["shape_decoder_mode"] = DENSE_SHAPE_DECODER_BD_BORDER
            item["joint_shape_confidence"] = float(
                joint_probabilities[index, height - 1, width - 1].item()
            )
        if include_raw_logits:
            item["color_logits"] = color_logits[index].detach().cpu()
            item["height_logits"] = height_logits[index].detach().cpu()
            item["width_logits"] = width_logits[index].detach().cpu()
            if joint_shape_logits is not None:
                item["joint_shape_logits"] = joint_shape_logits[index].detach().cpu()
        decoded.append(item)
    return decoded


class DenseGridQueryDecoder(nn.Module):
    def __init__(
        self,
        hidden_size: int = 384,
        num_heads: int = 6,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_size: int = 30,
        num_colors: int = 10,
        lambda_shape: float = 0.2,
        lambda_component: float = 0.0,
        recursive_steps: int = 1,
        recursive_loss_weighting: str = "hsw",
        hsw_lambda: float = 0.7,
        candidate_encoder_layers: int = 2,
        candidate_mlp_ratio: float = 4.0,
        candidate_dropout: float = 0.0,
        conditioning_mode: str = DENSE_CONDITIONING_SPLIT,
        same_coordinate_gate_initial_value: float = (
            DENSE_SAME_COORDINATE_GATE_INITIAL_VALUE
        ),
        paired_gate_initial_value: float = DENSE_GATED_PAIRED_GATE_INITIAL_VALUE,
        shape_decoder_mode: str = DENSE_SHAPE_DECODER_SEPARATE_HW,
        relational_memory_mode: str = DENSE_RELATIONAL_MEMORY_DISABLED,
        relational_gate_initial_value: float = (
            DENSE_RELATIONAL_MEMORY_GATE_INITIAL_VALUE
        ),
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads.")
        if max_size != 30 or num_colors != 10:
            raise ValueError("Phase 2A dense decoder requires max_size=30 and num_colors=10.")
        if num_layers < 1:
            raise ValueError("Dense decoder requires at least one layer.")
        if lambda_shape < 0 or lambda_component < 0:
            raise ValueError("lambda_shape and lambda_component must be non-negative.")
        if not 1 <= recursive_steps <= 8:
            raise ValueError("Dense recursive_steps must be in [1,8].")
        if recursive_loss_weighting not in RECURSIVE_LOSS_WEIGHTINGS:
            raise ValueError(
                f"recursive_loss_weighting must be one of {RECURSIVE_LOSS_WEIGHTINGS}."
            )
        if not 0.0 < hsw_lambda <= 1.0:
            raise ValueError("hsw_lambda must be in (0,1].")
        if candidate_encoder_layers < 1:
            raise ValueError("candidate_encoder_layers must be >=1.")
        if candidate_mlp_ratio <= 0 or not 0.0 <= candidate_dropout < 1.0:
            raise ValueError("Invalid candidate encoder MLP ratio/dropout.")
        if conditioning_mode not in DENSE_SUPPORTED_CONDITIONING_MODES:
            raise ValueError(
                "conditioning_mode must be one of "
                f"{DENSE_SUPPORTED_CONDITIONING_MODES}."
            )
        if not torch.isfinite(torch.tensor(same_coordinate_gate_initial_value)):
            raise ValueError("same_coordinate_gate_initial_value must be finite.")
        if not torch.isfinite(torch.tensor(paired_gate_initial_value)):
            raise ValueError("paired_gate_initial_value must be finite.")
        if relational_memory_mode not in DENSE_RELATIONAL_MEMORY_MODES:
            raise ValueError(
                "relational_memory_mode must be one of "
                f"{DENSE_RELATIONAL_MEMORY_MODES}."
            )
        if not torch.isfinite(torch.tensor(relational_gate_initial_value)):
            raise ValueError("relational_gate_initial_value must be finite.")
        relational_memory_enabled = (
            relational_memory_mode
            == DENSE_RELATIONAL_MEMORY_SET_CROSS_ATTENTION_V2
        )
        if shape_decoder_mode not in DENSE_SHAPE_DECODER_MODES:
            raise ValueError(
                f"shape_decoder_mode must be one of {DENSE_SHAPE_DECODER_MODES}."
            )
        if (
            conditioning_mode == DENSE_CONDITIONING_GATED_PAIRED
            and not relational_memory_enabled
            and not 0.0 < paired_gate_initial_value < 2.0
        ):
            raise ValueError(
                "gated paired initial effective scale must be strictly in (0,2)."
            )
        if relational_memory_enabled and (
            conditioning_mode != DENSE_CONDITIONING_GATED_PAIRED
        ):
            raise ValueError(
                "set_cross_attention_v2 is only supported with "
                "gated_paired_associative_v1 conditioning."
            )
        if relational_memory_enabled and not 0.0 < relational_gate_initial_value < 2.0:
            raise ValueError(
                "relational initial effective scale must be strictly in (0,2)."
            )
        if (
            conditioning_mode in (
                DENSE_CONDITIONING_SAME_COORDINATE,
                *DENSE_PAIRED_CONDITIONING_MODES,
            )
            and same_coordinate_gate_initial_value == 0.0
        ):
            raise ValueError(
                "same-coordinate query bridge gate must initialize non-zero."
            )
        if (
            conditioning_mode in DENSE_PAIRED_CONDITIONING_MODES
            and recursive_steps != 1
        ):
            raise ValueError(
                "paired-associative v1 modes are single-round diagnostics; "
                "recursive_steps must equal 1."
            )
        if (
            shape_decoder_mode == DENSE_SHAPE_DECODER_BD_BORDER
            and recursive_steps != 1
        ):
            raise ValueError(
                "bd_border_v1 is single-round only; recursive_steps must equal 1."
            )
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.mlp_ratio = float(mlp_ratio)
        self.dropout = float(dropout)
        self.max_size = int(max_size)
        self.num_colors = int(num_colors)
        self.lambda_shape = float(lambda_shape)
        self.lambda_component = float(lambda_component)
        self.recursive_steps = int(recursive_steps)
        self.recursive_loss_weighting = str(recursive_loss_weighting)
        self.hsw_lambda = float(hsw_lambda)
        self.candidate_encoder_layers = int(candidate_encoder_layers)
        self.candidate_mlp_ratio = float(candidate_mlp_ratio)
        self.candidate_dropout = float(candidate_dropout)
        self.conditioning_mode = str(conditioning_mode)
        self.same_coordinate_gate_initial_value = float(
            same_coordinate_gate_initial_value
        )
        self.paired_gate_initial_value = float(paired_gate_initial_value)
        self.relational_memory_mode = str(relational_memory_mode)
        self.relational_memory_enabled = bool(relational_memory_enabled)
        self.relational_gate_initial_value = float(relational_gate_initial_value)
        self.shape_decoder_mode = str(shape_decoder_mode)
        self.output_canvas_size = (
            DENSE_BD_CANVAS_SIZE
            if self.shape_decoder_mode == DENSE_SHAPE_DECODER_BD_BORDER
            else self.max_size
        )
        self.output_class_count = (
            DENSE_BD_NUM_CLASSES
            if self.shape_decoder_mode == DENSE_SHAPE_DECODER_BD_BORDER
            else self.num_colors
        )
        base_query_count = (
            self.output_canvas_size * self.output_canvas_size
            if self.conditioning_mode == DENSE_CONDITIONING_LEGACY
            else 1
        )
        self.base_query = nn.Parameter(
            torch.zeros(1, base_query_count, hidden_size)
        )
        self.output_row_embedding = nn.Embedding(self.output_canvas_size, hidden_size)
        self.output_col_embedding = nn.Embedding(self.output_canvas_size, hidden_size)
        self.layers = nn.ModuleList(
            [
                _DenseDecoderLayer(
                    hidden_size,
                    num_heads,
                    mlp_ratio,
                    dropout,
                    conditioning_mode=self.conditioning_mode,
                    same_coordinate_gate_initial_value=(
                        self.same_coordinate_gate_initial_value
                    ),
                    paired_gate_initial_value=self.paired_gate_initial_value,
                    relational_memory_mode=self.relational_memory_mode,
                    relational_gate_initial_value=(
                        self.relational_gate_initial_value
                    ),
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = RMSNorm(hidden_size)
        self.color_head = nn.Linear(hidden_size, self.output_class_count)
        if self.shape_decoder_mode == DENSE_SHAPE_DECODER_SEPARATE_HW:
            self.shape_norm: RMSNorm | None = RMSNorm(hidden_size)
            self.height_head: nn.Linear | None = nn.Linear(hidden_size, max_size)
            self.width_head: nn.Linear | None = nn.Linear(hidden_size, max_size)
        else:
            # BD owns shape through its structured L-border score.  Do not leave
            # independent learned H/W heads that could silently bypass it.
            self.shape_norm = None
            self.height_head = None
            self.width_head = None
        self.paired_associative_read: _PairedAssociativeRead | None = None
        if (
            self.conditioning_mode in DENSE_PAIRED_CONDITIONING_MODES
            and not self.relational_memory_enabled
        ):
            self.paired_associative_read = _PairedAssociativeRead(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
            )
        self.relational_memory_encoder: _RelationalSetMemoryEncoder | None = None
        if self.relational_memory_enabled:
            self.relational_memory_encoder = _RelationalSetMemoryEncoder(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
            )
        # Keep recursive_steps=1 parameter-identical to the pre-recursion
        # decoder so legacy checkpoints remain exactly loadable.
        # Initialize the legacy single-round parameter set first.  Candidate
        # construction is RNG-isolated so changing recursive_steps cannot alter
        # the proposal decoder or later LoRA initialization under the same seed.
        self.apply(initialize_vit5_module)
        # The outer apply above revisits every layer, so the zero-weight gate
        # contract must be restored here rather than only in the layer ctor.
        for layer in self.layers:
            layer.reset_paired_gate_initialization()
            layer.reset_relational_gate_initialization()
        nn.init.trunc_normal_(self.base_query, std=0.02)
        self.candidate_encoder: DenseCandidateEncoder | None = None
        self.candidate_refinement_layers = nn.ModuleList()
        self.candidate_shape_norm: RMSNorm | None = None
        self.register_parameter("candidate_shape_gamma", None)
        if recursive_steps > 1:
            with torch.random.fork_rng(devices=[]):
                self.candidate_encoder = DenseCandidateEncoder(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    num_layers=candidate_encoder_layers,
                    max_rounds=DENSE_CANDIDATE_MAX_ROUNDS,
                    mlp_ratio=candidate_mlp_ratio,
                    dropout=candidate_dropout,
                    max_size=max_size,
                    num_colors=num_colors,
                )
                self.candidate_refinement_layers = nn.ModuleList(
                    [
                        _CandidateCrossAttentionResidual(
                            hidden_size=hidden_size,
                            num_heads=num_heads,
                            dropout=dropout,
                        )
                        for _ in range(num_layers)
                    ]
                )
                self.candidate_shape_norm = RMSNorm(hidden_size)
                self.candidate_shape_gamma = nn.Parameter(
                    torch.zeros(hidden_size)
                )

    def config_dict(self) -> dict[str, Any]:
        forced_paired = (
            self.conditioning_mode == DENSE_CONDITIONING_FORCED_PAIRED
        )
        gated_conditioning = (
            self.conditioning_mode == DENSE_CONDITIONING_GATED_PAIRED
        )
        gated_paired = gated_conditioning and not self.relational_memory_enabled
        relational_mode = gated_conditioning and self.relational_memory_enabled
        paired_mode = forced_paired or gated_paired
        config = {
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout,
            "max_size": self.max_size,
            "num_colors": self.num_colors,
            "lambda_shape": self.lambda_shape,
            "lambda_component": self.lambda_component,
            "recursive_steps": self.recursive_steps,
            "recursive_loss_weighting": self.recursive_loss_weighting,
            "hsw_lambda": self.hsw_lambda,
            "candidate_encoder_layers": self.candidate_encoder_layers,
            "candidate_mlp_ratio": self.candidate_mlp_ratio,
            "candidate_dropout": self.candidate_dropout,
            "conditioning_mode": self.conditioning_mode,
            "conditioning_schema_version": (
                DENSE_GATED_RELATIONAL_SCHEMA_VERSION
                if relational_mode
                else (
                    0
                    if self.conditioning_mode == DENSE_CONDITIONING_LEGACY
                    else (
                        (
                            DENSE_FORCED_PAIRED_SCHEMA_VERSION
                            if forced_paired
                            else DENSE_GATED_PAIRED_SCHEMA_VERSION
                        )
                        if paired_mode
                        else (
                            DENSE_SAME_COORDINATE_SCHEMA_VERSION
                            if self.conditioning_mode
                            == DENSE_CONDITIONING_SAME_COORDINATE
                            else DENSE_CONDITIONING_SCHEMA_VERSION
                        )
                    )
                )
            ),
            "conditioning_fusion_type": (
                DENSE_RELATIONAL_MEMORY_FUSION_TYPE
                if relational_mode
                else (
                    "single_packed_cross_attention"
                    if self.conditioning_mode == DENSE_CONDITIONING_LEGACY
                    else (
                        (
                            DENSE_FORCED_PAIRED_FUSION_TYPE
                            if forced_paired
                            else DENSE_GATED_PAIRED_FUSION_TYPE
                        )
                        if paired_mode
                        else (
                            DENSE_SAME_COORDINATE_FUSION_TYPE
                            if self.conditioning_mode
                            == DENSE_CONDITIONING_SAME_COORDINATE
                            else DENSE_CONDITIONING_FUSION_TYPE
                        )
                    )
                )
            ),
            "base_query_parameterization": (
                "per_position"
                if self.conditioning_mode == DENSE_CONDITIONING_LEGACY
                else "shared"
            ),
            "query_injection_position": (
                None
                if self.conditioning_mode == DENSE_CONDITIONING_LEGACY
                else DENSE_QUERY_INJECTION_POSITION
            ),
            "query_injection_is_final_residual": (
                None
                if self.conditioning_mode == DENSE_CONDITIONING_LEGACY
                else True
            ),
            "context_memory_mode": (
                "packed_all_tokens"
                if self.conditioning_mode == DENSE_CONDITIONING_LEGACY
                else (
                    "disabled_for_color_path"
                    if forced_paired
                    else DENSE_CONTEXT_MEMORY_MODE
                )
            ),
            "query_branch_scale_initial_value": (
                None
                if self.conditioning_mode == DENSE_CONDITIONING_LEGACY
                else (
                    self.same_coordinate_gate_initial_value
                    if self.conditioning_mode in (
                        DENSE_CONDITIONING_SAME_COORDINATE,
                        *DENSE_PAIRED_CONDITIONING_MODES,
                    )
                    else DENSE_QUERY_BRANCH_SCALE_INITIAL_VALUE
                )
            ),
            "context_branch_scale_initial_value": (
                None
                if self.conditioning_mode in (
                    DENSE_CONDITIONING_LEGACY,
                    DENSE_CONDITIONING_FORCED_PAIRED,
                )
                else DENSE_CONTEXT_BRANCH_SCALE_INITIAL_VALUE
            ),
            "same_coordinate_source": (
                (
                    "pre_fusion_query_cell_memory_scattered_by_flat_indices"
                    if paired_mode or relational_mode
                    else "query_cell_memory_scattered_by_flat_indices"
                )
                if self.conditioning_mode in (
                    DENSE_CONDITIONING_SAME_COORDINATE,
                    *DENSE_PAIRED_CONDITIONING_MODES,
                )
                else None
            ),
            "same_coordinate_padding_policy": (
                "masked_no_injection"
                if self.conditioning_mode in (
                    DENSE_CONDITIONING_SAME_COORDINATE,
                    *DENSE_PAIRED_CONDITIONING_MODES,
                )
                else None
            ),
            "same_coordinate_gate_parameterization": (
                DENSE_SAME_COORDINATE_GATE_PARAMETERIZATION
                if self.conditioning_mode in (
                    DENSE_CONDITIONING_SAME_COORDINATE,
                    *DENSE_PAIRED_CONDITIONING_MODES,
                )
                else None
            ),
            "same_coordinate_gate_initial_value": (
                self.same_coordinate_gate_initial_value
                if self.conditioning_mode in (
                    DENSE_CONDITIONING_SAME_COORDINATE,
                    *DENSE_PAIRED_CONDITIONING_MODES,
                )
                else None
            ),
            "generic_context_color_path": (
                False if forced_paired else (True if gated_conditioning else None)
            ),
            "paired_memory_source": (
                DENSE_FORCED_PAIRED_SOURCE if paired_mode else None
            ),
            "paired_memory_pairing": (
                DENSE_FORCED_PAIRED_PAIRING if paired_mode else None
            ),
            "paired_memory_coordinate_policy": (
                "exact_input_output_coordinate_set_required"
                if paired_mode
                else None
            ),
            "paired_memory_no_context_policy": (
                DENSE_FORCED_PAIRED_NO_CONTEXT_POLICY if paired_mode else None
            ),
            "paired_memory_gate": (
                "fixed_one_no_gate"
                if forced_paired
                else (
                    DENSE_GATED_PAIRED_GATE_PARAMETERIZATION
                    if gated_paired
                    else None
                )
            ),
            "paired_memory_persistent_injection": (
                True if paired_mode else None
            ),
            "candidate_max_rounds": DENSE_CANDIDATE_MAX_ROUNDS,
            "candidate_detach": True,
            "candidate_fusion_type": DENSE_CANDIDATE_FUSION_TYPE,
            "candidate_gate_parameterization": (
                DENSE_CANDIDATE_GATE_PARAMETERIZATION
            ),
            "candidate_gate_initial_value": DENSE_CANDIDATE_GATE_INITIAL_VALUE,
            "candidate_shape_gate_parameterization": "per_channel_tanh",
            "candidate_shape_gate_initial_value": 0.0,
            "recursion_schema_version": DENSE_RECURSION_SCHEMA_VERSION,
            "loss_schema_version": VIT5_DENSE_LOSS_SCHEMA_VERSION,
            "target_component_mode": TARGET_COMPONENT_MODE,
            "component_loss_type": COMPONENT_LOSS_TYPE,
        }
        # Do not add Gated-only keys (even with ``None`` values) to the
        # established Forced v3 dictionary: its canonical hash is part of the
        # strict progress-resume contract.  This preserves old Forced runs
        # while giving Gated v4 an explicit, independently versioned contract.
        if gated_paired:
            config.update(
                {
                    "paired_memory_gate_initial_value": (
                        self.paired_gate_initial_value
                    ),
                    "paired_memory_gate_initial_effective_scale": (
                        self.paired_gate_initial_value
                    ),
                    "paired_memory_gate_input": DENSE_GATED_PAIRED_GATE_INPUT,
                    "paired_memory_gate_range": "open_0_2",
                    "paired_memory_injection_position": (
                        DENSE_GATED_PAIRED_INJECTION_POSITION
                    ),
                }
            )
        if relational_mode:
            config.update(
                {
                    "relational_memory_mode": self.relational_memory_mode,
                    "relational_memory_schema_version": (
                        DENSE_RELATIONAL_MEMORY_SCHEMA_VERSION
                    ),
                    "relational_memory_source": DENSE_RELATIONAL_MEMORY_SOURCE,
                    "relational_memory_pairing": DENSE_RELATIONAL_MEMORY_PAIRING,
                    "relational_memory_geometry": DENSE_RELATIONAL_MEMORY_GEOMETRY,
                    "relational_memory_relation_tokens": (
                        "one_per_valid_demonstration_output_cell"
                    ),
                    "relational_memory_example_isolation": True,
                    "relational_memory_input_output_lengths_may_differ": True,
                    "relational_memory_no_context_policy": "exact_zero_residual",
                    "relational_memory_gate": (
                        DENSE_RELATIONAL_MEMORY_GATE_PARAMETERIZATION
                    ),
                    "relational_memory_gate_initial_value": (
                        self.relational_gate_initial_value
                    ),
                    "relational_memory_gate_initial_effective_scale": (
                        self.relational_gate_initial_value
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
            )
        if self.shape_decoder_mode == DENSE_SHAPE_DECODER_BD_BORDER:
            config.update(
                {
                    "shape_decoder_mode": DENSE_SHAPE_DECODER_BD_BORDER,
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
                }
            )
        return config

    def _output_queries(self, batch_size: int) -> torch.Tensor:
        row_ids = torch.arange(
            self.output_canvas_size, device=self.base_query.device
        ).view(-1, 1)
        col_ids = torch.arange(
            self.output_canvas_size, device=self.base_query.device
        ).view(1, -1)
        positional = (
            self.output_row_embedding(row_ids) + self.output_col_embedding(col_ids)
        ).reshape(
            1,
            self.output_canvas_size * self.output_canvas_size,
            self.hidden_size,
        )
        if self.conditioning_mode == DENSE_CONDITIONING_LEGACY:
            content = self.base_query
        else:
            content = self.base_query.expand(
                -1, self.output_canvas_size * self.output_canvas_size, -1
            )
        return (content + positional).expand(batch_size, -1, -1)

    def _remap_query_cell_flat_indices(
        self, query_cell_flat_indices: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Map encoder 30-stride cell indices onto the output query canvas."""
        if query_cell_flat_indices is None or self.output_canvas_size == self.max_size:
            return query_cell_flat_indices
        valid = query_cell_flat_indices >= 0
        valid_indices = query_cell_flat_indices[valid]
        if bool((valid_indices >= self.max_size * self.max_size).any().item()):
            raise ValueError("Valid query-cell flat indices are out of 30x30 range.")
        remapped = query_cell_flat_indices.clone()
        remapped[valid] = (
            valid_indices.div(self.max_size, rounding_mode="floor")
            * self.output_canvas_size
            + valid_indices.remainder(self.max_size)
        )
        return remapped

    def _forward_round(
        self,
        encoder_memory: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        query_cell_memory: torch.Tensor | None,
        query_cell_mask: torch.Tensor | None,
        query_cell_flat_indices: torch.Tensor | None,
        paired_demo_input_memory: torch.Tensor | None,
        paired_demo_output_memory: torch.Tensor | None,
        paired_demo_mask: torch.Tensor | None,
        relational_demo_input_memory: torch.Tensor | None,
        relational_demo_input_mask: torch.Tensor | None,
        relational_demo_input_example_ids: torch.Tensor | None,
        relational_demo_input_geometry: torch.Tensor | None,
        relational_demo_output_memory: torch.Tensor | None,
        relational_demo_output_mask: torch.Tensor | None,
        relational_demo_output_example_ids: torch.Tensor | None,
        relational_demo_output_geometry: torch.Tensor | None,
        task_summary: torch.Tensor,
        query_grid_summary: torch.Tensor,
        candidate_memory: torch.Tensor | None = None,
        candidate_attention_mask: torch.Tensor | None = None,
        candidate_summary: torch.Tensor | None = None,
    ) -> DenseDecoderOutput:
        batch_size = encoder_memory.shape[0]
        queries = self._output_queries(batch_size).to(dtype=encoder_memory.dtype)
        output_query_flat_indices = self._remap_query_cell_flat_indices(
            query_cell_flat_indices
        )
        paired_query_read = None
        relational_memory = None
        relational_memory_mask = None
        if self.relational_memory_enabled:
            if self.relational_memory_encoder is None:
                raise RuntimeError("Relational memory encoder is missing.")
            relational_values = (
                relational_demo_input_memory,
                relational_demo_input_mask,
                relational_demo_input_example_ids,
                relational_demo_input_geometry,
                relational_demo_output_memory,
                relational_demo_output_mask,
                relational_demo_output_example_ids,
                relational_demo_output_geometry,
            )
            if any(value is None for value in relational_values):
                raise ValueError(
                    "Relational v2 round requires complete input/output set tensors."
                )
            assert relational_demo_input_memory is not None
            assert relational_demo_input_mask is not None
            assert relational_demo_input_example_ids is not None
            assert relational_demo_input_geometry is not None
            assert relational_demo_output_memory is not None
            assert relational_demo_output_mask is not None
            assert relational_demo_output_example_ids is not None
            assert relational_demo_output_geometry is not None
            relational_memory, relational_memory_mask = (
                self.relational_memory_encoder(
                    relational_demo_input_memory,
                    relational_demo_input_mask,
                    relational_demo_input_example_ids,
                    relational_demo_input_geometry,
                    relational_demo_output_memory,
                    relational_demo_output_mask,
                    relational_demo_output_example_ids,
                    relational_demo_output_geometry,
                )
            )
        elif self.conditioning_mode in DENSE_PAIRED_CONDITIONING_MODES:
            if self.paired_associative_read is None:
                raise RuntimeError("Paired associative reader is missing.")
            if (
                query_cell_memory is None
                or query_cell_mask is None
                or paired_demo_input_memory is None
                or paired_demo_output_memory is None
                or paired_demo_mask is None
            ):
                raise ValueError(
                    "Paired-associative round requires query Q and paired K/V tensors."
                )
            paired_query_read = self.paired_associative_read(
                query_cell_memory,
                query_cell_mask,
                paired_demo_input_memory,
                paired_demo_output_memory,
                paired_demo_mask,
            )
        if (candidate_memory is None) != (candidate_attention_mask is None):
            raise ValueError(
                "candidate_memory and candidate_attention_mask must be provided together."
            )
        if candidate_memory is not None and len(self.candidate_refinement_layers) != len(
            self.layers
        ):
            raise RuntimeError("Candidate refinement layers are not initialized.")
        for layer_index, layer in enumerate(self.layers):
            defer_query_injection = (
                candidate_memory is not None
                and self.conditioning_mode != DENSE_CONDITIONING_LEGACY
            )
            queries = layer(
                queries,
                encoder_memory,
                encoder_attention_mask,
                query_cell_memory=query_cell_memory,
                query_cell_mask=query_cell_mask,
                query_cell_flat_indices=output_query_flat_indices,
                paired_query_read=paired_query_read,
                relational_memory=relational_memory,
                relational_memory_mask=relational_memory_mask,
                defer_query_injection=defer_query_injection,
            )
            if candidate_memory is not None:
                assert candidate_attention_mask is not None
                queries = self.candidate_refinement_layers[layer_index](
                    queries,
                    candidate_memory,
                    candidate_attention_mask,
                )
                if defer_query_injection:
                    queries = layer.inject_query_cells(
                        queries,
                        query_cell_memory,
                        query_cell_mask,
                        output_query_flat_indices,
                        paired_query_read=paired_query_read,
                    )
                    queries = layer.inject_relational_memory(
                        queries,
                        relational_memory,
                        relational_memory_mask,
                    )
        queries = self.output_norm(queries)
        color_logits = self.color_head(queries).reshape(
            batch_size,
            self.output_canvas_size,
            self.output_canvas_size,
            self.output_class_count,
        )
        if self.shape_decoder_mode == DENSE_SHAPE_DECODER_BD_BORDER:
            joint_shape_logits = bd_joint_shape_logits(color_logits)
            height_logits = torch.logsumexp(joint_shape_logits, dim=2)
            width_logits = torch.logsumexp(joint_shape_logits, dim=1)
            return DenseDecoderOutput(
                color_logits=color_logits,
                height_logits=height_logits,
                width_logits=width_logits,
                query_hidden_state=queries,
                joint_shape_logits=joint_shape_logits,
            )
        # In the strict Forced diagnostic even the shape feature must not leak
        # demonstration content back into the color decision through a shared
        # hidden state.  The synthetic task is same-shape, so the independently
        # encoded query summary is the complete shape input.
        shape_hidden = (
            query_grid_summary
            if self.conditioning_mode == DENSE_CONDITIONING_FORCED_PAIRED
            else task_summary + query_grid_summary
        )
        if candidate_summary is not None:
            if self.candidate_shape_norm is None or self.candidate_shape_gamma is None:
                raise RuntimeError("Candidate shape conditioning is not initialized.")
            shape_hidden = shape_hidden + torch.tanh(self.candidate_shape_gamma) * (
                self.candidate_shape_norm(candidate_summary)
            )
        if self.shape_norm is None or self.height_head is None or self.width_head is None:
            raise RuntimeError("Separate H/W shape heads are not initialized.")
        shape_feature = self.shape_norm(shape_hidden)
        return DenseDecoderOutput(
            color_logits=color_logits,
            height_logits=self.height_head(shape_feature),
            width_logits=self.width_head(shape_feature),
            query_hidden_state=queries,
        )

    def _apply_supervision(
        self,
        output: DenseDecoderOutput,
        dense_target_ids: torch.Tensor,
        dense_target_mask: torch.Tensor,
        dense_target_component_ids: torch.Tensor,
        target_height: torch.Tensor,
        target_width: torch.Tensor,
    ) -> DenseDecoderOutput:
        if self.shape_decoder_mode == DENSE_SHAPE_DECODER_BD_BORDER:
            targets = build_bd_border_targets(
                dense_target_ids,
                dense_target_mask,
                target_height,
                target_width,
                max_size=self.max_size,
            )
            per_position_loss = F.cross_entropy(
                output.color_logits.permute(0, 3, 1, 2),
                targets["target_ids"],
                reduction="none",
            )
            color_mask = targets["color_mask"]
            border_mask = targets["border_mask"]
            per_sample_color = (
                (per_position_loss * color_mask).flatten(1).sum(dim=1)
                / color_mask.flatten(1).sum(dim=1).to(per_position_loss.dtype)
            )
            per_sample_border = (
                (per_position_loss * border_mask).flatten(1).sum(dim=1)
                / border_mask.flatten(1).sum(dim=1).to(per_position_loss.dtype)
            )
            raw_cell_loss = per_sample_color.mean()
            border_loss = per_sample_border.mean()
            component_loss = component_balanced_mean_ce(
                per_position_loss[:, : self.max_size, : self.max_size],
                dense_target_mask,
                dense_target_component_ids,
            )
            color_loss = raw_cell_loss + self.lambda_component * component_loss
            if output.joint_shape_logits is None:
                raise RuntimeError("BD output is missing structured joint shape logits.")
            # These marginal CEs remain useful diagnostics for existing logs;
            # only the independently normalized BD loss enters dense_loss.
            height_loss = F.cross_entropy(
                output.height_logits, target_height.long() - 1
            )
            width_loss = F.cross_entropy(
                output.width_logits, target_width.long() - 1
            )
            dense_loss = color_loss + self.lambda_shape * border_loss
            return DenseDecoderOutput(
                color_logits=output.color_logits,
                height_logits=output.height_logits,
                width_logits=output.width_logits,
                raw_cell_loss=raw_cell_loss,
                component_balanced_loss=component_loss,
                color_loss=color_loss,
                height_loss=height_loss,
                width_loss=width_loss,
                border_loss=border_loss,
                dense_loss=dense_loss,
                metrics=dense_decoder_metrics(
                    output.color_logits,
                    output.height_logits,
                    output.width_logits,
                    dense_target_ids,
                    dense_target_mask,
                    target_height,
                    target_width,
                    joint_shape_logits=output.joint_shape_logits,
                ),
                query_hidden_state=output.query_hidden_state,
                joint_shape_logits=output.joint_shape_logits,
            )
        per_cell_loss = F.cross_entropy(
            output.color_logits.permute(0, 3, 1, 2),
            dense_target_ids.long(),
            reduction="none",
        )
        valid_counts = dense_target_mask.flatten(1).sum(dim=1).to(
            per_cell_loss.dtype
        )
        per_sample_color = (
            (per_cell_loss * dense_target_mask).flatten(1).sum(dim=1)
            / valid_counts
        )
        raw_cell_loss = per_sample_color.mean()
        component_loss = component_balanced_mean_ce(
            per_cell_loss,
            dense_target_mask,
            dense_target_component_ids,
        )
        color_loss = raw_cell_loss + self.lambda_component * component_loss
        height_loss = F.cross_entropy(
            output.height_logits, target_height.long() - 1
        )
        width_loss = F.cross_entropy(output.width_logits, target_width.long() - 1)
        dense_loss = color_loss + self.lambda_shape * (height_loss + width_loss)
        return DenseDecoderOutput(
            color_logits=output.color_logits,
            height_logits=output.height_logits,
            width_logits=output.width_logits,
            raw_cell_loss=raw_cell_loss,
            component_balanced_loss=component_loss,
            color_loss=color_loss,
            height_loss=height_loss,
            width_loss=width_loss,
            dense_loss=dense_loss,
            metrics=dense_decoder_metrics(
                output.color_logits,
                output.height_logits,
                output.width_logits,
                dense_target_ids,
                dense_target_mask,
                target_height,
                target_width,
            ),
            query_hidden_state=output.query_hidden_state,
        )

    def forward(
        self,
        encoder_memory: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        memory_layout: list[dict[str, Any]],
        task_summary: torch.Tensor,
        query_grid_summary: torch.Tensor,
        query_cell_memory: torch.Tensor | None = None,
        query_cell_mask: torch.Tensor | None = None,
        query_cell_flat_indices: torch.Tensor | None = None,
        context_attention_mask: torch.Tensor | None = None,
        paired_demo_input_memory: torch.Tensor | None = None,
        paired_demo_output_memory: torch.Tensor | None = None,
        paired_demo_mask: torch.Tensor | None = None,
        relational_demo_input_memory: torch.Tensor | None = None,
        relational_demo_input_mask: torch.Tensor | None = None,
        relational_demo_input_example_ids: torch.Tensor | None = None,
        relational_demo_input_geometry: torch.Tensor | None = None,
        relational_demo_output_memory: torch.Tensor | None = None,
        relational_demo_output_mask: torch.Tensor | None = None,
        relational_demo_output_example_ids: torch.Tensor | None = None,
        relational_demo_output_geometry: torch.Tensor | None = None,
        dense_target_ids: torch.Tensor | None = None,
        dense_target_mask: torch.Tensor | None = None,
        dense_target_component_ids: torch.Tensor | None = None,
        target_height: torch.Tensor | None = None,
        target_width: torch.Tensor | None = None,
    ) -> DenseDecoderOutput:
        if encoder_memory.ndim != 3 or encoder_memory.shape[-1] != self.hidden_size:
            raise ValueError("encoder_memory must have shape [B,S,hidden_size].")
        batch_size, sequence_length, _ = encoder_memory.shape
        if encoder_attention_mask.shape != (batch_size, sequence_length):
            raise ValueError("encoder_attention_mask must have shape [B,S].")
        if len(memory_layout) != batch_size:
            raise ValueError("memory_layout must contain one entry per batch sample.")
        if task_summary.shape != (batch_size, self.hidden_size):
            raise ValueError("task_summary must have shape [B,hidden_size].")
        if query_grid_summary.shape != (batch_size, self.hidden_size):
            raise ValueError("query_grid_summary must have shape [B,hidden_size].")
        if self.conditioning_mode != DENSE_CONDITIONING_LEGACY:
            if query_cell_memory is None or query_cell_mask is None:
                raise ValueError(
                    f"{self.conditioning_mode} requires explicit query-cell memory."
                )
            if query_cell_memory.ndim != 3 or query_cell_memory.shape[0] != batch_size:
                raise ValueError(
                    "query_cell_memory must have shape [B,Q,hidden_size]."
                )
            if query_cell_memory.shape[-1] != self.hidden_size:
                raise ValueError(
                    "query_cell_memory must use the Dense hidden size."
                )
            if query_cell_mask.shape != query_cell_memory.shape[:2]:
                raise ValueError("query_cell_mask must have shape [B,Q].")
            if bool((query_cell_mask.bool().sum(dim=1) < 1).any()):
                raise ValueError("Every sample requires at least one query cell.")
            if self.conditioning_mode in (
                DENSE_CONDITIONING_SAME_COORDINATE,
                *DENSE_PAIRED_CONDITIONING_MODES,
            ):
                if query_cell_flat_indices is None:
                    raise ValueError(
                        "same_coordinate_bridge_v1 requires query_cell_flat_indices."
                    )
                if query_cell_flat_indices.shape != query_cell_mask.shape:
                    raise ValueError(
                        "query_cell_flat_indices must have shape [B,Q]."
                    )
            if self.conditioning_mode in DENSE_PAIRED_CONDITIONING_MODES:
                if self.relational_memory_enabled:
                    relational_values = (
                        relational_demo_input_memory,
                        relational_demo_input_mask,
                        relational_demo_input_example_ids,
                        relational_demo_input_geometry,
                        relational_demo_output_memory,
                        relational_demo_output_mask,
                        relational_demo_output_example_ids,
                        relational_demo_output_geometry,
                    )
                    if any(value is None for value in relational_values):
                        raise ValueError(
                            "set_cross_attention_v2 requires complete flattened "
                            "demonstration input/output sets."
                        )
                else:
                    if (
                        paired_demo_input_memory is None
                        or paired_demo_output_memory is None
                        or paired_demo_mask is None
                    ):
                        raise ValueError(
                            f"{self.conditioning_mode} requires paired demo K/V."
                        )
                    if paired_demo_input_memory.ndim != 3:
                        raise ValueError(
                            "paired_demo_input_memory must have shape [B,R,H]."
                        )
                    if paired_demo_output_memory.shape != paired_demo_input_memory.shape:
                        raise ValueError(
                            "paired demo input/output memories must match exactly."
                        )
                    if paired_demo_input_memory.shape[0] != batch_size or (
                        paired_demo_input_memory.shape[-1] != self.hidden_size
                    ):
                        raise ValueError(
                            "paired demo memory has incompatible B/H dimensions."
                        )
                    if paired_demo_mask.shape != paired_demo_input_memory.shape[:2]:
                        raise ValueError("paired_demo_mask must have shape [B,R].")
                if self.conditioning_mode == DENSE_CONDITIONING_GATED_PAIRED:
                    if context_attention_mask is None:
                        raise ValueError(
                            "gated_paired_associative_v1 requires "
                            "context_attention_mask."
                        )
                    if context_attention_mask.shape != (
                        batch_size,
                        sequence_length,
                    ):
                        raise ValueError(
                            "context_attention_mask must have shape [B,S]."
                        )
                    if bool((context_attention_mask.bool().sum(dim=1) < 1).any()):
                        raise ValueError(
                            "Every Gated paired sample requires non-empty "
                            "generic context memory."
                        )
                    resolved_context_mask = context_attention_mask.bool()
                else:
                    # Forced does not instantiate or execute generic context
                    # attention; the mask is carried only to satisfy the common
                    # layer interface.
                    resolved_context_mask = encoder_attention_mask.bool()
            else:
                if context_attention_mask is None:
                    raise ValueError(
                        f"{self.conditioning_mode} requires context_attention_mask."
                    )
                if context_attention_mask.shape != (batch_size, sequence_length):
                    raise ValueError("context_attention_mask must have shape [B,S].")
                if bool((context_attention_mask.bool().sum(dim=1) < 1).any()):
                    raise ValueError("Every sample requires non-empty context memory.")
                resolved_context_mask = context_attention_mask.bool()
        else:
            resolved_context_mask = encoder_attention_mask.bool()
        targets = (
            dense_target_ids,
            dense_target_mask,
            dense_target_component_ids,
            target_height,
            target_width,
        )
        supervised = not all(value is None for value in targets)
        if supervised and any(value is None for value in targets):
            raise ValueError("Dense supervision requires ids, mask, height and width together.")
        if supervised:
            assert dense_target_ids is not None and dense_target_mask is not None
            assert dense_target_component_ids is not None
            assert target_height is not None and target_width is not None
            _validate_dense_targets(
                dense_target_ids,
                dense_target_mask,
                target_height,
                target_width,
                self.max_size,
            )
            validate_target_component_ids(
                dense_target_component_ids, dense_target_mask
            )

        round_outputs: list[DenseDecoderOutput] = []
        for round_index in range(self.recursive_steps):
            candidate_memory = None
            candidate_attention_mask = None
            candidate_summary = None
            if round_index > 0:
                if self.candidate_encoder is None:
                    raise RuntimeError("Recursive Dense candidate encoder is missing.")
                previous = round_outputs[-1]
                candidate = project_dense_candidate(
                    previous.color_logits,
                    previous.height_logits,
                    previous.width_logits,
                )
                encoded_candidate = self.candidate_encoder(candidate, round_index)
                candidate_memory = encoded_candidate.memory.to(
                    dtype=encoder_memory.dtype
                )
                candidate_summary = encoded_candidate.summary.to(
                    dtype=task_summary.dtype
                )
                candidate_attention_mask = encoded_candidate.attention_mask.bool()
            round_output = self._forward_round(
                encoder_memory,
                resolved_context_mask,
                query_cell_memory,
                query_cell_mask.bool() if query_cell_mask is not None else None,
                query_cell_flat_indices,
                paired_demo_input_memory,
                paired_demo_output_memory,
                paired_demo_mask.bool() if paired_demo_mask is not None else None,
                relational_demo_input_memory,
                (
                    relational_demo_input_mask.bool()
                    if relational_demo_input_mask is not None
                    else None
                ),
                relational_demo_input_example_ids,
                relational_demo_input_geometry,
                relational_demo_output_memory,
                (
                    relational_demo_output_mask.bool()
                    if relational_demo_output_mask is not None
                    else None
                ),
                relational_demo_output_example_ids,
                relational_demo_output_geometry,
                task_summary,
                query_grid_summary,
                candidate_memory,
                candidate_attention_mask,
                candidate_summary,
            )
            if supervised:
                round_output = self._apply_supervision(
                    round_output,
                    dense_target_ids,
                    dense_target_mask,
                    dense_target_component_ids,
                    target_height,
                    target_width,
                )
            round_outputs.append(round_output)

        final = round_outputs[-1]
        weight_dtype = (
            final.dense_loss.dtype
            if final.dense_loss is not None
            else encoder_memory.dtype
        )
        weights = recursive_supervision_weights(
            self.recursive_steps,
            self.recursive_loss_weighting,
            self.hsw_lambda,
            device=encoder_memory.device,
            dtype=weight_dtype,
        )
        if not supervised or self.recursive_steps == 1:
            return DenseDecoderOutput(
                color_logits=final.color_logits,
                height_logits=final.height_logits,
                width_logits=final.width_logits,
                raw_cell_loss=final.raw_cell_loss,
                component_balanced_loss=final.component_balanced_loss,
                color_loss=final.color_loss,
                height_loss=final.height_loss,
                width_loss=final.width_loss,
                border_loss=final.border_loss,
                dense_loss=final.dense_loss,
                metrics=final.metrics,
                query_hidden_state=final.query_hidden_state,
                round_outputs=tuple(round_outputs),
                round_loss_weights=weights,
                recursive_metrics=(
                    dense_recursive_metrics(
                        round_outputs,
                        dense_target_ids,
                        dense_target_mask,
                        target_height,
                        target_width,
                    )
                    if supervised
                    else None
                ),
                joint_shape_logits=final.joint_shape_logits,
            )

        def weighted(field: str) -> torch.Tensor:
            values = [getattr(output, field) for output in round_outputs]
            if any(value is None for value in values):
                raise RuntimeError(f"Recursive supervised field {field} is missing.")
            return torch.sum(
                torch.stack([value for value in values if value is not None])
                * weights
            )

        return DenseDecoderOutput(
            color_logits=final.color_logits,
            height_logits=final.height_logits,
            width_logits=final.width_logits,
            raw_cell_loss=weighted("raw_cell_loss"),
            component_balanced_loss=weighted("component_balanced_loss"),
            color_loss=weighted("color_loss"),
            height_loss=weighted("height_loss"),
            width_loss=weighted("width_loss"),
            border_loss=(
                weighted("border_loss")
                if all(output.border_loss is not None for output in round_outputs)
                else None
            ),
            dense_loss=weighted("dense_loss"),
            metrics=final.metrics,
            query_hidden_state=final.query_hidden_state,
            round_outputs=tuple(round_outputs),
            round_loss_weights=weights,
            recursive_metrics=dense_recursive_metrics(
                round_outputs,
                dense_target_ids,
                dense_target_mask,
                target_height,
                target_width,
            ),
            joint_shape_logits=final.joint_shape_logits,
        )

    @torch.no_grad()
    def predict_dense_grid(
        self,
        encoder_memory: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        memory_layout: list[dict[str, Any]],
        task_summary: torch.Tensor,
        query_grid_summary: torch.Tensor,
        query_cell_memory: torch.Tensor | None = None,
        query_cell_mask: torch.Tensor | None = None,
        query_cell_flat_indices: torch.Tensor | None = None,
        context_attention_mask: torch.Tensor | None = None,
        paired_demo_input_memory: torch.Tensor | None = None,
        paired_demo_output_memory: torch.Tensor | None = None,
        paired_demo_mask: torch.Tensor | None = None,
        relational_demo_input_memory: torch.Tensor | None = None,
        relational_demo_input_mask: torch.Tensor | None = None,
        relational_demo_input_example_ids: torch.Tensor | None = None,
        relational_demo_input_geometry: torch.Tensor | None = None,
        relational_demo_output_memory: torch.Tensor | None = None,
        relational_demo_output_mask: torch.Tensor | None = None,
        relational_demo_output_example_ids: torch.Tensor | None = None,
        relational_demo_output_geometry: torch.Tensor | None = None,
        include_raw_logits: bool = False,
        include_recursive_rounds: bool = False,
    ) -> list[dict[str, Any]]:
        output = self(
            encoder_memory=encoder_memory,
            encoder_attention_mask=encoder_attention_mask,
            memory_layout=memory_layout,
            task_summary=task_summary,
            query_grid_summary=query_grid_summary,
            query_cell_memory=query_cell_memory,
            query_cell_mask=query_cell_mask,
            query_cell_flat_indices=query_cell_flat_indices,
            context_attention_mask=context_attention_mask,
            paired_demo_input_memory=paired_demo_input_memory,
            paired_demo_output_memory=paired_demo_output_memory,
            paired_demo_mask=paired_demo_mask,
            relational_demo_input_memory=relational_demo_input_memory,
            relational_demo_input_mask=relational_demo_input_mask,
            relational_demo_input_example_ids=relational_demo_input_example_ids,
            relational_demo_input_geometry=relational_demo_input_geometry,
            relational_demo_output_memory=relational_demo_output_memory,
            relational_demo_output_mask=relational_demo_output_mask,
            relational_demo_output_example_ids=relational_demo_output_example_ids,
            relational_demo_output_geometry=relational_demo_output_geometry,
        )
        decoded = decode_dense_output(
            output.color_logits,
            output.height_logits,
            output.width_logits,
            include_raw_logits=include_raw_logits,
            joint_shape_logits=output.joint_shape_logits,
        )
        if include_recursive_rounds:
            if output.round_outputs is None:
                raise RuntimeError("Dense recursive round outputs are unavailable.")
            round_decoded = [
                decode_dense_output(
                    item.color_logits,
                    item.height_logits,
                    item.width_logits,
                    include_raw_logits=False,
                    joint_shape_logits=item.joint_shape_logits,
                )
                for item in output.round_outputs
            ]
            for batch_index, item in enumerate(decoded):
                item["recursive_rounds"] = [
                    values[batch_index] for values in round_decoded
                ]
        return decoded
