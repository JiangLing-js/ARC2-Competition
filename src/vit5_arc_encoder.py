"""Hierarchical ViT-5 Small encoder for discrete ARC task grids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .vit5_hybrid_batch import (
    ARC_COLOR_PAD_ID,
    ARC_MAX_OBJECTS,
    ROLE_CONTEXT_INPUT,
    ROLE_CONTEXT_OUTPUT,
    ROLE_QUERY_INPUT,
)
from .vit5_layers import (
    RMSNorm,
    RotaryLayout,
    ViT5Block,
    initialize_vit5_module,
)


VIT5_SMALL_DIM = 384
VIT5_SMALL_DEPTH = 12
VIT5_SMALL_HEADS = 6
VIT5_SMALL_REGISTERS = 4


@dataclass
class GridEncoderOutput:
    cell_hidden_state: torch.Tensor
    cell_mask: torch.Tensor
    cell_flat_indices: torch.Tensor
    cell_color_ids: torch.Tensor
    grid_summary: torch.Tensor
    register_hidden_state: torch.Tensor
    sequence_attention_mask: torch.Tensor


@dataclass
class TaskFusionOutput:
    task_summary: torch.Tensor
    grid_summaries: torch.Tensor


@dataclass
class ViT5ArcEncoderOutput:
    last_hidden_state: torch.Tensor
    attention_mask: torch.Tensor
    context_attention_mask: torch.Tensor
    memory_layout: list[dict[str, Any]]
    task_summary: torch.Tensor
    grid_summaries: torch.Tensor
    grid_mask: torch.Tensor
    query_grid_summary: torch.Tensor
    query_grid_slots: torch.Tensor
    query_cell_memory: torch.Tensor
    query_cell_mask: torch.Tensor
    query_cell_row_ids: torch.Tensor
    query_cell_col_ids: torch.Tensor
    query_cell_flat_indices: torch.Tensor
    query_original_size: torch.Tensor
    encoded_grid_count: int
    grid_encoder_microbatch: int | None
    forced_query_cell_memory: torch.Tensor | None = None
    forced_query_cell_mask: torch.Tensor | None = None
    forced_query_cell_flat_indices: torch.Tensor | None = None
    forced_query_grid_summary: torch.Tensor | None = None
    paired_demo_input_memory: torch.Tensor | None = None
    paired_demo_output_memory: torch.Tensor | None = None
    paired_demo_mask: torch.Tensor | None = None
    paired_demo_example_ids: torch.Tensor | None = None
    paired_demo_flat_indices: torch.Tensor | None = None
    relational_demo_input_memory: torch.Tensor | None = None
    relational_demo_input_mask: torch.Tensor | None = None
    relational_demo_input_example_ids: torch.Tensor | None = None
    relational_demo_input_flat_indices: torch.Tensor | None = None
    relational_demo_input_row_ids: torch.Tensor | None = None
    relational_demo_input_col_ids: torch.Tensor | None = None
    relational_demo_input_original_sizes: torch.Tensor | None = None
    relational_demo_input_geometry: torch.Tensor | None = None
    relational_demo_output_memory: torch.Tensor | None = None
    relational_demo_output_mask: torch.Tensor | None = None
    relational_demo_output_example_ids: torch.Tensor | None = None
    relational_demo_output_flat_indices: torch.Tensor | None = None
    relational_demo_output_row_ids: torch.Tensor | None = None
    relational_demo_output_col_ids: torch.Tensor | None = None
    relational_demo_output_original_sizes: torch.Tensor | None = None
    relational_demo_output_geometry: torch.Tensor | None = None


class ArcGridEmbedding(nn.Module):
    """Discrete color, spatial APE and task-identity embeddings for ARC cells."""

    def __init__(
        self,
        dim: int = VIT5_SMALL_DIM,
        max_h: int = 30,
        max_w: int = 30,
        max_grids: int = 21,
        max_examples: int = 11,
        num_roles: int = 4,
        max_objects: int = ARC_MAX_OBJECTS,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.max_h = int(max_h)
        self.max_w = int(max_w)
        self.max_grids = int(max_grids)
        self.max_examples = int(max_examples)
        self.max_objects = int(max_objects)
        self.color_embedding = nn.Embedding(ARC_COLOR_PAD_ID + 1, dim, padding_idx=ARC_COLOR_PAD_ID)
        self.row_abs_embedding = nn.Embedding(max_h + 1, dim, padding_idx=max_h)
        self.col_abs_embedding = nn.Embedding(max_w + 1, dim, padding_idx=max_w)
        self.ape_residual_2d = nn.Parameter(torch.zeros(max_h, max_w, dim))
        self.role_embedding = nn.Embedding(num_roles, dim, padding_idx=0)
        self.example_embedding = nn.Embedding(max_examples + 1, dim, padding_idx=max_examples)
        self.grid_embedding = nn.Embedding(max_grids + 1, dim, padding_idx=max_grids)
        self.object_embedding = nn.Embedding(max_objects + 1, dim, padding_idx=0)
        self.apply(initialize_vit5_module)
        with torch.no_grad():
            self.color_embedding.weight[ARC_COLOR_PAD_ID].zero_()
            self.row_abs_embedding.weight[max_h].zero_()
            self.col_abs_embedding.weight[max_w].zero_()
            self.role_embedding.weight.zero_()
            self.example_embedding.weight.zero_()
            self.grid_embedding.weight.zero_()
            # Preserve exact pre-object model behavior at initialization while
            # allowing every non-background object row to learn immediately.
            self.object_embedding.weight.zero_()

    def initialize_ape_from_checkpoint(self, pos_embed: torch.Tensor) -> None:
        """Interpolate 14x14 APE and decompose it into row/column/residual terms."""
        if tuple(pos_embed.shape) != (1, 196, self.dim):
            raise ValueError(
                f"Expected pos_embed shape (1,196,{self.dim}), got {tuple(pos_embed.shape)}."
            )
        source = pos_embed.float().reshape(1, 14, 14, self.dim).permute(0, 3, 1, 2)
        interpolated = F.interpolate(
            source,
            size=(self.max_h, self.max_w),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1)[0]
        row_init = interpolated.mean(dim=1)
        col_init = interpolated.mean(dim=0) - interpolated.mean(dim=(0, 1))
        # P_rc = R_r+C_r+Q_rc
        residual = interpolated - row_init[:, None, :] - col_init[None, :, :]
        with torch.no_grad():
            self.row_abs_embedding.weight[: self.max_h].copy_(
                row_init.to(self.row_abs_embedding.weight.dtype)
            )
            self.row_abs_embedding.weight[self.max_h].zero_()
            self.col_abs_embedding.weight[: self.max_w].copy_(
                col_init.to(self.col_abs_embedding.weight.dtype)
            )
            self.col_abs_embedding.weight[self.max_w].zero_()
            self.ape_residual_2d.copy_(residual.to(self.ape_residual_2d.dtype))

    def identity_embedding(
        self,
        role_ids: torch.Tensor,
        example_ids: torch.Tensor,
        grid_ids: torch.Tensor,
    ) -> torch.Tensor:
        if (example_ids >= self.max_examples).any():
            raise ValueError(f"example_ids must be below {self.max_examples}.")
        if (grid_ids >= self.max_grids).any():
            raise ValueError(f"grid_ids must be below {self.max_grids}.")
        return (
            self.role_embedding(role_ids)
            + self.example_embedding(example_ids)
            + self.grid_embedding(grid_ids)
        )

    def forward(
        self,
        color_ids: torch.Tensor,
        row_ids: torch.Tensor,
        col_ids: torch.Tensor,
        role_ids: torch.Tensor,
        example_ids: torch.Tensor,
        grid_ids: torch.Tensor,
        object_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if color_ids.shape != row_ids.shape or color_ids.shape != col_ids.shape:
            raise ValueError("Compact color/row/column tensors must have equal shapes.")
        if object_ids is None:
            object_ids = torch.zeros_like(color_ids)
        if object_ids.shape != color_ids.shape:
            raise ValueError("Compact object IDs must match compact color tensor shape.")
        if object_ids.numel() and (
            bool((object_ids < 0).any()) or bool((object_ids > self.max_objects).any())
        ):
            raise ValueError(f"object_ids must be in [0,{self.max_objects}].")
        identity = self.identity_embedding(role_ids, example_ids, grid_ids)
        valid_cells = color_ids.ne(ARC_COLOR_PAD_ID).unsqueeze(-1)
        residual = self.ape_residual_2d[
            row_ids.clamp_max(self.max_h - 1), col_ids.clamp_max(self.max_w - 1)
        ] * valid_cells.to(self.ape_residual_2d.dtype)
        return (
            self.color_embedding(color_ids)
            + self.row_abs_embedding(row_ids)
            + self.col_abs_embedding(col_ids)
            + self.object_embedding(object_ids)
            + residual
            + identity.unsqueeze(1)
        )


class ViT5ArcGridEncoder(nn.Module):
    """Shared checkpoint-compatible ViT-5 Small applied independently per grid."""

    def __init__(
        self,
        max_h: int = 30,
        max_w: int = 30,
        max_grids: int = 21,
        max_examples: int = 11,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        use_sdpa: bool = True,
        max_objects: int = ARC_MAX_OBJECTS,
    ) -> None:
        super().__init__()
        self.embed_dim = VIT5_SMALL_DIM
        self.num_registers = VIT5_SMALL_REGISTERS
        self.max_h = int(max_h)
        self.max_w = int(max_w)
        self.arc_embedding = ArcGridEmbedding(
            dim=self.embed_dim,
            max_h=max_h,
            max_w=max_w,
            max_grids=max_grids,
            max_examples=max_examples,
            max_objects=max_objects,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.reg_token = nn.Parameter(torch.zeros(1, self.num_registers, self.embed_dim))
        self.blocks = nn.ModuleList(
            [
                ViT5Block(
                    dim=self.embed_dim,
                    num_heads=VIT5_SMALL_HEADS,
                    mlp_ratio=4.0,
                    qkv_bias=False,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_rate,
                    init_values=1e-4,
                    rope_size=14,
                    rope_reg_size=2,
                    reg_theta=100.0,
                    num_registers=self.num_registers,
                    qk_norm=True,
                    layer_scale=True,
                    use_sdpa=use_sdpa,
                )
                for _ in range(VIT5_SMALL_DEPTH)
            ]
        )
        self.norm = RMSNorm(self.embed_dim, eps=1e-6)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.reg_token, std=0.02)

    def initialize_ape_from_checkpoint(self, pos_embed: torch.Tensor) -> None:
        # 把 checkpoint 的14×14 APE → 30×30 插值 → row + column + residual
        self.arc_embedding.initialize_ape_from_checkpoint(pos_embed)

    def _compact_cells(
        self,
        color_ids: torch.Tensor,
        cell_mask: torch.Tensor,
        row_ids: torch.Tensor,
        col_ids: torch.Tensor,
        object_ids: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        grid_count, height, width = color_ids.shape
        if (height, width) != (self.max_h, self.max_w):
            raise ValueError(
                f"Grid tensor shape {(height, width)} != {(self.max_h, self.max_w)}."
            )
        counts = cell_mask.flatten(1).sum(dim=1)
        if (counts <= 0).any():
            raise ValueError("Every encoded grid must contain at least one real ARC cell.")
        compact_length = int(counts.max().item())
        compact_colors = torch.full(
            (grid_count, compact_length),
            ARC_COLOR_PAD_ID,
            dtype=torch.long,
            device=color_ids.device,
        )
        compact_rows = torch.full(
            (grid_count, compact_length), self.max_h, dtype=torch.long, device=color_ids.device
        )
        compact_cols = torch.full(
            (grid_count, compact_length), self.max_w, dtype=torch.long, device=color_ids.device
        )
        compact_mask = torch.zeros(
            grid_count, compact_length, dtype=torch.bool, device=color_ids.device
        )
        compact_objects = torch.zeros(
            (grid_count, compact_length), dtype=torch.long, device=color_ids.device
        )
        flat_indices = torch.full(
            (grid_count, compact_length), -1, dtype=torch.long, device=color_ids.device
        )
        flat_colors = color_ids.flatten(1)
        flat_rows = row_ids.flatten(1)
        flat_cols = col_ids.flatten(1)
        flat_objects = object_ids.flatten(1)
        flat_masks = cell_mask.flatten(1)
        for index in range(grid_count):
            selected = torch.nonzero(flat_masks[index], as_tuple=False).flatten()
            length = int(selected.numel())
            compact_colors[index, :length] = flat_colors[index].index_select(0, selected)
            compact_rows[index, :length] = flat_rows[index].index_select(0, selected)
            compact_cols[index, :length] = flat_cols[index].index_select(0, selected)
            compact_objects[index, :length] = flat_objects[index].index_select(0, selected)
            compact_mask[index, :length] = True
            flat_indices[index, :length] = selected
        return (
            compact_colors, compact_rows, compact_cols, compact_objects,
            compact_mask, flat_indices,
        )

    def _forward_sequence_chunks(
        self,
        sequence: torch.Tensor,
        attention_mask: torch.Tensor,
        rotary_layout: RotaryLayout,
        microbatch: int | None,
    ) -> torch.Tensor:
        grid_count = sequence.shape[0]
        chunk_size = grid_count if not microbatch or microbatch <= 0 else int(microbatch)
        if chunk_size < 1:
            raise ValueError("grid_encoder_microbatch must be positive or omitted.")
        outputs: list[torch.Tensor] = []
        for start in range(0, grid_count, chunk_size):
            end = min(start + chunk_size, grid_count)
            hidden = sequence[start:end]
            mask = attention_mask[start:end]
            layout = RotaryLayout(
                row_ids=rotary_layout.row_ids[start:end],
                col_ids=rotary_layout.col_ids[start:end],
                spatial_mask=rotary_layout.spatial_mask[start:end],
                register_mask=rotary_layout.register_mask[start:end],
                register_position_ids=rotary_layout.register_position_ids[start:end],
                coordinate_scale=rotary_layout.coordinate_scale,
            )
            for block in self.blocks:
                hidden = block(hidden, attention_mask=mask, rotary_layout=layout)
            outputs.append(self.norm(hidden) * mask.unsqueeze(-1).to(hidden.dtype))
        return torch.cat(outputs, dim=0)

    def forward(
        self,
        color_ids: torch.Tensor,
        cell_mask: torch.Tensor,
        row_ids: torch.Tensor,
        col_ids: torch.Tensor,
        role_ids: torch.Tensor,
        example_ids: torch.Tensor,
        grid_ids: torch.Tensor,
        object_ids: torch.Tensor | None = None,
        microbatch: int | None = None,
    ) -> GridEncoderOutput:
        if object_ids is None:
            object_ids = torch.zeros_like(color_ids)
        (
            compact_colors, compact_rows, compact_cols, compact_objects,
            compact_mask, flat_indices,
        ) = (
            self._compact_cells(color_ids, cell_mask, row_ids, col_ids, object_ids)
        )
        cell_embeddings = self.arc_embedding(
            compact_colors,
            compact_rows,
            compact_cols,
            role_ids,
            example_ids,
            grid_ids,
            compact_objects,
        )
        identity = self.arc_embedding.identity_embedding(role_ids, example_ids, grid_ids)
        grid_count, compact_length = compact_colors.shape
        cls = self.cls_token.expand(grid_count, -1, -1) + identity.unsqueeze(1)
        registers = self.reg_token.expand(grid_count, -1, -1) + identity.unsqueeze(1)
        sequence = torch.cat((cls, cell_embeddings, registers), dim=1)
        attention_mask = torch.cat(
            (
                torch.ones(grid_count, 1, dtype=torch.bool, device=sequence.device),
                compact_mask,
                torch.ones(
                    grid_count,
                    self.num_registers,
                    dtype=torch.bool,
                    device=sequence.device,
                ),
            ),
            dim=1,
        )
        length = sequence.shape[1]
        layout_rows = torch.zeros(grid_count, length, dtype=torch.long, device=sequence.device)
        layout_cols = torch.zeros_like(layout_rows)
        layout_rows[:, 1 : 1 + compact_length] = compact_rows.clamp_max(self.max_h - 1)
        layout_cols[:, 1 : 1 + compact_length] = compact_cols.clamp_max(self.max_w - 1)
        spatial_mask = torch.zeros_like(attention_mask)
        spatial_mask[:, 1 : 1 + compact_length] = compact_mask
        register_mask = torch.zeros_like(attention_mask)
        register_mask[:, -self.num_registers :] = True
        register_positions = torch.zeros_like(layout_rows)
        register_positions[:, -self.num_registers :] = torch.arange(
            self.num_registers, device=sequence.device
        )
        rotary_layout = RotaryLayout(
            row_ids=layout_rows,
            col_ids=layout_cols,
            spatial_mask=spatial_mask,
            register_mask=register_mask,
            register_position_ids=register_positions,
            coordinate_scale=14.0 / float(max(self.max_h, self.max_w)),
        )
        encoded = self._forward_sequence_chunks(
            sequence, attention_mask, rotary_layout, microbatch=microbatch
        )
        return GridEncoderOutput(
            cell_hidden_state=encoded[:, 1 : 1 + compact_length],
            cell_mask=compact_mask,
            cell_flat_indices=flat_indices,
            cell_color_ids=compact_colors,
            grid_summary=encoded[:, 0],
            register_hidden_state=encoded[:, -self.num_registers :],
            sequence_attention_mask=attention_mask,
        )


class _TaskFusionLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int = 6, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = RMSNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.apply(initialize_vit5_module)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(hidden)
        attended, _ = self.self_attn(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~attention_mask.bool(),
            need_weights=False,
        )
        hidden = hidden + attended
        hidden = hidden + self.mlp(self.norm2(hidden))
        return hidden * attention_mask.unsqueeze(-1).to(hidden.dtype)


class ArcTaskFusion(nn.Module):
    """创建一个全任务级别的 [TASK] token，把它和各个 grid_summary 拼起来，经过 1～2 层任务融合，最后拆回任务摘要和网格摘要。"""

    def __init__(self, dim: int = VIT5_SMALL_DIM, num_layers: int = 2) -> None:
        super().__init__()
        if not 1 <= num_layers <= 2:
            raise ValueError("First-stage task_fusion_layers must be 1 or 2.")
        self.task_summary_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.layers = nn.ModuleList([_TaskFusionLayer(dim) for _ in range(num_layers)])
        self.norm = RMSNorm(dim)
        nn.init.trunc_normal_(self.task_summary_token, std=0.02)

    def forward(
        self, grid_summaries: torch.Tensor, grid_mask: torch.Tensor
    ) -> TaskFusionOutput:
        batch = grid_summaries.shape[0]
        task_token = self.task_summary_token.expand(batch, -1, -1)
        hidden = torch.cat((task_token, grid_summaries), dim=1)
        mask = torch.cat(
            (
                torch.ones(batch, 1, dtype=torch.bool, device=grid_mask.device),
                grid_mask.bool(),
            ),
            dim=1,
        )
        for layer in self.layers:
            hidden = layer(hidden, mask)
        hidden = self.norm(hidden) * mask.unsqueeze(-1).to(hidden.dtype)
        return TaskFusionOutput(task_summary=hidden[:, 0], grid_summaries=hidden[:, 1:])


class HierarchicalViT5ArcEncoder(nn.Module):
    """Per-grid ViT-5 plus task-summary fusion and packed decoder memory."""

    def __init__(
        self,
        max_h: int = 30,
        max_w: int = 30,
        max_grids: int = 21,
        max_examples: int = 11,
        task_fusion_layers: int = 2,
        grid_encoder_microbatch: int | None = None,
        use_sdpa: bool = True,
        max_objects: int = ARC_MAX_OBJECTS,
        enable_forced_paired_diagnostic: bool = False,
        enable_relational_memory_v2: bool = False,
    ) -> None:
        super().__init__()
        self.max_h = int(max_h)
        self.max_w = int(max_w)
        self.max_grids = int(max_grids)
        self.grid_encoder_microbatch = grid_encoder_microbatch
        self.enable_forced_paired_diagnostic = bool(
            enable_forced_paired_diagnostic
        )
        self.enable_relational_memory_v2 = bool(enable_relational_memory_v2)
        # 输出 cell_hidden_state + grid_summary + register_hidden_state
        self.grid_encoder = ViT5ArcGridEncoder(
            max_h=max_h,
            max_w=max_w,
            max_grids=max_grids,
            max_examples=max_examples,
            use_sdpa=use_sdpa,
            max_objects=max_objects,
        )
        # 输出 task summary + fused grid summaries
        self.task_fusion = ArcTaskFusion(num_layers=task_fusion_layers)
        self.cell_refine_norm = RMSNorm(VIT5_SMALL_DIM)
        # cross attention: Query：所有真实 cell Key：融合后的 grid summaries Value：融合后的 grid summaries
        self.cell_to_grid_attention = nn.MultiheadAttention(
            VIT5_SMALL_DIM, VIT5_SMALL_HEADS, batch_first=True
        )
        self.cell_refine_gamma = nn.Parameter(torch.full((VIT5_SMALL_DIM,), 1e-4))
        self.cell_to_grid_attention.apply(initialize_vit5_module)

    def _flatten_valid_grids(
        self,
        color_ids: torch.Tensor,
        cell_mask: torch.Tensor,
        row_ids: torch.Tensor,
        col_ids: torch.Tensor,
        object_ids: torch.Tensor,
        grid_mask: torch.Tensor,
        role_ids: torch.Tensor,
        example_ids: torch.Tensor,
        grid_ids: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        locations = torch.nonzero(grid_mask.bool(), as_tuple=False)
        if locations.numel() == 0:
            raise ValueError("A hybrid encoder batch must contain at least one valid grid.")
        batch_indices = locations[:, 0]
        grid_slots = locations[:, 1]
        flattened = {
            "color_ids": color_ids[batch_indices, grid_slots],
            "cell_mask": cell_mask[batch_indices, grid_slots],
            "row_ids": row_ids[batch_indices, grid_slots],
            "col_ids": col_ids[batch_indices, grid_slots],
            "object_ids": object_ids[batch_indices, grid_slots],
            "role_ids": role_ids[batch_indices, grid_slots],
            "example_ids": example_ids[batch_indices, grid_slots],
            "grid_ids": grid_ids[batch_indices, grid_slots],
        }
        return flattened, batch_indices, grid_slots

    def _refine_cells(
        self,
        cells: torch.Tensor,
        cell_mask: torch.Tensor,
        batch_indices: torch.Tensor,
        grid_slots: torch.Tensor,
        fused_grid_summaries: torch.Tensor,
        grid_mask: torch.Tensor,
    ) -> torch.Tensor:
        refined = cells.clone()
        for batch_index in range(grid_mask.shape[0]):
            grid_rows = torch.nonzero(batch_indices == batch_index, as_tuple=False).flatten()
            valid_summaries = fused_grid_summaries[batch_index, grid_mask[batch_index]]
            query_parts: list[torch.Tensor] = []
            lengths: list[int] = []
            for flat_index in grid_rows.tolist():
                current = cells[flat_index, cell_mask[flat_index]]
                query_parts.append(current)
                lengths.append(current.shape[0])
            queries = torch.cat(query_parts, dim=0).unsqueeze(0)
            normalized = self.cell_refine_norm(queries)
            attended, _ = self.cell_to_grid_attention(
                normalized,
                valid_summaries.unsqueeze(0),
                valid_summaries.unsqueeze(0),
                need_weights=False,
            )
            conditioned = queries + self.cell_refine_gamma * attended
            offset = 0
            for flat_index, length in zip(grid_rows.tolist(), lengths):
                refined[flat_index, cell_mask[flat_index]] = conditioned[0, offset : offset + length]
                offset += length
        return refined

    def _pack_memory(
        self,
        grid_output: GridEncoderOutput,
        refined_cells: torch.Tensor,
        fusion: TaskFusionOutput,
        grid_mask: torch.Tensor,
        batch_indices: torch.Tensor,
        grid_slots: torch.Tensor,
        role_ids: torch.Tensor,
        example_ids: torch.Tensor,
        grid_ids: torch.Tensor,
        cell_mask_4d: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        '''
        对于每个 task，最终排列为：

        [TASK SUMMARY]

        [GRID SUMMARY 0]
        [GRID SUMMARY 1]
        ...
        [GRID SUMMARY G]

        [GRID 0 CELLS]
        [GRID 0 REGISTERS]

        [GRID 1 CELLS]
        [GRID 1 REGISTERS]

        ...

        这是整个模型与 decoder 连接的关键接口。
        '''
        sequences: list[torch.Tensor] = []
        layouts: list[dict[str, Any]] = []
        for batch_index in range(grid_mask.shape[0]):
            valid_slots = torch.nonzero(grid_mask[batch_index], as_tuple=False).flatten().tolist()
            parts = [fusion.task_summary[batch_index].unsqueeze(0)]
            cursor = 1
            layout: dict[str, Any] = {
                "task_summary_index": 0,
                "grid_summary_span": (cursor, cursor + len(valid_slots)),
                "grids": [],
            }
            parts.append(fusion.grid_summaries[batch_index, valid_slots])
            cursor += len(valid_slots)
            for slot in valid_slots:
                match = torch.nonzero(
                    (batch_indices == batch_index) & (grid_slots == slot), as_tuple=False
                ).flatten()
                if match.numel() != 1:
                    raise RuntimeError("Valid grid did not map to exactly one encoded grid.")
                flat_index = int(match.item())
                valid_cells = refined_cells[flat_index, grid_output.cell_mask[flat_index]]
                cell_start = cursor
                parts.append(valid_cells)
                cursor += valid_cells.shape[0]
                register_start = cursor
                parts.append(grid_output.register_hidden_state[flat_index])
                cursor += grid_output.register_hidden_state.shape[1]
                spatial_mask = cell_mask_4d[batch_index, slot]
                height = int(spatial_mask.any(dim=1).sum().item())
                width = int(spatial_mask.any(dim=0).sum().item())
                layout["grids"].append(
                    {
                        "grid_slot": int(slot),
                        "grid_id": int(grid_ids[batch_index, slot].item()),
                        "role_id": int(role_ids[batch_index, slot].item()),
                        "example_id": int(example_ids[batch_index, slot].item()),
                        "original_size": (height, width),
                        "cell_span": (cell_start, register_start),
                        "register_span": (register_start, cursor),
                    }
                )
            sequence = torch.cat(parts, dim=0)
            layout["sequence_length"] = int(sequence.shape[0])
            sequences.append(sequence)
            layouts.append(layout)
        max_length = max(sequence.shape[0] for sequence in sequences)
        padded = [F.pad(sequence, (0, 0, 0, max_length - sequence.shape[0])) for sequence in sequences]
        masks = [
            torch.cat(
                (
                    torch.ones(sequence.shape[0], dtype=torch.bool, device=sequence.device),
                    torch.zeros(max_length - sequence.shape[0], dtype=torch.bool, device=sequence.device),
                )
            )
            for sequence in sequences
        ]
        return torch.stack(padded), torch.stack(masks), layouts

    def _gather_query_cell_memory(
        self,
        grid_output: GridEncoderOutput,
        refined_cells: torch.Tensor,
        fusion: TaskFusionOutput,
        batch_indices: torch.Tensor,
        grid_slots: torch.Tensor,
        grid_mask: torch.Tensor,
        role_ids: torch.Tensor,
        cell_mask_4d: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Build the explicit batched query-cell branch without unpacking memory.

        Query identity comes from the role contract, not from assuming that the
        final valid grid happens to be the query.  The returned cell states are
        the post-fusion refined states used by the packed-memory path.
        """
        batch_size, grid_count = grid_mask.shape
        query_grid_mask = grid_mask.bool() & role_ids.eq(ROLE_QUERY_INPUT)
        query_counts = query_grid_mask.sum(dim=1)
        if not torch.equal(query_counts, torch.ones_like(query_counts)):
            raise ValueError(
                "Each ARC encoder sample must contain exactly one valid query grid; "
                f"counts={query_counts.detach().cpu().tolist()}."
            )
        query_grid_slots = query_grid_mask.long().argmax(dim=1)
        flat_grid_lookup = torch.full(
            (batch_size, grid_count),
            -1,
            dtype=torch.long,
            device=grid_mask.device,
        )
        flat_grid_lookup[batch_indices, grid_slots] = torch.arange(
            batch_indices.numel(), device=grid_mask.device
        )
        batch_arange = torch.arange(batch_size, device=grid_mask.device)
        query_flat_grid_indices = flat_grid_lookup[batch_arange, query_grid_slots]
        if bool((query_flat_grid_indices < 0).any()):
            raise RuntimeError("A query grid did not map to an encoded grid.")

        full_query_mask = grid_output.cell_mask[query_flat_grid_indices].bool()
        valid_cell_counts = full_query_mask.sum(dim=1)
        if bool((valid_cell_counts < 1).any()):
            raise ValueError("Every ARC query grid must contain at least one valid cell.")
        query_length = int(valid_cell_counts.max().item())
        query_cell_mask = full_query_mask[:, :query_length]
        query_cell_memory = refined_cells[
            query_flat_grid_indices, :query_length
        ] * query_cell_mask.unsqueeze(-1).to(refined_cells.dtype)
        query_cell_flat_indices = grid_output.cell_flat_indices[
            query_flat_grid_indices, :query_length
        ].long()
        query_cell_flat_indices = query_cell_flat_indices.masked_fill(
            ~query_cell_mask, -1
        )
        valid_flat = query_cell_flat_indices[query_cell_mask]
        if bool((valid_flat < 0).any()) or bool(
            (valid_flat >= self.max_h * self.max_w).any()
        ):
            raise RuntimeError("Query cell flat indices are outside the ARC canvas.")
        query_cell_row_ids = torch.where(
            query_cell_mask,
            query_cell_flat_indices // self.max_w,
            torch.full_like(query_cell_flat_indices, self.max_h),
        )
        query_cell_col_ids = torch.where(
            query_cell_mask,
            query_cell_flat_indices % self.max_w,
            torch.full_like(query_cell_flat_indices, self.max_w),
        )
        query_spatial_mask = cell_mask_4d[batch_arange, query_grid_slots].bool()
        expected_counts = query_spatial_mask.flatten(1).sum(dim=1)
        if not torch.equal(valid_cell_counts, expected_counts):
            raise RuntimeError("Compacted query cells differ from the source cell mask.")
        query_original_size = torch.stack(
            (
                query_spatial_mask.any(dim=2).sum(dim=1),
                query_spatial_mask.any(dim=1).sum(dim=1),
            ),
            dim=1,
        ).long()
        return {
            "query_grid_summary": fusion.grid_summaries[
                batch_arange, query_grid_slots
            ],
            "query_grid_slots": query_grid_slots,
            "query_cell_memory": query_cell_memory,
            "query_cell_mask": query_cell_mask,
            "query_cell_row_ids": query_cell_row_ids,
            "query_cell_col_ids": query_cell_col_ids,
            "query_cell_flat_indices": query_cell_flat_indices,
            "query_original_size": query_original_size,
        }

    def _gather_pre_fusion_query_memory(
        self,
        grid_output: GridEncoderOutput,
        batch_indices: torch.Tensor,
        grid_slots: torch.Tensor,
        grid_mask: torch.Tensor,
        role_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return raw independently encoded query cells for relation reads."""

        batch_size, grid_count = grid_mask.shape
        flat_grid_lookup = torch.full(
            (batch_size, grid_count),
            -1,
            dtype=torch.long,
            device=grid_mask.device,
        )
        flat_grid_lookup[batch_indices, grid_slots] = torch.arange(
            batch_indices.numel(), device=grid_mask.device
        )
        query_sequences: list[torch.Tensor] = []
        query_flat_sequences: list[torch.Tensor] = []
        query_summaries: list[torch.Tensor] = []
        for batch_index in range(batch_size):
            valid_slots = torch.nonzero(
                grid_mask[batch_index].bool(), as_tuple=False
            ).flatten()
            query_slots = valid_slots[
                role_ids[batch_index, valid_slots].eq(ROLE_QUERY_INPUT)
            ]
            if query_slots.numel() != 1:
                raise ValueError(
                    "Pre-fusion relation memory requires exactly one query "
                    f"grid per sample; batch={batch_index}, "
                    f"count={query_slots.numel()}."
                )
            query_slot = int(query_slots.item())
            query_flat_grid = int(
                flat_grid_lookup[batch_index, query_slot].item()
            )
            if query_flat_grid < 0:
                raise RuntimeError("The raw query grid was not independently encoded.")
            query_valid = grid_output.cell_mask[query_flat_grid].bool()
            query_sequence = grid_output.cell_hidden_state[
                query_flat_grid, query_valid
            ]
            query_flats = grid_output.cell_flat_indices[
                query_flat_grid, query_valid
            ].long()
            if query_sequence.shape[0] < 1:
                raise ValueError("A pre-fusion relation query cannot be empty.")
            query_sequences.append(query_sequence)
            query_flat_sequences.append(query_flats)
            query_summaries.append(grid_output.grid_summary[query_flat_grid])

        max_query_length = max(value.shape[0] for value in query_sequences)
        forced_query = grid_output.cell_hidden_state.new_zeros(
            (batch_size, max_query_length, VIT5_SMALL_DIM)
        )
        forced_query_mask = torch.zeros(
            batch_size,
            max_query_length,
            dtype=torch.bool,
            device=grid_mask.device,
        )
        forced_query_flat = torch.full(
            (batch_size, max_query_length),
            -1,
            dtype=torch.long,
            device=grid_mask.device,
        )
        for batch_index in range(batch_size):
            query_length = query_sequences[batch_index].shape[0]
            forced_query[batch_index, :query_length] = query_sequences[
                batch_index
            ]
            forced_query_mask[batch_index, :query_length] = True
            forced_query_flat[batch_index, :query_length] = (
                query_flat_sequences[batch_index]
            )
        return {
            "forced_query_cell_memory": forced_query,
            "forced_query_cell_mask": forced_query_mask,
            "forced_query_cell_flat_indices": forced_query_flat,
            "forced_query_grid_summary": torch.stack(query_summaries),
        }

    def _gather_forced_paired_memory(
        self,
        grid_output: GridEncoderOutput,
        batch_indices: torch.Tensor,
        grid_slots: torch.Tensor,
        grid_mask: torch.Tensor,
        role_ids: torch.Tensor,
        example_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Build the diagnostic-only pre-fusion Q and explicitly paired K/V.

        The Forced diagnostic must not reuse ``refined_cells`` because those
        states have already attended fused task summaries.  Every tensor here
        therefore comes directly from the independently encoded per-grid ViT
        output.  Demonstration cells are paired by semantic example identity
        and exact flat coordinate; a shape-changing pair is rejected rather
        than silently converted into an incomplete relation memory.
        """

        batch_size, grid_count = grid_mask.shape
        flat_grid_lookup = torch.full(
            (batch_size, grid_count),
            -1,
            dtype=torch.long,
            device=grid_mask.device,
        )
        flat_grid_lookup[batch_indices, grid_slots] = torch.arange(
            batch_indices.numel(), device=grid_mask.device
        )

        query_branch = self._gather_pre_fusion_query_memory(
            grid_output,
            batch_indices,
            grid_slots,
            grid_mask,
            role_ids,
        )
        relation_inputs: list[torch.Tensor] = []
        relation_outputs: list[torch.Tensor] = []
        relation_examples: list[torch.Tensor] = []
        relation_flats: list[torch.Tensor] = []

        for batch_index in range(batch_size):
            valid_slots = torch.nonzero(
                grid_mask[batch_index].bool(), as_tuple=False
            ).flatten()
            by_example: dict[int, dict[int, int]] = {}
            for slot_tensor in valid_slots:
                slot = int(slot_tensor.item())
                role = int(role_ids[batch_index, slot].item())
                if role not in (ROLE_CONTEXT_INPUT, ROLE_CONTEXT_OUTPUT):
                    continue
                example = int(example_ids[batch_index, slot].item())
                role_slots = by_example.setdefault(example, {})
                if role in role_slots:
                    raise ValueError(
                        "Forced paired diagnostic requires one grid per "
                        f"(example_id, role); batch={batch_index}, "
                        f"example={example}, role={role}."
                    )
                role_slots[role] = slot

            input_parts: list[torch.Tensor] = []
            output_parts: list[torch.Tensor] = []
            example_parts: list[torch.Tensor] = []
            flat_parts: list[torch.Tensor] = []
            for example in sorted(by_example):
                role_slots = by_example[example]
                if set(role_slots) != {ROLE_CONTEXT_INPUT, ROLE_CONTEXT_OUTPUT}:
                    raise ValueError(
                        "Forced paired diagnostic found an unpaired "
                        f"demonstration grid; batch={batch_index}, "
                        f"example={example}, roles={sorted(role_slots)}."
                    )
                input_flat_grid = int(
                    flat_grid_lookup[
                        batch_index, role_slots[ROLE_CONTEXT_INPUT]
                    ].item()
                )
                output_flat_grid = int(
                    flat_grid_lookup[
                        batch_index, role_slots[ROLE_CONTEXT_OUTPUT]
                    ].item()
                )
                if input_flat_grid < 0 or output_flat_grid < 0:
                    raise RuntimeError("A paired demonstration grid was not encoded.")
                input_valid = grid_output.cell_mask[input_flat_grid].bool()
                output_valid = grid_output.cell_mask[output_flat_grid].bool()
                input_flats = grid_output.cell_flat_indices[
                    input_flat_grid, input_valid
                ].long()
                output_flats = grid_output.cell_flat_indices[
                    output_flat_grid, output_valid
                ].long()
                if not torch.equal(input_flats, output_flats):
                    raise ValueError(
                        "forced_paired_associative_v1 requires demonstration "
                        "input/output to have identical valid coordinates; "
                        f"batch={batch_index}, example={example}."
                    )
                input_part = grid_output.cell_hidden_state[
                    input_flat_grid, input_valid
                ]
                output_part = grid_output.cell_hidden_state[
                    output_flat_grid, output_valid
                ]
                input_parts.append(input_part)
                output_parts.append(output_part)
                flat_parts.append(input_flats)
                example_parts.append(
                    torch.full_like(input_flats, int(example))
                )

            if input_parts:
                relation_inputs.append(torch.cat(input_parts, dim=0))
                relation_outputs.append(torch.cat(output_parts, dim=0))
                relation_examples.append(torch.cat(example_parts, dim=0))
                relation_flats.append(torch.cat(flat_parts, dim=0))
            else:
                relation_inputs.append(
                    grid_output.cell_hidden_state.new_zeros((0, VIT5_SMALL_DIM))
                )
                relation_outputs.append(
                    grid_output.cell_hidden_state.new_zeros((0, VIT5_SMALL_DIM))
                )
                relation_examples.append(
                    torch.empty(0, dtype=torch.long, device=grid_mask.device)
                )
                relation_flats.append(
                    torch.empty(0, dtype=torch.long, device=grid_mask.device)
                )

        # Keep one physical all-zero slot when C removes all demonstrations.
        # The decoder skips all-empty rows instead of feeding an all-masked
        # key sequence to MultiheadAttention.
        max_relation_length = max(
            1, max(value.shape[0] for value in relation_inputs)
        )
        dtype = grid_output.cell_hidden_state.dtype
        device = grid_output.cell_hidden_state.device
        paired_input = torch.zeros(
            batch_size, max_relation_length, VIT5_SMALL_DIM,
            dtype=dtype, device=device,
        )
        paired_output = torch.zeros_like(paired_input)
        paired_mask = torch.zeros(
            batch_size, max_relation_length, dtype=torch.bool, device=device
        )
        paired_examples = torch.full(
            (batch_size, max_relation_length), -1,
            dtype=torch.long, device=device,
        )
        paired_flats = torch.full_like(paired_examples, -1)
        for batch_index in range(batch_size):
            relation_length = relation_inputs[batch_index].shape[0]
            if relation_length:
                paired_input[batch_index, :relation_length] = relation_inputs[
                    batch_index
                ]
                paired_output[batch_index, :relation_length] = relation_outputs[
                    batch_index
                ]
                paired_mask[batch_index, :relation_length] = True
                paired_examples[batch_index, :relation_length] = relation_examples[
                    batch_index
                ]
                paired_flats[batch_index, :relation_length] = relation_flats[
                    batch_index
                ]
        return {
            **query_branch,
            "paired_demo_input_memory": paired_input,
            "paired_demo_output_memory": paired_output,
            "paired_demo_mask": paired_mask,
            "paired_demo_example_ids": paired_examples,
            "paired_demo_flat_indices": paired_flats,
        }

    def _gather_relational_memory_v2(
        self,
        grid_output: GridEncoderOutput,
        batch_indices: torch.Tensor,
        grid_slots: torch.Tensor,
        grid_mask: torch.Tensor,
        role_ids: torch.Tensor,
        example_ids: torch.Tensor,
        cell_mask_4d: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Gather unequal-shape demonstration sets without inventing cell pairs.

        ``paired_associative_v1`` deliberately zips an input cell and output
        cell at the same flat coordinate.  That contract is useful for its
        diagnostic, but cannot represent crop, padding, scaling or any other
        transformation whose two grids have different coordinate sets.

        Relational memory v2 instead returns two independently padded token
        sets.  Semantic ``example_id`` metadata says which input grid belongs
        to which output grid; no input/output cell correspondence is assumed.
        Already-encoded cells are packed canonically by
        ``(example_id, flat_index)``.  This makes the set contract independent
        of source-slot traversal order; the hidden states themselves may still
        encode the model's learned grid-identity embedding.

        All hidden states are taken directly from the independent per-grid
        encoder, before task fusion.  Query-output targets are not part of the
        encoder batch and this gather accepts only context input/output roles.
        """

        batch_size, grid_count = grid_mask.shape
        if tuple(role_ids.shape) != (batch_size, grid_count):
            raise ValueError("Relational memory role_ids must match grid_mask.")
        if tuple(example_ids.shape) != (batch_size, grid_count):
            raise ValueError("Relational memory example_ids must match grid_mask.")
        if tuple(cell_mask_4d.shape[:2]) != (batch_size, grid_count):
            raise ValueError("Relational memory cell masks must match grid_mask.")
        if tuple(cell_mask_4d.shape[-2:]) != (self.max_h, self.max_w):
            raise ValueError(
                "Relational memory cell-mask canvas must match encoder bounds; "
                f"got={tuple(cell_mask_4d.shape[-2:])}, "
                f"expected={(self.max_h, self.max_w)}."
            )

        query_branch = self._gather_pre_fusion_query_memory(
            grid_output,
            batch_indices,
            grid_slots,
            grid_mask,
            role_ids,
        )

        flat_grid_lookup = torch.full(
            (batch_size, grid_count),
            -1,
            dtype=torch.long,
            device=grid_mask.device,
        )
        flat_grid_lookup[batch_indices, grid_slots] = torch.arange(
            batch_indices.numel(), device=grid_mask.device
        )

        side_sequences: dict[str, list[dict[str, torch.Tensor]]] = {
            "input": [],
            "output": [],
        }
        role_for_side = {
            "input": ROLE_CONTEXT_INPUT,
            "output": ROLE_CONTEXT_OUTPUT,
        }
        hidden_template = grid_output.cell_hidden_state

        for batch_index in range(batch_size):
            valid_slots = torch.nonzero(
                grid_mask[batch_index].bool(), as_tuple=False
            ).flatten()
            by_example: dict[int, dict[int, int]] = {}
            for slot_tensor in valid_slots:
                slot = int(slot_tensor.item())
                role = int(role_ids[batch_index, slot].item())
                if role not in (ROLE_CONTEXT_INPUT, ROLE_CONTEXT_OUTPUT):
                    continue
                example = int(example_ids[batch_index, slot].item())
                if example < 0:
                    raise ValueError(
                        "Relational memory requires non-negative context "
                        f"example IDs; batch={batch_index}, slot={slot}."
                    )
                role_slots = by_example.setdefault(example, {})
                if role in role_slots:
                    raise ValueError(
                        "Relational memory requires one grid per "
                        f"(example_id, role); batch={batch_index}, "
                        f"example={example}, role={role}."
                    )
                role_slots[role] = slot

            side_parts: dict[str, dict[str, list[torch.Tensor]]] = {
                side: {
                    "memory": [],
                    "example_ids": [],
                    "flat_indices": [],
                    "row_ids": [],
                    "col_ids": [],
                    "original_sizes": [],
                    "normalized_geometry": [],
                }
                for side in role_for_side
            }
            for example in sorted(by_example):
                role_slots = by_example[example]
                if set(role_slots) != {ROLE_CONTEXT_INPUT, ROLE_CONTEXT_OUTPUT}:
                    raise ValueError(
                        "Relational memory found an unpaired demonstration "
                        f"grid; batch={batch_index}, example={example}, "
                        f"roles={sorted(role_slots)}."
                    )
                for side, role in role_for_side.items():
                    slot = role_slots[role]
                    flat_grid = int(
                        flat_grid_lookup[batch_index, slot].item()
                    )
                    if flat_grid < 0:
                        raise RuntimeError(
                            "A relational demonstration grid was not encoded."
                        )
                    valid = grid_output.cell_mask[flat_grid].bool()
                    if not bool(valid.any()):
                        raise ValueError(
                            "Relational demonstrations cannot contain empty grids."
                        )
                    flats = grid_output.cell_flat_indices[
                        flat_grid, valid
                    ].long()
                    if bool((flats < 0).any()) or bool(
                        (flats >= self.max_h * self.max_w).any()
                    ):
                        raise RuntimeError(
                            "Relational demonstration flat indices are outside "
                            "the ARC canvas."
                        )
                    order = torch.argsort(flats)
                    flats = flats.index_select(0, order)
                    if flats.numel() > 1 and bool(
                        flats[1:].eq(flats[:-1]).any()
                    ):
                        raise RuntimeError(
                            "Relational demonstration flat indices must be unique."
                        )
                    memory = grid_output.cell_hidden_state[
                        flat_grid, valid
                    ].index_select(0, order)

                    spatial_mask = cell_mask_4d[batch_index, slot].bool()
                    height = int(spatial_mask.any(dim=1).sum().item())
                    width = int(spatial_mask.any(dim=0).sum().item())
                    if height < 1 or width < 1:
                        raise ValueError(
                            "Relational demonstrations cannot contain empty grids."
                        )
                    expected_flats = (
                        torch.arange(height, device=flats.device)[:, None]
                        * self.max_w
                        + torch.arange(width, device=flats.device)[None, :]
                    ).reshape(-1)
                    if not torch.equal(flats, expected_flats):
                        raise RuntimeError(
                            "Relational demonstration cells differ from the "
                            "source rectangular grid mask."
                        )
                    rows = flats // self.max_w
                    cols = flats % self.max_w
                    geometry_dtype = memory.dtype
                    normalized_rows = rows.to(geometry_dtype) / float(
                        max(height - 1, 1)
                    )
                    normalized_cols = cols.to(geometry_dtype) / float(
                        max(width - 1, 1)
                    )
                    normalized_geometry = torch.stack(
                        (
                            normalized_rows,
                            normalized_cols,
                            torch.full_like(
                                normalized_rows, float(height) / self.max_h
                            ),
                            torch.full_like(
                                normalized_cols, float(width) / self.max_w
                            ),
                        ),
                        dim=-1,
                    )
                    length = int(flats.numel())
                    side_parts[side]["memory"].append(memory)
                    side_parts[side]["example_ids"].append(
                        torch.full(
                            (length,),
                            example,
                            dtype=torch.long,
                            device=flats.device,
                        )
                    )
                    side_parts[side]["flat_indices"].append(flats)
                    side_parts[side]["row_ids"].append(rows)
                    side_parts[side]["col_ids"].append(cols)
                    side_parts[side]["original_sizes"].append(
                        torch.tensor(
                            (height, width),
                            dtype=torch.long,
                            device=flats.device,
                        ).expand(length, -1)
                    )
                    side_parts[side]["normalized_geometry"].append(
                        normalized_geometry
                    )

            for side in role_for_side:
                parts = side_parts[side]
                if parts["memory"]:
                    side_sequences[side].append(
                        {
                            name: torch.cat(values, dim=0)
                            for name, values in parts.items()
                        }
                    )
                else:
                    side_sequences[side].append(
                        {
                            "memory": hidden_template.new_zeros(
                                (0, VIT5_SMALL_DIM)
                            ),
                            "example_ids": torch.empty(
                                0, dtype=torch.long, device=grid_mask.device
                            ),
                            "flat_indices": torch.empty(
                                0, dtype=torch.long, device=grid_mask.device
                            ),
                            "row_ids": torch.empty(
                                0, dtype=torch.long, device=grid_mask.device
                            ),
                            "col_ids": torch.empty(
                                0, dtype=torch.long, device=grid_mask.device
                            ),
                            "original_sizes": torch.empty(
                                0, 2, dtype=torch.long, device=grid_mask.device
                            ),
                            "normalized_geometry": hidden_template.new_zeros(
                                (0, 4)
                            ),
                        }
                    )

        result: dict[str, torch.Tensor] = dict(query_branch)
        for side in ("input", "output"):
            sequences = side_sequences[side]
            # Keep one physical zero slot for a query-only C condition.  Its
            # false mask makes the logical relation set exactly empty.
            max_length = max(
                1, max(sequence["memory"].shape[0] for sequence in sequences)
            )
            memory = hidden_template.new_zeros(
                (batch_size, max_length, VIT5_SMALL_DIM)
            )
            mask = torch.zeros(
                batch_size,
                max_length,
                dtype=torch.bool,
                device=grid_mask.device,
            )
            long_metadata = {
                name: torch.full(
                    (batch_size, max_length),
                    -1,
                    dtype=torch.long,
                    device=grid_mask.device,
                )
                for name in (
                    "example_ids",
                    "flat_indices",
                    "row_ids",
                    "col_ids",
                )
            }
            original_sizes = torch.zeros(
                batch_size,
                max_length,
                2,
                dtype=torch.long,
                device=grid_mask.device,
            )
            normalized_geometry = hidden_template.new_zeros(
                (batch_size, max_length, 4)
            )
            for batch_index, sequence in enumerate(sequences):
                length = int(sequence["memory"].shape[0])
                if length == 0:
                    continue
                memory[batch_index, :length] = sequence["memory"]
                mask[batch_index, :length] = True
                for name, values in long_metadata.items():
                    values[batch_index, :length] = sequence[name]
                original_sizes[batch_index, :length] = sequence[
                    "original_sizes"
                ]
                normalized_geometry[batch_index, :length] = sequence[
                    "normalized_geometry"
                ]
            prefix = f"relational_demo_{side}_"
            result.update(
                {
                    f"{prefix}memory": memory,
                    f"{prefix}mask": mask,
                    **{
                        f"{prefix}{name}": value
                        for name, value in long_metadata.items()
                    },
                    f"{prefix}original_sizes": original_sizes,
                    f"{prefix}geometry": normalized_geometry,
                }
            )
        return result

    def forward(
        self,
        encoder_color_ids: torch.Tensor,
        encoder_cell_mask: torch.Tensor,
        encoder_row_ids: torch.Tensor,
        encoder_col_ids: torch.Tensor,
        encoder_grid_mask: torch.Tensor,
        encoder_role_ids: torch.Tensor,
        encoder_example_ids: torch.Tensor,
        encoder_grid_ids: torch.Tensor,
        encoder_object_ids: torch.Tensor | None = None,
    ) -> ViT5ArcEncoderOutput:
        if encoder_object_ids is None:
            encoder_object_ids = torch.zeros_like(encoder_color_ids)
        flattened, batch_indices, grid_slots = self._flatten_valid_grids(
            encoder_color_ids,
            encoder_cell_mask,
            encoder_row_ids,
            encoder_col_ids,
            encoder_object_ids,
            encoder_grid_mask,
            encoder_role_ids,
            encoder_example_ids,
            encoder_grid_ids,
        )
        grid_output = self.grid_encoder(
            **flattened,
            microbatch=self.grid_encoder_microbatch,
        )
        batch_size, grid_count = encoder_grid_mask.shape
        summary_tensor = torch.zeros(
            batch_size,
            grid_count,
            VIT5_SMALL_DIM,
            dtype=grid_output.grid_summary.dtype,
            device=grid_output.grid_summary.device,
        )
        summary_tensor[batch_indices, grid_slots] = grid_output.grid_summary
        fusion = self.task_fusion(summary_tensor, encoder_grid_mask)
        refined_cells = self._refine_cells(
            grid_output.cell_hidden_state,
            grid_output.cell_mask,
            batch_indices,
            grid_slots,
            fusion.grid_summaries,
            encoder_grid_mask,
        )
        memory, memory_mask, layout = self._pack_memory(
            grid_output,
            refined_cells,
            fusion,
            encoder_grid_mask,
            batch_indices,
            grid_slots,
            encoder_role_ids,
            encoder_example_ids,
            encoder_grid_ids,
            encoder_cell_mask,
        )
        query_branch = self._gather_query_cell_memory(
            grid_output,
            refined_cells,
            fusion,
            batch_indices,
            grid_slots,
            encoder_grid_mask,
            encoder_role_ids,
            encoder_cell_mask,
        )
        forced_pair_branch: dict[str, torch.Tensor] = {}
        if self.enable_forced_paired_diagnostic:
            forced_pair_branch = self._gather_forced_paired_memory(
                grid_output,
                batch_indices,
                grid_slots,
                encoder_grid_mask,
                encoder_role_ids,
                encoder_example_ids,
            )
        relational_branch: dict[str, torch.Tensor] = {}
        if self.enable_relational_memory_v2:
            relational_branch = self._gather_relational_memory_v2(
                grid_output,
                batch_indices,
                grid_slots,
                encoder_grid_mask,
                encoder_role_ids,
                encoder_example_ids,
                encoder_cell_mask,
            )
            # Both experimental branches expose the same raw-query contract.
            # Avoid duplicate dataclass keywords if a diagnostic explicitly
            # enables both; the independently gathered values are identical.
            if forced_pair_branch:
                for name in (
                    "forced_query_cell_memory",
                    "forced_query_cell_mask",
                    "forced_query_cell_flat_indices",
                    "forced_query_grid_summary",
                ):
                    relational_branch.pop(name)
        # The context branch retains summaries, demonstrations and registers,
        # but query cell tokens cannot compete with the dedicated query-cell
        # attention branch in the same softmax.
        context_memory_mask = memory_mask.clone()
        for batch_index, sample_layout in enumerate(layout):
            query_slot = int(query_branch["query_grid_slots"][batch_index].item())
            matches = [
                item
                for item in sample_layout["grids"]
                if int(item["grid_slot"]) == query_slot
            ]
            if len(matches) != 1:
                raise RuntimeError("Packed memory must contain exactly one query grid.")
            cell_start, cell_end = matches[0]["cell_span"]
            context_memory_mask[batch_index, int(cell_start) : int(cell_end)] = False
        if bool((context_memory_mask.sum(dim=1) < 1).any()):
            raise RuntimeError("The Dense context branch cannot be empty.")
        return ViT5ArcEncoderOutput(
            last_hidden_state=memory,
            attention_mask=memory_mask,
            context_attention_mask=context_memory_mask,
            memory_layout=layout,
            task_summary=fusion.task_summary,
            grid_summaries=fusion.grid_summaries,
            grid_mask=encoder_grid_mask,
            **query_branch,
            **forced_pair_branch,
            **relational_branch,
            encoded_grid_count=int(batch_indices.numel()),
            grid_encoder_microbatch=self.grid_encoder_microbatch,
        )
