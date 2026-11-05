"""Project-local 2D relative position bias utilities for ViTARC-style batches."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


def make_alibi_slopes(num_heads: int) -> torch.Tensor:
    """Return deterministic positive ALiBi-style slopes for attention heads.

    The exact ALiBi paper schedule is not required for this smoke layer. We use
    a simple decreasing geometric schedule so nearby positions receive bias
    close to zero and distant positions receive more negative bias.
    """
    if num_heads < 1:
        raise ValueError("num_heads must be at least 1.")
    return torch.tensor([1.0 / (2 ** head_idx) for head_idx in range(num_heads)], dtype=torch.float32)


def _validate_position_shapes(
    row_ids: torch.Tensor,
    col_ids: torch.Tensor,
    position_mask: torch.Tensor | None = None,
) -> None:
    if row_ids.ndim != 2 or col_ids.ndim != 2:
        raise ValueError("row_ids and col_ids must have shape [B, L].")
    if row_ids.shape != col_ids.shape:
        raise ValueError(
            f"row_ids shape {tuple(row_ids.shape)} must match col_ids shape {tuple(col_ids.shape)}."
        )
    if position_mask is not None and position_mask.shape != row_ids.shape:
        raise ValueError(
            f"position_mask shape {tuple(position_mask.shape)} must match row_ids shape {tuple(row_ids.shape)}."
        )


def pairwise_2d_distances(
    row_ids: torch.Tensor,
    col_ids: torch.Tensor,
    position_mask: torch.Tensor | None = None,
    special_distance: int = 0,
    metric: str = "manhattan",
) -> torch.Tensor:
    """Compute pairwise 2D distances between row/column id sequences.

    Args:
        row_ids: Tensor of row ids with shape ``[B, L]``.
        col_ids: Tensor of column ids with shape ``[B, L]``.
        position_mask: Optional mask with shape ``[B, L]`` where truthy values
            mark real grid-body positions. Pairs touching a masked-out token use
            ``special_distance``.
        special_distance: Distance assigned to special/padding token pairs.
        metric: Either ``"manhattan"`` or ``"chebyshev"``.

    Returns:
        Tensor of pairwise distances with shape ``[B, L, L]``.
    """
    _validate_position_shapes(row_ids, col_ids, position_mask)

    row_delta = (row_ids.unsqueeze(2) - row_ids.unsqueeze(1)).abs()
    col_delta = (col_ids.unsqueeze(2) - col_ids.unsqueeze(1)).abs()
    if metric == "manhattan":
        distances = row_delta + col_delta
    elif metric == "chebyshev":
        distances = torch.maximum(row_delta, col_delta)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

    if position_mask is not None:
        mask = position_mask.to(dtype=torch.bool)
        pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
        distances = distances.clone()
        distances = distances.masked_fill(~pair_mask, int(special_distance))

    return distances


def build_2d_alibi_bias(
    row_ids: torch.Tensor,
    col_ids: torch.Tensor,
    num_heads: int,
    position_mask: torch.Tensor | None = None,
    metric: str = "manhattan",
    slopes: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a 2D ALiBi-style bias tensor shaped ``[B, heads, L, L]``."""
    distances = pairwise_2d_distances(
        row_ids,
        col_ids,
        position_mask=position_mask,
        special_distance=0,
        metric=metric,
    ).to(dtype=torch.float32)

    if slopes is None:
        slopes = make_alibi_slopes(num_heads)
    if slopes.ndim != 1 or slopes.shape[0] != num_heads:
        raise ValueError(
            f"slopes must have shape [{num_heads}], got {tuple(slopes.shape)}."
        )

    slopes = slopes.to(device=row_ids.device, dtype=distances.dtype)
    return -slopes.view(1, num_heads, 1, 1) * distances.unsqueeze(1)


def apply_2d_rpe_bias(
    attention_scores: torch.Tensor,
    rpe_bias: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    mask_value: float = -1e9,
) -> torch.Tensor:
    """Add 2D-RPE bias to attention scores and optionally mask key positions."""
    if attention_scores.ndim != 4 or rpe_bias.ndim != 4:
        raise ValueError("attention_scores and rpe_bias must have shape [B, H, L, L].")
    if attention_scores.shape != rpe_bias.shape:
        raise ValueError(
            f"attention_scores shape {tuple(attention_scores.shape)} must match "
            f"rpe_bias shape {tuple(rpe_bias.shape)}."
        )

    result = attention_scores + rpe_bias.to(device=attention_scores.device, dtype=attention_scores.dtype)
    if attention_mask is not None:
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [B, L].")
        batch, _, _, seq_len = result.shape
        if attention_mask.shape != (batch, seq_len):
            raise ValueError(
                f"attention_mask shape {tuple(attention_mask.shape)} must be {(batch, seq_len)}."
            )
        key_mask = attention_mask.to(device=result.device, dtype=torch.bool).view(batch, 1, 1, seq_len)
        result = result + (~key_mask).to(dtype=result.dtype) * mask_value
    return result


class TinySelfAttentionWith2DRPE(nn.Module):
    """A tiny standalone self-attention block that adds 2D ALiBi-style bias."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        metric: str = "manhattan",
    ) -> None:
        super().__init__()
        if num_heads < 1:
            raise ValueError("num_heads must be at least 1.")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if metric not in {"manhattan", "chebyshev"}:
            raise ValueError(f"Unknown distance metric: {metric}")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.metric = metric
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def _split_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        return (
            hidden_states.view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def _merge_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, _, seq_len, _ = hidden_states.shape
        return (
            hidden_states.transpose(1, 2)
            .contiguous()
            .view(batch, seq_len, self.d_model)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        row_ids: torch.Tensor,
        col_ids: torch.Tensor,
        position_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run self-attention with project-local 2D relative position bias."""
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [B, L, D].")
        batch, seq_len, d_model = hidden_states.shape
        if d_model != self.d_model:
            raise ValueError(f"Expected hidden size {self.d_model}, got {d_model}.")
        if row_ids.shape != (batch, seq_len) or col_ids.shape != (batch, seq_len):
            raise ValueError("row_ids and col_ids must match hidden_states [B, L].")

        query = self._split_heads(self.q_proj(hidden_states))
        key = self._split_heads(self.k_proj(hidden_states))
        value = self._split_heads(self.v_proj(hidden_states))

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        rpe_bias = build_2d_alibi_bias(
            row_ids,
            col_ids,
            num_heads=self.num_heads,
            position_mask=position_mask,
            metric=self.metric,
        ).to(dtype=attention_scores.dtype)
        biased_scores = apply_2d_rpe_bias(
            attention_scores,
            rpe_bias,
            attention_mask=attention_mask,
        )
        attention_probs = torch.softmax(biased_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        output = self.out_proj(self._merge_heads(torch.matmul(attention_probs, value)))

        info = {
            "attention_probs_shape": tuple(attention_probs.shape),
            "rpe_bias_shape": tuple(rpe_bias.shape),
            "distance_metric": self.metric,
            "rpe_bias_min": float(rpe_bias.min().item()),
            "rpe_bias_max": float(rpe_bias.max().item()),
        }
        return output, info
