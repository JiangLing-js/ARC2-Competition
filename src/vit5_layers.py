"""ViT-5 Small layers adapted for discrete, masked ARC grids.

The parameter names and shapes intentionally mirror the official ViT-5
implementation.  The only behavioral extensions are coordinate-driven rotary
embeddings, key-padding masks, and optional PyTorch SDPA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent feature pairs as ``(-x_1, x_0)``."""
    paired = x.reshape(*x.shape[:-1], -1, 2)
    first, second = paired.unbind(dim=-1)
    return torch.stack((-second, first), dim=-1).flatten(-2)


class RMSNorm(nn.Module):
    """Bias-free RMSNorm matching ``ViT-5-main/models_vit5.py``."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        variance = normalized.pow(2).mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * normalized.to(input_dtype)


class ArcRotaryEmbedding(nn.Module):
    """Coordinate-aware half-axis rotary embedding.

    ``dim`` has the same meaning as the official ``VisionRotaryEmbedding``:
    for a 64-dimensional attention head, ``dim=32`` and the generated row and
    column features concatenate to 64 dimensions.
    """

    def __init__(
        self,
        dim: int,
        pt_seq_len: int = 14,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError("Rotary axis dimension must be even.")
        freqs = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[: dim // 2] / dim)
        )
        self.pt_seq_len = int(pt_seq_len)
        self.register_buffer("freqs", freqs)

    def cos_sin(
        self,
        row_ids: torch.Tensor,
        col_ids: torch.Tensor,
        coordinate_scale: float = 1.0,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return row+column rotary cos/sin for integer or float coordinates."""
        if row_ids.shape != col_ids.shape:
            raise ValueError("row_ids and col_ids must have matching shapes.")
        target_dtype = dtype or self.freqs.dtype
        freqs = self.freqs.to(device=row_ids.device)
        rows = row_ids.float() * float(coordinate_scale)
        cols = col_ids.float() * float(coordinate_scale)
        row_angles = torch.einsum("...n,f->...nf", rows, freqs).repeat_interleave(2, dim=-1)
        col_angles = torch.einsum("...n,f->...nf", cols, freqs).repeat_interleave(2, dim=-1)
        angles = torch.cat((row_angles, col_angles), dim=-1)
        return angles.cos().to(target_dtype), angles.sin().to(target_dtype)


@dataclass
class RotaryLayout:
    """Masks and positions controlling spatial and register RoPE."""

    row_ids: torch.Tensor
    col_ids: torch.Tensor
    spatial_mask: torch.Tensor
    register_mask: torch.Tensor
    register_position_ids: torch.Tensor
    coordinate_scale: float = 1.0


class DropPath(nn.Module):
    """Per-sample stochastic depth without a timm dependency."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ViT5MLP(nn.Module):
    """Biased GeLU MLP matching timm's ViT MLP key layout."""

    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ViT5Attention(nn.Module):
    """ViT-5 attention with fused bias-free QKV, QK-Norm and masked RoPE."""

    def __init__(
        self,
        dim: int = 384,
        num_heads: int = 6,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope_size: int = 14,
        rope_reg_size: int = 2,
        num_registers: int = 4,
        reg_theta: float = 100.0,
        qk_norm: bool = True,
        use_sdpa: bool = True,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads.")
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.scale = float(qk_scale or self.head_dim**-0.5)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)
        self.num_registers = int(num_registers)
        self.rope = (
            ArcRotaryEmbedding(self.head_dim // 2, pt_seq_len=rope_size, theta=10000.0)
            if rope_size > 0
            else None
        )
        self.rope_reg = (
            ArcRotaryEmbedding(self.head_dim // 2, pt_seq_len=rope_reg_size, theta=reg_theta)
            if rope_reg_size > 0
            else None
        )
        self.qk_norm = bool(qk_norm)
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.use_sdpa = bool(use_sdpa and hasattr(F, "scaled_dot_product_attention"))
        self.last_backend = "uninitialized"

    def _apply_rotary(
        self,
        tensor: torch.Tensor,
        layout: RotaryLayout | None,
    ) -> torch.Tensor:
        if layout is None:
            return tensor
        batch, length = tensor.shape[:2]
        expected = (batch, length)
        for name, value in (
            ("row_ids", layout.row_ids),
            ("col_ids", layout.col_ids),
            ("spatial_mask", layout.spatial_mask),
            ("register_mask", layout.register_mask),
            ("register_position_ids", layout.register_position_ids),
        ):
            if tuple(value.shape) != expected:
                raise ValueError(f"{name} shape {tuple(value.shape)} != {expected}.")

        cos = torch.ones(
            batch, length, self.head_dim, dtype=tensor.dtype, device=tensor.device
        )
        sin = torch.zeros_like(cos)
        spatial_mask = layout.spatial_mask.bool()
        if self.rope is not None and spatial_mask.any():
            spatial_cos, spatial_sin = self.rope.cos_sin(
                layout.row_ids,
                layout.col_ids,
                coordinate_scale=layout.coordinate_scale,
                dtype=tensor.dtype,
            )
            cos = torch.where(spatial_mask.unsqueeze(-1), spatial_cos, cos)
            sin = torch.where(spatial_mask.unsqueeze(-1), spatial_sin, sin)

        register_mask = layout.register_mask.bool()
        if self.rope_reg is not None and register_mask.any():
            register_ids = layout.register_position_ids.long().clamp_min(0)
            side = int(round(self.num_registers**0.5))
            if side * side != self.num_registers:
                raise ValueError("num_registers must be a square number for register RoPE.")
            register_rows = register_ids // side
            register_cols = register_ids % side
            register_cos, register_sin = self.rope_reg.cos_sin(
                register_rows,
                register_cols,
                coordinate_scale=1.0,
                dtype=tensor.dtype,
            )
            cos = torch.where(register_mask.unsqueeze(-1), register_cos, cos)
            sin = torch.where(register_mask.unsqueeze(-1), register_sin, sin)
        return tensor * cos.unsqueeze(2) + rotate_half(tensor) * sin.unsqueeze(2)

    def _manual_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        q = q.transpose(1, 2) * self.scale
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = q @ k.transpose(-2, -1)
        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :].bool()
            scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1)
        weights = self.attn_drop(weights)
        return (weights @ v).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        rotary_layout: RotaryLayout | None = None,
        force_manual: bool = False,
    ) -> torch.Tensor:
        batch, length, channels = x.shape
        if attention_mask is not None and tuple(attention_mask.shape) != (batch, length):
            raise ValueError("attention_mask must have shape [B,N].")
        qkv = self.qkv(x).reshape(
            batch, length, 3, self.num_heads, channels // self.num_heads
        )
        q, k, v = qkv.unbind(dim=2)
        if self.qk_norm:
            q_dtype = q.dtype
            q = self.q_norm(q).to(q_dtype)
            k = self.k_norm(k).to(q_dtype)
        q = self._apply_rotary(q, rotary_layout)
        k = self._apply_rotary(k, rotary_layout)

        use_sdpa = self.use_sdpa and not force_manual
        if use_sdpa:
            q_t = q.transpose(1, 2)
            default_scale = self.head_dim**-0.5
            if self.scale != default_scale:
                q_t = q_t * (self.scale / default_scale)
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)
            sdpa_mask = None
            if attention_mask is not None:
                sdpa_mask = attention_mask[:, None, None, :].bool()
            dropout_p = self.attn_drop.p if self.training else 0.0
            output = F.scaled_dot_product_attention(
                q_t,
                k_t,
                v_t,
                attn_mask=sdpa_mask,
                dropout_p=dropout_p,
            ).transpose(1, 2)
            self.last_backend = "sdpa"
        else:
            output = self._manual_attention(q, k, v, attention_mask)
            self.last_backend = "manual"
        output = output.reshape(batch, length, channels)
        output = self.proj(output)
        return self.proj_drop(output)


class ViT5Block(nn.Module):
    """One checkpoint-compatible ViT-5 transformer block."""

    def __init__(
        self,
        dim: int = 384,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: float = 1e-4,
        rope_size: int = 14,
        rope_reg_size: int = 2,
        reg_theta: float = 100.0,
        num_registers: int = 4,
        qk_norm: bool = True,
        layer_scale: bool = True,
        use_sdpa: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=1e-6)
        self.attn = ViT5Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            rope_size=rope_size,
            rope_reg_size=rope_reg_size,
            num_registers=num_registers,
            reg_theta=reg_theta,
            qk_norm=qk_norm,
            use_sdpa=use_sdpa,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = RMSNorm(dim, eps=1e-6)
        self.mlp = ViT5MLP(dim, int(dim * mlp_ratio), drop=drop)
        self.layer_scale = bool(layer_scale)
        if self.layer_scale:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        rotary_layout: RotaryLayout | None = None,
        force_manual: bool = False,
    ) -> torch.Tensor:
        attention = self.attn(
            self.norm1(x),
            attention_mask=attention_mask,
            rotary_layout=rotary_layout,
            force_manual=force_manual,
        )
        if self.layer_scale:
            x = x + self.drop_path(self.gamma_1 * attention)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(attention)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(x.dtype)
        return x


def initialize_vit5_module(module: nn.Module, std: float = 0.02) -> None:
    """Initialize new Linear/Embedding modules with ViT-style defaults."""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=std)


def build_full_grid_rotary_layout(
    batch_size: int,
    grid_size: int,
    num_registers: int = 4,
    device: torch.device | str | None = None,
) -> RotaryLayout:
    """Build the official ``CLS + square grid + registers`` layout for tests."""
    target_device = torch.device(device) if device is not None else None
    rows = torch.arange(grid_size, device=target_device).repeat_interleave(grid_size)
    cols = torch.arange(grid_size, device=target_device).repeat(grid_size)
    spatial_count = grid_size * grid_size
    length = 1 + spatial_count + num_registers
    row_ids = torch.zeros(batch_size, length, dtype=torch.long, device=target_device)
    col_ids = torch.zeros_like(row_ids)
    row_ids[:, 1 : 1 + spatial_count] = rows
    col_ids[:, 1 : 1 + spatial_count] = cols
    spatial_mask = torch.zeros(batch_size, length, dtype=torch.bool, device=target_device)
    spatial_mask[:, 1 : 1 + spatial_count] = True
    register_mask = torch.zeros_like(spatial_mask)
    register_mask[:, -num_registers:] = True
    register_position_ids = torch.zeros_like(row_ids)
    register_position_ids[:, -num_registers:] = torch.arange(
        num_registers, device=target_device
    )
    return RotaryLayout(
        row_ids=row_ids,
        col_ids=col_ids,
        spatial_mask=spatial_mask,
        register_mask=register_mask,
        register_position_ids=register_position_ids,
        coordinate_scale=1.0,
    )
