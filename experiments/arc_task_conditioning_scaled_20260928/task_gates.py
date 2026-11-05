"""Differentiable task-conditioned IA3-style projections for original Qwen3.

Plain Transformers training uses explicit gates. Inference exports one task's
fixed gates into ordinary BF16 weights before Unsloth attaches its LoRA.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import weakref

import torch
from torch import nn
import torch.nn.functional as F


CODE_DIM = 64
INIT_SEED = 20260925
KINDS = ("k_proj", "v_proj", "down_proj")


class GatedLinear(nn.Module):
    def __init__(self, base: nn.Linear, bank: "TaskGateBank", layer: int, kind: str):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"{kind} is not an ordinary Linear: {type(base)}")
        self.base = base
        self.layer = layer
        self.kind = kind
        object.__setattr__(self, "_bank_ref", weakref.ref(bank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bank = self._bank_ref()
        if bank is None or bank.active_code is None:
            return self.base(x)
        gate = bank.gate(self.layer, self.kind, bank.active_code)
        if self.kind == "down_proj":
            return self.base(x * gate.to(dtype=x.dtype))
        output = self.base(x)
        return output * gate.to(dtype=output.dtype)


class TaskGateBank(nn.Module):
    def __init__(self, model: nn.Module, seed: int = INIT_SEED):
        super().__init__()
        layers = list(model.model.layers)
        if not layers:
            raise ValueError("Qwen model has no decoder layers")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.maps = nn.ParameterDict()
        self.dimensions = {}
        object.__setattr__(self, "active_code", None)
        for index, layer in enumerate(layers):
            modules = {"k_proj": layer.self_attn.k_proj,
                       "v_proj": layer.self_attn.v_proj,
                       "down_proj": layer.mlp.down_proj}
            for kind, linear in modules.items():
                if not isinstance(linear, nn.Linear):
                    raise TypeError(f"Unexpected projection {index}.{kind}: {type(linear)}")
                channels = linear.in_features if kind == "down_proj" else linear.out_features
                key = f"L{index}_{kind}"
                self.maps[key] = nn.Parameter(torch.randn(channels, CODE_DIM, generator=generator,
                                                          dtype=torch.float32) * 0.02)
                self.dimensions[key] = channels
        self.layer_count = len(layers)

    def gate(self, layer: int, kind: str, code: torch.Tensor) -> torch.Tensor:
        mapping = self.maps[f"L{layer}_{kind}"]
        if code.dtype != torch.float32 or code.shape != (CODE_DIM,):
            raise ValueError("Task code must be a 64-dimensional FP32 vector")
        return torch.exp(0.5 * torch.tanh(mapping @ code))

    def bind(self, code: torch.Tensor | None) -> None:
        # A fitted z is transient, even when it is an nn.Parameter for Adam.
        # Bypass Module.__setattr__ so it never enters the persistent state.
        object.__setattr__(self, "active_code", code)

    def install(self, model: nn.Module) -> None:
        if len(model.model.layers) != self.layer_count:
            raise ValueError("Decoder layer count differs")
        for index, layer in enumerate(model.model.layers):
            layer.self_attn.k_proj = GatedLinear(layer.self_attn.k_proj, self, index, "k_proj")
            layer.self_attn.v_proj = GatedLinear(layer.self_attn.v_proj, self, index, "v_proj")
            layer.mlp.down_proj = GatedLinear(layer.mlp.down_proj, self, index, "down_proj")

    def configuration(self) -> dict:
        return {"code_dim": CODE_DIM, "seed": INIT_SEED, "layer_count": self.layer_count,
                "map_shapes": {name: list(value.shape) for name, value in self.maps.items()},
                "gate": "exp(0.5*tanh(C@z))", "gate_compute_dtype": "float32"}


def answer_nll(model: nn.Module, view: dict) -> torch.Tensor:
    prompt, answer = view["prompt_tokens"], view["answer_tokens"]
    if not prompt or not answer or answer[-1] != 15 or len(prompt) + len(answer) > 8192:
        raise ValueError("Invalid complete 16-token answer row")
    device = next(model.parameters()).device
    ids = torch.tensor(prompt + answer, dtype=torch.long, device=device).unsqueeze(0)
    target = torch.tensor(answer, dtype=torch.long, device=device)
    outputs = model(input_ids=ids, use_cache=False)
    positions = outputs.logits[0, len(prompt) - 1:-1]
    if positions.shape[0] != target.shape[0] or positions.shape[-1] != 16:
        raise ValueError("Unexpected causal answer slice or vocabulary")
    return F.cross_entropy(positions.float(), target, reduction="mean")


def fit_code(model: nn.Module, bank: TaskGateBank, support_rows: list[dict],
             *, steps: int, learning_rate: float = 0.1) -> tuple[torch.Tensor, list[float]]:
    if not support_rows or steps < 1:
        raise ValueError("No support rows/steps")
    device = next(model.parameters()).device
    code = nn.Parameter(torch.zeros(CODE_DIM, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam([code], lr=learning_rate, weight_decay=0)
    previous_flags = [p.requires_grad for p in bank.parameters()]
    try:
        for parameter in bank.parameters():
            parameter.requires_grad_(False)
        losses = []
        for index in range(steps):
            optimizer.zero_grad(set_to_none=True)
            bank.bind(code)
            loss = answer_nll(model, support_rows[index % len(support_rows)])
            objective = loss + 1e-3 * code.square().mean()
            objective.backward()
            norm = torch.nn.utils.clip_grad_norm_([code], 1.0)
            if not torch.isfinite(norm) or not torch.isfinite(objective):
                raise ValueError("Nonfinite code fitting loss/gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
        return code.detach(), losses
    finally:
        bank.bind(None)
        for parameter, flag in zip(bank.parameters(), previous_flags):
            parameter.requires_grad_(flag)


def gates_for_code(bank: TaskGateBank, code: torch.Tensor) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        return {f"L{index}_{kind}": bank.gate(index, kind, code).detach().cpu().contiguous()
                for index in range(bank.layer_count) for kind in KINDS}


def fold_into_model(model: nn.Module, gates: dict[str, torch.Tensor]) -> dict:
    """Fold fixed gates once into an original, unwrapped Qwen model."""
    expected = {f"L{index}_{kind}" for index in range(len(model.model.layers)) for kind in KINDS}
    if set(gates) != expected:
        raise ValueError("Fixed gate inventory differs from model")
    summaries = []
    with torch.no_grad():
        for index, layer in enumerate(model.model.layers):
            modules = {"k_proj": layer.self_attn.k_proj,
                       "v_proj": layer.self_attn.v_proj,
                       "down_proj": layer.mlp.down_proj}
            for kind, linear in modules.items():
                if isinstance(linear, GatedLinear):
                    linear = linear.base
                if not isinstance(linear, nn.Linear):
                    raise TypeError(f"Unexpected projection {index}.{kind}")
                gate = gates[f"L{index}_{kind}"].to(device=linear.weight.device, dtype=torch.float32)
                axis = 1 if kind == "down_proj" else 0
                expected_size = linear.weight.shape[axis]
                if gate.shape != (expected_size,) or not torch.isfinite(gate).all():
                    raise ValueError(f"Invalid fixed gate L{index}_{kind}")
                original = linear.weight.detach().float()
                scaled = original * (gate[None, :] if axis == 1 else gate[:, None])
                linear.weight.copy_(scaled.to(dtype=linear.weight.dtype))
                if kind != "down_proj" and linear.bias is not None:
                    linear.bias.copy_((linear.bias.detach().float() * gate).to(linear.bias.dtype))
                summaries.append({"layer": index, "kind": kind, "channels": expected_size,
                                  "gate_min": float(gate.min()), "gate_max": float(gate.max())})
    return {"folded_projections": len(summaries), "gate_ranges": summaries}


def gate_file_manifest(path: Path, gates: dict[str, torch.Tensor]) -> dict:
    from safetensors.torch import save_file
    save_file(gates, str(path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"sha256": digest, "bytes": path.stat().st_size,
            "gate_count": len(gates), "min": min(float(g.min()) for g in gates.values()),
            "max": max(float(g.max()) for g in gates.values())}


def explicit_fold_check(model: nn.Module, bank: TaskGateBank, view: dict) -> dict:
    """One existing answer row, nonzero code, explicit versus folded logits."""
    device = next(model.parameters()).device
    code = torch.full((CODE_DIM,), 0.2, dtype=torch.float32, device=device)
    ids = torch.tensor(view["prompt_tokens"] + view["answer_tokens"], device=device).unsqueeze(0)
    originals = {}
    for index, layer in enumerate(model.model.layers):
        for kind, module in (("k_proj", layer.self_attn.k_proj),
                             ("v_proj", layer.self_attn.v_proj),
                             ("down_proj", layer.mlp.down_proj)):
            originals[f"L{index}_{kind}"] = (module.base.weight.detach().clone(),
                                              module.base.bias.detach().clone() if module.base.bias is not None else None)
    try:
        model.eval()
        with torch.no_grad():
            bank.bind(code)
            explicit = model(input_ids=ids, use_cache=False).logits.detach().float()
            bank.bind(None)
            fold_into_model(model, gates_for_code(bank, code))
            folded = model(input_ids=ids, use_cache=False).logits.detach().float()
        delta = (explicit - folded).abs()
        if not torch.isfinite(delta).all():
            raise ValueError("Explicit/folded comparison yielded nonfinite logits")
        return {"sequence_tokens": ids.shape[-1], "max_abs_logit_difference": float(delta.max()),
                "mean_abs_logit_difference": float(delta.mean()),
                "argmax_disagreements": int((explicit.argmax(-1) != folded.argmax(-1)).sum())}
    finally:
        bank.bind(None)
        with torch.no_grad():
            for index, layer in enumerate(model.model.layers):
                for kind, module in (("k_proj", layer.self_attn.k_proj),
                                     ("v_proj", layer.self_attn.v_proj),
                                     ("down_proj", layer.mlp.down_proj)):
                    weight, bias = originals[f"L{index}_{kind}"]
                    module.base.weight.copy_(weight)
                    if bias is not None:
                        module.base.bias.copy_(bias)
