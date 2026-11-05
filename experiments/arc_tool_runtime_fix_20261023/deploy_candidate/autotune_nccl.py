"""4-GPU SHM PyNCCL & Tiled Star-SHM CUDA-graph micro-autotuner for Kaggle 4xL4 (TP=4).

Runs immediately after `stage_model.py` finishes (idle disk & GPU) before `vllm` starts.
Hardware ground truth from `v13` & `v14`:
- GCE `g2-standard` IOMMU blocks `PHB` peer BAR1 P2P DMA.
- `v14` proved `nccl==2.29.7` 2-channel `default` (`SHM/direct/direct`) already saturates NCCL Ring
  SHM (`57.57 ms` on `MTP=4` decode, `19.33 ms` on `[6144, 5120]` prefill, `142 MiB` driver VRAM);
  4..16 channels regressed decode (`58.60..62.35 ms`) and `NCCL_SHM_USE_CUDA_MEMCPY=1` (`CE`) hung.

Stage-H (`v15`) benchmarks (`~55s` total):
1. Baseline `"default"` (`SAFE_SHM`, `MTP=3` `K=3` 145-collective decode CUDAGraph + `[6144, 5120]` prefill).
2. Tiled `sm_89` cooperative-launch 1-hop `StarShm` (`star_shm_b32`, `star_shm_b48`, `1.00S TX + 1.50S RX`,
   `37.66 MiB` `K=4` `memfd` arena at `capacity = 192 * 5120`, 1 tile for decode & 32 tiles for 6144 prefill),
   qualified separately across `"both"`, `"decode_only"`, and `"prefill_only"` regimes with multi-seed
   changing-input CUDAGraph & tile-boundary exactness checks and persisted to `/opt/arc_pi_cache/arc_shm_ar/`.
"""
from __future__ import annotations

import ctypes
import datetime
import hashlib
import importlib.util
import json
import mmap
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

SAFE_SHM: dict[str, str] = {
    "NCCL_P2P_DISABLE": "1",
    "NCCL_SHM_DISABLE": "0",
    "NCCL_PROTO": "^LL128",
    "NCCL_SHM_USE_CUDA_MEMCPY": "0",
    "NCCL_SHM_MEMCPY_MODE": "1",
    "NCCL_SHM_LOCALITY": "2",
}

TUNED_KEYS: tuple[str, ...] = (
    "NCCL_P2P_DISABLE",
    "NCCL_P2P_LEVEL",
    "NCCL_SHM_DISABLE",
    "NCCL_SHM_USE_CUDA_MEMCPY",
    "NCCL_SHM_MEMCPY_MODE",
    "NCCL_SHM_LOCALITY",
    "NCCL_MIN_NCHANNELS",
    "NCCL_MAX_NCHANNELS",
    "NCCL_MIN_NRINGS",
    "NCCL_MAX_NRINGS",
    "NCCL_MIN_CTAS",
    "NCCL_MAX_CTAS",
    "NCCL_BUFFSIZE",
    "NCCL_ALGO",
    "NCCL_PROTO",
)


def shm_profile(channels: int, buff_mib: int, ce_mode: int = 0) -> dict[str, str]:
    return {
        **SAFE_SHM,
        "NCCL_MIN_NCHANNELS": str(channels),
        "NCCL_MAX_NCHANNELS": str(channels),
        "NCCL_BUFFSIZE": str(buff_mib * 1024 * 1024),
        "NCCL_SHM_USE_CUDA_MEMCPY": str(int(ce_mode != 0)),
        "NCCL_SHM_MEMCPY_MODE": str(ce_mode or 1),
    }


def profile_environment(parent: dict[str, str], overrides: dict[str, str]) -> dict[str, str]:
    env = {k: v for k, v in parent.items() if k not in TUNED_KEYS}
    env.update(SAFE_SHM)
    env.update(overrides)
    return env


CANDIDATE_PROFILES: list[tuple[str, dict[str, str]]] = [
    ("default", dict(SAFE_SHM)),
]

STAR_ARENA_CAPACITY_P48 = 48 * 5120
STAR_ARENA_CAPACITY_P96 = 96 * 5120
STAR_ARENA_CAPACITY = STAR_ARENA_CAPACITY_P96
ARC_MAX_NUMEL = 6144 * 5120
DEFAULT_SPEC_K = 3
# (profile_name, blocks, arena_capacity, cpu_workers, pipe_d)
STAR_CANDIDATE_PROFILES: list[tuple[str, int, int, int, int]] = [
    ("star_cpu_p48_b32_w8_d2", 32, STAR_ARENA_CAPACITY_P48, 8, 2),
    ("star_cpu_p96_b32_w8_d2", 32, STAR_ARENA_CAPACITY_P96, 8, 2),
    ("star_cpu_p48_b32_w16_d2", 32, STAR_ARENA_CAPACITY_P48, 16, 2),
    ("star_cpu_p96_b32_w16_d3", 32, STAR_ARENA_CAPACITY_P96, 16, 3),
]
PROFILE_ENV_DICT: dict[str, dict[str, str]] = {
    **dict(CANDIDATE_PROFILES),
    "shm_direct_c4_b2": shm_profile(4, 2, 0),
    "shm_direct_c8_b1": shm_profile(8, 1, 0),
    "shm_direct_c12_b1": shm_profile(12, 1, 0),
    "shm_direct_c16_b1": shm_profile(16, 1, 0),
    "shm_recv_ce_c8_b1": shm_profile(8, 1, 2),
    "shm_send_ce_c8_b1": shm_profile(8, 1, 1),
    "shm_both_ce_c8_b1": shm_profile(8, 1, 3),
}

# Ground-truth 900s 4xL4 telemetry with Mamba decode block pinning (v17):
# ~55.78 x 6144-tok chunks (342,713 uncached prefill tokens), 129 ARs/chunk,
# across ~5,820 decode steps for MTP=3 (135,037 drafts / ~23.2 active reqs) or 4,880 steps for MTP=4.
PREFILL_STEP_WEIGHT_K3 = (342_713.0 / 6144.0) / 5820.0
PREFILL_STEP_WEIGHT_K4 = (342_713.0 / 6144.0) / 4880.0
PREFILL_STEP_WEIGHT = PREFILL_STEP_WEIGHT_K3
PREFILL_ARS_PER_CHUNK = 129.0


def prefill_factor_for_k(spec_k: int | None = None) -> float:
    k = DEFAULT_SPEC_K if spec_k is None else int(spec_k)
    weight = PREFILL_STEP_WEIGHT_K4 if k >= 4 else PREFILL_STEP_WEIGHT_K3
    return weight * PREFILL_ARS_PER_CHUNK


def compute_effective_step_ms(entry: dict[str, Any], spec_k: int | None = None) -> float:
    decode_ms = float(entry["decode_step_ms"])
    prefill_ar_ms = float(entry.get("prefill_ar6144_ms") or 0.0)
    k = entry.get("spec_k") if spec_k is None else spec_k
    return decode_ms + prefill_factor_for_k(None if k is None else int(k)) * prefill_ar_ms


def _unset_keys_for(selected_env: dict[str, str]) -> list[str]:
    return [k for k in TUNED_KEYS if k not in selected_env]


def _disabled_star_selection(reason: str = "not_selected") -> dict[str, Any]:
    return {
        "enabled": False,
        "reason": reason,
        "profile": None,
        "regime": None,
        "blocks": 0,
        "workers": 0,
        "pipe_d": 0,
        "so_path": "",
        "so_sha256": "",
        "capacity": STAR_ARENA_CAPACITY,
        "min_numel": 0,
        "max_numel": 0,
    }


def select_best_profile(
    results: list[dict[str, Any]],
    min_gain_ms: float = 1.5,
    max_decode_regression_ms: float = 0.5,
    max_prefill_regression_ms: float = 0.25,
    max_extra_nccl_mib: float = 140.0,
) -> dict[str, Any]:
    """Select the fastest verified production SHM PyNCCL or StarShm regime within per-rank VRAM budget."""
    by_name = {r.get("profile"): r for r in results if isinstance(r, dict)}
    default_entry = by_name.get("default")
    fallback_env = dict(SAFE_SHM)
    fallback_unset = _unset_keys_for(fallback_env)
    if (
        not default_entry
        or default_entry.get("returncode", 0) != 0
        or default_entry.get("status") != "ok"
        or not default_entry.get("numerical_ok")
    ):
        return {
            "selected_profile": "default",
            "selected_env": fallback_env,
            "unset_env": fallback_unset,
            "star_shm_selection": _disabled_star_selection("default_baseline_unavailable"),
            "reason": "default_baseline_unavailable",
            "default_decode_ms": None,
            "selected_decode_ms": None,
            "default_prefill_ar6144_ms": None,
            "selected_prefill_ar6144_ms": None,
            "default_effective_ms": None,
            "selected_effective_ms": None,
            "gain_decode_ms": 0.0,
            "gain_effective_ms": 0.0,
            "extra_nccl_mib": 0.0,
        }

    spec_k = int(default_entry.get("spec_k") or DEFAULT_SPEC_K)
    pf_factor = prefill_factor_for_k(spec_k)
    default_decode_ms = float(default_entry["decode_step_ms"])
    default_prefill_ms = float(default_entry.get("prefill_ar6144_ms") or 0.0)
    default_eff_ms = default_decode_ms + pf_factor * default_prefill_ms
    default_nccl_mib = float(default_entry.get("max_nccl_driver_mib") or 0.0)
    default_ranks = default_entry.get("per_rank") or []
    default_rank_mib = {
        int(r["rank"]): float(r.get("nccl_driver_mib") or 0.0)
        for r in default_ranks
        if isinstance(r, dict) and "rank" in r
    }

    best_choice: dict[str, Any] | None = None
    best_eff_ms = default_eff_ms

    for cand in results:
        name = cand.get("profile")
        if not name or name == "default":
            continue
        if cand.get("diagnostic_only"):
            continue
        if (
            cand.get("returncode", 0) != 0
            or cand.get("status") != "ok"
            or not cand.get("numerical_ok")
        ):
            continue

        extra_max_mib = float(cand.get("max_nccl_driver_mib") or 0.0) - default_nccl_mib
        cand_ranks = cand.get("per_rank") or []
        per_rank_extra = [
            float(r.get("nccl_driver_mib") or 0.0) - default_rank_mib.get(int(r["rank"]), default_nccl_mib)
            for r in cand_ranks
            if isinstance(r, dict) and "rank" in r
        ]
        worst_extra_mib = max([extra_max_mib, *per_rank_extra])
        if worst_extra_mib > max_extra_nccl_mib:
            continue

        cand_decode_ms = float(cand["decode_step_ms"])
        cand_prefill_ms = float(cand.get("prefill_ar6144_ms") or 0.0)
        is_star = bool(cand.get("backend") == "star_shm" or cand.get("star_so_path"))

        if is_star:
            if not cand.get("star_multiseed_ok") or not cand.get("star_so_sha256") or not cand.get("star_so_path"):
                continue
            cap = int(cand.get("star_capacity") or STAR_ARENA_CAPACITY)
            decode_rows = int(cand.get("decode_rows") or (24 * (spec_k + 1)))
            decode_max_numel = max(cap, decode_rows * 5120)
            max_n = int(cand.get("star_max_numel") or ARC_MAX_NUMEL)
            blocks = int(cand.get("star_blocks") or 32)
            workers = int(cand.get("star_workers") or 8)
            pipe_d = int(cand.get("star_pipe_d") or 2)
            # Evaluate all 3 rank-consistent routing regimes and pick the fastest qualified regime.
            regimes: list[tuple[str, int, int, float, float]] = []
            if (
                (cand_decode_ms - default_decode_ms) <= max_decode_regression_ms
                and (cand_prefill_ms - default_prefill_ms) <= max_prefill_regression_ms
            ):
                regimes.append(("both", 1, max_n, cand_decode_ms, cand_prefill_ms))
            if (default_decode_ms - cand_decode_ms) >= min_gain_ms:
                regimes.append(("decode_only", 1, decode_max_numel, cand_decode_ms, default_prefill_ms))
            if pf_factor * (default_prefill_ms - cand_prefill_ms) >= min_gain_ms:
                regimes.append(("prefill_only", decode_max_numel + 1, max_n, default_decode_ms, cand_prefill_ms))

            for regime, min_n, max_num, eff_dec, eff_pf in regimes:
                eff_ms = eff_dec + pf_factor * eff_pf
                if (default_eff_ms - eff_ms) < min_gain_ms:
                    continue
                if best_choice is None or eff_ms < best_eff_ms:
                    best_eff_ms = eff_ms
                    best_choice = {
                        "profile": str(name),
                        "env": profile_environment({}, cand.get("env") or SAFE_SHM),
                        "decode_ms": eff_dec,
                        "prefill_ms": eff_pf,
                        "eff_ms": eff_ms,
                        "extra_mib": extra_max_mib,
                        "star_shm_selection": {
                            "enabled": True,
                            "reason": f"verified_faster_{regime}_within_vram_budget",
                            "profile": str(name),
                            "regime": regime,
                            "blocks": blocks,
                            "workers": workers,
                            "pipe_d": pipe_d,
                            "so_path": str(cand["star_so_path"]),
                            "so_sha256": str(cand["star_so_sha256"]),
                            "capacity": cap,
                            "min_numel": int(min_n),
                            "max_numel": int(max_num),
                        },
                    }
        else:
            if name not in PROFILE_ENV_DICT:
                continue
            cand_eff_ms = cand_decode_ms + pf_factor * cand_prefill_ms
            if (cand_decode_ms - default_decode_ms) > max_decode_regression_ms:
                continue
            if (default_eff_ms - cand_eff_ms) < min_gain_ms:
                continue
            if best_choice is None or cand_eff_ms < best_eff_ms:
                best_eff_ms = cand_eff_ms
                best_choice = {
                    "profile": str(name),
                    "env": profile_environment({}, PROFILE_ENV_DICT[str(name)]),
                    "decode_ms": cand_decode_ms,
                    "prefill_ms": cand_prefill_ms,
                    "eff_ms": cand_eff_ms,
                    "extra_mib": extra_max_mib,
                    "star_shm_selection": _disabled_star_selection("pynccl_profile_selected"),
                }

    if best_choice is None:
        return {
            "selected_profile": "default",
            "selected_env": fallback_env,
            "unset_env": fallback_unset,
            "star_shm_selection": _disabled_star_selection("no_candidate_exceeded_gain_and_memory_thresholds"),
            "reason": "no_candidate_exceeded_gain_and_memory_thresholds",
            "default_decode_ms": round(default_decode_ms, 3),
            "selected_decode_ms": round(default_decode_ms, 3),
            "default_prefill_ar6144_ms": round(default_prefill_ms, 3),
            "selected_prefill_ar6144_ms": round(default_prefill_ms, 3),
            "default_effective_ms": round(default_eff_ms, 3),
            "selected_effective_ms": round(default_eff_ms, 3),
            "gain_decode_ms": 0.0,
            "gain_effective_ms": 0.0,
            "extra_nccl_mib": 0.0,
        }

    winner_env = dict(best_choice["env"])
    winner_env["NCCL_P2P_DISABLE"] = "1"
    unset_keys = _unset_keys_for(winner_env)
    winner_decode_ms = float(best_choice["decode_ms"])
    winner_prefill_ms = float(best_choice["prefill_ms"])
    winner_eff_ms = float(best_choice["eff_ms"])
    return {
        "selected_profile": best_choice["profile"],
        "selected_env": winner_env,
        "unset_env": unset_keys,
        "star_shm_selection": best_choice["star_shm_selection"],
        "reason": "verified_faster_within_vram_budget",
        "default_decode_ms": round(default_decode_ms, 3),
        "selected_decode_ms": round(winner_decode_ms, 3),
        "default_prefill_ar6144_ms": round(default_prefill_ms, 3),
        "selected_prefill_ar6144_ms": round(winner_prefill_ms, 3),
        "default_effective_ms": round(default_eff_ms, 3),
        "selected_effective_ms": round(winner_eff_ms, 3),
        "gain_decode_ms": round(default_decode_ms - winner_decode_ms, 3),
        "gain_effective_ms": round(default_eff_ms - winner_eff_ms, 3),
        "extra_nccl_mib": round(float(best_choice["extra_mib"]), 2),
    }


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _has_four_nvidia_gpus() -> bool:
    if Path("/dev/nvidia3").exists():
        return True
    cvis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvis:
        parts = [x.strip() for x in cvis.split(",") if x.strip()]
        return len(parts) >= 4
    return False


def _extract_nccl_log_highlights(log_dir: Path) -> list[str]:
    keywords = (
        "Version",
        "Channel ",
        "via SHM",
        "SHM/",
        "via P2P",
        "Trees ",
        "Ring ",
        "Connected",
        "BUFFSIZE",
        "NCHANNELS",
        "MIN_CTAS",
        "SHM_USE_CUDA_MEMCPY",
    )
    lines: list[str] = []
    seen: set[str] = set()
    for p in sorted(log_dir.glob("nccl.*.log")):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for raw in text.splitlines():
            ln = raw.strip()
            if not ln or not any(k in ln for k in keywords):
                continue
            dedup_key = ln.split(" NCCL INFO ", 1)[-1] if " NCCL INFO " in ln else ln
            if dedup_key not in seen:
                seen.add(dedup_key)
                lines.append(ln[:240])
            if len(lines) >= 28:
                return lines
    return lines


NCCL_UNIQUE_ID_BYTES = 128
NCCL_DATATYPE_INT32 = 2
NCCL_DATATYPE_FLOAT32 = 7
NCCL_DATATYPE_BFLOAT16 = 9
NCCL_REDOP_SUM = 0


class _NcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_ubyte * NCCL_UNIQUE_ID_BYTES)]


def _load_nccl_cdll() -> tuple[ctypes.CDLL, str]:
    """Load the exact `libnccl.so.2` used by vLLM (`nvidia-nccl-cu12` wheel first) without importing vLLM."""
    candidates: list[str] = []
    env_so = os.environ.get("VLLM_NCCL_SO_PATH", "").strip()
    if env_so:
        candidates.append(env_so)
    try:
        spec = importlib.util.find_spec("nvidia.nccl")
        if spec and spec.submodule_search_locations:
            for loc in spec.submodule_search_locations:
                wheel_so = Path(loc) / "lib" / "libnccl.so.2"
                if wheel_so.exists():
                    candidates.append(str(wheel_so))
    except Exception:
        pass
    candidates.append("libnccl.so.2")

    last_err: Exception | None = None
    for cand in candidates:
        try:
            lib = ctypes.CDLL(cand)
            return lib, cand
        except OSError as exc:
            last_err = exc
    raise RuntimeError(f"Unable to load libnccl.so.2 from {candidates}: {last_err}")


class _LiteNcclComm:
    """Minimal zero-`vllm` ctypes communicator matching `PyNcclCommunicator` C calls 1:1."""

    def __init__(self, group: Any, device: Any) -> None:
        import torch
        import torch.distributed as dist

        self._torch = torch
        self.device = device
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.lib, self.so_path = _load_nccl_cdll()
        self._bind_signatures()

        ver = ctypes.c_int(0)
        self._check(self.lib.ncclGetVersion(ctypes.byref(ver)), "ncclGetVersion")
        self.nccl_version = int(ver.value)

        uid = _NcclUniqueId()
        if self.rank == 0:
            self._check(self.lib.ncclGetUniqueId(ctypes.byref(uid)), "ncclGetUniqueId")
            uid_tensor = torch.tensor(list(uid.internal), dtype=torch.uint8)
        else:
            uid_tensor = torch.zeros(NCCL_UNIQUE_ID_BYTES, dtype=torch.uint8)
        dist.broadcast(uid_tensor, src=0, group=group)
        raw_bytes = bytes(int(x) for x in uid_tensor.tolist())
        ctypes.memmove(ctypes.byref(uid), raw_bytes, NCCL_UNIQUE_ID_BYTES)

        self.comm = ctypes.c_void_p()
        with torch.cuda.device(self.device):
            self._check(
                self.lib.ncclCommInitRank(
                    ctypes.byref(self.comm),
                    int(self.world_size),
                    uid,
                    int(self.rank),
                ),
                "ncclCommInitRank",
            )
            stream = torch.cuda.current_stream(self.device)
            warm = torch.zeros(1, dtype=torch.float32, device=self.device)
            self.all_reduce(warm, warm, stream=stream)
            stream.synchronize()
            del warm

    def _bind_signatures(self) -> None:
        self.lib.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.lib.ncclGetVersion.restype = ctypes.c_int
        self.lib.ncclGetErrorString.argtypes = [ctypes.c_int]
        self.lib.ncclGetErrorString.restype = ctypes.c_char_p
        self.lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(_NcclUniqueId)]
        self.lib.ncclGetUniqueId.restype = ctypes.c_int
        self.lib.ncclCommInitRank.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            _NcclUniqueId,
            ctypes.c_int,
        ]
        self.lib.ncclCommInitRank.restype = ctypes.c_int
        self.lib.ncclCommAbort.argtypes = [ctypes.c_void_p]
        self.lib.ncclCommAbort.restype = ctypes.c_int
        self.lib.ncclAllReduce.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.ncclAllReduce.restype = ctypes.c_int
        self.lib.ncclAllGather.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.ncclAllGather.restype = ctypes.c_int

    def _check(self, code: int, fn_name: str) -> None:
        if code != 0:
            err_ptr = self.lib.ncclGetErrorString(code)
            msg = err_ptr.decode("utf-8", errors="replace") if err_ptr else f"code={code}"
            raise RuntimeError(f"{fn_name} failed ({code}): {msg}")

    def _nccl_dtype(self, dtype: Any) -> int:
        torch = self._torch
        if dtype == torch.bfloat16:
            return NCCL_DATATYPE_BFLOAT16
        if dtype == torch.float32:
            return NCCL_DATATYPE_FLOAT32
        if dtype == torch.int32:
            return NCCL_DATATYPE_INT32
        raise ValueError(f"Unsupported dtype in _LiteNcclComm: {dtype}")

    def all_reduce(self, in_tensor: Any, out_tensor: Any = None, stream: Any = None) -> Any:
        torch = self._torch
        if out_tensor is None:
            out_tensor = torch.empty_like(in_tensor)
        if stream is None:
            stream = torch.cuda.current_stream(self.device)
        self._check(
            self.lib.ncclAllReduce(
                ctypes.c_void_p(in_tensor.data_ptr()),
                ctypes.c_void_p(out_tensor.data_ptr()),
                ctypes.c_size_t(in_tensor.numel()),
                self._nccl_dtype(in_tensor.dtype),
                NCCL_REDOP_SUM,
                self.comm,
                ctypes.c_void_p(stream.cuda_stream),
            ),
            "ncclAllReduce",
        )
        return out_tensor

    def all_gather(self, output_tensor: Any, input_tensor: Any, stream: Any = None) -> None:
        torch = self._torch
        if stream is None:
            stream = torch.cuda.current_stream(self.device)
        self._check(
            self.lib.ncclAllGather(
                ctypes.c_void_p(input_tensor.data_ptr()),
                ctypes.c_void_p(output_tensor.data_ptr()),
                ctypes.c_size_t(input_tensor.numel()),
                self._nccl_dtype(input_tensor.dtype),
                self.comm,
                ctypes.c_void_p(stream.cuda_stream),
            ),
            "ncclAllGather",
        )

    def destroy(self) -> None:
        torch = self._torch
        if self.comm and self.comm.value:
            comm_handle = self.comm
            self.comm = ctypes.c_void_p()

            def _abort() -> None:
                with torch.cuda.device(self.device):
                    self.lib.ncclCommAbort(comm_handle)

            abort_thread = threading.Thread(target=_abort, daemon=True)
            abort_thread.start()
            abort_thread.join(timeout=3.0)


# Track J2: Host-CPU AVX-512 persistent pinned shard-owner worker pool + warp-specialized sm_89 1.00S TX/RX Star-SHM AllReduce
ARC_SHM_STAR_CPU_CPP_SOURCE = r"""
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <immintrin.h>
#include <xmmintrin.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <mutex>

#ifndef ARC_BLOCKS
#define ARC_BLOCKS 32
#endif
#ifndef ARC_WORKERS
#define ARC_WORKERS 8
#endif

constexpr int B = ARC_BLOCKS;
constexpr int K = 4;
constexpr int W = ARC_WORKERS;
static_assert(B > 0 && W > 0 && (B % W) == 0, "ARC_BLOCKS must be divisible by ARC_WORKERS");
constexpr int SHARDS_PER_WORKER = B / W;

struct alignas(128) Flag {
    uint64_t value;
};

struct alignas(128) ShardDesc {
    uint64_t tile_n;
};

struct Control {
    Flag ready[4];
    Flag reduced;
    Flag done[4];
    ShardDesc desc;
};
static_assert(sizeof(Control) == 10 * 128, "Control layout mismatch");

constexpr size_t HEADER = ((sizeof(Control) * size_t(K) * size_t(B) + 4095) / 4096) * 4096;

static inline size_t arc_shard_stride(size_t capacity) {
    // Exact worst-case shard count across all 0 <= tile_n <= capacity and 0 <= b < B:
    // shard B-1 receives up to ceil((capacity >> 3) / B) * 8 + 7 elements.
    const size_t max_n8 = capacity >> 3;
    const size_t max_count = (((max_n8 + size_t(B) - 1) / size_t(B)) << 3) + 7;
    return (max_count + 31) & ~size_t(31);
}

static inline void arc_shard_bounds(size_t tile_n, int b, size_t& begin, size_t& end) {
    const size_t n8 = tile_n >> 3;
    begin = ((n8 * size_t(b)) / size_t(B)) << 3;
    end = (b == B - 1) ? tile_n : (((n8 * size_t(b + 1)) / size_t(B)) << 3);
}

static inline float bf16_bits_to_f32(uint16_t bits) {
    uint32_t u = uint32_t(bits) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(float));
    return f;
}

static inline uint16_t f32_to_bf16_rne_canonical(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(uint32_t));
    if ((u & 0x7fffffffU) > 0x7f800000U) {
        return 0x7fffU;
    }
    const uint32_t lsb = (u >> 16) & 1U;
    return uint16_t((u + 0x00007fffU + lsb) >> 16);
}

__attribute__((target("avx512f,avx512bw,avx512dq")))
static void reduce_shard_avx512(
    const uint16_t* __restrict__ in0,
    const uint16_t* __restrict__ in1,
    const uint16_t* __restrict__ in2,
    const uint16_t* __restrict__ in3,
    uint16_t* __restrict__ out,
    size_t count
) {
    const __m512i v_abs_mask = _mm512_set1_epi32(0x7fffffff);
    const __m512i v_inf_bits = _mm512_set1_epi32(0x7f800000);
    const __m512i v_one      = _mm512_set1_epi32(1);
    const __m512i v_7fff     = _mm512_set1_epi32(0x00007fff);

    size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        __m512 f0 = _mm512_castsi512_ps(_mm512_slli_epi32(
            _mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(in0 + i))), 16));
        __m512 f1 = _mm512_castsi512_ps(_mm512_slli_epi32(
            _mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(in1 + i))), 16));
        __m512 f2 = _mm512_castsi512_ps(_mm512_slli_epi32(
            _mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(in2 + i))), 16));
        __m512 f3 = _mm512_castsi512_ps(_mm512_slli_epi32(
            _mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(in3 + i))), 16));

        __m512 s01 = _mm512_add_round_ps(f0, f1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m512 s23 = _mm512_add_round_ps(f2, f3, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m512 sum = _mm512_add_round_ps(s01, s23, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        __m512i u = _mm512_castps_si512(sum);
        __mmask16 is_nan = _mm512_cmpgt_epu32_mask(_mm512_and_si512(u, v_abs_mask), v_inf_bits);
        __m512i lsb = _mm512_and_si512(_mm512_srli_epi32(u, 16), v_one);
        __m512i rounded = _mm512_srli_epi32(_mm512_add_epi32(u, _mm512_add_epi32(v_7fff, lsb)), 16);
        __m512i res32 = _mm512_mask_mov_epi32(rounded, is_nan, v_7fff);
        __m256i out16 = _mm512_cvtepi32_epi16(res32);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), out16);
    }

    for (; i < count; ++i) {
        const float f0 = bf16_bits_to_f32(in0[i]);
        const float f1 = bf16_bits_to_f32(in1[i]);
        const float f2 = bf16_bits_to_f32(in2[i]);
        const float f3 = bf16_bits_to_f32(in3[i]);
        const float s01 = f0 + f1;
        const float s23 = f2 + f3;
        const float sum = s01 + s23;
        out[i] = f32_to_bf16_rne_canonical(sum);
    }
}

struct WorkerPool;

struct WorkerArg {
    WorkerPool* pool;
    int worker_id;
    int pin_cpu;
};

struct WorkerPool {
    void* host;
    void* mapped;
    size_t bytes;
    size_t capacity;
    int stop;
    pthread_t threads[W];
    WorkerArg args[W];
};

__attribute__((target("avx512f,avx512bw,avx512dq")))
static void worker_loop_avx512(WorkerArg* wa) {
    if (wa->pin_cpu >= 0) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(wa->pin_cpu, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    }
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);
    _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);

    WorkerPool* pool = wa->pool;
    Control* controls = static_cast<Control*>(pool->host);
    uint16_t* payload = reinterpret_cast<uint16_t*>(static_cast<char*>(pool->host) + HEADER);
    const size_t C = arc_shard_stride(pool->capacity);

    int shards[SHARDS_PER_WORKER];
    uint64_t next_q[SHARDS_PER_WORKER];
    for (int m = 0; m < SHARDS_PER_WORKER; ++m) {
        shards[m] = wa->worker_id + m * W;
        next_q[m] = 1;
    }

    uint32_t idle_iters = 0;
    while (true) {
        bool did_work = false;
        for (int m = 0; m < SHARDS_PER_WORKER; ++m) {
            const int b = shards[m];
            const uint64_t q = next_q[m];
            const int slot = int((q - 1) & (K - 1));
            Control& c = controls[slot * B + b];
            if (__atomic_load_n(&c.ready[0].value, __ATOMIC_ACQUIRE) >= q &&
                __atomic_load_n(&c.ready[1].value, __ATOMIC_ACQUIRE) >= q &&
                __atomic_load_n(&c.ready[2].value, __ATOMIC_ACQUIRE) >= q &&
                __atomic_load_n(&c.ready[3].value, __ATOMIC_ACQUIRE) >= q) {
                const size_t tile_n = size_t(__atomic_load_n(&c.desc.tile_n, __ATOMIC_RELAXED));
                size_t begin = 0, end = 0;
                arc_shard_bounds(tile_n, b, begin, end);
                const size_t count = end - begin;
                if (count > 0 && count <= C) {
                    const uint16_t* base_in = payload + size_t(slot * B + b) * 5 * C;
                    uint16_t* out = const_cast<uint16_t*>(base_in) + 4 * C;
                    reduce_shard_avx512(
                        base_in + 0 * C,
                        base_in + 1 * C,
                        base_in + 2 * C,
                        base_in + 3 * C,
                        out,
                        count
                    );
                }
                __atomic_store_n(&c.reduced.value, q, __ATOMIC_RELEASE);
                next_q[m] = q + 1;
                did_work = true;
            }
        }
        if (did_work) {
            idle_iters = 0;
            continue;
        }
        ++idle_iters;
        if ((idle_iters & 63U) == 0U || idle_iters > 2000000U) {
            if (__atomic_load_n(&pool->stop, __ATOMIC_ACQUIRE) != 0) {
                break;
            }
        }
        if (idle_iters <= 2000000U) {
            _mm_pause();
        } else {
            usleep(25);
        }
    }
}

static void* worker_trampoline(void* p) {
    worker_loop_avx512(static_cast<WorkerArg*>(p));
    return nullptr;
}

static int read_primary_sibling_cpu(int cpu) {
    char path[128];
    std::snprintf(
        path,
        sizeof(path),
        "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list",
        cpu
    );
    FILE* f = std::fopen(path, "r");
    if (!f) return cpu;
    int first = cpu;
    if (std::fscanf(f, "%d", &first) != 1) {
        first = cpu;
    }
    std::fclose(f);
    return first;
}

static void select_worker_cpus(int out_cpus[W]) {
    for (int w = 0; w < W; ++w) out_cpus[w] = -1;
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &allowed) != 0) {
        return;
    }
    std::vector<int> all_cpus;
    std::vector<int> phys_primary;
    std::vector<int> seen_keys;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (!CPU_ISSET(c, &allowed)) continue;
        all_cpus.push_back(c);
        const int key = read_primary_sibling_cpu(c);
        if (std::find(seen_keys.begin(), seen_keys.end(), key) == seen_keys.end()) {
            seen_keys.push_back(key);
            phys_primary.push_back(c);
        }
    }
    const std::vector<int>& chosen = (int(phys_primary.size()) >= W) ? phys_primary : all_cpus;
    if (chosen.empty()) return;
    const int total = int(chosen.size());
    const int start = (total >= W) ? (total - W) : 0;
    for (int w = 0; w < W; ++w) {
        out_cpus[w] = chosen[(start + w) % total];
    }
}

struct RegisteredMapping {
    void* host;
    void* mapped;
    size_t bytes;
};

static std::mutex g_pool_mu;
static std::vector<RegisteredMapping> g_mappings;
static std::vector<WorkerPool*> g_pools;

static bool cpu_supports_avx512_bw() {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512dq");
}

static int start_pool_locked(void* host, void* mapped, size_t bytes, size_t capacity) {
    if (!host || capacity == 0) return -1003;
    for (WorkerPool* p : g_pools) {
        if (p->host == host) {
            if (p->capacity == capacity) {
                if (mapped && !p->mapped) p->mapped = mapped;
                return 0;
            }
            return -1003;
        }
    }
    if (!cpu_supports_avx512_bw()) {
        return -1001;
    }
    int cpus[W];
    select_worker_cpus(cpus);
    auto* pool = new WorkerPool{};
    pool->host = host;
    pool->mapped = mapped;
    pool->bytes = bytes;
    pool->capacity = capacity;
    pool->stop = 0;

    int created = 0;
    for (int w = 0; w < W; ++w) {
        pool->args[w] = WorkerArg{pool, w, cpus[w]};
        if (pthread_create(&pool->threads[w], nullptr, worker_trampoline, &pool->args[w]) != 0) {
            __atomic_store_n(&pool->stop, 1, __ATOMIC_RELEASE);
            for (int j = 0; j < created; ++j) {
                pthread_join(pool->threads[j], nullptr);
            }
            delete pool;
            return -1002;
        }
        ++created;
    }
    g_pools.push_back(pool);
    return 0;
}

extern "C" void arc_note_registered(void* host, void* mapped, size_t bytes) {
    std::lock_guard<std::mutex> lk(g_pool_mu);
    for (auto& m : g_mappings) {
        if (m.host == host) {
            m.mapped = mapped;
            m.bytes = bytes;
            return;
        }
    }
    g_mappings.push_back(RegisteredMapping{host, mapped, bytes});
}

extern "C" int arc_init_workers(void* host, size_t bytes, size_t capacity, int rank) {
    if (rank != 0) return 0;
    std::lock_guard<std::mutex> lk(g_pool_mu);
    void* mapped = nullptr;
    for (const auto& m : g_mappings) {
        if (m.host == host) {
            mapped = m.mapped;
            if (bytes == 0) bytes = m.bytes;
            break;
        }
    }
    return start_pool_locked(host, mapped, bytes, capacity);
}

extern "C" int arc_ensure_workers_for_mapped(void* mapped, size_t capacity, int rank) {
    if (rank != 0) return 0;
    std::lock_guard<std::mutex> lk(g_pool_mu);
    for (WorkerPool* p : g_pools) {
        if (p->mapped == mapped || p->host == mapped) {
            return (p->capacity == capacity) ? 0 : -1003;
        }
    }
    for (const auto& m : g_mappings) {
        if (m.mapped == mapped || m.host == mapped) {
            return start_pool_locked(m.host, m.mapped, m.bytes, capacity);
        }
    }
    return -1003;
}

extern "C" int arc_workers() { return W; }

extern "C" void arc_stop_workers(void* host) {
    WorkerPool* target = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_pool_mu);
        g_mappings.erase(
            std::remove_if(
                g_mappings.begin(),
                g_mappings.end(),
                [host](const RegisteredMapping& m) { return m.host == host; }
            ),
            g_mappings.end()
        );
        for (auto it = g_pools.begin(); it != g_pools.end(); ++it) {
            if ((*it)->host == host) {
                target = *it;
                g_pools.erase(it);
                break;
            }
        }
    }
    if (target) {
        __atomic_store_n(&target->stop, 1, __ATOMIC_RELEASE);
        for (int w = 0; w < W; ++w) {
            pthread_join(target->threads[w], nullptr);
        }
        delete target;
    }
}
"""

ARC_SHM_STAR_CU_SOURCE = r"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstddef>

#ifndef ARC_BLOCKS
#define ARC_BLOCKS 32
#endif
#ifndef ARC_THREADS
#define ARC_THREADS 256
#endif
#ifndef ARC_PIPE_D
#define ARC_PIPE_D 2
#endif

constexpr int B = ARC_BLOCKS;
constexpr int T = ARC_THREADS;
static_assert(T == 256, "Warp-specialized TX/RX expects 256 threads (128 TX + 128 RX)");
constexpr int HALF_T = 128;
constexpr int K = 4;
constexpr int PIPE_D = ARC_PIPE_D;
static_assert(PIPE_D > 0 && PIPE_D < K, "PIPE_D must be in [1, K-1]");
constexpr size_t ARC_MAX_NUMEL = size_t(6144) * 5120;

struct alignas(128) Flag {
    uint64_t value;
};

struct alignas(128) ShardDesc {
    uint64_t tile_n;
};

struct Control {
    Flag ready[4];
    Flag reduced;
    Flag done[4];
    ShardDesc desc;
};
static_assert(sizeof(Control) == 10 * 128, "Control layout mismatch");

constexpr size_t HEADER = ((sizeof(Control) * size_t(K) * size_t(B) + 4095) / 4096) * 4096;

union Vec8Bf16 {
    uint4 u4;
    uint16_t u16[8];
};

struct Args {
    void* mapped;
    uint64_t* seq;
    const uint16_t* x;
    uint16_t* y;
    size_t n;
    size_t capacity;
    int rank;
};

extern "C" void arc_note_registered(void* host, void* mapped, size_t bytes);
extern "C" int arc_init_workers(void* host, size_t bytes, size_t capacity, int rank);
extern "C" int arc_ensure_workers_for_mapped(void* mapped, size_t capacity, int rank);
extern "C" void arc_stop_workers(void* host);

__host__ __device__ __forceinline__ size_t arc_shard_stride(size_t capacity) {
    const size_t max_n8 = capacity >> 3;
    const size_t max_count = (((max_n8 + size_t(B) - 1) / size_t(B)) << 3) + 7;
    return (max_count + 31) & ~size_t(31);
}

__host__ __device__ __forceinline__ void arc_shard_bounds(
    size_t tile_n, int b, size_t& begin, size_t& end
) {
    const size_t n8 = tile_n >> 3;
    begin = ((n8 * size_t(b)) / size_t(B)) << 3;
    end = (b == B - 1) ? tile_n : (((n8 * size_t(b + 1)) / size_t(B)) << 3);
}

__device__ __forceinline__ uint64_t acquire_flag(const uint64_t* p) {
    uint64_t v;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void release_flag(uint64_t* p, uint64_t v) {
    asm volatile("st.release.sys.global.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}

__device__ __forceinline__ void store_host_u64(uint64_t* p, uint64_t v) {
    asm volatile("st.relaxed.sys.global.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}

__device__ __forceinline__ uint64_t acquire_cta_u64(const uint64_t* p) {
    uint64_t v;
    asm volatile("ld.acquire.cta.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void release_cta_u64(uint64_t* p, uint64_t v) {
    asm volatile("st.release.cta.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}

__device__ __forceinline__ uint16_t load_host_u16(const uint16_t* p) {
    uint16_t v;
    asm volatile("ld.global.cv.u16 %0, [%1];" : "=h"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void store_host_u16(uint16_t* p, uint16_t v) {
    asm volatile("st.global.wt.u16 [%0], %1;" :: "l"(p), "h"(v) : "memory");
}

__device__ __forceinline__ uint4 load_host_u128(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.cv.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void store_host_u128(uint4* p, uint4 v) {
    asm volatile("st.global.wt.v4.u32 [%0], {%1, %2, %3, %4};"
                 :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

__device__ __forceinline__ void alias_fence() {
    asm volatile("fence.proxy.alias;" ::: "memory");
}

__device__ __forceinline__ void bar_tx() {
    asm volatile("bar.sync 1, 128;" ::: "memory");
}

__device__ __forceinline__ void bar_rx() {
    asm volatile("bar.sync 2, 128;" ::: "memory");
}

__global__ void star_sum(Args a) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const size_t num_tiles = (a.n + a.capacity - 1) / a.capacity;
    const size_t C = arc_shard_stride(a.capacity);
    auto* payload = reinterpret_cast<uint16_t*>(static_cast<char*>(a.mapped) + HEADER);

    __shared__ uint64_t s_q_base;
    __shared__ uint64_t s_rx_done;
    if (tid == 0) {
        const uint64_t prev = a.seq[b];
        if (prev + num_tiles < prev) asm volatile("trap;");
        s_q_base = prev + 1;
        a.seq[b] = prev + num_tiles;
        release_cta_u64(&s_rx_done, 0);
    }
    __syncthreads();
    const uint64_t q_base = s_q_base;

    if (tid < HALF_T) {
        for (size_t t = 0; t < num_tiles; ++t) {
            const uint64_t q = q_base + t;
            const int slot = int((q - 1) & (K - 1));
            Control& c = static_cast<Control*>(a.mapped)[slot * B + b];

            if (tid == 0) {
                if (t >= PIPE_D) {
                    const uint64_t min_rx = uint64_t(t - PIPE_D + 1);
                    while (acquire_cta_u64(&s_rx_done) < min_rx) {
                        __nanosleep(32);
                    }
                }
                if (q > K) {
                    const uint64_t prev_q = q - K;
                    #pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        while (acquire_flag(&c.done[r].value) < prev_q) {
                            __nanosleep(48);
                        }
                    }
                }
            }
            bar_tx();
            alias_fence();

            const size_t base = t * a.capacity;
            const size_t rem = a.n - base;
            const size_t tile_n = rem < a.capacity ? rem : a.capacity;
            size_t begin = 0, end = 0;
            arc_shard_bounds(tile_n, b, begin, end);
            const size_t count = end - begin;
            if (tid == 0 && count > C) asm volatile("trap;");
            const uint16_t* x_shard = a.x + base + begin;
            uint16_t* in_r = payload + (size_t(slot * B + b) * 5 + size_t(a.rank)) * C;

            const size_t count_v = count >> 3;
            auto* in_v = reinterpret_cast<uint4*>(in_r);
            if ((uintptr_t(x_shard) & 15u) == 0u) {
                const auto* x_v = reinterpret_cast<const uint4*>(x_shard);
                for (size_t j = tid; j < count_v; j += HALF_T) {
                    store_host_u128(in_v + j, x_v[j]);
                }
            } else {
                for (size_t j = tid; j < count_v; j += HALF_T) {
                    Vec8Bf16 v;
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        v.u16[k] = x_shard[(j << 3) + k];
                    }
                    store_host_u128(in_v + j, v.u4);
                }
            }
            for (size_t j = (count_v << 3) + tid; j < count; j += HALF_T) {
                store_host_u16(in_r + j, x_shard[j]);
            }
            if (a.rank == 0 && tid == 0) {
                store_host_u64(&c.desc.tile_n, uint64_t(tile_n));
            }

            alias_fence();
            __threadfence_system();
            bar_tx();
            if (tid == 0) {
                release_flag(&c.ready[a.rank].value, q);
            }
            bar_tx();
        }
    } else {
        const int rx_tid = tid - HALF_T;
        for (size_t t = 0; t < num_tiles; ++t) {
            const uint64_t q = q_base + t;
            const int slot = int((q - 1) & (K - 1));
            Control& c = static_cast<Control*>(a.mapped)[slot * B + b];

            if (rx_tid == 0) {
                while (acquire_flag(&c.reduced.value) < q) {
                    __nanosleep(32);
                }
            }
            bar_rx();
            alias_fence();

            const size_t base = t * a.capacity;
            const size_t rem = a.n - base;
            const size_t tile_n = rem < a.capacity ? rem : a.capacity;
            size_t begin = 0, end = 0;
            arc_shard_bounds(tile_n, b, begin, end);
            const size_t count = end - begin;
            if (rx_tid == 0 && count > C) asm volatile("trap;");
            uint16_t* y_shard = a.y + base + begin;
            const uint16_t* out = payload + (size_t(slot * B + b) * 5 + 4) * C;

            const size_t count_v = count >> 3;
            const auto* out_v = reinterpret_cast<const uint4*>(out);
            if ((uintptr_t(y_shard) & 15u) == 0u) {
                auto* y_v = reinterpret_cast<uint4*>(y_shard);
                for (size_t j = rx_tid; j < count_v; j += HALF_T) {
                    y_v[j] = load_host_u128(out_v + j);
                }
            } else {
                for (size_t j = rx_tid; j < count_v; j += HALF_T) {
                    Vec8Bf16 v;
                    v.u4 = load_host_u128(out_v + j);
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        y_shard[(j << 3) + k] = v.u16[k];
                    }
                }
            }
            for (size_t j = (count_v << 3) + rx_tid; j < count; j += HALF_T) {
                y_shard[j] = load_host_u16(out + j);
            }

            alias_fence();
            bar_rx();
            if (rx_tid == 0) {
                release_flag(&c.done[a.rank].value, q);
                release_cta_u64(&s_rx_done, uint64_t(t + 1));
            }
            bar_rx();
        }
    }
    __syncthreads();
}

extern "C" size_t arc_bytes(size_t capacity) {
    const size_t C = arc_shard_stride(capacity);
    const size_t bytes = HEADER + size_t(K) * size_t(B) * 5 * C * sizeof(uint16_t);
    return ((bytes + 4095) / 4096) * 4096;
}

extern "C" int arc_blocks() { return B; }
extern "C" int arc_pipe_d() { return PIPE_D; }

extern "C" const char* arc_error(int code) {
    if (code == -1001) return "CPU lacks AVX-512F/AVX-512BW/AVX-512DQ support";
    if (code == -1002) return "pthread_create failed in arc_init_workers";
    if (code == -1003) return "invalid arguments or mismatched capacity in arc_init_workers";
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}

extern "C" int arc_register(void* host, size_t bytes, void** device) {
    int dev = 0, active = 0;
    cudaDeviceProp prop{};
    cudaError_t e = cudaGetDevice(&dev);
    if (e != cudaSuccess) return e;
    e = cudaGetDeviceProperties(&prop, dev);
    if (e != cudaSuccess) return e;
    if (!prop.canMapHostMemory || !prop.cooperativeLaunch)
        return cudaErrorNotSupported;

    e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, star_sum, T, 0);
    if (e != cudaSuccess) return e;
    if (B > active * prop.multiProcessorCount)
        return cudaErrorCooperativeLaunchTooLarge;

    e = cudaHostRegister(host, bytes, cudaHostRegisterPortable | cudaHostRegisterMapped);
    if (e != cudaSuccess) return e;

    e = cudaHostGetDevicePointer(device, host, 0);
    if (e != cudaSuccess) {
        cudaHostUnregister(host);
        return e;
    }
    arc_note_registered(host, *device, bytes);
    return cudaSuccess;
}

extern "C" int arc_unregister(void* host) {
    arc_stop_workers(host);
    return cudaHostUnregister(host);
}

extern "C" int arc_launch(
    void* mapped, void* seq, const void* x, void* y,
    size_t n, size_t capacity, int rank, void* stream) {
    if (!mapped || !seq || !x || !y ||
        n == 0 || capacity == 0 || (capacity % (B * 8)) != 0 ||
        n > ARC_MAX_NUMEL || rank < 0 || rank >= 4)
        return cudaErrorInvalidValue;

    if (rank == 0) {
        int wrc = arc_ensure_workers_for_mapped(mapped, capacity, rank);
        if (wrc != 0) return wrc;
    }

    Args a{
        mapped, static_cast<uint64_t*>(seq),
        static_cast<const uint16_t*>(x), static_cast<uint16_t*>(y),
        n, capacity, rank
    };
    void* params[] = {&a};
    return cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(star_sum),
        dim3(B), dim3(T), params, 0,
        reinterpret_cast<cudaStream_t>(stream));
}
"""


def _compile_star_shm_so(
    build_dir: Path,
    blocks: int = 32,
    workers: int = 8,
    pipe_d: int = 2,
) -> Path | None:
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    gxx = shutil.which("g++") or "/usr/bin/g++"
    if not Path(nvcc).exists() or not Path(gxx).exists():
        return None
    tag = f"b{int(blocks)}_w{int(workers)}_d{int(pipe_d)}"
    cpp_path = build_dir / f"arc_shm_cpu_{tag}.cpp"
    obj_path = build_dir / f"arc_shm_cpu_{tag}.o"
    cu_path = build_dir / f"arc_shm_gpu_{tag}.cu"
    so_path = build_dir / f"libarc_shm_ar_{tag}.so"
    if so_path.exists():
        return so_path

    cpp_path.write_text(ARC_SHM_STAR_CPU_CPP_SOURCE, encoding="utf-8")
    cu_path.write_text(ARC_SHM_STAR_CU_SOURCE, encoding="utf-8")

    proc_cpu = subprocess.run(
        [
            gxx,
            "-O3",
            "-std=c++17",
            "-fPIC",
            "-fno-fast-math",
            "-pthread",
            f"-DARC_BLOCKS={int(blocks)}",
            f"-DARC_WORKERS={int(workers)}",
            "-c",
            str(cpp_path),
            "-o",
            str(obj_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=20,
        check=False,
    )
    if proc_cpu.returncode != 0 or not obj_path.exists():
        (build_dir / f"nvcc_error_{tag}.txt").write_text(
            f"[g++ failed rc={proc_cpu.returncode}]\n{proc_cpu.stdout or ''}",
            encoding="utf-8",
        )
        return None

    proc_gpu = subprocess.run(
        [
            nvcc,
            "-O3",
            "-std=c++17",
            "-arch=sm_89",
            f"-DARC_BLOCKS={int(blocks)}",
            f"-DARC_PIPE_D={int(pipe_d)}",
            "--shared",
            "--cudart",
            "shared",
            "-Xcompiler",
            "-fPIC,-pthread",
            "--fmad=false",
            "--ftz=false",
            str(cu_path),
            str(obj_path),
            "-lpthread",
            "-o",
            str(so_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=25,
        check=False,
    )
    if proc_gpu.returncode != 0 or not so_path.exists():
        (build_dir / f"nvcc_error_{tag}.txt").write_text(
            f"[nvcc failed rc={proc_gpu.returncode}]\n{proc_gpu.stdout or ''}",
            encoding="utf-8",
        )
        return None
    return so_path


class _StarShmComm:
    """Tiled 1-hop Star-SHM AllReduce communicator backed by Linux `memfd_create`."""

    def __init__(self, so_path: str, group: Any, device: Any, capacity: int = STAR_ARENA_CAPACITY) -> None:
        import torch
        import torch.distributed as dist

        self._torch = torch
        self._dist = dist
        self.group = group
        self.device = device
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        if self.world_size != 4:
            raise ValueError(f"_StarShmComm requires world_size=4, got {self.world_size}")
        self.capacity = int(capacity)
        self.so_path = so_path
        self.so_sha256 = hashlib.sha256(Path(so_path).read_bytes()).hexdigest()

        lib = self.lib = ctypes.CDLL(so_path)
        lib.arc_bytes.argtypes = [ctypes.c_size_t]
        lib.arc_bytes.restype = ctypes.c_size_t
        lib.arc_blocks.argtypes = []
        lib.arc_blocks.restype = ctypes.c_int
        lib.arc_error.argtypes = [ctypes.c_int]
        lib.arc_error.restype = ctypes.c_char_p
        lib.arc_register.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.arc_register.restype = ctypes.c_int
        lib.arc_unregister.argtypes = [ctypes.c_void_p]
        lib.arc_unregister.restype = ctypes.c_int
        lib.arc_launch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.arc_launch.restype = ctypes.c_int
        if hasattr(lib, "arc_init_workers"):
            lib.arc_init_workers.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            lib.arc_init_workers.restype = ctypes.c_int

        if hasattr(lib, "arc_workers"):
            lib.arc_workers.argtypes = []
            lib.arc_workers.restype = ctypes.c_int
        if hasattr(lib, "arc_pipe_d"):
            lib.arc_pipe_d.argtypes = []
            lib.arc_pipe_d.restype = ctypes.c_int

        self.blocks = int(lib.arc_blocks())
        self.workers = int(lib.arc_workers()) if hasattr(lib, "arc_workers") else 8
        self.pipe_d = int(lib.arc_pipe_d()) if hasattr(lib, "arc_pipe_d") else 2
        self.bytes = int(lib.arc_bytes(self.capacity))
        self._root_fd: int | None = None
        if self.rank == 0:
            self._root_fd = os.memfd_create("arc_star_shm", 0)
            os.ftruncate(self._root_fd, self.bytes)
            if hasattr(os, "posix_fallocate"):
                os.posix_fallocate(self._root_fd, 0, self.bytes)
            share_info = [f"/proc/{os.getpid()}/fd/{self._root_fd}"]
        else:
            share_info = [None]
        group_ranks = dist.get_process_group_ranks(group)
        dist.broadcast_object_list(share_info, src=group_ranks[0], group=group)

        open_fd = os.open(str(share_info[0]), os.O_RDWR)
        try:
            self.mapping = mmap.mmap(
                open_fd,
                self.bytes,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
        finally:
            os.close(open_fd)

        dist.barrier(group=group)
        if self._root_fd is not None:
            os.close(self._root_fd)
            self._root_fd = None

        self.anchor = ctypes.c_char.from_buffer(self.mapping)
        self.host = ctypes.c_void_p(ctypes.addressof(self.anchor))
        self.mapped = ctypes.c_void_p()
        with torch.cuda.device(self.device):
            self._check(lib.arc_register(self.host, self.bytes, ctypes.byref(self.mapped)), "arc_register")
            if hasattr(lib, "arc_init_workers"):
                self._check(
                    lib.arc_init_workers(
                        self.host,
                        ctypes.c_size_t(self.bytes),
                        ctypes.c_size_t(self.capacity),
                        int(self.rank),
                    ),
                    "arc_init_workers",
                )
            self.seq = torch.zeros(int(lib.arc_blocks()), device=self.device, dtype=torch.int64)
            warm = torch.zeros(8, device=self.device, dtype=torch.bfloat16)
            self.all_reduce(warm, warm, stream=torch.cuda.current_stream(self.device))
            torch.cuda.synchronize(self.device)
            del warm
        dist.barrier(group=group)

    def _check(self, code: int, fn_name: str) -> None:
        if code != 0:
            err_ptr = self.lib.arc_error(code)
            msg = err_ptr.decode("utf-8", errors="replace") if err_ptr else f"code={code}"
            raise RuntimeError(f"{fn_name} failed ({code}): {msg}")

    def all_reduce(self, in_tensor: Any, out_tensor: Any = None, stream: Any = None) -> Any:
        torch = self._torch
        if out_tensor is None:
            out_tensor = torch.empty_like(in_tensor)
        if stream is None:
            stream = torch.cuda.current_stream(self.device)
        self._check(
            self.lib.arc_launch(
                self.mapped,
                ctypes.c_void_p(self.seq.data_ptr()),
                ctypes.c_void_p(in_tensor.data_ptr()),
                ctypes.c_void_p(out_tensor.data_ptr()),
                ctypes.c_size_t(in_tensor.numel()),
                ctypes.c_size_t(self.capacity),
                int(self.rank),
                ctypes.c_void_p(stream.cuda_stream),
            ),
            "arc_launch",
        )
        return out_tensor

    def destroy(self) -> None:
        torch = self._torch
        try:
            torch.cuda.synchronize(self.device)
        except Exception:
            pass
        if getattr(self, "host", None) is not None and self.host.value:
            host_ptr = self.host
            self.host = ctypes.c_void_p()
            try:
                with torch.cuda.device(self.device):
                    self.lib.arc_unregister(host_ptr)
            except Exception:
                pass
        if hasattr(self, "anchor"):
            del self.anchor
        if getattr(self, "mapping", None) is not None:
            try:
                self.mapping.close()
            except Exception:
                pass
            self.mapping = None
        if self._root_fd is not None:
            try:
                os.close(self._root_fd)
            except Exception:
                pass
            self._root_fd = None


def _verify_star_shm_multiseed(star: _StarShmComm, device: Any, capture_stream: Any) -> bool:
    """Verify non-NaN bitwise & NaN-classification FP32-pairwise->BF16 exactness across >K eager, tile-boundary, 6144-prefill & CUDAGraph calls.

    All tensors are generated directly on `device` inside `capture_stream` (eliminating cross-stream races and
    single-threaded CPU RNG overhead), and loops never break/return early on a single rank so collective
    `star.all_reduce` calls stay strictly lockstep across all 4 ranks.
    """
    import torch

    cap = int(star.capacity)
    shapes = [
        (48, 5120),
        (49, 5120),
        (96, 5120),
        (97, 5120),
        (24, 5120),
        (192, 5120),
        (193, 5120),
        (500, 5120),
        (6144, 5120),
        (1, 5123),
        (1, max(8, cap - 7)),
        (1, max(8, cap - 1)),
        (1, min(ARC_MAX_NUMEL, cap + 1)),
        (1, min(ARC_MAX_NUMEL, 4 * cap + 13)),
    ]
    ok = True
    with torch.cuda.stream(capture_stream):
        for step_idx, shape in enumerate(shapes):
            xs = []
            for r in range(4):
                gen = torch.Generator(device=device)
                gen.manual_seed(9001 + step_idx * 37 + r)
                scale = 4.0 if r % 2 == 0 else -4.0
                t_gpu = (
                    torch.randn(shape, generator=gen, dtype=torch.float32, device=device) * scale
                ).to(torch.bfloat16)
                xs.append(t_gpu)
            expected = ((xs[0].float() + xs[1].float()) + (xs[2].float() + xs[3].float())).to(torch.bfloat16)
            out = torch.zeros_like(xs[star.rank])
            star.all_reduce(xs[star.rank], out, stream=capture_stream)
            capture_stream.synchronize()
            if not bool(torch.equal(out.view(torch.int16), expected.view(torch.int16))):
                ok = False

        # IEEE-754 special-values test: +-0, BF16 subnormals, subnormal->normal carry,
        # tie-to-even midpoints, exact cancellation, overflow -> +-Inf, +-Inf, and NaN
        special_u16_patterns = [
            0x0000,  # +0.0
            0x8000,  # -0.0
            0x0001,  # min +subnormal
            0x8001,  # min -subnormal
            0x003F,  # mid +subnormal
            0x007E,  # near-max +subnormal (even)
            0x007F,  # max +subnormal (odd -> carry to 0x0080 normal)
            0x807F,  # max -subnormal
            0x0080,  # min +normal
            0x8080,  # min -normal
            0x3F80,  # +1.0
            0xBF80,  # -1.0
            0x3F81,  # +1.0 + 2^-7 (odd ulp)
            0xBF81,  # -1.0 - 2^-7
            0x3B80,  # +2^-8 (half ulp at 1.0 -> tie-to-even midpoint!)
            0xBB80,  # -2^-8
            0x7F7F,  # max finite +BF16 (overflows to +Inf on sum)
            0xFF7F,  # max finite -BF16 (overflows to -Inf on sum)
            0x7F80,  # +Inf
            0xFF80,  # -Inf
            0x7FC0,  # quiet NaN
            0x7FFF,  # canonical NaN
        ]
        n_pat = len(special_u16_patterns)
        signed_patterns = [p if p < 0x8000 else p - 0x10000 for p in special_u16_patterns]
        idx_grid = torch.arange(4096, device=device, dtype=torch.int64)
        pat_gpu = torch.tensor(signed_patterns, device=device, dtype=torch.int16)
        xs_spec = []
        for r in range(4):
            sel = ((idx_grid // (n_pat**r if r <= 2 else 7)) + r * 5) % n_pat
            xs_spec.append(pat_gpu[sel].view(torch.bfloat16).reshape(1, 4096).contiguous())
        exp_spec = (
            (xs_spec[0].float() + xs_spec[1].float()) + (xs_spec[2].float() + xs_spec[3].float())
        ).to(torch.bfloat16)
        out_spec = torch.zeros_like(xs_spec[star.rank])
        star.all_reduce(xs_spec[star.rank], out_spec, stream=capture_stream)
        capture_stream.synchronize()
        nan_out = torch.isnan(out_spec)
        nan_exp = torch.isnan(exp_spec)
        if not bool(torch.equal(nan_out, nan_exp)):
            ok = False
        non_nan = ~nan_exp
        if not bool(
            torch.equal(out_spec[non_nan].view(torch.int16), exp_spec[non_nan].view(torch.int16))
        ):
            ok = False

        # Capture a 3-op graph (96-row decode + 24-row draft + 384-row 2-tile multi-slot op) and replay 5 times (> K=4)
        g_in96 = torch.zeros((96, 5120), dtype=torch.bfloat16, device=device)
        g_out96 = torch.zeros_like(g_in96)
        g_in24 = torch.zeros((24, 5120), dtype=torch.bfloat16, device=device)
        g_out24 = torch.zeros_like(g_in24)
        g_in384 = torch.zeros((384, 5120), dtype=torch.bfloat16, device=device)
        g_out384 = torch.zeros_like(g_in384)
    capture_stream.synchronize()

    g_test = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_test, stream=capture_stream):
        star.all_reduce(g_in96, g_out96, stream=capture_stream)
        star.all_reduce(g_in24, g_out24, stream=capture_stream)
        star.all_reduce(g_in384, g_out384, stream=capture_stream)
    capture_stream.synchronize()

    try:
        for rep in range(5):
            with torch.cuda.stream(capture_stream):
                xs96, xs24, xs384 = [], [], []
                for r in range(4):
                    g1 = torch.Generator(device=device).manual_seed(40000 + rep * 19 + r)
                    g2 = torch.Generator(device=device).manual_seed(80000 + rep * 23 + r)
                    g3 = torch.Generator(device=device).manual_seed(120000 + rep * 29 + r)
                    xs96.append(torch.randn((96, 5120), generator=g1, dtype=torch.float32, device=device).to(torch.bfloat16))
                    xs24.append(torch.randn((24, 5120), generator=g2, dtype=torch.float32, device=device).to(torch.bfloat16))
                    xs384.append(torch.randn((384, 5120), generator=g3, dtype=torch.float32, device=device).to(torch.bfloat16))
                exp96 = ((xs96[0].float() + xs96[1].float()) + (xs96[2].float() + xs96[3].float())).to(torch.bfloat16)
                exp24 = ((xs24[0].float() + xs24[1].float()) + (xs24[2].float() + xs24[3].float())).to(torch.bfloat16)
                exp384 = ((xs384[0].float() + xs384[1].float()) + (xs384[2].float() + xs384[3].float())).to(torch.bfloat16)
                g_in96.copy_(xs96[star.rank])
                g_in24.copy_(xs24[star.rank])
                g_in384.copy_(xs384[star.rank])
                g_out96.zero_()
                g_out24.zero_()
                g_out384.zero_()
                g_test.replay()
            capture_stream.synchronize()
            if not bool(torch.equal(g_out96.view(torch.int16), exp96.view(torch.int16))):
                ok = False
            if not bool(torch.equal(g_out24.view(torch.int16), exp24.view(torch.int16))):
                ok = False
            if not bool(torch.equal(g_out384.view(torch.int16), exp384.view(torch.int16))):
                ok = False
    finally:
        del g_test
        torch.cuda.synchronize(device)
    return ok


def _write_pool_checkpoint(
    checkpoint_path: str,
    rank: int,
    t0: float,
    history: list[dict[str, Any]],
    phase: str,
    **extra: Any,
) -> None:
    if rank != 0 or not checkpoint_path:
        return
    item: dict[str, Any] = {
        "phase": phase,
        "elapsed_seconds": round(time.monotonic() - t0, 3),
        **extra,
    }
    history.append(item)
    p = Path(checkpoint_path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    try:
        tmp.write_text(
            json.dumps({"last_phase": phase, "events": history}, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(p)
    except OSError:
        pass


def _benchmark_profile_in_pool(
    *,
    rank: int,
    world_size: int,
    device: Any,
    comm: _LiteNcclComm,
    capture_stream: Any,
    free_baseline: int,
    res_baseline: int,
    total_bytes: int,
    spec_k: int,
    profile_name: str,
    star_so_path: str,
    star_capacity: int = STAR_ARENA_CAPACITY,
    diagnostic_only: bool,
    checkpoint_path: str,
    pool_t0: float,
    ckpt_history: list[dict[str, Any]],
) -> dict[str, Any] | None:
    import torch
    import torch.distributed as dist

    prof_t0 = time.monotonic()
    m_main = 24 * (spec_k + 1)
    star: _StarShmComm | None = None
    graph = None

    _write_pool_checkpoint(
        checkpoint_path, rank, pool_t0, ckpt_history, f"{profile_name}:start"
    )
    try:
        torch.cuda.synchronize(device)
        dist.barrier()
        if star_so_path:
            star = _StarShmComm(
                so_path=star_so_path,
                group=dist.group.WORLD,
                device=device,
                capacity=int(star_capacity or STAR_ARENA_CAPACITY),
            )
            _write_pool_checkpoint(
                checkpoint_path,
                rank,
                pool_t0,
                ckpt_history,
                f"{profile_name}:star_registered",
                star_blocks=star.blocks,
                star_shm_mib=round(star.bytes / (1024.0**2), 2),
            )

        # 1) Eager numerical verification (AllReduce + AllGather) on capture_stream
        with torch.cuda.stream(capture_stream):
            chk_in = torch.full((m_main, 5120), float(rank + 1), dtype=torch.bfloat16, device=device)
            chk_out = (
                star.all_reduce(chk_in, stream=capture_stream)
                if star is not None
                else comm.all_reduce(chk_in, stream=capture_stream)
            )
            ag_chk_in = torch.full((m_main, 1280), float(rank + 10), dtype=torch.bfloat16, device=device)
            ag_chk_out = torch.empty((world_size * m_main, 1280), dtype=torch.bfloat16, device=device)
            comm.all_gather(ag_chk_out, ag_chk_in, stream=capture_stream)
        capture_stream.synchronize()

        expected_sum = float(sum(range(1, world_size + 1)))
        ar_eager_ok = bool(torch.all(chk_out == expected_sum).item())
        ag_eager_ok = all(
            bool(torch.all(ag_chk_out[r * m_main : (r + 1) * m_main] == float(r + 10)).item())
            for r in range(world_size)
        )
        del chk_in, chk_out, ag_chk_in, ag_chk_out

        local_star_multiseed_ok = True
        if star is not None:
            local_star_multiseed_ok = _verify_star_shm_multiseed(star, device, capture_stream)
        # Aggregate multiseed verification collectively across all 4 ranks before proceeding
        ms_vec = torch.tensor([1 if local_star_multiseed_ok else 0], dtype=torch.int32)
        dist.all_reduce(ms_vec, op=dist.ReduceOp.MIN)
        star_multiseed_ok = bool(int(ms_vec.item()) == 1)
        if star is not None:
            _write_pool_checkpoint(
                checkpoint_path,
                rank,
                pool_t0,
                ckpt_history,
                f"{profile_name}:star_verified",
                star_multiseed_ok=star_multiseed_ok,
            )

        # 2) Pre-allocate rotating buffers on capture_stream for exact Qwen3.8-27B MTP=K step:
        #    132 x AllReduce([M, 5120] bf16) + 3*(K-1) x AllReduce([24, 5120] bf16) + (2*K+1) AllGathers
        n_rot_main = 8
        n_rot24 = 4
        with torch.cuda.stream(capture_stream):
            ar_main_in = [
                torch.full((m_main, 5120), float(rank + 1) * 0.125, dtype=torch.bfloat16, device=device)
                for _ in range(n_rot_main)
            ]
            ar_main_out = [torch.zeros_like(x) for x in ar_main_in]
            ar24_in = [
                torch.full((24, 5120), float(rank + 1) * 0.25, dtype=torch.bfloat16, device=device)
                for _ in range(n_rot24)
            ]
            ar24_out = [torch.zeros_like(x) for x in ar24_in]

            ag_main_fc_in = torch.full((m_main, 1280), float(rank + 1), dtype=torch.bfloat16, device=device)
            ag_main_fc_out = torch.zeros((world_size * m_main, 1280), dtype=torch.bfloat16, device=device)
            ag24_fc_in = torch.full((24, 1280), float(rank + 1), dtype=torch.bfloat16, device=device)
            ag24_fc_out = torch.zeros((world_size * 24, 1280), dtype=torch.bfloat16, device=device)

            ag_main_tk_in = torch.full((m_main, 128), rank + 1, dtype=torch.int32, device=device)
            ag_main_tk_out = torch.zeros((world_size * m_main, 128), dtype=torch.int32, device=device)
            ag24_tk_in = torch.full((24, 128), rank + 1, dtype=torch.int32, device=device)
            ag24_tk_out = torch.zeros((world_size * 24, 128), dtype=torch.int32, device=device)
        capture_stream.synchronize()

        ar_fn = star.all_reduce if star is not None else comm.all_reduce

        def _run_one_decode_step(s: torch.cuda.Stream) -> None:
            for i in range(129):
                b = i & (n_rot_main - 1)
                ar_fn(ar_main_in[b], ar_main_out[b], stream=s)
            comm.all_gather(ag_main_tk_out, ag_main_tk_in, stream=s)

            ar_fn(ar_main_in[1], ar_main_out[1], stream=s)
            comm.all_gather(ag_main_fc_out, ag_main_fc_in, stream=s)
            ar_fn(ar_main_in[2], ar_main_out[2], stream=s)
            ar_fn(ar_main_in[3], ar_main_out[3], stream=s)
            comm.all_gather(ag24_tk_out, ag24_tk_in, stream=s)

            idx24 = 0
            for _ in range(spec_k - 1):
                b0 = idx24 & (n_rot24 - 1)
                ar_fn(ar24_in[b0], ar24_out[b0], stream=s)
                comm.all_gather(ag24_fc_out, ag24_fc_in, stream=s)
                b1 = (idx24 + 1) & (n_rot24 - 1)
                b2 = (idx24 + 2) & (n_rot24 - 1)
                ar_fn(ar24_in[b1], ar24_out[b1], stream=s)
                ar_fn(ar24_in[b2], ar24_out[b2], stream=s)
                comm.all_gather(ag24_tk_out, ag24_tk_in, stream=s)
                idx24 += 3

        torch.cuda.synchronize(device)
        dist.barrier()
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                _run_one_decode_step(capture_stream)
        capture_stream.synchronize()
        dist.barrier()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            _run_one_decode_step(capture_stream)
        capture_stream.synchronize()
        dist.barrier()

        # Verify graph replay numerical exactness with stream-ordered output resets
        with torch.cuda.stream(capture_stream):
            for t in ar_main_out:
                t.zero_()
            for t in ar24_out:
                t.zero_()
            ag_main_fc_out.zero_()
            ag24_tk_out.zero_()
            graph.replay()
        capture_stream.synchronize()

        exp_main = expected_sum * 0.125
        exp24 = expected_sum * 0.25
        graph_ok = (
            all(bool(torch.all(t == exp_main).item()) for t in ar_main_out)
            and all(bool(torch.all(t == exp24).item()) for t in ar24_out)
            and all(
                bool(torch.all(ag_main_fc_out[r * m_main : (r + 1) * m_main] == float(r + 1)).item())
                for r in range(world_size)
            )
            and all(
                bool(torch.all(ag24_tk_out[r * 24 : (r + 1) * 24] == (r + 1)).item())
                for r in range(world_size)
            )
        )
        local_numerical_ok = bool(ar_eager_ok and ag_eager_ok and graph_ok and star_multiseed_ok)

        torch.cuda.synchronize(device)
        free_after_graph, _ = torch.cuda.mem_get_info(device)
        res_after_graph = torch.cuda.memory_reserved(device)
        nccl_driver_mib_graph = max(
            0.0,
            ((free_baseline - free_after_graph) - (res_after_graph - res_baseline)) / (1024.0**2),
        )

        # Warmup & time 40 full-step CUDA-graph replays
        warmup_iters = 12
        timed_iters = 40
        with torch.cuda.stream(capture_stream):
            for _ in range(warmup_iters):
                graph.replay()
        capture_stream.synchronize()
        dist.barrier()

        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(capture_stream):
            ev_start.record(capture_stream)
            for _ in range(timed_iters):
                graph.replay()
            ev_end.record(capture_stream)
        capture_stream.synchronize()
        decode_step_ms = float(ev_start.elapsed_time(ev_end)) / float(timed_iters)

        # Time 8 chunked-prefill-sized [6144, 5120] BF16 AllReduces allocated on capture_stream
        with torch.cuda.stream(capture_stream):
            pf_in = torch.full((6144, 5120), float(rank + 1) * 0.125, dtype=torch.bfloat16, device=device)
            pf_out = torch.empty_like(pf_in)
            for _ in range(2):
                ar_fn(pf_in, pf_out, stream=capture_stream)
        capture_stream.synchronize()
        dist.barrier()

        pf_start = torch.cuda.Event(enable_timing=True)
        pf_end = torch.cuda.Event(enable_timing=True)
        pf_iters = 8
        with torch.cuda.stream(capture_stream):
            pf_start.record(capture_stream)
            for _ in range(pf_iters):
                ar_fn(pf_in, pf_out, stream=capture_stream)
            pf_end.record(capture_stream)
        capture_stream.synchronize()
        prefill_ar6144_ms = float(pf_start.elapsed_time(pf_end)) / float(pf_iters)
        pf_ok = bool(torch.all(pf_out == exp_main).item())
        local_numerical_ok = bool(local_numerical_ok and pf_ok)

        # Collectively reduce pass/fail across all 4 ranks
        ok_vec = torch.tensor([1 if local_numerical_ok else 0], dtype=torch.int32)
        dist.all_reduce(ok_vec, op=dist.ReduceOp.MIN)
        numerical_ok = bool(int(ok_vec.item()) == 1)

        torch.cuda.synchronize(device)
        free_after_pf, _ = torch.cuda.mem_get_info(device)
        res_after_pf = torch.cuda.memory_reserved(device)
        nccl_driver_mib_pf = max(
            0.0,
            ((free_baseline - free_after_pf) - (res_after_pf - res_baseline)) / (1024.0**2),
        )
        nccl_driver_mib = max(nccl_driver_mib_graph, nccl_driver_mib_pf)
        total_used_mib = (free_baseline - free_after_graph) / (1024.0**2)
        del pf_in, pf_out

        local_report: dict[str, Any] = {
            "rank": rank,
            "spec_k": spec_k,
            "decode_rows": m_main,
            "decode_step_ms": round(decode_step_ms, 4),
            "prefill_ar6144_ms": round(prefill_ar6144_ms, 4),
            "nccl_driver_mib": round(nccl_driver_mib, 2),
            "nccl_driver_mib_graph": round(nccl_driver_mib_graph, 2),
            "total_used_mib": round(total_used_mib, 2),
            "gpu_total_mib": round(total_bytes / (1024.0**2), 1),
            "numerical_ok": numerical_ok,
            "nccl_raw_version": int(getattr(comm, "nccl_version", 0) or 0),
            "nccl_so_path": str(getattr(comm, "so_path", "")),
        }
        if star is not None:
            local_report["star_multiseed_ok"] = star_multiseed_ok
            local_report["star_shm_mib"] = round(star.bytes / (1024.0**2), 2)
            local_report["star_so_sha256"] = star.so_sha256
            local_report["star_blocks"] = star.blocks
            local_report["star_workers"] = star.workers
            local_report["star_pipe_d"] = star.pipe_d
            local_report["star_capacity"] = star.capacity

        gathered: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(gathered, local_report)
        prof_wall = round(time.monotonic() - prof_t0, 3)

        summary: dict[str, Any] | None = None
        if rank == 0:
            ranks_data = [g for g in gathered if isinstance(g, dict)]
            summary = {
                "profile": profile_name,
                "env": dict(SAFE_SHM),
                "diagnostic_only": diagnostic_only,
                "wall_seconds": prof_wall,
                "returncode": 0,
                "status": "ok",
                "backend": "star_shm" if star is not None else "pynccl",
                "spec_k": spec_k,
                "decode_rows": m_main,
                "numerical_ok": all(bool(r.get("numerical_ok")) for r in ranks_data),
                "decode_step_ms": round(max(float(r["decode_step_ms"]) for r in ranks_data), 4),
                "prefill_ar6144_ms": round(max(float(r["prefill_ar6144_ms"]) for r in ranks_data), 4),
                "max_nccl_driver_mib": round(max(float(r["nccl_driver_mib"]) for r in ranks_data), 2),
                "max_total_used_mib": round(max(float(r["total_used_mib"]) for r in ranks_data), 2),
                "nccl_raw_version": ranks_data[0].get("nccl_raw_version"),
                "nccl_so_path": ranks_data[0].get("nccl_so_path"),
                "per_rank": ranks_data,
            }
            if star is not None:
                summary["star_multiseed_ok"] = all(bool(r.get("star_multiseed_ok")) for r in ranks_data)
                summary["star_shm_mib"] = ranks_data[0].get("star_shm_mib")
                summary["star_so_sha256"] = ranks_data[0].get("star_so_sha256")
                summary["star_so_path"] = star_so_path
                summary["star_blocks"] = ranks_data[0].get("star_blocks")
                summary["star_workers"] = ranks_data[0].get("star_workers")
                summary["star_pipe_d"] = ranks_data[0].get("star_pipe_d")
                summary["star_capacity"] = ranks_data[0].get("star_capacity")
                summary["star_max_numel"] = ARC_MAX_NUMEL
            summary["effective_step_ms"] = round(compute_effective_step_ms(summary), 4)
            _write_pool_checkpoint(
                checkpoint_path,
                rank,
                pool_t0,
                ckpt_history,
                f"{profile_name}:done",
                decode_step_ms=summary["decode_step_ms"],
                prefill_ar6144_ms=summary["prefill_ar6144_ms"],
                numerical_ok=summary["numerical_ok"],
                wall_seconds=prof_wall,
            )
        return summary
    finally:
        if graph is not None:
            del graph
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass
        if star is not None:
            try:
                star.destroy()
            except Exception:
                pass
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            if dist.is_initialized():
                dist.barrier()
        except Exception:
            pass


def _pool_rank_worker(
    rank: int,
    world_size: int,
    port: int,
    pool_plan: list[dict[str, Any]],
    results_path: str,
    checkpoint_path: str,
) -> None:
    pool_t0 = time.monotonic()
    ckpt_history: list[dict[str, Any]] = []
    _write_pool_checkpoint(checkpoint_path, rank, pool_t0, ckpt_history, "worker_spawned")

    import torch
    import torch.distributed as dist

    spec_k = max(1, int(os.environ.get("ARC_AUTOTUNE_SPEC_K", str(DEFAULT_SPEC_K))))
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.synchronize(device)
    free_baseline, total_bytes = torch.cuda.mem_get_info(device)
    res_baseline = torch.cuda.memory_reserved(device)
    _write_pool_checkpoint(checkpoint_path, rank, pool_t0, ckpt_history, "import_cuda_ready")

    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=30),
    )
    _write_pool_checkpoint(checkpoint_path, rank, pool_t0, ckpt_history, "gloo_ready")

    comm = _LiteNcclComm(group=dist.group.WORLD, device=device)
    capture_stream = torch.cuda.Stream(device=device)
    _write_pool_checkpoint(
        checkpoint_path,
        rank,
        pool_t0,
        ckpt_history,
        "nccl_ready",
        nccl_raw_version=int(getattr(comm, "nccl_version", 0) or 0),
    )

    completed_summaries: list[dict[str, Any]] = []
    try:
        for item in pool_plan:
            summary = _benchmark_profile_in_pool(
                rank=rank,
                world_size=world_size,
                device=device,
                comm=comm,
                capture_stream=capture_stream,
                free_baseline=free_baseline,
                res_baseline=res_baseline,
                total_bytes=total_bytes,
                spec_k=spec_k,
                profile_name=str(item["profile"]),
                star_so_path=str(item.get("star_so_path") or ""),
                star_capacity=int(item.get("star_capacity") or STAR_ARENA_CAPACITY),
                diagnostic_only=bool(item.get("diagnostic_only", False)),
                checkpoint_path=checkpoint_path,
                pool_t0=pool_t0,
                ckpt_history=ckpt_history,
            )
            if rank == 0 and summary is not None:
                completed_summaries.append(summary)
                res_p = Path(results_path)
                tmp_p = res_p.with_suffix(res_p.suffix + ".tmp")
                tmp_p.write_text(json.dumps(completed_summaries), encoding="utf-8")
                tmp_p.replace(res_p)
    finally:
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass
        try:
            comm.destroy()
        except Exception:
            pass
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass
        _write_pool_checkpoint(checkpoint_path, rank, pool_t0, ckpt_history, "teardown_done")


def _run_pool_entry(
    spec_path: str,
    results_path: str,
    checkpoint_path: str,
    max_pool_seconds: float = 75.0,
) -> int:
    # Avoid importing `torch` in the parent coordinator process (~4.5s saved!)
    pool_plan = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    if not _has_four_nvidia_gpus():
        skipped = [
            {
                "profile": str(item["profile"]),
                "env": dict(SAFE_SHM),
                "diagnostic_only": bool(item.get("diagnostic_only", False)),
                "returncode": 0,
                "status": "skipped",
                "reason": "expected 4 GPUs (/dev/nvidia3 absent)",
            }
            for item in pool_plan
        ]
        Path(results_path).write_text(json.dumps(skipped), encoding="utf-8")
        return 0

    port = _find_free_port()
    ctx = mp.get_context("spawn")
    procs = []
    for r in range(4):
        p = ctx.Process(
            target=_pool_rank_worker,
            args=(r, 4, port, pool_plan, results_path, checkpoint_path),
        )
        p.start()
        procs.append(p)
    exit_codes = []
    deadline = time.monotonic() + max_pool_seconds
    for p in procs:
        rem = max(0.2, deadline - time.monotonic())
        p.join(timeout=rem)
        if p.is_alive():
            p.kill()
            p.join(timeout=1.0)
        exit_codes.append(p.exitcode)
    if any(code != 0 for code in exit_codes):
        ckpt_p = Path(checkpoint_path)
        prev: dict[str, Any] = {}
        if ckpt_p.exists():
            try:
                prev = json.loads(ckpt_p.read_text(encoding="utf-8"))
            except Exception:
                prev = {}
        prev["exit_codes"] = exit_codes
        try:
            ckpt_p.write_text(json.dumps(prev, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
        return 1
    return 0


def _run_single_profile_entry(profile_name: str, result_path: str, star_so_path: str = "") -> int:
    with tempfile.TemporaryDirectory(prefix=f"arc_single_{profile_name}_") as tmp_s:
        tmpdir = Path(tmp_s)
        spec_file = tmpdir / "spec.json"
        pool_res = tmpdir / "pool_results.json"
        ckpt_file = tmpdir / "checkpoint.json"
        spec_file.write_text(
            json.dumps(
                [{"profile": profile_name, "star_so_path": star_so_path, "diagnostic_only": False}]
            ),
            encoding="utf-8",
        )
        rc = _run_pool_entry(str(spec_file), str(pool_res), str(ckpt_file), max_pool_seconds=45.0)
        if pool_res.exists():
            items = json.loads(pool_res.read_text(encoding="utf-8"))
            if items:
                Path(result_path).write_text(json.dumps(items[0]), encoding="utf-8")
                return rc
        if rc != 0 and not Path(result_path).exists():
            ckpt_data = json.loads(ckpt_file.read_text(encoding="utf-8")) if ckpt_file.exists() else {}
            Path(result_path).write_text(
                json.dumps(
                    {
                        "status": "error",
                        "exit_codes": ckpt_data.get("exit_codes"),
                        "checkpoint": ckpt_data,
                    }
                ),
                encoding="utf-8",
            )
        return rc


def run_autotune(output_json_path: Path, max_wall_seconds: float = 170.0) -> dict[str, Any]:
    t0 = time.monotonic()
    deadline = t0 + max_wall_seconds
    inherited_env = {
        k: v
        for k, v in sorted(os.environ.items())
        if any(tok in k for tok in ("NCCL", "CUDA", "VLLM", "OMP", "ARC_"))
    }
    # Enable verified Host-CPU AVX-512 StarShm candidates by default (strictly gated behind
    # bitwise FP32-tree + IEEE special-value + >K CUDAGraph checks and >= 1.5 ms step gain).
    allow_fp32_tree_star = os.environ.get("ARC_ALLOW_FP32_TREE_STAR", "1").strip() not in (
        "0",
        "false",
        "False",
    )
    star_diagnostic_only = not allow_fp32_tree_star

    persist_dir = Path(
        os.environ.get("ARC_STAR_SHM_PERSIST_DIR", "/opt/arc_pi_cache/arc_shm_ar")
    )
    results: list[dict[str, Any]] = []
    pool_checkpoint: dict[str, Any] = {}
    pool_wall_seconds = 0.0

    with tempfile.TemporaryDirectory(prefix="arc_star_shm_pool_") as work_s:
        workdir = Path(work_s)
        build_dir = workdir / "build"
        build_dir.mkdir(parents=True, exist_ok=True)

        pool_plan: list[dict[str, Any]] = []
        for profile_name, _overrides in CANDIDATE_PROFILES:
            pool_plan.append(
                {
                    "profile": profile_name,
                    "star_so_path": "",
                    "star_capacity": 0,
                    "diagnostic_only": False,
                }
            )

        compile_skipped_entries: list[dict[str, Any]] = []
        compiled_so_cache: dict[tuple[int, int, int], Path | None] = {}
        for star_entry in STAR_CANDIDATE_PROFILES:
            if len(star_entry) == 5:
                star_name, star_blocks, star_cap, star_workers, star_pipe_d = star_entry
            elif len(star_entry) == 3:
                star_name, star_blocks, star_cap = star_entry  # type: ignore[misc]
                star_workers, star_pipe_d = 8, 2
            else:
                star_name, star_blocks = star_entry  # type: ignore[misc]
                star_cap, star_workers, star_pipe_d = STAR_ARENA_CAPACITY, 8, 2
            cache_key = (int(star_blocks), int(star_workers), int(star_pipe_d))
            if cache_key not in compiled_so_cache:
                compiled_so_cache[cache_key] = _compile_star_shm_so(
                    build_dir,
                    blocks=int(star_blocks),
                    workers=int(star_workers),
                    pipe_d=int(star_pipe_d),
                )
            so_path = compiled_so_cache[cache_key]
            if so_path is not None:
                pool_plan.append(
                    {
                        "profile": star_name,
                        "star_so_path": str(so_path),
                        "star_capacity": int(star_cap),
                        "diagnostic_only": star_diagnostic_only,
                    }
                )
            else:
                tag = f"b{int(star_blocks)}_w{int(star_workers)}_d{int(star_pipe_d)}"
                err_file = build_dir / f"nvcc_error_{tag}.txt"
                compile_skipped_entries.append(
                    {
                        "profile": star_name,
                        "env": dict(SAFE_SHM),
                        "diagnostic_only": star_diagnostic_only,
                        "returncode": 0,
                        "status": "compile_skipped",
                        "reason": (
                            err_file.read_text(encoding="utf-8")[:600]
                            if err_file.exists()
                            else "nvcc_unavailable"
                        ),
                    }
                )

        spec_file = workdir / "pool_spec.json"
        res_file = workdir / "pool_results.json"
        ckpt_file = workdir / "pool_checkpoint.json"
        spec_file.write_text(json.dumps(pool_plan), encoding="utf-8")

        child_env = profile_environment(dict(os.environ), dict(SAFE_SHM))
        child_env["PYTHONNOUSERSITE"] = "1"
        child_env["OMP_NUM_THREADS"] = "1"
        child_env["MKL_NUM_THREADS"] = "1"
        child_env["OPENBLAS_NUM_THREADS"] = "1"
        child_env["NCCL_DEBUG"] = "INFO"
        child_env["NCCL_DEBUG_SUBSYS"] = "INIT,P2P,SHM,TUNING"
        child_env["NCCL_DEBUG_FILE"] = str(workdir / "nccl.%p.log")

        pool_timeout = max(20.0, min(85.0, deadline - time.monotonic()))
        inner_timeout = max(15.0, pool_timeout - 5.0)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--run-pool",
            str(spec_file),
            str(res_file),
            str(ckpt_file),
            f"{inner_timeout:.1f}",
        ]

        pool_t0 = time.monotonic()
        proc = subprocess.Popen(
            cmd,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            out_txt, _ = proc.communicate(timeout=pool_timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                out_txt, _ = proc.communicate(timeout=1.0)
            except Exception:
                out_txt = ""
            rc = -9
        pool_wall_seconds = round(time.monotonic() - pool_t0, 3)

        if ckpt_file.exists():
            try:
                pool_checkpoint = json.loads(ckpt_file.read_text(encoding="utf-8"))
            except Exception as exc:
                pool_checkpoint = {"parse_error": str(exc)}
        if out_txt:
            pool_checkpoint["output_tail"] = out_txt[-800:]
        pool_checkpoint["pool_returncode"] = rc
        pool_checkpoint["pool_wall_seconds"] = pool_wall_seconds

        nccl_highlights = _extract_nccl_log_highlights(workdir)
        completed_by_name: dict[str, dict[str, Any]] = {}
        if res_file.exists():
            try:
                raw_list = json.loads(res_file.read_text(encoding="utf-8"))
                if isinstance(raw_list, list):
                    for entry in raw_list:
                        if isinstance(entry, dict) and entry.get("profile"):
                            completed_by_name[str(entry["profile"])] = entry
            except Exception as exc:
                pool_checkpoint["results_parse_error"] = str(exc)

        for item in pool_plan:
            pname = str(item["profile"])
            if pname in completed_by_name:
                entry = completed_by_name[pname]
                if pname == "default":
                    entry["nccl_log_highlights"] = nccl_highlights
                if out_txt and entry.get("returncode", 0) != 0:
                    entry["output_tail"] = out_txt[-800:]
                results.append(entry)
            else:
                fail_entry: dict[str, Any] = {
                    "profile": pname,
                    "env": dict(SAFE_SHM),
                    "diagnostic_only": bool(item.get("diagnostic_only", False)),
                    "wall_seconds": pool_wall_seconds,
                    "returncode": rc if rc != 0 else 1,
                    "status": "timeout" if rc == -9 else "failed",
                    "last_checkpoint_phase": pool_checkpoint.get("last_phase"),
                    "exit_codes": pool_checkpoint.get("exit_codes"),
                }
                if out_txt:
                    fail_entry["output_tail"] = out_txt[-800:]
                if pname == "default":
                    fail_entry["nccl_log_highlights"] = nccl_highlights
                results.append(fail_entry)

        results.extend(compile_skipped_entries)

        selection = select_best_profile(results)
        star_sel = selection.get("star_shm_selection") or _disabled_star_selection()
        if star_sel.get("enabled"):
            try:
                src_so = Path(str(star_sel["so_path"]))
                persist_dir.mkdir(parents=True, exist_ok=True)
                dst_so = persist_dir / src_so.name
                tmp_so = dst_so.with_suffix(".so.tmp")
                shutil.copy2(src_so, tmp_so)
                os.chmod(tmp_so, 0o755)
                tmp_so.replace(dst_so)
                copied_sha = hashlib.sha256(dst_so.read_bytes()).hexdigest()
                if copied_sha != star_sel.get("so_sha256"):
                    raise RuntimeError("persisted StarShm .so sha256 mismatch")
                star_sel["so_path"] = str(dst_so)
                selection["star_shm_selection"] = star_sel
            except Exception as persist_exc:
                # Fail closed back to pure SAFE_SHM PyNCCL if persisting .so fails
                fallback_env = dict(SAFE_SHM)
                selection = {
                    **selection,
                    "selected_profile": "default",
                    "selected_env": fallback_env,
                    "unset_env": _unset_keys_for(fallback_env),
                    "star_shm_selection": {
                        **_disabled_star_selection("persist_failed"),
                        "error": f"{type(persist_exc).__name__}: {persist_exc}",
                    },
                    "reason": "star_shm_persist_failed_fallback_default",
                    "selected_decode_ms": selection.get("default_decode_ms"),
                    "selected_prefill_ar6144_ms": selection.get("default_prefill_ar6144_ms"),
                    "selected_effective_ms": selection.get("default_effective_ms"),
                    "gain_decode_ms": 0.0,
                    "gain_effective_ms": 0.0,
                    "extra_nccl_mib": 0.0,
                }

    report = {
        "at_epoch": time.time(),
        "total_wall_seconds": round(time.monotonic() - t0, 3),
        "allow_fp32_tree_star": allow_fp32_tree_star,
        "inherited_env": inherited_env,
        "pool_checkpoint": pool_checkpoint,
        "selection": selection,
        "selected_profile": selection["selected_profile"],
        "selected_env": selection["selected_env"],
        "unset_env": selection["unset_env"],
        "star_shm_selection": selection.get("star_shm_selection") or _disabled_star_selection(),
        "profiles": results,
    }
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = output_json_path.with_suffix(output_json_path.suffix + ".tmp")
    tmp_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_out.replace(output_json_path)
    print(
        "NCCL_AUTOTUNE_DONE",
        json.dumps(
            {
                "selected_profile": selection["selected_profile"],
                "selected_env": selection["selected_env"],
                "star_shm_selection": selection.get("star_shm_selection"),
                "default_decode_ms": selection.get("default_decode_ms"),
                "selected_decode_ms": selection.get("selected_decode_ms"),
                "default_prefill_ar6144_ms": selection.get("default_prefill_ar6144_ms"),
                "selected_prefill_ar6144_ms": selection.get("selected_prefill_ar6144_ms"),
                "gain_decode_ms": selection.get("gain_decode_ms"),
                "gain_effective_ms": selection.get("gain_effective_ms"),
                "extra_nccl_mib": selection.get("extra_nccl_mib"),
                "wall_seconds": report["total_wall_seconds"],
            }
        ),
        flush=True,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) in (4, 5) and args[0] == "--run-pool":
        max_s = float(args[4]) if len(args) == 5 else 75.0
        return _run_pool_entry(args[1], args[2], args[3], max_pool_seconds=max_s)
    if len(args) in (3, 4) and args[0] == "--run-profile":
        star_so = args[3] if len(args) == 4 else ""
        return _run_single_profile_entry(args[1], args[2], star_so)
    if len(args) != 1:
        raise SystemExit("Usage: python autotune_nccl.py <output_json_path>")
    run_autotune(Path(args[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
