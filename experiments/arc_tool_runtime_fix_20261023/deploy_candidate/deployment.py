"""Run the Pi medium ARC solver on the complete Kaggle challenge input."""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
CAP = 43200
SAVE_MARGIN = 60
PANEL_DRAIN_SECONDS = 5
WATCHDOG_GRACE_SECONDS = 15
KV_PAUSE_AT = 0.84
KV_RESUME_AT = 0.78
INITIAL_TASKS = 24
WAVES = 3
WAVE_SEEDS = "42,1337,2026"
ENABLE_WAVE2_GUIDED_CONTROLS = False  # True = R1, False = R0 (verified stronger on G4)
WAVE2_UNVERIFIED_ONLY_CAPS = True
WAVE2_TURN1_THINK_CAP = 16000
WAVE2_TURN_THINK_CAP = 22000
WAVE2_CODE_RESERVE = 4096
WAVE2_FINALIZE_RESERVE = 4608
WAVE2_DUPLICATE_EXTEND = 0
GOVERNOR_DEFAULT_TPS = 300
# 12.0 GiB KV/GPU (+256 MiB over 11.75 GiB) with MTP=3 preserves ~620 MiB physical headroom (> 512 MiB floor) alongside the 303 MiB FP8 draft lm_head.
KV_GIB = 12
KV_CACHE_BYTES = int(KV_GIB * 1024**3)
MIN_FREE_MIB = 512
EXPECTED_MAMBA_PIN_GROUPS = 3
PREFILL_BATCH_CAP = 6144
NUM_SPECULATIVE_TOKENS = 3
MAX_CUDAGRAPH_CAPTURE_SIZE = 96


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def stop_groups(pids, grace_seconds=8):
    """One shared grace window for every owned group, including descendants."""
    pending = set(pids)
    for pid in list(pending):
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pending.discard(pid)
    until = time.monotonic() + max(0, grace_seconds)
    while pending and time.monotonic() < until:
        for pid in list(pending):
            try:
                os.killpg(pid, 0)
            except ProcessLookupError:
                pending.discard(pid)
        if pending:
            time.sleep(min(.05, max(0, until - time.monotonic())))
    for pid in pending:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def stop_group(pid, grace_seconds=8):
    stop_groups([pid], grace_seconds)


def stop_owned_processes(groups, processes, cutoff):
    stop_groups(groups.values(), grace_seconds=1 if time.time() >= cutoff - 30 else 8)
    # Reap children against a shared deadline, never N independent timeouts.
    until = time.monotonic() + 1
    for proc in processes.values():
        try:
            proc.wait(timeout=max(0, until - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass


def archive_allowed(task_limit, cutoff):
    return task_limit > 0 and time.time() < cutoff - 30


def monitor_panel(panel, server, cutoff, on_poll):
    while panel.poll() is None:
        if time.time() >= cutoff + PANEL_DRAIN_SECONDS:
            raise TimeoutError("Panel save window ended")
        on_poll()
        if server.poll() is not None:
            # Ask the panel to persist its partial answers before group cleanup.
            try:
                os.kill(panel.pid, signal.SIGUSR2)
            except ProcessLookupError:
                pass
            try:
                panel.wait(timeout=max(0, min(PANEL_DRAIN_SECONDS,
                    cutoff + PANEL_DRAIN_SECONDS - time.time())))
            except subprocess.TimeoutExpired:
                pass
            raise RuntimeError("Model server exited during panel; partial answers retained")
        time.sleep(min(2, max(0, cutoff + PANEL_DRAIN_SECONDS - time.time())))


def watchdog(output, origin):
    output = Path(output)
    done = output / "DEPLOYMENT_DONE.json"
    cutoff = origin + CAP - SAVE_MARGIN
    while time.time() < cutoff and not done.exists():
        time.sleep(min(5, max(.1, cutoff - time.time())))
    if done.exists():
        return
    write(output / "HARD_STOP.json", {"at": now(), "save_cutoff": cutoff, "hard_deadline": origin + CAP})
    # Independent backstop; the controller normally finishes its shorter drain
    # and group shutdown first. Leave most of the 60s margin for kernel exit.
    until = cutoff + WATCHDOG_GRACE_SECONDS
    while time.time() < until and not done.exists():
        time.sleep(min(1, max(0, until - time.time())))
    if done.exists():
        return
    groups = read(output / "PROCESS_GROUPS.json") if (output / "PROCESS_GROUPS.json").exists() else {}
    stop_groups(groups.values(), grace_seconds=1)


def expose_driver(output):
    driver = ctypes.CDLL("libcuda.so.1")
    paths = {Path(line.split(maxsplit=5)[5]).resolve()
             for line in Path("/proc/self/maps").read_text().splitlines()
             if len(line.split(maxsplit=5)) == 6 and Path(line.split(maxsplit=5)[5]).name.startswith("libcuda.so")}
    actual = sorted(paths)[0]
    directory = Path("/opt/arc_pi_cuda_driver")
    directory.mkdir(exist_ok=True)
    for name in ("libcuda.so", "libcuda.so.1"):
        target = directory / name
        if not target.exists():
            target.symlink_to(actual)
    for key in ("LIBRARY_PATH", "LD_LIBRARY_PATH"):
        os.environ[key] = str(directory) + os.pathsep + os.environ.get(key, "")
    write(output / "CUDA_DRIVER_LINK.json", {"driver": str(actual), "link": str(directory)})


def locate_one(pattern, description):
    found = list(Path("/kaggle/input").glob(pattern))
    if len(found) != 1:
        raise RuntimeError(f"Expected exactly one {description}; found {len(found)}")
    return found[0]


def build_panel(challenges_path, data_root, task_limit=0):
    challenges = json.loads(Path(challenges_path).read_text(encoding="utf-8"))
    if not isinstance(challenges, dict) or not challenges:
        raise ValueError("Competition challenge input is empty")
    data_root = Path(data_root)
    task_dir = data_root / "solver_tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    records, task_order, query_counts = [], [], {}
    full_task_order, full_query_counts = [], {}
    task_ids_env = [t.strip() for t in os.environ.get("ARC_TASK_IDS", "").split(",") if t.strip()]
    if task_ids_env:
        selected = [(tid, challenges[tid]) for tid in task_ids_env if tid in challenges]
    elif task_limit > 0:
        selected = list(challenges.items())[:task_limit]
    else:
        selected = list(challenges.items())
    for task_id, task in challenges.items():
        full_task_order.append(task_id)
        full_query_counts[task_id] = len(task["test"])
    for task_id, task in selected:
        if not isinstance(task_id, str) or len(task_id) != 8 or any(c not in "0123456789abcdef" for c in task_id):
            raise ValueError("Unexpected competition task identity")
        if not isinstance(task, dict) or set(task) != {"train", "test"}:
            raise ValueError("Unexpected competition task schema")
        path = task_dir / (task_id + ".json")
        path.write_text(json.dumps(task, separators=(",", ":")), encoding="utf-8")
        rel = path.relative_to(data_root).as_posix()
        task_order.append(task_id); query_counts[task_id] = len(task["test"])
        records.append({"task_id": task_id, "path": rel, "query_count": len(task["test"])})
    manifest = {"schema": "arc-gr-pi-kaggle-input-v1", "tasks": records,
                "count": len(records), "query_count": sum(query_counts.values())}
    write(data_root / "manifest.json", manifest)
    write(data_root / "selection_manifest.json", {"count": len(records), "task_ids": task_order,
                                                    "query_count": manifest["query_count"]})
    write(data_root.parent.parent / "INPUT_MANIFEST.json", {"schema": "arc-gr48k-competition-input-v1",
        "task_order": full_task_order, "queries_per_task": full_query_counts, "query_gold_used": False,
        "task_count": len(full_task_order), "query_count": sum(full_query_counts.values()),
        "inference_task_order": task_order})
    return manifest


def server_command(python, model):
    command = [str(python), "-m", "vllm.entrypoints.openai.api_server", "--model", str(model),
        "--served-model-name", "gr48k", "--host", "127.0.0.1", "--port", "8000",
        "--tensor-parallel-size", "4", "--max-model-len", "131072", "--max-num-seqs", "24",
        "--max-num-batched-tokens", str(PREFILL_BATCH_CAP), "--gpu-memory-utilization", "0.95",
        "--kv-cache-memory-bytes", str(KV_CACHE_BYTES),
        "--language-model-only", "--dtype", "bfloat16", "--kv-cache-dtype", "fp8",
        "--mamba-ssm-cache-dtype", "float32", "--chat-template", str(Path(model) / "chat_template.jinja"),
        "--quantization", "compressed-tensors", "--linear-backend", "cutlass",
        "--load-format", "safetensors", "--safetensors-load-strategy", "lazy",
        "--generation-config", "vllm", "--reasoning-parser", "qwen3",
        "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
        "--enable-prefix-caching", "--prefix-match-unit", "16",
        "--mamba-cache-mode", "align", "--enable-mamba-fine-grained-prefix-cache",
        "--enable-prompt-tokens-details", "--enable-request-id-headers",
        "--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": NUM_SPECULATIVE_TOKENS, "draft_sample_method": "probabilistic"}),
        "--attention-backend", "TRITON_ATTN",
        "--compilation-config", json.dumps({"max_cudagraph_capture_size": MAX_CUDAGRAPH_CAPTURE_SIZE, "pass_config": {"fuse_norm_quant": True, "fuse_act_quant": True}}),
        "--scheduling-policy", "priority", "--scheduler-reserve-full-isl", "--watermark", "0.02"]
    if ENABLE_WAVE2_GUIDED_CONTROLS:
        command.extend(["--reasoning-config", json.dumps({
            "reasoning_start_str": "<think>", "reasoning_end_str": "</think>"})])
    blocks_override = os.environ.get("ARC_NUM_GPU_BLOCKS_OVERRIDE", "").strip()
    if blocks_override:
        command.extend(["--num-gpu-blocks-override", blocks_override])
    return command


def panel_command(node, python, data_root, output, cutoff, task_count=120):
    output = Path(output)
    command = [str(node), str(HERE / "pi/panel-kv-kaggle.bundle.mjs"),
        "--data", str(data_root), "--output", str(output / "panel"),
        "--bridge", str(HERE / "pi/bridge.py"), "--python", str(python),
        "--deadline", str(cutoff), "--concurrency", "24", "--output-limit", "65000",
        "--reasoning-effort", "medium", "--panel-kind", "kaggle",
        "--waves", str(WAVES), "--wave-seeds", WAVE_SEEDS, "--wave3-transform", "transpose",
        "--python-limit", "24", "--python-notice-after", "16",
        "--wave2-duplicate-extend", str(WAVE2_DUPLICATE_EXTEND),
        "--initial-tasks", str(INITIAL_TASKS), "--ramp-interval-s", "2", "--cold-interval-s", "1",
        "--pause-at", str(KV_PAUSE_AT), "--resume-at", str(KV_RESUME_AT),
        # cutoff already excludes the 60-second kernel-exit margin.
        "--governor", "on", "--governor-reserve-seconds", "0", "--governor-default-tps", str(GOVERNOR_DEFAULT_TPS),
        "--min-output-limit", "24000",
        "--server-log", str(output / "server.log"), "--submission", "/kaggle/working/submission.json"]
    if ENABLE_WAVE2_GUIDED_CONTROLS:
        command.extend([
            "--wave2-turn1-think-cap", str(WAVE2_TURN1_THINK_CAP),
            "--wave2-turn-think-cap", str(WAVE2_TURN_THINK_CAP),
            "--wave2-code-reserve", str(WAVE2_CODE_RESERVE),
            "--wave2-finalize-reserve", str(WAVE2_FINALIZE_RESERVE)])
    return command


def is_panel_complete(panel_returncode, summary, probe_stopped=False,
                      panel_max_seconds=0, panel_cutoff=0, now_seconds=None):
    if now_seconds is None:
        now_seconds = time.time()
    live_tokens = int(summary.get("total_live_output_tokens") or summary.get("output_tokens") or 0)
    return (panel_returncode == 0 and summary.get("status") == "completed") or (
        probe_stopped and live_tokens > 0
    ) or (
        panel_max_seconds > 0
        and panel_returncode in (0, 2)
        and not summary.get("engine_failure")
        and int(summary.get("task_errors") or 0) == 0
        and (panel_returncode == 0 or now_seconds >= panel_cutoff - PANEL_DRAIN_SECONDS)
        and summary.get("status") in ("completed", "partial")
        and live_tokens > 0
    )


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    hard, cutoff = args.origin + CAP, args.origin + CAP - SAVE_MARGIN
    # Setup and inference share the original 12-hour allocation clock.
    # The old 45-minute setup timer killed a still-progressing weight load.
    startup_deadline = cutoff
    write(output / "ALLOCATION.json", {"first_instruction": args.origin, "hard_deadline": hard,
        "save_cutoff": cutoff, "provider_cap_seconds": CAP, "controller_pid": os.getpid()})
    write(output / "PROCESS_GROUPS.json", {})
    watchdog_proc = subprocess.Popen([sys.executable, str(HERE / "deployment.py"), "watchdog",
        "--output", str(output), "--origin", str(args.origin)], start_new_session=True)
    groups, processes, logs = {}, {}, []
    status = {"phase": "preparing", "started_at": now(), "submitted_to_competition": False,
              "reasoning_effort": "medium", "output_cap": 65000, "concurrency": 24}

    def phase(value, **extra):
        status.update(phase=value, updated_at=now(), **extra)
        write(output / "STATUS.json", status)
        print("PHASE", json.dumps(status), flush=True)

    def spawn(name, command, env=None):
        log = (output / (name + ".log")).open("xb")
        logs.append(log)
        proc = subprocess.Popen(command, cwd=PACKAGE, stdout=log, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
        groups[name], processes[name] = proc.pid, proc
        write(output / "PROCESS_GROUPS.json", groups)
        return proc

    def wait_command(name, proc, deadline):
        try:
            code = proc.wait(timeout=max(.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            stop_group(proc.pid, grace_seconds=1 if time.time() >= cutoff - 30 else 8)
            try: proc.wait(timeout=1)
            except subprocess.TimeoutExpired: pass
            raise TimeoutError(name + " exceeded fixed window") from None
        finally:
            groups.pop(name, None); write(output / "PROCESS_GROUPS.json", groups)
        if code:
            raise RuntimeError(name + " failed; see saved log")

    def command(name, argv, deadline, env=None):
        wait_command(name, spawn(name, argv, env), deadline)

    try:
        challenges = locate_one(getattr(args, "challenges_pattern", None) or
            os.environ.get("ARC_CHALLENGES_PATTERN", "**/arc-agi_test_challenges.json"), "challenge file")
        model_index = locate_one("**/model.safetensors.index.json", "Qwen model index")
        model_source = model_index.parent
        export = read(model_source / "EXPORT_COMPLETE.json")
        if export.get("runtime_lora_required") is not False:
            raise RuntimeError("Attach the completed Drive FP8_DYNAMIC model with merged GR48K")
        hardware = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                            "--format=csv,noheader"], text=True).splitlines()
        if len(hardware) != 4 or any(not line.startswith("NVIDIA L4,") for line in hardware):
            raise RuntimeError("Expected four NVIDIA L4 cards")
        topo_info = {}
        for topo_key, topo_cmd in (
            ("topo_m", ["nvidia-smi", "topo", "-m"]),
            ("topo_p2p_r", ["nvidia-smi", "topo", "-p2p", "r"]),
            ("topo_p2p_w", ["nvidia-smi", "topo", "-p2p", "w"]),
        ):
            try:
                topo_info[topo_key] = subprocess.check_output(topo_cmd, text=True, stderr=subprocess.STDOUT, timeout=10).strip()
            except Exception as topo_exc:
                topo_info[topo_key] = f"ERR: {topo_exc}"
        write(output / "HARDWARE.json", {"rows": hardware, "tensor_parallel": 4, "provider": "Kaggle", **topo_info})
        expose_driver(output)
        venv = Path("/opt/arc_pi_venv")
        command("create_venv", [sys.executable, "-m", "venv", "--without-pip", str(venv)], startup_deadline)
        python = venv / "bin/python"
        environment = locate_one("**/ARC_PI_MEDIUM_PY312_ENV.tar.gz.bin", "prebuilt medium environment")
        model = Path("/opt/arc_pi_model")
        phase("staging_model_and_unpacking_environment", model_source=str(model_source), model=str(model))
        stage = spawn("stage_model", [sys.executable, str(HERE / "stage_model.py"),
                      str(model_source), str(model), str(output / "MODEL_STAGE.json")])
        command("unpack_environment", ["tar", "-xzf", str(environment), "-C", str(venv)], startup_deadline)
        command("mamba_pin_patch", [str(python), str(HERE / "install_mamba_pin_patch.py"),
                str(venv), str(output / "MAMBA_PIN_PATCH.json")], startup_deadline)
        base_env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_NO_USAGE_STATS="1",
                        VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC="0", VLLM_BATCH_INVARIANT="0",
                        VLLM_USE_V2_MODEL_RUNNER="1", VLLM_GDN_DECODE_KERNEL="cuda",
                        DO_NOT_TRACK="1", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1",
                        VLLM_CACHE_ROOT="/opt/arc_pi_cache/vllm", TRITON_CACHE_DIR="/opt/arc_pi_cache/triton",
                        TORCHINDUCTOR_CACHE_DIR="/opt/arc_pi_cache/inductor",
                        PATH=str(python.parent) + os.pathsep + os.environ.get("PATH", ""),
                        PYTHONPATH=str(PACKAGE) + os.pathsep + os.environ.get("PYTHONPATH", ""))
        wait_command("stage_model", stage, startup_deadline)
        _safe_shm_fallback_env = {
            "NCCL_P2P_DISABLE": "1",
            "NCCL_SHM_DISABLE": "0",
            "NCCL_PROTO": "^LL128",
            "NCCL_SHM_USE_CUDA_MEMCPY": "0",
            "NCCL_SHM_MEMCPY_MODE": "1",
            "NCCL_SHM_LOCALITY": "2",
        }
        _safe_shm_unset = [
            "NCCL_P2P_LEVEL",
            "NCCL_MIN_NCHANNELS",
            "NCCL_MAX_NCHANNELS",
            "NCCL_MIN_NRINGS",
            "NCCL_MAX_NRINGS",
            "NCCL_MIN_CTAS",
            "NCCL_MAX_CTAS",
            "NCCL_BUFFSIZE",
            "NCCL_ALGO",
        ]
        nccl_autotune = {
            "selected_profile": "default",
            "selected_env": dict(_safe_shm_fallback_env),
            "unset_env": list(_safe_shm_unset),
            "selection": {"reason": "not_run"},
        }
        autotune_env = dict(
            base_env,
            ARC_AUTOTUNE_SPEC_K=str(NUM_SPECULATIVE_TOKENS),
            ARC_STAR_SHM_PERSIST_DIR="/opt/arc_pi_cache/arc_shm_ar",
        )
        try:
            autotune_deadline = min(startup_deadline, time.time() + 185.0)
            command("nccl_autotune", [str(python), str(HERE / "autotune_nccl.py"),
                    str(output / "NCCL_AUTOTUNE.json")], autotune_deadline, autotune_env)
            if (output / "NCCL_AUTOTUNE.json").exists():
                nccl_autotune = read(output / "NCCL_AUTOTUNE.json")
        except Exception as _nccl_exc:
            nccl_autotune = {
                "selected_profile": "default",
                "selected_env": dict(_safe_shm_fallback_env),
                "unset_env": list(_safe_shm_unset),
                "selection": {"reason": "fallback_on_error", "error": f"{type(_nccl_exc).__name__}: {_nccl_exc}"},
            }
            write(output / "NCCL_AUTOTUNE.json", nccl_autotune)
        env = dict(base_env)
        server_env = dict(base_env)
        for _unset_k in (nccl_autotune.get("unset_env") or _safe_shm_unset):
            server_env.pop(_unset_k, None)
        for _star_k in (
            "ARC_FAST_SHM_AR_SO",
            "ARC_FAST_SHM_AR_SHA256",
            "ARC_FAST_SHM_AR_CAPACITY",
            "ARC_FAST_SHM_AR_MIN_NUMEL",
            "ARC_FAST_SHM_AR_MAX_NUMEL",
        ):
            server_env.pop(_star_k, None)
        selected_nccl_env = dict(_safe_shm_fallback_env)
        selected_nccl_env.update({str(k): str(v) for k, v in (nccl_autotune.get("selected_env") or {}).items()})
        selected_nccl_env["NCCL_P2P_DISABLE"] = "1"
        server_env.update(selected_nccl_env)
        star_sel = (
            nccl_autotune.get("star_shm_selection")
            or (nccl_autotune.get("selection") or {}).get("star_shm_selection")
            or {}
        )
        star_active = False
        star_env = {}
        if star_sel.get("enabled") is True:
            _so_cand = Path(str(star_sel.get("so_path") or ""))
            _exp_sha = str(star_sel.get("so_sha256") or "")
            if _so_cand.is_file() and _exp_sha:
                _act_sha = __import__("hashlib").sha256(_so_cand.read_bytes()).hexdigest()
                if _act_sha == _exp_sha:
                    star_active = True
                    star_env = {
                        "ARC_FAST_SHM_AR_SO": str(_so_cand),
                        "ARC_FAST_SHM_AR_SHA256": _exp_sha,
                        "ARC_FAST_SHM_AR_CAPACITY": str(int(star_sel.get("capacity") or (192 * 5120))),
                        "ARC_FAST_SHM_AR_MIN_NUMEL": str(int(star_sel.get("min_numel") or 1)),
                        "ARC_FAST_SHM_AR_MAX_NUMEL": str(int(star_sel.get("max_numel") or (6144 * 5120))),
                    }
                    server_env.update(star_env)
        active_ar_backend = "star_shm+pynccl" if star_active else "pynccl"
        data_root = output / "data/public"
        manifest = build_panel(challenges, data_root, args.task_limit)
        write(output / "MODEL_CONFIG.json", {"served_model": "gr48k", "model": str(model),
            "format": "FP8_DYNAMIC", "runtime_lora_required": False,
            "source_export": export, "load_strategy": "lazy",
            "model_source": str(model_source), "staged_on_kaggle_vm": True,
            "reasoning_effort": "medium", "max_tokens": 65000,
            "manifest": manifest, "query_gold_used": False})
        argv = server_command(python, model)
        write(output / "SERVER_COMMAND.json", {"argv": argv,
            "environment": {
                "VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC": "0",
                "VLLM_BATCH_INVARIANT": "0",
                "VLLM_USE_V2_MODEL_RUNNER": "1",
                "VLLM_GDN_DECODE_KERNEL": "cuda",
                **selected_nccl_env,
                **star_env,
            },
            "allreduce_backend": active_ar_backend,
            "nccl_autotune_profile": nccl_autotune.get("selected_profile", "default"),
            "nccl_autotune_selection": nccl_autotune.get("selection"),
            "star_shm_selection": star_sel,
            "kv_cache_bytes_per_gpu": KV_CACHE_BYTES, "max_batched_tokens": PREFILL_BATCH_CAP,
            "num_speculative_tokens": NUM_SPECULATIVE_TOKENS, "draft_sample_method": "probabilistic",
            "max_cudagraph_capture_size": MAX_CUDAGRAPH_CAPTURE_SIZE,
            "mamba_pin_patch": read(output / "MAMBA_PIN_PATCH.json"),
            "task_output_budget": 65000, "max_concurrency": 24})
        server = spawn("server", argv, server_env)
        phase("loading_model", preparation_seconds=time.time() - args.origin)
        while time.time() < startup_deadline:
            if server.poll() is not None: raise RuntimeError("Model server exited; no restart")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=3) as response:
                    ids = {item["id"] for item in json.load(response)["data"]}
                if "gr48k" in ids: break
            except (OSError, ValueError, KeyError): pass
            time.sleep(3)
        else: raise TimeoutError("Model startup reached the allocation save cutoff")
        server_log = (output / "server.log").read_text(errors="replace")
        ipc_lines = [line for line in server_log.splitlines()
                     if "Initialized FlashInfer PCIe IPC all-reduce for TP4" in line]
        if ipc_lines:
            raise RuntimeError("FlashInfer PCIe IPC unexpectedly initialized in PYNCCL evaluation")
        pin_lines = [line for line in server_log.splitlines() if "ARC_MAMBA_PIN init" in line]
        star_lines = [line for line in server_log.splitlines() if "ARC_FAST_SHM_AR attached" in line]
        draft_fp8_lines = [line for line in server_log.splitlines() if "ARC_DRAFT_FP8_LM_HEAD" in line]
        triton_lines = [line for line in server_log.splitlines() if any(k in line for k in ("ARC_TRITON_ATTN_PATCH", "ARC_SMALL_TOPK_PATCH", "ARC_FAST_SHM_AR", "ARC_DRAFT_FP8_LM_HEAD"))]
        cg_lines = [line for line in server_log.splitlines() if any(k in line for k in (
            "Model Runner", "attention backend", "cudagraph_mode", "Capturing CUDA graphs",
            "Capturing draft", "Graph capturing finished", "ARC_SMALL_TOPK_PATCH", "ARC_FAST_SHM_AR", "ARC_DRAFT_FP8_LM_HEAD"))]
        write(output / "MAMBA_PIN_ACTIVE.json", {"at": now(), "init_lines": len(pin_lines),
              "expected_mamba_groups": EXPECTED_MAMBA_PIN_GROUPS, "first": pin_lines[:1],
              "star_shm_lines": star_lines,
              "draft_fp8_lm_head_lines": draft_fp8_lines,
              "triton_patch_lines": triton_lines, "cudagraph_lines": cg_lines})
        if len(pin_lines) != EXPECTED_MAMBA_PIN_GROUPS:
            raise RuntimeError(f"Mamba pin patch active in {len(pin_lines)} groups, expected {EXPECTED_MAMBA_PIN_GROUPS}")
        star_fallback_lines = [line for line in server_log.splitlines() if "ARC_FAST_SHM_AR fallback_pynccl" in line]
        if star_active and len(star_lines) not in (0, 4):
            raise RuntimeError(f"ARC_FAST_SHM_AR attached asymmetrically on {len(star_lines)} workers, expected 4 or 0")
        if star_active and len(star_lines) == 0 and not star_fallback_lines:
            raise RuntimeError("ARC_FAST_SHM_AR requested, but neither attached nor logged fallback_pynccl")
        star_runtime_active = bool(star_active and len(star_lines) == 4)
        active_ar_backend = "star_shm+pynccl" if star_runtime_active else "pynccl"
        free_mib = [int(value.strip()) for value in subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).splitlines()]
        write(output / "PYNCCL_ACTIVE.json", {"at": now(), "allreduce_backend": active_ar_backend,
              "nccl_autotune_profile": nccl_autotune.get("selected_profile", "default"),
              "nccl_selected_env": selected_nccl_env,
              "star_shm_active": star_runtime_active,
              "star_shm_requested": star_active,
              "star_shm_attached_workers": len(star_lines),
              "star_shm_fallback_lines": star_fallback_lines,
              "star_shm_selection": star_sel,
              "draft_fp8_lm_head_workers": len(draft_fp8_lines),
              "free_mib_after_startup": free_mib, "kv_cache_gib_per_gpu": KV_CACHE_BYTES / 1024**3,
              "min_free_mib_required": MIN_FREE_MIB,
              "prefill_batch_cap": PREFILL_BATCH_CAP,
              "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
              "max_cudagraph_capture_size": MAX_CUDAGRAPH_CAPTURE_SIZE,
              "additional_model_generations": 0})
        if min(free_mib) < MIN_FREE_MIB:
            raise RuntimeError(f"Less than {MIN_FREE_MIB} MiB physical headroom after PYNCCL/graphs; stop before ARC dispatch")
        node_root = output / "node"; node_root.mkdir()
        with __import__("tarfile").open(HERE / "support/node-v24.12.0-linux-x64.tar.xz") as tar:
            tar.extractall(node_root, filter="data")
        node = node_root / "node-v24.12.0-linux-x64/bin/node"
        warmup_t0 = time.time()
        try:
            import concurrent.futures as _cf
            _w_tools = [
                {"type": "function", "function": {"name": "python", "description": "Execute Python in the persistent task workspace. Print observations.", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"], "additionalProperties": False}}},
                {"type": "function", "function": {"name": "submit_program", "description": "Check a complete transform(grid) program in a fresh workspace on every demo and test input.", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"], "additionalProperties": False}}},
            ]
            _w_grid_row = "[0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5],"
            def _send_warmup(slot_idx, n_rows, max_tok):
                u_txt = f"warmup_slot_{slot_idx}\n" + (_w_grid_row * n_rows)
                body = json.dumps({
                    "model": "gr48k",
                    "messages": [
                        {"role": "system", "content": "Solve the ARC grid transformation using the Python tools."},
                        {"role": "user", "content": u_txt},
                    ],
                    "tools": _w_tools,
                    "tool_choice": "auto",
                    "parallel_tool_calls": False,
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": 20,
                    "seed": 1000 + slot_idx,
                    "max_tokens": max_tok,
                    "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "medium", "preserve_thinking": True},
                }).encode("utf-8")
                req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    resp.read()
            # 1) Single >1700-tok warmup (compiles FLA chunked prefill, Mamba block boundary, 3D Split-K, Small-K sampler & MTP=4 rejection)
            _send_warmup(0, 72, 16)
            # 2) 24-way concurrent >1100-tok warmup (compiles batch=24 / num_tokens=120 3D Split-K + GDN + Small-K MTP=4 rejection)
            with _cf.ThreadPoolExecutor(max_workers=24) as _ex:
                list(_ex.map(lambda idx: _send_warmup(idx, 42, 15), range(1, 25)))
            server_log_post = (output / "server.log").read_text(errors="replace")
            triton_lines_post = [line for line in server_log_post.splitlines() if any(k in line for k in ("ARC_TRITON_ATTN_PATCH", "ARC_SMALL_TOPK_PATCH", "ARC_FAST_SHM_AR", "ARC_DRAFT_FP8_LM_HEAD"))]
            if len(triton_lines_post) > len(triton_lines):
                write(output / "MAMBA_PIN_ACTIVE.json", {"at": now(), "init_lines": len(pin_lines),
                      "expected_mamba_groups": EXPECTED_MAMBA_PIN_GROUPS, "first": pin_lines[:1],
                      "star_shm_lines": star_lines,
                      "draft_fp8_lm_head_lines": draft_fp8_lines,
                      "triton_patch_lines": triton_lines_post, "cudagraph_lines": cg_lines})
            write(output / "WARMUP_COMPLETE.json", {"at": now(), "seconds": round(time.time() - warmup_t0, 3), "ok": True})
        except Exception as _w_exc:
            write(output / "WARMUP_COMPLETE.json", {"at": now(), "seconds": round(time.time() - warmup_t0, 3), "ok": False, "error": str(_w_exc)})
        phase("running_panel", formal_generation_observed=False,
              panel_start_seconds=time.time() - args.origin, first_arc_target_seconds=1800)
        panel_started = time.time()
        probe_seconds = max(0, int(os.environ.get("ARC_SPEED_PROBE_SECONDS", "0").strip() or "0"))
        panel_max_seconds = max(0, int(os.environ.get("ARC_PANEL_MAX_SECONDS", "0").strip() or "0"))
        panel_cutoff = min(cutoff, int(panel_started + panel_max_seconds)) if panel_max_seconds > 0 else cutoff
        probe_stopped = False
        panel = spawn("panel", panel_command(node, python, data_root, output, panel_cutoff, manifest.get("count", 120)), env)
        first = False
        def observe_panel():
            nonlocal first, probe_stopped
            if (output / "panel/FIRST_GENERATION_OBSERVED.json").exists() and not first:
                first = True
                elapsed = read(output / "panel/FIRST_GENERATION_OBSERVED.json")["at"] - args.origin
                write(output / "FIRST_GENERATION_OBSERVED.json", {
                    "at": now(), "source": "formal Pi panel", "allocation_seconds": elapsed,
                    "within_30_minutes": elapsed <= 1800})
                phase("running_panel", formal_generation_observed=True, first_arc_generation_seconds=elapsed)
            if probe_seconds > 0 and not probe_stopped and time.time() - panel_started >= probe_seconds:
                probe_stopped = True
                try:
                    with urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=5) as m_resp:
                        (output / "SPEED_PROBE_METRICS.txt").write_bytes(m_resp.read())
                except OSError:
                    pass
                try:
                    os.kill(panel.pid, signal.SIGUSR2)
                except ProcessLookupError:
                    pass
                try:
                    panel.wait(timeout=PANEL_DRAIN_SECONDS)
                except subprocess.TimeoutExpired:
                    stop_group(panel.pid, grace_seconds=3)
        monitor_panel(panel, server, panel_cutoff, observe_panel)
        summary = read(output / "panel/summary.json") if (output / "panel/summary.json").exists() else {}
        first = first or (output / "panel/FIRST_GENERATION_OBSERVED.json").exists()
        status.update(panel_returncode=panel.returncode, completed_tasks=summary.get("completed_tasks"),
                      eligible_tasks=summary.get("eligible_tasks"),
                      fallback_tasks=summary.get("fallback_tasks"),
                      task_errors=summary.get("task_errors"),
                      output_tokens=summary.get("output_tokens"),
                      total_live_output_tokens=summary.get("total_live_output_tokens"),
                      panel_seconds=summary.get("seconds"),
                      speed_probe_seconds=probe_seconds,
                      panel_max_seconds=panel_max_seconds,
                      panel_cutoff=panel_cutoff,
                      formal_generation_observed=first)
        complete = is_panel_complete(
            panel.returncode, summary, probe_stopped=probe_stopped,
            panel_max_seconds=panel_max_seconds, panel_cutoff=panel_cutoff)
        phase("completed" if complete else "failed_partial")
        first_result = next((v for v in summary.get("tasks", {}).values() if v.get("output_tokens", 0) > 0), None)
        write(output / "MODEL_GENERATION.json", {"model": "gr48k", "requested_model": "gr48k",
            "generation_observed": bool(first and first_result), "runtime_lora_required": False,
            "formal_panel_evidence": True, "usage": {"prompt_tokens": int((first_result or {}).get("input_tokens", 1)),
            "completion_tokens": int((first_result or {}).get("output_tokens", 1))}})
    except Exception as exc:
        status.update(phase="failed_or_partial", error=type(exc).__name__ + ": " + str(exc))
        write(output / "FAILURE.json", {"at": now(), "error": status["error"], "retry": False})
    finally:
        stop_owned_processes(groups, processes, cutoff)
        for log in logs: log.close()
        status.update(finished_at=now(), allocation_wall_seconds=time.time() - args.origin,
                      owned_model_processes_stopped=all(p.poll() is not None for p in processes.values()))
        write(output / "STATUS.json", status)
        archive = output.parent / "ARC_PI_MEDIUM_L4_65K_RESULTS.zip"
        make_archive = archive_allowed(args.task_limit, cutoff)
        if make_archive and not archive.exists():
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
                for file in output.rglob("*"):
                    relative = file.relative_to(output)
                    if file.is_file() and file.suffix != ".tmp" and relative.parts[0] != "node":
                        z.write(file, relative.as_posix())
        write(output / "DEPLOYMENT_DONE.json", {"at": now(), "archive": archive.name if make_archive else None,
            "archive_bytes": archive.stat().st_size if make_archive else 0,
            "archive_skipped_reason": None if make_archive else "competition_rerun_or_near_cutoff",
            "owned_model_processes_stopped": status["owned_model_processes_stopped"],
            "model_bytes_on_local_pc": 0})
        try:
            watchdog_proc.terminate(); watchdog_proc.wait(timeout=.2)
        except subprocess.TimeoutExpired:
            watchdog_proc.kill()
            try: watchdog_proc.wait(timeout=.2)
            except subprocess.TimeoutExpired: pass
        except ProcessLookupError: pass
        print("DEPLOYMENT_DONE", json.dumps(status), flush=True)
    return 0 if status.get("phase") == "completed" else 2


def parse_args(argv=None):
    p = argparse.ArgumentParser(); p.add_argument("mode", choices=["run", "watchdog"])
    p.add_argument("--output", required=True); p.add_argument("--origin", required=True, type=float)
    p.add_argument("--task-limit", type=int, default=0)
    p.add_argument("--challenges-pattern", default=os.environ.get(
        "ARC_CHALLENGES_PATTERN", "**/arc-agi_test_challenges.json"))
    return p.parse_args(argv)


if __name__ == "__main__":
    a = parse_args()
    raise SystemExit(watchdog(a.output, a.origin) or 0) if a.mode == "watchdog" else SystemExit(run(a))
