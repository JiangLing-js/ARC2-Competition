"""One task, one A100, equal elapsed work caps including import/model loading."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback

from equal_budget_helpers import PROTOCOL, pass_configs
from runtime_helpers import atomic_json, finalize_equal


def local_worker(settings):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
    import torch
    torch.set_default_device("cpu")
    from arc_solver import worker
    worker(settings)


def recover_summary(settings, reason):
    """Retain denominator and completed pass evidence after a killed worker."""
    key = settings["selected_task_ids"][0]
    path = Path(settings["log_dir"]) / f"task_{key}_primary.json"
    summary = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    passes = []
    for config in pass_configs(settings["arm"], settings["started_at"], settings["work_deadline"]):
        pass_path = Path(settings["log_dir"]) / f"task_{key}_{config['pass']}.json"
        if pass_path.exists():
            passes.append(json.loads(pass_path.read_text(encoding="utf-8")))
    summary.update(task_id=key, phase="primary", arm=settings["arm"], passes=passes,
                   status="error", error=reason, forced_stop=True,
                   elapsed_seconds=time.time()-settings["started_at"],
                   global_step=sum(p.get("global_step", 0) for p in passes),
                   training_rows=sum(p.get("training_rows", 0) for p in passes),
                   completed_views=sum(p.get("completed_views", 0) for p in passes),
                   expected_training_rows=128,
                   expected_views=16*len(settings["challenges"][key]["test"]))
    atomic_json(summary, path)
    return summary


def main(args):
    started = time.time()
    if args.num_workers != 1:
        raise ValueError("This paired experiment requires exactly one GPU worker")
    selected = [x.strip() for x in args.task_ids.split(",") if x.strip()]
    if len(selected) != 1:
        raise ValueError("Each invocation must select exactly one task; controller pairs tasks")
    if args.retry_seconds != 0:
        raise ValueError("There is no extra retry budget in the equal-budget experiment")
    if args.task_work_seconds <= 0 or args.finalization_reserve_seconds <= 0:
        raise ValueError("Work and finalization budgets must be positive")
    path = Path(args.test_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    challenges = {key: data[key] for key in selected}
    root = Path(args.artifact_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing nonempty artifact directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    hard_end = min(started + args.duration_seconds, args.end_time or float("inf"))
    work_end = min(started + args.task_work_seconds, hard_end - args.finalization_reserve_seconds)
    if work_end <= started:
        raise ValueError("No time remains for model loading/work")
    os.environ["PYTHONHASHSEED"] = "0"
    settings = {"protocol": PROTOCOL, "arm": args.arm, "started_at": started,
                "work_deadline": work_end, "hard_end_time": hard_end,
                "selected_task_ids": selected, "challenges": challenges,
                "model_path": args.model_path, "test_path": str(path.resolve()),
                "artifact_root": str(root), "query_dir": str(root / "query_results"),
                "log_dir": str(root / "task_logs"), "submission_path": str(root / "submission.json"),
                "selection_manifest": args.selection_manifest}
    for directory in (settings["query_dir"], settings["log_dir"]):
        Path(directory).mkdir(parents=True, exist_ok=True)
    manifest = {k: v for k, v in settings.items() if k != "challenges"}
    manifest.update(task_work_seconds=args.task_work_seconds, effective_work_seconds=work_end-started,
                    finalization_reserve_seconds=args.finalization_reserve_seconds,
                    hash_seed="0", pass_configs=pass_configs(args.arm, started, work_end),
                    selection="source-style self-scored pooled KGMon; split-A top1 locked; scores uncalibrated")
    atomic_json(manifest, root / "nvarc_run_manifest.json")
    finalize_equal(challenges, settings)
    atomic_json({"status": "starting", "failure": None, "elapsed_seconds": time.time()-started},
                root / "nvarc_run_status.json")
    process, failure, forced_stop = None, None, False
    try:
        if not Path(args.model_path).is_dir():
            raise FileNotFoundError(args.model_path)
        context = mp.get_context("spawn")
        process = context.Process(target=local_worker, args=(settings,), name="nvarc-equal-budget-gpu")
        process.start()
        refresh_at = time.time()
        while process.is_alive():
            now = time.time()
            if now >= hard_end:
                forced_stop = True
                failure = "Worker exceeded work plus finalization deadline"
                break
            if now >= refresh_at:
                # Worker exclusively owns query/prediction writes while alive.
                atomic_json({"status": "finalizing" if now >= work_end else "running", "failure": None,
                             "work_deadline_reached": now >= work_end, "elapsed_seconds": now-started},
                            root / "nvarc_run_status.json")
                print(f"[{args.arm}/{selected[0]}] elapsed={now-started:.1f}s remaining_work={max(0,work_end-now):.1f}s", flush=True)
                refresh_at = now + 30
            time.sleep(.5)
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        if process is not None and process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            if process.exitcode != 0 and not failure:
                failure = f"Worker exit code {process.exitcode}"
        summary_path = Path(settings["log_dir"]) / f"task_{selected[0]}_primary.json"
        if failure or not summary_path.exists():
            failure = failure or "Worker exited without aggregate task summary"
            summary = recover_summary(settings, failure)
        else:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            failure = summary.get("error")
        coverage = finalize_equal(challenges, settings)
        status = {"status": "error" if failure else summary.get("status", "incomplete"),
                  "failure": failure, "deadline_stopped": time.time() >= work_end,
                  "forced_stop": forced_stop, "completed_primary_task_ids": selected if not failure else [],
                  "exit_codes": {"worker": process.exitcode if process else None},
                  "elapsed_seconds": time.time()-started, "coverage": coverage}
        atomic_json(status, root / "nvarc_run_status.json")
        finalize_equal(challenges, settings, freeze=True, elapsed_seconds=time.time()-started, gpu_count=1)
    if failure:
        raise RuntimeError(failure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("baseline", "split"))
    parser.add_argument("--task-work-seconds", type=float, default=1200)
    parser.add_argument("--duration-seconds", type=float, default=1320)
    parser.add_argument("--finalization-reserve-seconds", type=float, default=120)
    parser.add_argument("--end-time", type=float, default=0)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--task-ids", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--selection-manifest")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--retry-seconds", type=float, default=0)
    main(parser.parse_args())
