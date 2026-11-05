"""Small CPU-only scheduling, persistence, and submission helpers."""
from __future__ import annotations

import bz2
import json
import math
import os
import pickle
from pathlib import Path
import time
import hashlib

DIAGNOSTIC_IDS = ["0934a4d8", "36a08778", "981571dc", "aa4ec2a5"]


def rerun_enabled(value):
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def select_task_ids(challenges, rerun, override=None):
    if rerun:
        return sorted(challenges)
    selected = [part.strip() for part in override.split(",") if part.strip()] if override else DIAGNOSTIC_IDS
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Diagnostic IDs must be nonempty and distinct")
    absent = set(selected) - set(challenges)
    if absent:
        raise ValueError(f"Diagnostic IDs absent from challenge file: {sorted(absent)}")
    return sorted(selected)


def retry_admission(primary_elapsed, ttt_seconds, now, global_end, query_counts, task_cap=1200):
    missing = [int(index) for index, count in query_counts.items() if int(count) == 0]
    available = max(0.0, min(float(global_end) - float(now), float(task_cap) - float(primary_elapsed)))
    measured = math.isfinite(float(ttt_seconds)) and float(ttt_seconds) > 0
    required = 2 * float(ttt_seconds) + 60 if measured else None
    reason = "admitted"
    if not missing:
        reason = "all_queries_have_candidates"
    elif not measured:
        reason = "no_measured_ttt_cost"
    elif available < required:
        reason = "insufficient_remaining_budget"
    return {"admitted": reason == "admitted", "reason": reason,
            "missing_query_ids": sorted(missing), "available_seconds": available,
            "required_seconds": required, "deadline": float(now) + available}


def primary_completed(shared, task_id):
    """Called only after any retry job has entered the retry queue."""
    with shared["lock"]:
        if task_id in shared["completed"]:
            raise RuntimeError(f"Duplicate primary completion: {task_id}")
        shared["completed"][task_id] = True
        shared["remaining"].value -= 1


def atomic_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_view_dump(payload, path):
    """Keep temporary files outside the directory consumed by ArcDecoder."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = path.parent.parent / "inference_write_tmp"
    temporary_dir.mkdir(parents=True, exist_ok=True)
    temporary = temporary_dir / (path.name + f".{os.getpid()}.tmp")
    with bz2.BZ2File(temporary, "w") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def log_event(log_dir, worker_rank, event, **fields):
    root = Path(log_dir)
    root.mkdir(parents=True, exist_ok=True)
    row = {"time": time.time(), "rank": worker_rank, "event": event, **fields}
    with (root / f"worker_{worker_rank}.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def native_grid(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list) or not 1 <= len(value) <= 30:
        raise ValueError("Invalid grid height")
    if not isinstance(value[0], list) or not 1 <= len(value[0]) <= 30:
        raise ValueError("Invalid grid width")
    width = len(value[0])
    if any(not isinstance(row, list) or len(row) != width for row in value):
        raise ValueError("Nonrectangular grid")
    if any(type(cell) is not int or not 0 <= cell <= 9 for row in value for cell in row):
        raise ValueError("Invalid grid color")
    return value


def finalize_submission(challenges, output_dir, submission_path):
    """Apply unchanged upstream KGMon selection; no solution file is read."""
    from arc_decoder import score_kgmon
    grouped, unreadable = {}, []
    for path in sorted(Path(output_dir).glob("*.bz2")):
        try:
            base_key = path.name.split(".")[0]
            task_id, query_id = base_key.rsplit("_", 1)
            if task_id not in challenges or not 0 <= int(query_id) < len(challenges[task_id]["test"]):
                raise ValueError("Candidate file has an unknown task/query")
            with bz2.BZ2File(path, "rb") as handle:
                records = pickle.load(handle)
            for index, record in enumerate(records):
                native_grid(record["solution"])
                if len(record["score_aug"]) != 8 or not all(math.isfinite(float(x)) for x in record["score_aug"]):
                    raise ValueError("Candidate lacks eight finite author scores")
                grouped.setdefault(base_key, {})[f"{path.name}.out{index}"] = record
        except Exception as error:
            unreadable.append({"file": path.name, "error": f"{type(error).__name__}: {error}"})
    submission, model_outputs = {}, 0
    for task_id, task in challenges.items():
        attempts = []
        for query_id, query in enumerate(task["test"]):
            records = grouped.get(f"{task_id}_{query_id}", {})
            grids = []
            if records:
                for grid in score_kgmon(records):
                    grid = native_grid(grid)
                    if grid not in grids:
                        grids.append(grid)
                    if len(grids) == 2:
                        break
                model_outputs += bool(grids)
            while len(grids) < 2:
                grids.append([[0]])
            attempts.append({"attempt_1": grids[0], "attempt_2": grids[1]})
        submission[task_id] = attempts
    atomic_json(submission, submission_path)
    coverage = {"tasks": len(challenges), "test_inputs": sum(len(t["test"]) for t in challenges.values()),
                "model_outputs": model_outputs, "fallback_only": model_outputs == 0,
                "unreadable_files": unreadable}
    atomic_json(coverage, Path(submission_path).with_suffix(".coverage.json"))
    return coverage


def view_counts(states):
    rows = list(states.values()) if isinstance(states, dict) else list(states)
    return {"planned": len(rows), "attempted": sum(bool(x.get("attempted")) for x in rows),
            "completed": sum(bool(x.get("completed")) for x in rows),
            "timed_out": sum(bool(x.get("timed_out")) for x in rows),
            "empty": sum(bool(x.get("empty")) for x in rows),
            "errors": sum(bool(x.get("error")) for x in rows)}


def read_query_records(output_dir, task_id, query_id, pass_name):
    records = []
    for path in sorted(Path(output_dir).glob(f"{task_id}_{query_id}.*.bz2")):
        with bz2.BZ2File(path, "rb") as handle:
            for value in pickle.load(handle):
                records.append({**value, "solution": native_grid(value["solution"]),
                                "pass": pass_name, "view_id": value.get("view_id", path.name)})
    return records


def primary_query_document(task_id, query_id, records, views):
    from arc_decoder import score_kgmon
    from selection_helpers import uncertainty_gate
    ranked = [native_grid(g) for g in score_kgmon({str(i): r for i, r in enumerate(records)})]
    counts = view_counts(views)
    top2 = ranked[:2]
    padded = top2 + [[[0]]] * (2 - len(top2))
    gate = uncertainty_gate(records, top2[0] if top2 else None, counts["completed"])
    gate["attempted_views"] = counts["attempted"]
    gate["top1_support_fraction_attempted"] = gate["top1_support_views"] / counts["attempted"] if counts["attempted"] else None
    return {"task_id": task_id, "query_id": query_id,
            "primary": {"top2": top2, "ranked_grids": ranked, "records": records,
                        "view_counts": counts, "views": views, "gate": gate},
            "retry": {"status": "not_selected", "records": [], "view_counts": view_counts([])},
            "common_rescore": {"model_pass": "retry", "expected_views": 8, "candidates": [], "complete": False},
            "baseline_top2": padded, "selected_top2": padded,
            "selection_audit": {"rule": "original_primary_kgmon", "top1_locked": bool(top2)}}


def retry_budget(ttt_seconds, now, work_deadline, maximum_seconds=600):
    """A new retry budget from global spare time, never 1200-primary elapsed."""
    available = max(0.0, min(float(maximum_seconds), float(work_deadline)-float(now)))
    valid_cost = math.isfinite(float(ttt_seconds)) and float(ttt_seconds) > 0
    required = 2*float(ttt_seconds)+60 if valid_cost else None
    admitted = valid_cost and available >= required
    return {"admitted": bool(admitted), "available_seconds": available,
            "required_seconds": required, "deadline": float(now)+available,
            "reason": "admitted" if admitted else ("insufficient_global_retry_budget" if valid_cost else "no_measured_ttt_cost")}


def finalize_v2(challenges, settings, freeze=False, elapsed_seconds=None, gpu_count=None):
    root = Path(settings["artifact_root"])
    root.mkdir(parents=True, exist_ok=True)
    baseline, improved = {}, {}
    for task_id, task in challenges.items():
        baseline[task_id], improved[task_id] = [], []
        for query_id in range(len(task["test"])):
            path = Path(settings["query_dir"]) / f"{task_id}_{query_id}.json"
            if path.exists():
                document = json.loads(path.read_text(encoding="utf-8"))
            else:
                records = read_query_records(settings["output_dir"], task_id, query_id, "primary")
                document = primary_query_document(task_id, query_id, records, [])
                atomic_json(document, path)
            for target, field in ((baseline, "baseline_top2"), (improved, "selected_top2")):
                grids = document[field]
                if len(grids) != 2:
                    raise ValueError(f"Expected exactly two attempts in {path.name}")
                target[task_id].append({"attempt_1": native_grid(grids[0]), "attempt_2": native_grid(grids[1])})
    atomic_json(baseline, root / "predictions_baseline.json")
    atomic_json(improved, root / "predictions_v2.json")
    atomic_json(improved, settings["submission_path"])
    if freeze:
        paths = [root / "predictions_baseline.json", root / "predictions_v2.json"]
        paths += sorted(Path(settings["query_dir"]).glob("*.json"))
        paths += sorted(Path(settings["log_dir"]).glob("task_*.json"))
        files = {str(path.relative_to(root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        manifest_path = settings.get("selection_manifest")
        selection_hash = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest() if manifest_path else None
        atomic_json({"schema_version": 1, "files": files, "selection_manifest_sha256": selection_hash,
                     "elapsed_seconds": elapsed_seconds, "gpu_count": gpu_count}, root / "FREEZE.json")
    return {"tasks": len(challenges), "test_inputs": sum(len(t["test"]) for t in challenges.values()),
            "query_documents": sum(1 for _ in Path(settings["query_dir"]).glob("*.json"))}


def equal_query_document(task_id, query_id, settings, pass_update=None):
    """Recover durable candidates even if a worker died between file and JSON writes."""
    from equal_budget_helpers import rank_kgmon, choose_pooled, PROTOCOL
    path = Path(settings["query_dir"]) / f"{task_id}_{query_id}.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    arm = settings["arm"]
    names = ["baseline"] if arm == "baseline" else ["split_a", "split_b"]
    sections = []
    for index, name in enumerate(names):
        section_name = "passA" if index == 0 else "passB"
        section = dict(old.get(section_name, {}))
        if pass_update and pass_update["pass"] == name:
            section.update(pass_update)
        records = read_query_records(Path(settings["artifact_root"]) / "candidate_outputs" / name,
                                     task_id, query_id, name)
        ranked = rank_kgmon(records)
        views = section.get("views", [])
        section.update(pass_name=name, records=records, top2=[r["grid"] for r in ranked[:2]],
                       ranked_grids=[r["grid"] for r in ranked], view_counts=view_counts(views))
        section.setdefault("status", "not_started")
        sections.append(section)
    if len(sections) == 1:
        sections.append({"pass_name": None, "records": [], "top2": [], "ranked_grids": [],
                         "views": [], "view_counts": view_counts([]), "status": "not_applicable"})
    selection = choose_pooled(arm, sections[0]["records"], sections[1]["records"])
    baseline_top2 = sections[0]["top2"] + [[[0]]] * (2 - len(sections[0]["top2"]))
    document = {"protocol": PROTOCOL, "task_id": task_id, "query_id": query_id, "arm": arm,
                "primary": sections[0], "passA": sections[0], "passB": sections[1],
                "baseline_top2": baseline_top2, **selection}
    atomic_json(document, path)
    return document


def finalize_equal(challenges, settings, freeze=False, elapsed_seconds=None, gpu_count=None):
    from equal_budget_helpers import PROTOCOL
    root = Path(settings["artifact_root"])
    primary, selected = {}, {}
    model_outputs = 0
    for task_id, task in challenges.items():
        primary[task_id], selected[task_id] = [], []
        for query_id in range(len(task["test"])):
            document = equal_query_document(task_id, query_id, settings)
            model_outputs += bool(document["pooled"]["ranked"])
            for target, field in ((primary, "baseline_top2"), (selected, "selected_top2")):
                grids = document[field]
                if len(grids) != 2:
                    raise ValueError("Exactly two submission attempts are required")
                target[task_id].append({"attempt_1": native_grid(grids[0]), "attempt_2": native_grid(grids[1])})
    atomic_json(primary, root / "predictions_baseline.json")
    atomic_json(selected, root / "predictions_v2.json")
    atomic_json(selected, settings["submission_path"])
    coverage = {"tasks": len(challenges), "test_inputs": sum(len(t["test"]) for t in challenges.values()),
                "query_documents": sum(1 for _ in Path(settings["query_dir"]).glob("*.json")),
                "model_outputs": model_outputs, "fallback_only": model_outputs == 0}
    if freeze:
        paths = [root / "predictions_baseline.json", root / "predictions_v2.json", Path(settings["submission_path"])]
        paths += sorted(Path(settings["query_dir"]).glob("*.json"))
        paths += sorted(Path(settings["log_dir"]).glob("task_*.json"))
        paths += [root / name for name in ("nvarc_run_manifest.json", "nvarc_run_status.json") if (root / name).exists()]
        files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        selection_path = settings.get("selection_manifest")
        atomic_json({"schema_version": 1, "protocol": PROTOCOL, "arm": settings["arm"], "files": files,
                     "selection_manifest_sha256": hashlib.sha256(Path(selection_path).read_bytes()).hexdigest() if selection_path else None,
                     "elapsed_seconds": elapsed_seconds, "gpu_count": gpu_count}, root / "FREEZE.json")
    return coverage
