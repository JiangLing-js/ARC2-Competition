#!/usr/bin/env python3
"""Run frozen global 0-step/TTT evaluation on development or final tasks."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.arc_agi_io import load_arc_challenges, load_arc_solutions
from src.hf_arc_tokenizer import load_t5_tokenizer_with_arc_tokens
from src.model_checkpointing import build_model_from_checkpoint, load_checkpoint
from src.vit5_clean_task_split import load_frozen_split_manifest
from src.vit5_colab_baseline import resolved_environment
from src.vit5_global_ttt_evaluation import (
    evaluate_tasks_from_global_checkpoint,
    path_sha256,
    score_fixed_predictions,
    validate_frozen_evaluation_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project_dir",
        default="/content",
        help="Project root recorded for Colab command compatibility.",
    )
    parser.add_argument("--global_checkpoint", required=True)
    parser.add_argument("--frozen_evaluation_config", required=True)
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--split", choices=("development", "final_heldout"), required=True)
    parser.add_argument("--challenges", required=True)
    parser.add_argument("--solutions", default=None)
    parser.add_argument(
        "--task_id",
        default=None,
        help="Optional single task from the selected split; omit for the full split.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a partial run by skipping task artifacts already completed.",
    )
    parser.add_argument("--t5_path", required=True)
    parser.add_argument(
        "--vit5_checkpoint",
        default=None,
        help=(
            "Required for pretrained_lora checkpoints; omit for random_scratch "
            "ARC-full checkpoints."
        ),
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def _device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(value)


def _tokenizer_path(checkpoint: dict[str, Any]) -> Path:
    path = Path(checkpoint["model_config"]["tokenizer_dir"])
    if not path.is_absolute():
        path = Path(checkpoint["_checkpoint_base_dir"]) / path
    return path


def _challenge_only_task(task: dict[str, Any], task_id: str) -> dict[str, Any]:
    clean = copy.deepcopy(task)
    for test_index, item in enumerate(clean.get("test", [])):
        extra = set(item) - {"input"}
        if extra:
            raise ValueError(
                f"Challenge task {task_id} test {test_index} contains forbidden fields: {extra}"
            )
    return clean


def _payload_sha256(value: Any) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _top2_submissions(task_results: list[dict[str, Any]]) -> dict[str, Any]:
    submissions: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for task in task_results:
        task_id = str(task["task_id"])
        for branch in task["branches"]:
            name = str(branch["branch"])
            task_submission: list[dict[str, Any]] = []
            for prediction in branch["test_predictions"]:
                first = prediction.get(
                    "final_prediction", prediction.get("prediction")
                )
                if first is None:
                    raise ValueError("Fixed prediction is missing its first attempt.")
                second = prediction.get("second_prediction")
                if second is None:
                    tops = prediction.get("top_predictions", [])
                    if len(tops) >= 2:
                        second = (
                            tops[1].get("prediction")
                            if isinstance(tops[1], dict)
                            else tops[1]
                        )
                if second is None:
                    second = first
                task_submission.append(
                    {"attempt_1": first, "attempt_2": second}
                )
            submissions.setdefault(name, {})[task_id] = task_submission
    return submissions


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir).expanduser()
    if any(
        (output / name).exists()
        for name in ("predictions_fixed.json", "FINAL_EVALUATION_LOCK.json")
    ):
        raise FileExistsError(
            "This evaluation directory already contains fixed predictions; "
            "use a new output directory."
        )
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(
            "Evaluation output is non-empty. Use --resume for a partial run or "
            "choose a new directory."
        )
    output.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)
    challenges = load_arc_challenges(args.challenges)
    split = load_frozen_split_manifest(
        args.split_manifest, args.challenges, available_task_ids=set(challenges)
    )
    frozen = json.loads(Path(args.frozen_evaluation_config).read_text(encoding="utf-8"))
    branches = validate_frozen_evaluation_config(
        frozen,
        args.global_checkpoint,
        split["split_manifest_sha256"],
        require_frozen=args.split == "final_heldout",
    )
    ids_key = (
        "development_task_ids" if args.split == "development" else "final_heldout_task_ids"
    )
    external_final = bool(frozen.get("external_final_challenge_set", False))
    if external_final and args.split != "final_heldout":
        raise ValueError(
            "An external final challenge set may only run as split=final_heldout."
        )
    selected_task_ids = list(split[ids_key])
    if args.task_id is not None:
        if args.task_id not in selected_task_ids:
            raise ValueError(
                f"Task {args.task_id} is not in split {args.split}."
            )
        selected_task_ids = [args.task_id]
    tasks = {
        task_id: _challenge_only_task(challenges[task_id], task_id)
        for task_id in selected_task_ids
    }
    run_manifest = {
        "run_manifest_schema": "vit5_global_ttt_streaming_tasks_v1",
        "split": args.split,
        "task_ids": selected_task_ids,
        "single_task_id": args.task_id,
        "split_manifest_sha256": split["split_manifest_sha256"],
        "challenge_path_sha256": path_sha256(args.challenges),
        "frozen_config_sha256": frozen["frozen_config_sha256"],
        "global_checkpoint_sha256": path_sha256(args.global_checkpoint),
        "solutions_excluded_from_run_contract": True,
    }
    run_manifest["run_manifest_sha256"] = _payload_sha256(run_manifest)
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != run_manifest:
            raise ValueError("Partial-run manifest differs from the requested evaluation.")
    else:
        _atomic_json(manifest_path, run_manifest)
    checkpoint = load_checkpoint(args.global_checkpoint, map_location="cpu")
    checkpoint["model_config"]["pretrained_model_name_or_path"] = args.t5_path
    initialization = checkpoint["model_config"].get(
        "vit5_initialization", "pretrained_lora"
    )
    if initialization == "random_scratch":
        if args.vit5_checkpoint is not None:
            raise ValueError(
                "random_scratch evaluation must not receive --vit5_checkpoint."
            )
        checkpoint["model_config"]["vit5_checkpoint_path"] = None
    elif initialization == "pretrained_lora":
        if not args.vit5_checkpoint:
            raise ValueError(
                "pretrained_lora evaluation requires --vit5_checkpoint."
            )
        checkpoint["model_config"]["vit5_checkpoint_path"] = args.vit5_checkpoint
    else:
        raise ValueError(
            f"Unsupported checkpoint vit5_initialization={initialization!r}."
        )
    checkpoint["model_config"]["local_files_only"] = True
    model_config = checkpoint["model_config"]
    for key in ("lambda_component", "lambda_shape"):
        if float(model_config.get(key, -1)) != float(frozen[key]):
            raise ValueError(
                f"Frozen {key}={frozen[key]} differs from checkpoint {model_config.get(key)}."
            )
    if not bool(model_config.get("formal_baseline_eligible", False)):
        raise ValueError("Global checkpoint is not marked formal_baseline_eligible.")
    training_split_sha = frozen.get(
        "global_training_split_manifest_sha256",
        frozen["split_manifest_sha256"],
    )
    if model_config.get("split_manifest_sha256") != training_split_sha:
        raise ValueError(
            "Global checkpoint does not match the frozen global-training split "
            "manifest."
        )
    tokenizer, _ = load_t5_tokenizer_with_arc_tokens(
        _tokenizer_path(checkpoint), local_files_only=True
    )
    model = build_model_from_checkpoint(checkpoint, device=device)
    precision = str(frozen.get("precision", "bf16"))
    max_grids = int(model_config.get("max_grids", 21))
    environment = resolved_environment()
    _atomic_json(output / "environment.json", environment)
    _atomic_json(output / "frozen_evaluation_config.json", frozen)
    task_dir = output / "tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    completed: dict[str, dict[str, Any]] = {}
    for task_id in selected_task_ids:
        artifact_path = task_dir / f"{task_id}.json"
        if not artifact_path.exists():
            continue
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        if (
            artifact.get("run_manifest_sha256")
            != run_manifest["run_manifest_sha256"]
            or artifact.get("task_id") != task_id
            or not isinstance(artifact.get("task_result"), dict)
        ):
            raise ValueError(f"Invalid completed task artifact: {artifact_path}")
        completed[task_id] = artifact["task_result"]
    for task_id in selected_task_ids:
        if task_id in completed:
            print(f"resume: skipping completed task {task_id}", flush=True)
            continue
        task_result = evaluate_tasks_from_global_checkpoint(
            model,
            {task_id: tasks[task_id]},
            tokenizer,
            device,
            precision,
            max_grids,
            branches,
            tta=bool(frozen.get("tta", False)),
            tta_transform_names=tuple(
                frozen.get("tta_transform_names", ("identity",))
            ),
            prediction_protocol=str(
                frozen.get("prediction_protocol", "standard_d4_v1")
            ),
            prediction_view_seed=int(frozen.get("prediction_view_seed", 0)),
            prediction_batch_size=int(frozen.get("prediction_batch_size", 4)),
            ensemble_global_zero_step=bool(
                frozen.get("ensemble_global_zero_step", False)
            ),
        )[0]
        artifact = {
            "task_artifact_schema": "vit5_global_ttt_task_complete_v1",
            "run_manifest_sha256": run_manifest["run_manifest_sha256"],
            "task_id": task_id,
            "task_result": task_result,
        }
        _atomic_json(task_dir / f"{task_id}.json", artifact)
        completed[task_id] = task_result
        print(
            f"completed task {len(completed)}/{len(selected_task_ids)}: {task_id}",
            flush=True,
        )
    results = [completed[task_id] for task_id in selected_task_ids]
    fixed_payload = {
        "predictions_fixed": True,
        "split": args.split,
        "split_manifest_sha256": split["split_manifest_sha256"],
        "frozen_config_sha256": frozen["frozen_config_sha256"],
        "official_test_outputs_loaded": False,
        "task_results": results,
    }
    predictions_path = output / "predictions_fixed.json"
    _atomic_json(predictions_path, fixed_payload)
    submissions_path = output / "submissions_top2_fixed.json"
    _atomic_json(
        submissions_path,
        {
            "predictions_fixed": True,
            "frozen_config_sha256": frozen["frozen_config_sha256"],
            "branches": _top2_submissions(results),
        },
    )
    # Only now, after the prediction artifact is fixed, may diagnostics load solutions.
    scores = None
    if args.solutions:
        solutions = load_arc_solutions(args.solutions)
        scores = score_fixed_predictions(results, solutions)
        _atomic_json(output / "scores_after_predictions_fixed.json", scores)
    summary = {
        "split": args.split,
        "task_count": len(results),
        "predictions_fixed_path": str(predictions_path),
        "submissions_top2_fixed_path": str(submissions_path),
        "completed_task_artifact_count": len(completed),
        "solutions_loaded_after_predictions_fixed": bool(args.solutions),
        "scores": scores,
        "environment": environment,
    }
    _atomic_json(output / "evaluation_summary.json", summary)
    if args.split == "final_heldout":
        _atomic_json(
            output / "FINAL_EVALUATION_LOCK.json",
            {
                "one_shot_complete": True,
                "single_task_id": args.task_id,
                "frozen_config_sha256": frozen["frozen_config_sha256"],
                "split_manifest_sha256": split["split_manifest_sha256"],
            },
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
