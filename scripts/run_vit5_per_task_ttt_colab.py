"""Run strict Dense ViT-5 per-task TTT locally or in Google Colab."""

from __future__ import annotations

import argparse
import json
import os
import platform
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
from src.vit5_per_task_ttt import (
    D4_TRANSFORM_NAMES,
    METHOD_LABEL,
    PerTaskTTTConfig,
    adapt_one_task,
    attach_final_diagnostics,
    predict_task_tests,
    restore_global_base,
    save_run_outputs,
    snapshot_state,
    state_sha256,
    vit5_lora_named_parameters,
)


class _Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams
        self._closed = False

    def write(self, value: str) -> int:
        if self._closed:
            return 0
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        if self._closed:
            return
        for stream in self.streams:
            if not getattr(stream, "closed", False):
                stream.flush()

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Flush the wrapper without closing streams owned by the caller."""
        if not self._closed:
            self.flush()
            self._closed = True

    def isatty(self) -> bool:
        return bool(self.streams and getattr(self.streams[0], "isatty", lambda: False)())


def _parse_names(value: str) -> tuple[str, ...]:
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    if not names:
        raise argparse.ArgumentTypeError("Transform list cannot be empty.")
    return names


def _environment() -> dict[str, Any]:
    import tokenizers
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
    }


def _resolve_device(value: str) -> torch.device:
    resolved = "cuda" if value == "auto" and torch.cuda.is_available() else (
        "cpu" if value == "auto" else value
    )
    device = torch.device(resolved)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def _load_task_ids(path: str | Path, available: set[str]) -> list[str]:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, list) or not values or not all(isinstance(value, str) for value in values):
        raise ValueError("task_ids_json must contain a non-empty JSON list of strings.")
    if len(values) != len(set(values)):
        raise ValueError("task_ids_json contains duplicate task IDs.")
    missing = sorted(set(values) - available)
    if missing:
        raise ValueError(f"Unknown task IDs: {missing[:10]}")
    return list(values)


def _override_checkpoint_paths(
    checkpoint: dict[str, Any], t5_path: str | None, vit5_checkpoint: str | None
) -> None:
    config = checkpoint["model_config"]
    if t5_path:
        config["pretrained_model_name_or_path"] = t5_path
        tokenizer_dir = config.get("tokenizer_dir")
        if tokenizer_dir:
            candidate = Path(tokenizer_dir)
            if not candidate.is_absolute():
                candidate = Path(checkpoint.get("_checkpoint_base_dir", ".")) / candidate
            if not candidate.exists():
                config["tokenizer_dir"] = t5_path
                config["tokenizer_name_or_path"] = t5_path
        else:
            config["tokenizer_name_or_path"] = t5_path
    if vit5_checkpoint:
        config["vit5_checkpoint_path"] = vit5_checkpoint
    config["local_files_only"] = True


def _checkpoint_tokenizer_path(checkpoint: dict[str, Any]) -> str:
    config = checkpoint["model_config"]
    value = config.get("tokenizer_dir") or config.get("tokenizer_name_or_path")
    if not value:
        raise ValueError("Checkpoint has no tokenizer path.")
    path = Path(value)
    if not path.is_absolute():
        path = Path(checkpoint.get("_checkpoint_base_dir", ".")) / path
    return str(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_dir", default="/content")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--t5_path", default=None)
    parser.add_argument("--vit5_checkpoint", default=None)
    parser.add_argument("--challenges", default=None)
    parser.add_argument("--solutions", default=None)
    parser.add_argument("--task_ids_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--ttt_max_optimizer_updates", type=int, default=2000)
    parser.add_argument("--ttt_lr", type=float, default=1e-5)
    parser.add_argument("--ttt_min_lr", type=float, default=1e-6)
    parser.add_argument("--ttt_warmup_updates", type=int, default=50)
    parser.add_argument("--ttt_gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--ttt_gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--ttt_weight_decay", type=float, default=0.0)
    parser.add_argument("--ttt_validate_every_updates", type=int, default=25)
    parser.add_argument("--ttt_save_every_updates", type=int, default=25)
    parser.add_argument("--ttt_early_stopping_patience", type=int, default=20)
    parser.add_argument("--ttt_seed", type=int, default=0)
    parser.add_argument(
        "--ttt_transform_names", type=_parse_names, default=D4_TRANSFORM_NAMES
    )
    parser.add_argument(
        "--ttt_shuffle_demonstrations", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--ttt_color_permutation", action="store_true")
    parser.add_argument("--ttt_tta", action="store_true")
    parser.add_argument(
        "--ttt_tta_transform_names", type=_parse_names, default=D4_TRANSFORM_NAMES
    )
    parser.add_argument(
        "--ttt_fallback_to_base", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--save_task_lora_states", action="store_true")
    parser.add_argument("--max_grids", type=int, default=21)
    parser.add_argument("--mount_drive", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    project = Path(args.project_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    challenges_path = Path(args.challenges) if args.challenges else project / "problems" / "arc-agi_training_challenges.json"
    device = _resolve_device(args.device)
    precision = args.precision
    if device.type == "cpu" and precision != "fp32":
        print(f"precision_override: {precision} -> fp32 on CPU", flush=True)
        precision = "fp32"
    if device.type == "cuda" and precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 requested on a CUDA device without bf16 support.")
    config = PerTaskTTTConfig(
        max_optimizer_updates=args.ttt_max_optimizer_updates,
        lr=args.ttt_lr,
        min_lr=args.ttt_min_lr,
        warmup_updates=args.ttt_warmup_updates,
        gradient_accumulation_steps=args.ttt_gradient_accumulation_steps,
        gradient_clip_norm=args.ttt_gradient_clip_norm,
        weight_decay=args.ttt_weight_decay,
        validate_every_updates=args.ttt_validate_every_updates,
        save_every_updates=args.ttt_save_every_updates,
        early_stopping_patience=args.ttt_early_stopping_patience,
        seed=args.ttt_seed,
        transform_names=tuple(args.ttt_transform_names),
        shuffle_demonstrations=args.ttt_shuffle_demonstrations,
        color_permutation=args.ttt_color_permutation,
        tta=args.ttt_tta,
        tta_transform_names=tuple(args.ttt_tta_transform_names),
        fallback_to_base=args.ttt_fallback_to_base,
        precision=precision,
        max_grids=args.max_grids,
    )
    config.validate()
    environment = _environment()
    print("environment=" + json.dumps(environment, sort_keys=True), flush=True)
    print(f"method_label={METHOD_LABEL}", flush=True)
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    _override_checkpoint_paths(checkpoint, args.t5_path, args.vit5_checkpoint)
    model_config = checkpoint["model_config"]
    if model_config.get("model_type") != "vit5_hybrid":
        raise ValueError("Per-task TTT requires model_type=vit5_hybrid.")
    if model_config.get("decoder_mode") != "dense":
        raise ValueError("First-version per-task TTT requires a Dense-only checkpoint.")
    if not model_config.get("enable_vit5_lora", False):
        raise ValueError("Checkpoint must have ViT-5 LoRA enabled.")
    tokenizer, _ = load_t5_tokenizer_with_arc_tokens(
        _checkpoint_tokenizer_path(checkpoint), local_files_only=True
    )
    model = build_model_from_checkpoint(checkpoint, device=device)
    audited_lora = vit5_lora_named_parameters(model)
    print(
        f"checkpoint_format={checkpoint.get('checkpoint_format', 'full')} "
        f"decoder_mode={model_config.get('decoder_mode')} "
        f"vit5_lora_tensors={len(audited_lora)} "
        f"vit5_lora_parameters={sum(parameter.numel() for _, parameter in audited_lora)}",
        flush=True,
    )
    challenges = load_arc_challenges(challenges_path)
    task_ids = _load_task_ids(args.task_ids_json, set(challenges))
    # Challenge objects are never joined with solutions in this process.
    selected_tasks = {task_id: copy_task_without_test_outputs(challenges[task_id]) for task_id in task_ids}
    base_state = snapshot_state(model)
    base_sha = state_sha256(base_state)
    base_lora_sha: str | None = None
    results: list[dict[str, Any]] = []
    metrics_path = output_dir / "per_task_ttt_metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8", buffering=1) as metrics_handle:
        for task_number, task_id in enumerate(task_ids, start=1):
            task = selected_tasks[task_id]
            print(
                f"task_start={task_number}/{len(task_ids)} task_id={task_id} "
                f"train_pairs={len(task['train'])} test_inputs={len(task['test'])}",
                flush=True,
            )
            # An independent base prediction is compared with adapt_one_task's
            # post-restore step-0 prediction to audit output-level reset.
            restore_global_base(model, base_state, base_sha)
            expected_base = predict_task_tests(
                model, task, task_id, tokenizer, device, config, ("identity",)
            )
            progress_dir = output_dir / "task_progress"
            progress_dir.mkdir(parents=True, exist_ok=True)

            def save_progress(
                saved_task_id: str,
                update: int,
                current_lora: dict[str, torch.Tensor],
                best_lora: dict[str, torch.Tensor],
                best_metrics: dict[str, Any],
            ) -> None:
                torch.save(
                    {
                        "format": "vit5_per_task_ttt_progress_v1",
                        "task_id": saved_task_id,
                        "optimizer_update": update,
                        "current_lora_state": current_lora,
                        "best_lora_state": best_lora,
                        "best_metrics": best_metrics,
                        "base_state_sha256": base_sha,
                    },
                    progress_dir / f"{saved_task_id}.pt",
                )

            result = adapt_one_task(
                model=model,
                task=task,
                task_id=task_id,
                tokenizer=tokenizer,
                device=device,
                config=config,
                base_state=base_state,
                base_sha256=base_sha,
                metrics_handle=metrics_handle,
                progress_callback=lambda tid, update, train, validation: print(
                    f"task={tid} update={update} loss={train['dense_loss']:.6f} "
                    f"loo_exact={validation['exact_count']}/{validation['sample_count']} "
                    f"loo_pixel={validation['pixel_accuracy']:.6f} lr={train['learning_rate']:.3e}",
                    flush=True,
                ),
                save_callback=save_progress,
            )
            if result["base_test_predictions"] != expected_base:
                raise AssertionError(f"Task {task_id} did not start with global-base outputs.")
            if base_lora_sha is None:
                base_lora_sha = result["base_lora_sha256"]
            elif result["base_lora_sha256"] != base_lora_sha:
                raise AssertionError("A previous task polluted the next task's initial LoRA state.")
            best_lora_state = result.get("best_lora_state")
            if args.save_task_lora_states and best_lora_state is not None:
                lora_dir = output_dir / "task_lora_states"
                lora_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "format": "vit5_per_task_lora_v1",
                        "task_id": task_id,
                        "state_dict": best_lora_state,
                        "best_optimizer_update": result["best_optimizer_update"],
                        "base_checkpoint": str(args.checkpoint),
                        "base_state_sha256": base_sha,
                        "method_label": METHOD_LABEL,
                    },
                    lora_dir / f"{task_id}.pt",
                )
            result.pop("best_lora_state", None)
            results.append(result)
            print(
                f"task_done={task_id} best_update={result['best_optimizer_update']} "
                f"stopping_update={result['stopping_update']} fallback={result['fell_back_to_step0']} "
                f"elapsed_sec={result['elapsed_seconds']:.1f}",
                flush=True,
            )

    # This is intentionally the first point at which official solutions are read.
    diagnostics = None
    if args.solutions:
        solutions = load_arc_solutions(args.solutions)
        diagnostics = attach_final_diagnostics(results, solutions)
    summary = save_run_outputs(
        output_dir=output_dir,
        task_results=results,
        config=config,
        environment=environment,
        base_checkpoint=str(args.checkpoint),
        base_sha256=base_sha,
        diagnostics=diagnostics,
    )
    print("summary=" + json.dumps(summary, sort_keys=True), flush=True)
    return summary


def copy_task_without_test_outputs(task: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy a challenge task while dropping any accidental test outputs."""
    import copy

    result = {"train": copy.deepcopy(task.get("train", [])), "test": []}
    for item in task.get("test", []):
        result["test"].append({"input": copy.deepcopy(item["input"])})
    return result


def main() -> None:
    args = build_parser().parse_args()
    if args.mount_drive:
        try:
            from google.colab import drive
        except ImportError as exc:
            raise RuntimeError("--mount_drive is available only in Colab.") from exc
        drive.mount("/content/drive")
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_handle = (output_dir / "per_task_ttt_console.log").open("a", encoding="utf-8", buffering=1)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    stdout_tee = _Tee(original_stdout, log_handle)
    stderr_tee = _Tee(original_stderr, log_handle)
    sys.stdout = stdout_tee
    sys.stderr = stderr_tee
    try:
        os.chdir(Path(args.project_dir).expanduser().resolve())
        run(args)
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        stdout_tee.close()
        stderr_tee.close()
        log_handle.close()


if __name__ == "__main__":
    main()
