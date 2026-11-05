"""First-order MAML of the deployed NVARC rank-256 rsLoRA initializer.

The inner loop is the solver's support-only 128-row/128-step TTT. The separate
query answer contributes only to a first-order outer gradient at the adapted
weights. No gradient is taken through the inner optimizer trajectory.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import math
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import shutil
import sys
import time


BASE_IDENTITY = "c171d0f4152a237a63a3b3fa5c43ce52a85d15f6548f70bdb079d361f5dc2204"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
           "down_proj", "embed_tokens", "lm_head"]
OUTER_UPDATES = 80
INNER_STEPS = 128
PEAK_LR = 5e-5
FINAL_LR = 5e-6
WARMUP_UPDATES = 5


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8")
    temporary.replace(path)


def copy_verified(source: Path, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    shutil.copyfile(source, temporary)
    digest = sha(source)
    if sha(temporary) != digest:
        raise ValueError("Backup readback differs: " + str(destination))
    temporary.replace(destination)
    return {"bytes": source.stat().st_size, "sha256": digest}


def outer_lr(update: int) -> float:
    if not 1 <= update <= OUTER_UPDATES:
        raise ValueError(update)
    if update <= WARMUP_UPDATES:
        return PEAK_LR * update / WARMUP_UPDATES
    fraction = (update - WARMUP_UPDATES) / (OUTER_UPDATES - WARMUP_UPDATES)
    return FINAL_LR + (PEAK_LR - FINAL_LR) * (1 + math.cos(math.pi * fraction)) / 2


def control_deadline(path: Path) -> float:
    value = json.loads(Path(path).read_text(encoding="utf8"))
    if value.get("schema") != "arc_meta_ttt_control_v1":
        raise ValueError("Unknown M control schema")
    return float(value["train_stop_unix"])


def validate_inputs(args):
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf8"))
    if manifest.get("schema") != "arc_meta_episodes_v1" or manifest["inner_steps"] != INNER_STEPS:
        raise ValueError("Meta-episode manifest differs")
    if sha(Path(args.episodes)) != manifest["episodes_sha256"]:
        raise ValueError("Meta-episode file differs")
    if sha(Path(args.challenges)) != manifest["inner_challenges_sha256"]:
        raise ValueError("Inner challenges differ")
    episodes = [json.loads(line) for line in Path(args.episodes).read_text(encoding="utf8").splitlines() if line]
    if len(episodes) != 160 or len({e["task_id"] for e in episodes}) != 160:
        raise ValueError("Expected 160 distinct meta-training episodes")
    if [sum(e["kind"] == kind for e in episodes) for kind in ("weak", "other", "replay")] != [64, 64, 32]:
        raise ValueError("Meta-training mixture differs")
    challenges = json.loads(Path(args.challenges).read_text(encoding="utf8"))
    if set(challenges) != {e["task_id"] for e in episodes}:
        raise ValueError("Inner challenge IDs differ")
    if any(set(challenges[e["task_id"]]["test"][0]) != {"input"} for e in episodes):
        raise ValueError("Inner challenge contains query answer")
    for episode in episodes:
        if len(episode["outer_views"]) != 2:
            raise ValueError("Each episode needs two outer query views")
        for view in episode["outer_views"]:
            prompt, answer = view["prompt_tokens"], view["answer_tokens"]
            if not prompt or not answer or answer[-1] != 15 or len(prompt) + len(answer) > 8192:
                raise ValueError("Invalid whole outer view or missing EOS")
            if min(prompt + answer) < 0 or max(prompt + answer) > 15:
                raise ValueError("Outer view exceeds the 16-token vocabulary")
    return episodes, challenges, manifest


def train(args):
    os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")
    os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "12")
    import torch
    from datasets import Dataset
    from safetensors.torch import save_file
    from transformers import set_seed
    from score_gold import checkpoint_identity
    from preference_pipeline import logprobs

    torch.set_default_device("cpu")
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Candidate M requires the authorized A100")
    if not torch.cuda.is_bf16_supported() or torch.cuda.get_device_properties(0).total_memory < 35 * 2**30:
        raise RuntimeError("Candidate M requires the measured 40 GB BF16 A100 class")
    episodes, challenges, manifest = validate_inputs(args)
    identity, _ = checkpoint_identity(args.model)
    if identity != BASE_IDENTITY:
        raise ValueError("Candidate M must start from original NVARC weights")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "progress.json").exists() or (root / "training_status.json").exists():
        raise FileExistsError("A previous M training attempt has complete outer updates")
    backup = Path(args.backup)
    backup.mkdir(parents=True, exist_ok=True)
    started = time.time()
    write(root / "input.json", {"base_identity": identity, "episodes_sha256": sha(Path(args.episodes)),
          "inner_challenges_sha256": sha(Path(args.challenges)),
          "manifest_sha256": sha(Path(args.manifest)), "seed": 42,
          "rank": 256, "alpha": 32, "use_rslora": True, "dropout": 0,
          "targets": TARGETS, "inner_steps": INNER_STEPS, "inner_peak_lr": PEAK_LR,
          "inner_batch": 1, "inner_warmup_ratio": .1, "inner_scheduler": "cosine",
          "inner_weight_decay": 0, "outer_updates": OUTER_UPDATES, "outer_tasks_per_update": 2,
          "outer_views_per_task": 2, "outer_peak_lr": PEAK_LR, "outer_final_lr": FINAL_LR,
          "outer_warmup_updates": WARMUP_UPDATES, "outer_weight_decay": 0,
          "inner_and_outer_clip": 1, "outer_loss": "mean_answer_token_NLL_including_EOS",
          "first_order": True, "backbone_frozen": True, "working_dtype": "bfloat16",
          "master_dtype": "float32", "attention": "solver_Unsloth_default"})

    sys.path.insert(0, str(Path(args.runtime_dir).resolve()))
    import arc_solver as solver
    from arc_loader import ArcDataset, QwenFormatter

    set_seed(42)
    model, tokenizer = solver.FastLanguageModel.from_pretrained(
        model_name=args.model, dtype=torch.bfloat16, full_finetuning=False,
        load_in_4bit=False, local_files_only=True, use_gradient_checkpointing=False,
        max_seq_length=8192)
    for text, expected in {"0123456789": list(range(10)), "\n": [10],
                           "user": [11], "assistant": [12], "<|im_end|>": [15]}.items():
        if tokenizer.encode(text, add_special_tokens=False) != expected:
            raise ValueError("Original tokenizer contract failed: " + repr(text))
    model = solver.FastLanguageModel.get_peft_model(
        model, r=256, target_modules=TARGETS, lora_alpha=32, lora_dropout=0,
        bias="none", use_gradient_checkpointing=False, random_state=42,
        use_rslora=True, loftq_config=None)
    for parameter in model.parameters():
        if parameter.dtype == torch.float32:
            parameter.data = parameter.data.to(torch.bfloat16)
    trainables = {name: p for name, p in model.named_parameters() if p.requires_grad}
    # Unsloth/PEFT trains saved copies of the 16-token embedding and head in
    # this deployed configuration, alongside the rank-256 LoRA factors.
    # These two modules_to_save tensors are part of the solver adapter, while
    # the original backbone weights remain frozen.
    saved_grid_modules = ("embed_tokens.modules_to_save.default.weight",
                          "lm_head.modules_to_save.default.weight")
    unexpected = [name for name in trainables
                  if "lora_" not in name and not name.endswith(saved_grid_modules)]
    if not trainables or unexpected:
        raise ValueError("Unexpected trainable parameter: " + repr(unexpected))
    initial_hash = solver.adapter_hash(model)
    master = {name: torch.nn.Parameter(p.detach().cpu().float().clone())
              for name, p in trainables.items()}
    optimizer = torch.optim.AdamW(list(master.values()), lr=PEAK_LR,
                                   weight_decay=0, foreach=False)
    formatter = QwenFormatter(tokenizer=tokenizer)
    collator = solver.QwenDataCollatorForCompletionOnlyLM(tokenizer=tokenizer, mlm=False)
    dataset = ArcDataset(queries=challenges, is_orig=True)
    write(root / "adapter_initial.json", {"sha256": initial_hash,
          "trainable_tensors": len(trainables),
          "trainable_parameters": sum(p.numel() for p in trainables.values()),
          "base_identity": identity})
    progress = []

    def load_master(current_model):
        current = {name: p for name, p in current_model.named_parameters() if p.requires_grad}
        if set(current) != set(master):
            raise RuntimeError("Trainable adapter mapping changed")
        with torch.no_grad():
            for name, p in current.items():
                p.copy_(master[name].detach().to(device=p.device, dtype=p.dtype))
        current_model.zero_grad(set_to_none=True)
        return current

    def snapshot(update: int, label: str):
        # Save the exact FP32 global initializer; never save task-adapted factors.
        destination = root / "checkpoints" / label
        destination.mkdir(parents=True, exist_ok=False)
        save_file({name: p.detach().cpu().contiguous() for name, p in master.items()},
                  str(destination / "adapter_master.safetensors"))
        write(destination / "state.json", {"completed_outer_updates": update,
              "completed_episodes": update * 2, "adapter_master_sha256": sha(destination / "adapter_master.safetensors"),
              "episode_file_sha256": sha(Path(args.episodes))})
        for name in ("adapter_master.safetensors", "state.json"):
            copy_verified(destination / name, backup / "checkpoints" / label / name)
        write(backup / "checkpoints" / label / "backup_manifest.json", {
              "complete": True, "adapter_master_sha256": sha(backup / "checkpoints" / label / "adapter_master.safetensors"),
              "completed_outer_updates": update})
        print("M_CHECKPOINT", update, destination, flush=True)

    reason = "target_reached"
    pair_times = []
    for update in range(1, OUTER_UPDATES + 1):
        recent = max(pair_times[-3:], default=180.0)
        if time.time() + recent + 120 >= control_deadline(Path(args.control)):
            reason = "training_time_box"
            break
        pair = episodes[(update - 1) * 2:update * 2]
        if len(pair) != 2:
            raise ValueError("Frozen episode order changed")
        pair_start = time.time()
        outer_grads = {name: torch.zeros_like(p.detach()) for name, p in master.items()}
        task_records = []
        complete_pair = True
        for episode in pair:
            load_master(model)
            model = solver.FastLanguageModel.for_training(model)
            task_data = dataset.change_keys([episode["task_id"]])
            inner_data = task_data.augment(n=INNER_STEPS // 8, shfl_keys=True, seed=1)
            inner_data = inner_data.cut_to_len(formatter=formatter, name="text", max_len=8192)
            rows = inner_data.as_list(formatter)
            if len(rows) != INNER_STEPS:
                raise ValueError("Solver support-only training did not produce 128 rows")
            trainer_args = dict(per_device_eval_batch_size=1, per_device_train_batch_size=1,
                gradient_accumulation_steps=1, num_train_epochs=1, warmup_steps=0,
                warmup_ratio=.1, max_grad_norm=1.0, learning_rate=PEAK_LR,
                optim="adamw_torch", weight_decay=0, lr_scheduler_type="cosine",
                seed=42, report_to="none", save_strategy="no", eval_strategy="no",
                logging_strategy="no", fp16=False, bf16=True, fsdp="",
                ddp_find_unused_parameters=False, dataloader_num_workers=0,
                gradient_checkpointing=False)
            callback = solver.DeadlineCallback(control_deadline(Path(args.control)))
            task_start = time.time()
            with io.StringIO() as sink, redirect_stdout(sink), redirect_stderr(sink):
                trainer = solver.UnslothFixedTrainer(
                    model=model, tokenizer=tokenizer, data_collator=collator,
                    train_dataset=Dataset.from_list(rows), dataset_text_field="text",
                    max_seq_length=8192, args=solver.UnslothTrainingArguments(**trainer_args),
                    callbacks=[callback])
                stats = trainer.train()
                model = trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)
                trainer = None
            inner_seconds = time.time() - task_start
            if int(stats.global_step) != INNER_STEPS:
                reason = "inner_deadline_before_128_steps"
                complete_pair = False
                break
            current = {name: p for name, p in model.named_parameters() if p.requires_grad}
            if set(current) != set(master):
                raise RuntimeError("Adapter mapping changed after inner TTT")
            model.zero_grad(set_to_none=True)
            query_losses = []
            for view in episode["outer_views"]:
                chosen = logprobs(model, view["prompt_tokens"], view["answer_tokens"])
                if chosen.shape[0] != len(view["answer_tokens"]):
                    raise ValueError("Outer answer mask length differs")
                loss = -chosen.sum() / len(view["answer_tokens"])
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite outer query NLL")
                (loss / len(episode["outer_views"])).backward()
                query_losses.append(float(loss.detach()))
            for name, p in current.items():
                if p.grad is not None:
                    outer_grads[name].add_(p.grad.detach().float().cpu(), alpha=1 / len(pair))
            task_records.append({"task_id": episode["task_id"], "kind": episode["kind"],
                                 "inner_steps": int(stats.global_step), "inner_seconds": inner_seconds,
                                 "query_view_nll": query_losses,
                                 "query_mean_nll": sum(query_losses) / len(query_losses)})
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
        if not complete_pair:
            break
        lr = outer_lr(update)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        for name, p in master.items():
            p.grad = outer_grads[name]
        norm = torch.nn.utils.clip_grad_norm_(list(master.values()), 1.0)
        if not torch.isfinite(norm):
            raise ValueError("Nonfinite outer gradient norm")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        elapsed = time.time() - pair_start
        pair_times.append(elapsed)
        record = {"outer_update": update, "episodes": update * 2,
                  "lr": lr, "outer_gradient_norm_before_clip": float(norm),
                  "mean_query_nll": sum(t["query_mean_nll"] for t in task_records) / 2,
                  "tasks": task_records, "seconds": elapsed}
        progress.append(record)
        write(root / "progress.json", {"completed_outer_updates": update,
              "completed_episodes": update * 2, "planned_outer_updates": OUTER_UPDATES,
              "updates": progress})
        copy_verified(root / "progress.json", backup / "progress.json")
        print("M_OUTER_UPDATE", update, update * 2, round(elapsed, 2),
              round(record["mean_query_nll"], 6), flush=True)
        if update % args.checkpoint_every == 0 and update < OUTER_UPDATES:
            snapshot(update, f"update_{update:06d}")

    completed = len(progress)
    # A partial inner adaptation never changes the global initializer. Restore
    # phi before serializing the deployable adapter and its FP32 source.
    load_master(model)
    initializer = root / "initializer"
    initializer.mkdir(exist_ok=False)
    save_file({name: p.detach().cpu().contiguous() for name, p in master.items()},
              str(initializer / "adapter_master.safetensors"))
    model.save_pretrained(initializer / "adapter", safe_serialization=True)
    tokenizer.save_pretrained(initializer / "adapter")
    adapter_files = {path.relative_to(initializer).as_posix(): {"bytes": path.stat().st_size,
             "sha256": sha(path)} for path in sorted(initializer.rglob("*")) if path.is_file()}
    write(initializer / "manifest.json", {"schema": "arc_meta_initializer_v1",
          "base_identity": identity, "initial_adapter_sha256": initial_hash,
          "completed_outer_updates": completed, "completed_episodes": completed * 2,
          "rank": 256, "alpha": 32, "use_rslora": True, "dropout": 0,
          "targets": TARGETS, "working_dtype": "bfloat16", "master_dtype": "float32",
          "files": adapter_files})
    backup_files = {}
    for path in sorted(initializer.rglob("*")):
        if path.is_file():
            relative = path.relative_to(initializer)
            backup_files[relative.as_posix()] = copy_verified(path, backup / "initializer" / relative)
    write(backup / "initializer" / "backup_manifest.json", {
          "complete": True, "files": backup_files,
          "completed_outer_updates": completed,
          "manifest_sha256": sha(backup / "initializer" / "manifest.json")})
    status = {"complete_target": completed == OUTER_UPDATES, "reason": reason,
              "completed_outer_updates": completed, "completed_episodes": completed * 2,
              "planned_outer_updates": OUTER_UPDATES, "planned_episodes": 160,
              "elapsed_seconds": time.time() - started,
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
              "initializer_manifest_sha256": sha(initializer / "manifest.json"),
              "training_stop_unix": control_deadline(Path(args.control))}
    write(root / "training_status.json", status)
    copy_verified(root / "training_status.json", backup / "training_status.json")
    copy_verified(root / "input.json", backup / "input.json")
    copy_verified(root / "adapter_initial.json", backup / "adapter_initial.json")
    print("M_TRAIN_STATUS", json.dumps(status, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ("model", "runtime-dir", "episodes", "challenges", "manifest",
                 "output", "backup", "control"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
