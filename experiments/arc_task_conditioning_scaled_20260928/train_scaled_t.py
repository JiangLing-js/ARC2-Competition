"""Continue candidate T from update 256 on 4,096 disjoint frozen episodes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

BASE_IDENTITY = "c171d0f4152a237a63a3b3fa5c43ce52a85d15f6548f70bdb079d361f5dc2204"
UPDATES = 1024
EPISODES = 4096
PARENT_UPDATES = 256
PARENT_EPISODES = 1024
PARENT_MAPS_SHA256 = "0147981e50e4ef511da09f3bfad0450bec76e3560b1d04f0cd84a4bd162daf10"
PARENT_STATE_SHA256 = "2d3f370233f9722c172dcaccdd1b54ab83155419ac4618654d527f2751be8ad7"
PARENT_EPISODES_SHA256 = "1bf782153693631a5008eef987906a9c4d7a42fed6a9d419746f09072445a721"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(root: Path) -> str:
    if (root / "adapter_config.json").exists():
        raise ValueError("Scaled T requires the original full model, not an adapter")
    weights = sorted(root.glob("*.safetensors")) or sorted(root.glob("pytorch_model*.bin"))
    files = weights + [root / "config.json", root / "tokenizer.json"] + sorted(root.glob("*.index.json"))
    if not weights or any(not path.is_file() for path in files):
        raise ValueError("Original full-model checkpoint is incomplete")
    hashes = {path.name: sha(path) for path in files}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf8")).hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n", encoding="utf8")
    temp.replace(path)


def copy_verified(source: Path, target: Path) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".partial")
    shutil.copyfile(source, temp)
    digest = sha(source)
    if sha(temp) != digest:
        raise ValueError("Backup copy differs: " + str(target))
    temp.replace(target)
    return {"bytes": target.stat().st_size, "sha256": digest}


def schedule(update: int) -> float:
    if update <= 10:
        return 3e-5 + (3e-4 - 3e-5) * (update - 1) / 9
    fraction = (update - 10) / (UPDATES - 10)
    return 3e-5 + (3e-4 - 3e-5) * (1 + math.cos(math.pi * fraction)) / 2


def train(args) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import load_file, save_file
    from task_gates import TaskGateBank, answer_nll, explicit_fold_check, fit_code

    if not torch.cuda.is_available() or not any(name in torch.cuda.get_device_name(0) for name in ("A100", "H100")):
        raise RuntimeError("Scaled T requires one A100 or H100")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 support missing")
    base_identity = checkpoint_identity(args.model)
    if base_identity != BASE_IDENTITY:
        raise ValueError("Original NVARC checkpoint identity differs")
    source_manifest = json.loads(args.manifest.read_text(encoding="utf8"))
    if (source_manifest.get("schema") != "arc_task_conditioning_scaled_episodes_v1"
            or source_manifest.get("episodes") != EPISODES
            or source_manifest.get("counts") != {"weak": 1638, "other": 1638, "replay": 820}
            or sha(args.episodes) != source_manifest["episodes_sha256"]):
        raise ValueError("Frozen scaled-T episodes differ")
    episodes = [json.loads(line) for line in args.episodes.open(encoding="utf8")]
    if len(episodes) != EPISODES or [item["ordinal"] for item in episodes] != list(range(EPISODES)):
        raise ValueError("Episode order differs")
    for item in episodes:
        if len(item["support_rows"]) != 4 or len(item["query_views"]) != 2:
            raise ValueError("Episode support/query row count differs")
    args.output.mkdir(parents=True, exist_ok=True)
    args.backup.mkdir(parents=True, exist_ok=True)
    if (args.output / "configuration.json").exists() and args.resume_checkpoint is None:
        raise FileExistsError("Scaled-T output already exists; supply an explicit verified stage checkpoint")
    parent = args.parent_checkpoint
    parent_manifest = json.loads((parent / "manifest.json").read_text(encoding="utf8"))
    if (sha(parent / "maps.safetensors") != PARENT_MAPS_SHA256 or
            sha(parent / "optimizer_rng.pt") != PARENT_STATE_SHA256 or
            parent_manifest.get("completed_outer_updates") != PARENT_UPDATES or
            parent_manifest.get("completed_episodes") != PARENT_EPISODES or
            parent_manifest.get("base_identity") != base_identity or
            parent_manifest.get("files", {}).get("maps.safetensors", {}).get("sha256") != PARENT_MAPS_SHA256 or
            parent_manifest.get("files", {}).get("optimizer_rng.pt", {}).get("sha256") != PARENT_STATE_SHA256):
        raise ValueError("Original update-256 C/Adam/RNG checkpoint differs")
    started = time.time()
    config = {"schema": "arc_task_conditioning_scaled_training_v1", "base_identity": base_identity,
              "episodes_sha256": sha(args.episodes), "episode_manifest_sha256": sha(args.manifest),
              "parent_completed_outer_updates": PARENT_UPDATES,
              "parent_completed_episodes": PARENT_EPISODES,
              "parent_maps_sha256": PARENT_MAPS_SHA256,
              "parent_optimizer_rng_sha256": PARENT_STATE_SHA256,
              "code_dim": 64, "C_seed": 20260925, "C_initial_std": 0.02,
              "inner_updates": 4, "inner_lr": 0.1, "inner_weight_decay": 0,
              "inner_regularization": 1e-3, "inner_grad_clip": 1.0,
              "query_views": 2, "outer_batch_episodes": 4, "planned_episodes": EPISODES,
              "planned_outer_updates": UPDATES, "outer_peak_lr": 3e-4,
              "outer_initial_lr": 3e-5, "outer_final_lr": 3e-5,
              "outer_warmup_updates": 10, "stage_lr_restart": True,
              "outer_weight_decay": 0, "outer_grad_clip": 1.0,
              "stop_gradient_through_code_fit": True, "backbone_frozen": True,
              "base_dtype": "bfloat16", "C_and_code_dtype": "float32",
              "attention_backend": "sdpa", "use_cache": False,
              "activation_checkpointing": "non_reentrant_transformers_decoder_layers"}
    if args.resume_checkpoint is not None:
        if json.loads((args.output / "configuration.json").read_text(encoding="utf8")) != config:
            raise ValueError("Stage-resume configuration differs")
    else:
        write_json(args.output / "configuration.json", config)
    write_json(args.output / "launch.json", {"train_stop_unix": args.train_stop_unix,
               "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint else None,
               "launched_unix": started, "gpu_name": torch.cuda.get_device_name(0)})
    copy_verified(args.output / "configuration.json", args.backup / "configuration.json")
    copy_verified(args.output / "launch.json", args.backup / "launch.json")
    model = AutoModelForCausalLM.from_pretrained(str(args.model), torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa", local_files_only=True,
                                                 low_cpu_mem_usage=True).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    for text, expected in {"0123456789": list(range(10)), "\n": [10],
                           "user": [11], "assistant": [12], "<|im_end|>": [15]}.items():
        if tokenizer.encode(text, add_special_tokens=False) != expected:
            raise ValueError("Original 16-token tokenizer differs: " + repr(text))
    if model.config.vocab_size != 16:
        raise ValueError("Model vocabulary differs")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.eval()
    bank = TaskGateBank(model).to("cuda")
    bank.install(model)
    if args.resume_checkpoint is None:
        checkpoint = parent
        restored = torch.load(checkpoint / "optimizer_rng.pt", map_location="cpu", weights_only=False)
        if (restored.get("completed_outer_updates") != PARENT_UPDATES or
                restored.get("next_episode_index") != PARENT_EPISODES or
                restored.get("episodes_sha256") != PARENT_EPISODES_SHA256 or
                json.loads(args.previous_episode_manifest.read_text(encoding="utf8")).get("episodes_sha256") != PARENT_EPISODES_SHA256):
            raise ValueError("Parent optimizer state has a different completed dose or data identity")
        completed_stage_updates = 0
        progress = []
    else:
        checkpoint = args.resume_checkpoint
        stage_manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf8"))
        for name in ("maps.safetensors", "optimizer_rng.pt"):
            if sha(checkpoint / name) != stage_manifest.get("files", {}).get(name, {}).get("sha256"):
                raise ValueError("Stage-resume checkpoint hash differs: " + name)
        restored = torch.load(checkpoint / "optimizer_rng.pt", map_location="cpu", weights_only=False)
        completed_stage_updates = int(restored["stage_completed_outer_updates"])
        if (stage_manifest.get("schema") != "arc_task_conditioning_scaled_checkpoint_v1" or
                not stage_manifest.get("complete") or
                stage_manifest.get("stage_completed_outer_updates") != completed_stage_updates or
                restored.get("next_episode_index") != completed_stage_updates * 4 or
                restored.get("episodes_sha256") != config["episodes_sha256"] or
                restored.get("parent_maps_sha256") != PARENT_MAPS_SHA256 or
                restored.get("parent_optimizer_rng_sha256") != PARENT_STATE_SHA256 or
                stage_manifest.get("base_identity") != base_identity or
                stage_manifest.get("parent_maps_sha256") != PARENT_MAPS_SHA256 or
                stage_manifest.get("parent_optimizer_rng_sha256") != PARENT_STATE_SHA256 or
                not 1 <= completed_stage_updates < UPDATES):
            raise ValueError("Stage-resume state identity or cursor differs")
        saved_progress = json.loads((args.output / "progress.json").read_text(encoding="utf8"))["updates"]
        progress = [row for row in saved_progress if row["outer_update"] <= completed_stage_updates]
        if len(progress) != completed_stage_updates:
            raise ValueError("Stage-resume progress does not reach the saved checkpoint")
        if len(saved_progress) > len(progress):
            recovery_name = f"progress_before_resume_{int(started)}.json"
            write_json(args.output / recovery_name, {"updates": saved_progress,
                       "resume_checkpoint_update": completed_stage_updates})
            copy_verified(args.output / recovery_name, args.backup / recovery_name)
            write_json(args.output / "progress.json", {"updates": progress,
                       "completed_outer_updates": completed_stage_updates,
                       "completed_episodes": completed_stage_updates * 4})
            copy_verified(args.output / "progress.json", args.backup / "progress.json")
    bank.load_state_dict(load_file(str(checkpoint / "maps.safetensors"), device="cpu"), strict=True)
    write_json(args.output / "gate_configuration.json", bank.configuration())
    copy_verified(args.output / "gate_configuration.json", args.backup / "gate_configuration.json")
    check_view = min((view for ep in episodes[:32] for view in ep["support_rows"]),
                     key=lambda v: v["sequence_tokens"])
    check = explicit_fold_check(model, bank, check_view)
    write_json(args.output / "explicit_fold_check.json", check)
    copy_verified(args.output / "explicit_fold_check.json", args.backup / "explicit_fold_check.json")
    print("T_FOLD_CHECK", json.dumps(check), flush=True)
    if (getattr(model.config, "attention_dropout", 0) != 0 or
            any(isinstance(module, torch.nn.Dropout) and module.p != 0
                for module in model.modules())):
        raise ValueError("Nonzero dropout would change the original backbone")
    # Transformers applies its layer checkpointing only in training mode.
    # Frozen Qwen has no nonzero dropout, and only C receives outer gradients.
    model.train()
    optimizer = torch.optim.AdamW(bank.parameters(), lr=3e-4, weight_decay=0)
    optimizer.load_state_dict(restored["optimizer"])
    if len(optimizer.state) != len(list(bank.parameters())):
        raise ValueError("Restored Adam state does not cover every C-map parameter")
    adam_steps = {int(state["step"].item()) for state in optimizer.state.values()}
    if adam_steps != {PARENT_UPDATES + completed_stage_updates}:
        raise ValueError("Adam step counters differ from the verified completed dose")
    torch.set_rng_state(restored["torch_rng"])
    torch.cuda.set_rng_state(restored["cuda_rng"])
    write_json(args.output / "restored_checkpoint.json", {
        "checkpoint": str(checkpoint), "stage_completed_outer_updates": completed_stage_updates,
        "total_completed_outer_updates": PARENT_UPDATES + completed_stage_updates,
        "parent_maps_sha256": PARENT_MAPS_SHA256,
        "parent_optimizer_rng_sha256": PARENT_STATE_SHA256,
        "loaded_maps_sha256": sha(checkpoint / "maps.safetensors"),
        "loaded_optimizer_rng_sha256": sha(checkpoint / "optimizer_rng.pt"),
        "adam_step": next(iter(adam_steps))})
    copy_verified(args.output / "restored_checkpoint.json", args.backup / "restored_checkpoint.json")

    def snapshot(completed: int, label: str) -> dict:
        folder = args.output / "checkpoints" / label
        folder.mkdir(parents=True, exist_ok=False)
        map_path = folder / "maps.safetensors"
        state_path = folder / "optimizer_rng.pt"
        save_file({name: value.detach().cpu().contiguous()
                   for name, value in bank.state_dict().items()}, str(map_path))
        state = {"stage_completed_outer_updates": completed,
                 "total_completed_outer_updates": PARENT_UPDATES + completed,
                 "next_episode_index": completed * 4,
                 "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state(),
                 "episodes_sha256": config["episodes_sha256"],
                 "parent_maps_sha256": PARENT_MAPS_SHA256,
                 "parent_optimizer_rng_sha256": PARENT_STATE_SHA256}
        torch.save(state, state_path)
        files = {name: {"bytes": (folder / name).stat().st_size,
                        "sha256": sha(folder / name)} for name in ("maps.safetensors", "optimizer_rng.pt")}
        manifest = {"schema": "arc_task_conditioning_scaled_checkpoint_v1", "complete": True,
                    "stage_completed_outer_updates": completed,
                    "total_completed_outer_updates": PARENT_UPDATES + completed,
                    "stage_completed_episodes": completed * 4,
                    "total_completed_episodes": PARENT_EPISODES + completed * 4,
                    "parent_maps_sha256": PARENT_MAPS_SHA256,
                    "parent_optimizer_rng_sha256": PARENT_STATE_SHA256,
                    "files": files, "base_identity": base_identity}
        write_json(folder / "manifest.json", manifest)
        backup_folder = args.backup / "checkpoints" / label
        for name in ("maps.safetensors", "optimizer_rng.pt", "manifest.json"):
            copy_verified(folder / name, backup_folder / name)
        write_json(backup_folder / "backup_manifest.json", {"complete": True,
                   "source_manifest_sha256": sha(folder / "manifest.json"), "files": files})
        print("T_CHECKPOINT", completed, label, flush=True)
        return manifest

    reason = "target_reached"
    recent_update_times = []
    for update in range(completed_stage_updates + 1, UPDATES + 1):
        predicted = max(recent_update_times[-2:], default=180)
        if time.time() + predicted + 180 >= args.train_stop_unix:
            reason = "training_time_box_before_outer_update"
            break
        batch = episodes[(update - 1) * 4:update * 4]
        start_update = time.time()
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        query_losses = []
        support_final_losses = []
        fully_completed = True
        for item in batch:
            if time.time() + 180 >= args.train_stop_unix:
                reason = "training_time_box_mid_outer_update_no_step"
                fully_completed = False
                break
            start_episode = time.time()
            code, support_losses = fit_code(model, bank, item["support_rows"], steps=4)
            bank.bind(code.detach())
            try:
                view_losses = []
                for view in item["query_views"]:
                    query_loss = answer_nll(model, view)
                    if not torch.isfinite(query_loss):
                        raise ValueError("Nonfinite query loss")
                    (query_loss / (len(item["query_views"]) * len(batch))).backward()
                    view_losses.append(float(query_loss.detach()))
                    del query_loss
                query_losses.append(sum(view_losses) / len(view_losses))
                support_final_losses.append(support_losses[-1])
            finally:
                bank.bind(None)
            print("T_EPISODE", update, item["ordinal"], item["kind"],
                  round(time.time() - start_episode, 3), round(query_losses[-1], 6), flush=True)
        if not fully_completed:
            optimizer.zero_grad(set_to_none=True)
            break
        grad_norm = torch.nn.utils.clip_grad_norm_(bank.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise ValueError("Nonfinite C gradient")
        for group in optimizer.param_groups:
            group["lr"] = schedule(update)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        duration = time.time() - start_update
        recent_update_times.append(duration)
        record = {"outer_update": update, "episodes": update * 4,
                  "total_outer_updates": PARENT_UPDATES + update,
                  "total_episodes": PARENT_EPISODES + update * 4,
                  "seconds": duration, "mean_query_nll": sum(query_losses) / 4,
                  "mean_last_support_nll": sum(support_final_losses) / 4,
                  "C_grad_norm_before_clip": float(grad_norm), "lr": schedule(update),
                  "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
        progress.append(record)
        write_json(args.output / "progress.json", {"updates": progress,
                   "completed_outer_updates": update, "completed_episodes": update * 4})
        copy_verified(args.output / "progress.json", args.backup / "progress.json")
        print("T_OUTER_UPDATE", json.dumps(record, sort_keys=True), flush=True)
        if update == 1 or update % 16 == 0 or update in (256, 512, 1024):
            snapshot(update, f"update_{update:06d}")
    completed = len(progress)
    if completed and completed != 1 and completed % 16:
        snapshot(completed, f"update_{completed:06d}")
    status = {"schema": "arc_task_conditioning_scaled_training_status_v1",
              "complete_target": completed == UPDATES, "reason": reason,
              "completed_outer_updates": completed, "completed_episodes": completed * 4,
              "total_outer_updates": PARENT_UPDATES + completed,
              "total_episodes": PARENT_EPISODES + completed * 4,
              "planned_outer_updates": UPDATES, "planned_episodes": EPISODES,
              "elapsed_seconds": time.time() - started,
              "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
              "last_checkpoint": f"update_{completed:06d}" if completed else None}
    write_json(args.output / "status.json", status)
    copy_verified(args.output / "status.json", args.backup / "status.json")
    print("T_TRAIN_STATUS", json.dumps(status, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--previous-episode-manifest", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--train-stop-unix", type=float, required=True)
    train(parser.parse_args())
