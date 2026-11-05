from unsloth import FastLanguageModel, UnslothTrainingArguments, UnslothTrainer
from arc_loader import ArcDataset, QwenFormatter

import gc
import os
import io
import time
import torch
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from collections import defaultdict

from typing import Any, Union
from transformers import DataCollatorForLanguageModeling

import logging
from contextlib import redirect_stdout, redirect_stderr

from peft import get_peft_model_state_dict, set_peft_model_state_dict

import bz2
import pickle
import json
import math
import queue as queue_module
import traceback
import hashlib
from pathlib import Path
from transformers import TrainerCallback
from runtime_helpers import atomic_json, atomic_view_dump, log_event, primary_completed, retry_admission, native_grid
from runtime_helpers import view_counts, read_query_records, primary_query_document, retry_budget
from selection_helpers import grid_key, build_common_shortlist, select_locked_top1
from runtime_helpers import equal_query_document, finalize_equal
from equal_budget_helpers import orientation_batches, pass_configs

logging.disable(logging.WARNING)

ARC_VOCAB = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "Ċ": 10,
    "<|im_end|>": 15,
}

ARC_TOKENS = list(ARC_VOCAB.values())
USER_TOKEN_ID = 11
ASSISTANT_TOKEN_ID = 12
PAD_ID = 13
EOS_ID = 15


class UnslothFixedTrainer(UnslothTrainer):

    # Issue https://github.com/unslothai/unsloth/issues/2435

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Fixed compute_loss that handles Unsloth's view tensor issue"""
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, "_get_name") and "unsloth" in unwrapped_model._get_name().lower():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        # 🔧 KEY FIX: Clone the loss tensor before in-place operations
        if hasattr(loss, "clone"):
            loss = loss.clone()  # Converts view tensor to independent tensor
        # Now safe for DDP gradient scaling
        if self.accelerator.num_processes > 1:
            loss = loss * self.accelerator.num_processes
        return (loss, outputs) if return_outputs else loss


class QwenDataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        batch = super().torch_call(examples)
        for i in range(len(examples)):
            labels = batch["input_ids"][i].clone()
            user_start_idx = np.where(labels == USER_TOKEN_ID)[0].tolist()
            assistant_start_idx = np.where(labels == ASSISTANT_TOKEN_ID)[0].tolist()
            start_idx = sorted(user_start_idx + assistant_start_idx)
            end_idx = np.where(labels == EOS_ID)[0]
            batch["labels"][i, :] = -100
            for j, (start, end) in enumerate(zip(start_idx, end_idx)):
                assert start < end
                if j % 2 == 1:
                    start += 2
                    end += 1
                    batch["labels"][i, start:end] = labels[start:end]
        return batch


# Minimal performance patch: preserve the baseline beam set and ranking, but transfer
# only the 12 ARC-token NLL values to CPU instead of every Qwen vocabulary logit.
_ARC_TOKEN_ID_CACHE = {}


def _arc_token_ids(device):
    key = str(device)
    token_ids = _ARC_TOKEN_ID_CACHE.get(key)
    if token_ids is None:
        token_ids = torch.tensor(ARC_TOKENS, dtype=torch.long, device=device)
        _ARC_TOKEN_ID_CACHE[key] = token_ids
    return token_ids


def turbo_dfs(model, logits, max_new_tokens, max_score, scores, pos, cache, start_time, end_time, diagnostics=None) -> dict:

    n = logits.size(0)

    # Algebraically identical to: scores - logits.float().cpu().log_softmax(-1),
    # restricted to the same ARC_TOKENS used by the baseline DFS loop.
    logits_f = logits.float()
    token_ids = _arc_token_ids(logits.device)
    arc_logits = logits_f.index_select(-1, token_ids)
    nll = (
        torch.as_tensor(scores, dtype=torch.float32, device=logits.device).view(n, 1)
        + torch.logsumexp(logits_f, dim=-1, keepdim=True)
        - arc_logits
    ).cpu()

    suffixes = defaultdict(list)

    candidates = dict()

    for i in range(n):
        candidates[i] = []
        for token_idx, t in enumerate(ARC_TOKENS):
            score = nll[i, token_idx].item()
            if score < max_score:
                if t == EOS_ID:
                    suffixes[i].append((score, [t]))
                elif max_new_tokens > 1:
                    candidates[i].append((score, t))

    for i in range(n):
        candidates[i] = sorted(candidates[i], key=lambda x:x[0]) #[:5]
    
    while time.time() - start_time < 540 and time.time() < end_time:

        batch_tokens = []
        batch_scores = []
        num_alive_beams = 0

        for i in range(n):
            if len(candidates[i]) == 0:
                batch_tokens.append(PAD_ID)
                batch_scores.append(1000)
            else:
                score, t = candidates[i].pop(0)
                batch_tokens.append(t)
                batch_scores.append(score)
                num_alive_beams += 1

        if num_alive_beams == 0:
            break

        outputs = model(
            input_ids=torch.tensor(batch_tokens, device=model.device, dtype=torch.long).view(-1, 1),
            position_ids=torch.full((n, 1), pos, device=model.device),
            past_key_values=cache,
            return_dict=True,
            use_cache=True,
        )

        next_suffixes = turbo_dfs(
            model,
            logits=outputs.logits[:, -1],
            max_new_tokens=max_new_tokens-1,
            max_score=max_score,
            scores=batch_scores,
            pos=pos+1,
            cache=outputs.past_key_values,
            start_time=start_time,
            end_time=end_time,
            diagnostics=diagnostics,
        )

        for batch_id, beams in next_suffixes.items():
            for score, suffix_tokens in beams:
                suffix_tokens.insert(0, batch_tokens[batch_id])
                suffixes[batch_id].append((score, suffix_tokens))

    if diagnostics is not None and (time.time() - start_time >= 540 or time.time() >= end_time):
        for row, pending in candidates.items():
            if pending:
                diagnostics.setdefault("timed_out_rows", set()).add(row)
    return suffixes


@torch.no_grad()
def inference_turbo_dfs(model, prefix_tokens, max_new_tokens, max_score, end_time, diagnostics=None):
    if time.time() >= end_time:
        raise TimeoutError("Generation deadline reached before prefill")
    input_ids = torch.tensor(prefix_tokens, device=model.device, dtype=torch.long)
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
    suffixes = turbo_dfs(
        model,
        logits=outputs.logits[:, -1],
        max_new_tokens=max_new_tokens,
        max_score=max_score,
        scores=[0.0] * input_ids.size(0),
        pos=input_ids.size(1),
        cache=outputs.past_key_values,
        start_time=time.time(),
        end_time=end_time,
        diagnostics=diagnostics,
    )
    result = []
    for batch_id, beams in suffixes.items():
        sorted_beams = sorted(beams, key=lambda x:x[0])
        result.append((batch_id, sorted_beams))
    return result


class DeadlineCallback(TrainerCallback):
    def __init__(self, deadline):
        self.deadline = float(deadline)
        self.triggered = False

    def on_step_end(self, args, state, control, **kwargs):
        if time.time() >= self.deadline:
            control.should_training_stop = True
            self.triggered = True
        return control


def adapter_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(get_peft_model_state_dict(model, adapter_name="default").items()):
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def independently_reset_adapter(model, seed, initial_trainables):
    """Reset every PEFT LoRA layer, including embeddings, without touching base weights."""
    import random
    from peft.tuners.lora.layer import LoraLayer
    current = {name: value for name, value in model.named_parameters() if name in initial_trainables}
    if set(current) != set(initial_trainables):
        raise RuntimeError("Trainable parameter identities changed after first adaptation")
    with torch.no_grad():
        for name, value in current.items():
            value.copy_(initial_trainables[name])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    count = 0
    for module in model.modules():
        if isinstance(module, LoraLayer):
            if "default" in module.lora_A or "default" in module.lora_embedding_A:
                module.reset_lora_parameters("default", init_lora_weights=True)
                count += 1
    if not count:
        raise RuntimeError("No PEFT LoRA layers were reinitialized")
    # modules_to_save/full trainable heads must retain their original values.
    non_lora_checked = 0
    for name, value in current.items():
        if "lora_" not in name:
            if not torch.equal(value.detach().cpu(), initial_trainables[name]):
                raise RuntimeError(f"Non-LoRA trainable state leaked across passes: {name}")
            non_lora_checked += 1
    model.zero_grad(set_to_none=True)
    return count, non_lora_checked


def _candidate_scores(bk, solution, dataset, formatter, tokenizer, model, deadline):
    if time.time() >= deadline:
        raise TimeoutError("Candidate scoring deadline reached")
    score_ds = ArcDataset(keys=[bk], queries={bk: dataset.queries[bk]}, replies={bk: [native_grid(solution)]})
    seed = hash(bk) % 1024**2
    score_ds = score_ds.augment(seed=seed)
    score_ds = score_ds.cut_to_len(formatter=formatter, name="input", max_len=8192-formatter.max_new_tokens())
    rows = score_ds.as_list(formatter)
    if len(rows) != 8:
        raise ValueError("Expected eight self-score views")
    signature = hashlib.sha256(json.dumps([(r["key"], r["input"]) for r in rows]).encode()).hexdigest()
    queries, answers = [r["input"] for r in rows], [r["reply"] for r in rows]
    scores = calc_scores(queries[:4], answers[:4], tokenizer, model, deadline)
    scores += calc_scores(queries[4:], answers[4:], tokenizer, model, deadline)
    if len(scores) != 8 or not all(math.isfinite(float(x)) for x in scores):
        raise ValueError("Expected eight finite self scores")
    return {"score_aug": [float(x) for x in scores], "score_view_signature": signature,
            "score_view_ids": [r["key"] for r in rows], "score_view_seed": seed}


def _publish(key, query_ids, config, settings, summary, status="running"):
    for query_id in query_ids:
        views = [row for row in summary.get("views", {}).values() if row["query_id"] == query_id]
        equal_query_document(key, query_id, settings, {
            "pass": config["pass"], "views": views, "status": status, "error": summary.get("error"),
            "config": config, "adapter_initial_sha256": summary.get("adapter_initial_sha256"),
            "global_step": summary.get("global_step", 0), "training_rows": summary.get("training_rows", 0)})
    # One writer while the worker lives; starter rebuilds only after it exits.
    finalize_equal(settings["challenges"], settings)


def _decode_pass(key, config, model, tokenizer, formatter, puzzle_ds, settings, summary):
    dataset = puzzle_ds.split_multi_replies()
    query_ids = set(summary["query_ids"])
    phase, deadline = config["pass"], config["deadline"]
    eval_ds = dataset.augment(n=config["generation_views"] // 8, seed=config["generation_seed"])
    eval_ds = eval_ds.cut_to_len(formatter=formatter, name="input", max_len=8192-formatter.max_new_tokens())
    batches = orientation_batches(eval_ds.keys, query_ids, config["generation_views"])
    states = {subkey: {"view_id": phase + ":" + subkey, "augmentation_key": subkey,
                       "query_id": int(subkey.split(".")[0].rsplit("_", 1)[1]),
                       "pass": phase, "attempted": False, "completed": False, "timed_out": False,
                       "empty": False, "raw_valid_candidates": 0, "saved_candidates": 0}
              for batch in batches for subkey in batch}
    summary["views"] = states
    known_scores, signatures = {}, {}
    _publish(key, query_ids, config, settings, summary)
    try:
        for subkeys in batches:
            if time.time() >= config["generation_deadline"]:
                summary["stopped_by_deadline"] = True
                break
            query_id = states[subkeys[0]]["query_id"]
            tokens = [tokenizer.encode(eval_ds.get(subkey, formatter)["input"]) for subkey in subkeys]
            if len(subkeys) != 4 or len({len(row) for row in tokens}) != 1:
                raise ValueError("B4 orientation batch has unequal prompt lengths")
            for subkey in subkeys:
                states[subkey]["attempted"] = True
                log_event(settings["log_dir"], 0, "view_attempted", task_id=key, phase=phase,
                          view=states[subkey]["view_id"])
            diagnostics = {}
            began = time.time()
            dfs = inference_turbo_dfs(model, tokens, formatter.max_new_tokens(),
                                      -np.log(config["min_probability"]), config["generation_deadline"], diagnostics)
            summary["generation_seconds"] += time.time() - began
            by_row = dict(dfs)
            for row_id, subkey in enumerate(subkeys):
                state = states[subkey]
                state["timed_out"] = row_id in diagnostics.get("timed_out_rows", set())
                state["search_completed"] = not state["timed_out"]
                bk, decoded = subkey.split(".")[0], []
                score_complete = True
                for beam_score, generated in by_row.get(row_id, []):
                    array = formatter.convert_tokens_to_array(generated)
                    if array is None:
                        continue
                    solution = native_grid(dataset.invert_mod(array, subkey, inv_perm=True))
                    state["raw_valid_candidates"] += 1
                    cache_key = (bk, grid_key(solution))
                    if cache_key not in known_scores:
                        began = time.time()
                        try:
                            scored = _candidate_scores(bk, solution, dataset, formatter, tokenizer, model, deadline)
                            if query_id in signatures and signatures[query_id] != scored["score_view_signature"]:
                                raise ValueError("Self-scoring prompt signature changed within adapter")
                            signatures[query_id] = scored["score_view_signature"]
                            # Check the same eight scoring prompts across A and B.
                            prior_path = Path(settings["query_dir"]) / f"{key}_{query_id}.json"
                            prior = json.loads(prior_path.read_text(encoding="utf-8"))
                            prior_records = prior["passA"]["records"]
                            if phase == "split_b" and prior_records and any(
                                r["score_view_signature"] != scored["score_view_signature"] for r in prior_records):
                                raise ValueError("Scoring prompts differ between adapters")
                            known_scores[cache_key] = scored
                        except TimeoutError:
                            score_complete = False
                            state["timed_out"] = True
                            summary["stopped_by_deadline"] = True
                            break
                        finally:
                            summary["score_seconds"] += time.time() - began
                    decoded.append({"beam_score": float(beam_score), **known_scores[cache_key],
                                    "solution": solution, "view_id": state["view_id"], "pass": phase,
                                    "record_id": f"{phase}:{subkey}:out{len(decoded)}",
                                    "adapter_initial_sha256": summary["adapter_initial_sha256"]})
                if decoded:
                    atomic_view_dump(decoded, Path(settings["artifact_root"]) / "candidate_outputs" / phase / (subkey + ".bz2"))
                    summary["candidate_counts"][query_id] += len(decoded)
                state.update(saved_candidates=len(decoded), scoring_completed=score_complete,
                             completed=state["search_completed"] and score_complete,
                             empty=state["search_completed"] and state["raw_valid_candidates"] == 0)
                log_event(settings["log_dir"], 0, "view_finished", task_id=key, phase=phase, **state)
                _publish(key, query_ids, config, settings, summary)
    finally:
        summary["view_counts"] = view_counts(states)
        summary["completed_views"] = summary["view_counts"]["completed"]


def run_pass(key, config, model, tokenizer, formatter, collator, arc_test_set, settings, initial_hash):
    started = time.time()
    query_ids = list(range(len(arc_test_set.queries[key]["test"])))
    summary = {"task_id": key, "phase": config["pass"], "query_ids": query_ids, **config,
               "candidate_counts": {q: 0 for q in query_ids}, "error": None,
               "stopped_by_deadline": False, "training_rows": 0, "global_step": 0,
               "ttt_seconds": 0.0, "generation_seconds": 0.0, "score_seconds": 0.0,
               "expected_training_rows": config["training_rows"],
               "expected_views": config["generation_views"] * len(query_ids), "completed_views": 0,
               "adapter_initial_sha256": initial_hash, "optimizer_fresh": True, "views": {}}
    # The measured count must not inherit the nominal value from config.
    summary["training_rows"] = 0
    trainer = None
    log_event(settings["log_dir"], 0, "pass_start", **summary)
    _publish(key, query_ids, config, settings, summary)
    try:
        if time.time() >= config["deadline"]:
            raise TimeoutError("No pass time remains before adaptation")
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        model = FastLanguageModel.for_training(model)
        unexpected = {name for name, value in model.named_parameters() if value.requires_grad} - model._equal_initial_trainable_names
        if unexpected:
            raise RuntimeError(f"Untracked trainable parameters appeared: {sorted(unexpected)}")
        train_start = time.time()
        puzzle_ds = arc_test_set.change_keys([key])
        train_ds = puzzle_ds.augment(n=config["training_rows"] // 8, shfl_keys=True, seed=config["augmentation_seed"])
        train_ds = train_ds.cut_to_len(formatter=formatter, name="text", max_len=8192)
        rows = train_ds.as_list(formatter)
        summary["training_rows"] = len(rows)
        if len(rows) != config["training_rows"]:
            raise ValueError("Unexpected augmented training row count")
        callback = DeadlineCallback(config["deadline"])
        train_args = dict(per_device_eval_batch_size=1, per_device_train_batch_size=1,
            gradient_accumulation_steps=1, num_train_epochs=1, warmup_steps=0, warmup_ratio=.1,
            max_grad_norm=1.0, learning_rate=5e-5, optim="adamw_torch", weight_decay=0.0,
            lr_scheduler_type="cosine", seed=config["training_seed"], report_to="none",
            save_strategy="no", eval_strategy="no", logging_strategy="no", fp16=False, bf16=True,
            fsdp="", ddp_find_unused_parameters=False, dataloader_num_workers=0, gradient_checkpointing=False)
        with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
            trainer = UnslothFixedTrainer(model=model, tokenizer=tokenizer, data_collator=collator,
                train_dataset=Dataset.from_list(rows), dataset_text_field="text", max_seq_length=8192,
                args=UnslothTrainingArguments(**train_args), callbacks=[callback])
            if time.time() >= config["deadline"]:
                raise TimeoutError("Deadline passed while constructing trainer")
            stats = trainer.train()
            model = trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)
            trainer = None
        summary.update(ttt_seconds=time.time() - train_start, global_step=int(stats.global_step),
                       training_loss=float(stats.training_loss) if math.isfinite(float(stats.training_loss)) else None,
                       stopped_by_deadline=callback.triggered)
        log_event(settings["log_dir"], 0, "training_complete", task_id=key, phase=config["pass"],
                  global_step=summary["global_step"], training_rows=len(rows), ttt_seconds=summary["ttt_seconds"])
        model = FastLanguageModel.for_inference(model)
        gc.collect()
        torch.cuda.empty_cache()
        with torch.inference_mode():
            _decode_pass(key, config, model, tokenizer, formatter, puzzle_ds, settings, summary)
    except TimeoutError as error:
        summary["stopped_by_deadline"] = True
        summary["deadline_reason"] = str(error)
    except Exception as error:
        summary["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
        error.__traceback__ = None
    finally:
        if trainer is not None:
            summary["global_step"] = int(getattr(trainer.state, "global_step", 0))
            trainer = None
        model.zero_grad(set_to_none=True)
        model = FastLanguageModel.for_inference(model)
        gc.collect()
        torch.cuda.empty_cache()
        summary["elapsed_seconds"] = time.time() - started
        summary["deadline_overrun_seconds"] = max(0.0, time.time() - config["deadline"])
        summary["view_counts"] = view_counts(summary["views"])
        summary["completed_views"] = summary["view_counts"]["completed"]
        summary["peak_memory_mb"] = int(torch.cuda.max_memory_allocated() // 1024**2)
        incomplete = summary["global_step"] != config["training_rows"] or summary["completed_views"] != summary["expected_views"]
        status = "error" if summary["error"] else "incomplete" if incomplete else "completed"
        summary["status"] = status
        _publish(key, query_ids, config, settings, summary, status)
        atomic_json(summary, Path(settings["log_dir"]) / f"task_{key}_{config['pass']}.json")
        log_event(settings["log_dir"], 0, "pass_complete", **summary)
    return model, summary


def worker(settings):
    key = settings["selected_task_ids"][0]
    started = settings["started_at"]
    aggregate = {"task_id": key, "phase": "primary", "arm": settings["arm"], "error": None,
                 "passes": [], "started_at": started, "task_deadline": settings["work_deadline"]}
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(model_name=settings["model_path"],
            dtype=torch.bfloat16, full_finetuning=False, load_in_4bit=False, local_files_only=True,
            use_gradient_checkpointing=False, max_seq_length=8192)
        for text, expected in {"0123456789": list(range(10)), "\n": [10], "user": [11],
                               "assistant": [12], "<|im_end|>": [15]}.items():
            if tokenizer.encode(text, add_special_tokens=False) != expected:
                raise ValueError(f"Original ARC tokenizer contract failed for {text!r}")
        model = FastLanguageModel.get_peft_model(model, r=256,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
            lora_alpha=32, lora_dropout=0.0, bias="none", use_gradient_checkpointing=False,
            random_state=42, use_rslora=True, loftq_config=None)
        for parameter in model.parameters():
            if parameter.dtype == torch.float32:
                parameter.data = parameter.data.to(torch.bfloat16)
        initial_trainables = {name: value.detach().cpu().clone()
                              for name, value in model.named_parameters() if value.requires_grad}
        if not initial_trainables:
            raise RuntimeError("No trainable adapter parameters")
        model._equal_initial_trainable_names = set(initial_trainables)
        first_hash = adapter_hash(model)
        formatter = QwenFormatter(tokenizer=tokenizer)
        collator = QwenDataCollatorForCompletionOnlyLM(tokenizer=tokenizer, mlm=False)
        arc_test_set = ArcDataset.from_file(settings["test_path"])
        aggregate["load_and_first_init_seconds"] = time.time() - started
        log_event(settings["log_dir"], 0, "worker_loaded", elapsed_seconds=time.time()-started,
                  adapter_initial_sha256=first_hash, hash_seed=os.environ.get("PYTHONHASHSEED"))
        for config in pass_configs(settings["arm"], started, settings["work_deadline"]):
            reset_start = time.time()
            initial_hash = first_hash
            if config["pass"] == "split_b":
                if time.time() >= config["deadline"]:
                    aggregate["second_pass_not_started_reason"] = "work_deadline"
                    break
                reset_count, non_lora_checked = independently_reset_adapter(model, config["initialization_seed"], initial_trainables)
                initial_hash = adapter_hash(model)
                if initial_hash == first_hash:
                    raise RuntimeError("Second initialization unexpectedly equals first adapter")
                log_event(settings["log_dir"], 0, "independent_adapter_reset", seed=137,
                          reset_modules=reset_count, restored_trainable_tensors=len(initial_trainables),
                          non_lora_trainables_checked=non_lora_checked, adapter_initial_sha256=initial_hash)
            reset_seconds = time.time() - reset_start
            model, summary = run_pass(key, config, model, tokenizer, formatter, collator,
                                      arc_test_set, settings, initial_hash)
            summary["reset_seconds"] = reset_seconds
            aggregate["passes"].append(summary)
            aggregate["elapsed_seconds"] = time.time()-started
            aggregate["error"] = "; ".join(s["error"] for s in aggregate["passes"] if s["error"]) or None
            atomic_json(aggregate, Path(settings["log_dir"]) / f"task_{key}_primary.json")
            if summary["error"]:
                break
    except Exception as error:
        aggregate["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        aggregate["elapsed_seconds"] = time.time() - started
        aggregate["deadline_overrun_seconds"] = max(0.0, time.time() - settings["work_deadline"])
        aggregate["global_step"] = sum(s["global_step"] for s in aggregate["passes"])
        aggregate["training_rows"] = sum(s["training_rows"] for s in aggregate["passes"])
        aggregate["ttt_seconds"] = sum(s["ttt_seconds"] for s in aggregate["passes"])
        aggregate["completed_views"] = sum(s["completed_views"] for s in aggregate["passes"])
        aggregate["expected_views"] = 16 * len(settings["challenges"][key]["test"])
        aggregate["expected_training_rows"] = 128
        aggregate["status"] = "error" if aggregate["error"] else "completed" if (
            aggregate["global_step"] == 128 and aggregate["completed_views"] == aggregate["expected_views"]
        ) else "incomplete"
        atomic_json(aggregate, Path(settings["log_dir"]) / f"task_{key}_primary.json")
        log_event(settings["log_dir"], 0, "worker_finished", **aggregate)


@torch.no_grad()
def calc_scores(queries, answers, tokenizer, model, end_time=None):
    if end_time is not None and time.time() >= end_time:
        raise TimeoutError("Scoring deadline reached")
    batch_query_tokens = []
    batch_answer_tokens = []
    batch_tokens = []
    batch_lengths = []
    for query, answer in zip(queries, answers):
        query_tokens = tokenizer.encode(query)
        answer_tokens = tokenizer.encode(answer)
        tokens = query_tokens + answer_tokens
        batch_query_tokens.append(query_tokens)
        batch_answer_tokens.append(answer_tokens)
        batch_tokens.append(tokens)
        batch_lengths.append(len(tokens))
    max_len = max(batch_lengths)
    padded_tokens = []
    for tokens in batch_tokens:
        padded = tokens + [PAD_ID] * (max_len - len(tokens))
        padded_tokens.append(padded)
    input_ids = torch.tensor(padded_tokens, device=model.device, dtype=torch.long)

    # Keep logits on GPU and gather only the target-token scores. KV cache is not
    # consumed by teacher-forced scoring, so disabling it removes redundant writes.
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=False)
    batch_logits = outputs.logits.float()
    batch_log_norm = torch.logsumexp(batch_logits, dim=-1)
    result = []
    for row_id, (query_tokens, answer_tokens) in enumerate(zip(batch_query_tokens, batch_answer_tokens)):
        query_length = len(query_tokens)
        answer_length = len(answer_tokens)
        positions = torch.arange(
            query_length - 1,
            query_length - 1 + answer_length,
            device=model.device,
        )
        target_tokens = torch.tensor(answer_tokens, device=model.device, dtype=torch.long)
        answer_log_probs = (
            batch_logits[row_id, positions, target_tokens]
            - batch_log_norm[row_id, positions]
        )
        result.append(-answer_log_probs.sum().item())
    return result


