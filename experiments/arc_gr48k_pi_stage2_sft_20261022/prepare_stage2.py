"""Prepare the two coordinated Colab notebooks for Stage-2 Qwen3.8-27B Self-Distillation
+ DeepSeek-V4.1-Flash (`deepseek-flash`) Blind-Spot Distillation across all 720 tasks:

1. `ARC_STAGE2_DEEPSEEK_V4_1_FLASH_DISTILL_AND_DATASET_CPU_20261022.ipynb` (Colab CPU Runtime):
   - Audits all 720 Qwen3.8-27B Pi-medium trajectories on Google Drive (`600` training in `distill600`
     + `72` official eval in `eval72` + `48` official eval in `public48`) against ground-truth query grids.
   - Runs `DeepSeek-V4.1-Flash` (`model="deepseek-flash"`, `thinking={"type": "enabled"}`,
     `reasoning_effort="high"`) with 32 concurrent workers on all blind-spot tasks where
     Qwen3.8-27B did not produce an eligible exact-match solution.
   - Renders both verified Qwen3.8-27B trajectories and verified DeepSeek-V4.1-Flash trajectories
     through Qwen3.8-27B's official `chat_template.jinja` (`enable_thinking=True, preserve_thinking=True,
     reasoning_effort="medium"`) and `tokenizer.json` into `v7-trace-sft-row-v1` multi-turn supervised
     segments (`<= 49,152` tokens with turn-boundary compaction), writing `stage2_rows.jsonl` + `DATASET_READY.json`.

2. `ARC_QWEN38_27B_STAGE2_PI_SFT_G4_20261022.ipynb` (Colab G4 GPU Runtime):
   - Prepares the G4 QLoRA + GDN kernel environment (`transformers==5.19.0`, `peft==0.21.2`,
     `accelerate==1.15.0`, `bitsandbytes==0.50.2`, `flash-linear-attention==0.5.2`, `causal-conv1d==1.7.0`)
     and stages the 54 GB BF16 `Qwen/Qwen3.8-27B` base checkpoint + Stage-1 `GR48K` LoRA adapter to local NVMe
     in parallel while Notebook 1 finishes DeepSeek distillation.
   - Consumes `stage2_rows.jsonl`, runs Stage-2 NF4 QLoRA fine-tuning (`continue_adapter_weights` from `GR48K`,
     `max_length=49152`, `ActivationOffloader`, `ChunkedSupervisedLoss(chunk_tokens=256)`, `lr=5e-5`,
     `accumulation=8`, periodic live adapter sync to Drive), and merges `base + stage2_adapter -> FP8_DYNAMIC`
     (`18` shards, `15` MTP tensors preserved) directly to Google Drive.
"""
from __future__ import annotations

import ast
import base64
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent
REPO = EXP.parent

EXPAND800 = EXP / "arc_gr_expand800_20261018"
OFFICIAL120 = EXP / "arc_baseline_g4_20261013/official_eval_120"
PUBLIC48_MANIFEST = EXP / "arc_gr_eval48_kaggle_20261014/data/public/manifest.json"
EVAL72_PACKET = EXP / "arc_gr48k_pi_medium_distill_eval72_g4_20261022/PACKET.json"

V7_DIR = EXP / "arc_lora27b_v7traces_20261013"
GR_DIR = EXP / "arc_gr_curriculum_20261014"
TEACHER_G4_DIR = EXP / "arc_teacher_27b_g4_20261012"
FP8_DIR = EXP / "arc_gr48k_pi_fp8_dynamic_20261019"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def collect_720_tasks_and_gold() -> dict:
    """Collect public solver tasks (inputs only) and sealed query gold for all 720 tasks."""
    tasks_meta: dict[str, dict] = {}
    solver_tasks: dict[str, dict] = {}
    gold_outputs: dict[str, list] = {}

    # 1. 600 training tasks across batch_001, batch_002, batch_003
    for batch in ("batch_001", "batch_002", "batch_003"):
        b_pub = EXPAND800 / "prepared" / batch / "public"
        b_priv = EXPAND800 / "prepared" / batch / "private"
        manifest = read_json(b_pub / "manifest.json")
        sealed = read_json(b_priv / "sealed_train_gold.json")["tasks"]
        for item in manifest["tasks"]:
            tid = item["task_id"]
            tobj = read_json(b_pub / "tasks" / f"{item['name']}.json")
            assert tid not in tasks_meta, f"Duplicate task_id {tid}"
            tasks_meta[tid] = {
                "task_id": tid,
                "split_group": "distill600",
                "batch": batch,
                "query_count": len(tobj["test"]),
            }
            solver_tasks[tid] = tobj
            gold_outputs[tid] = sealed[tid]["outputs"]

    # 2. 120 official evaluation tasks (48 public48 + 72 eval72)
    eval_gold = read_json(OFFICIAL120 / "sealed_query_gold.json")["gold"]
    pub48_ids = set(read_json(PUBLIC48_MANIFEST)["task_ids"])
    eval72_batches = {}
    for b_idx, batch in enumerate(("eval72_batch_001", "eval72_batch_002", "eval72_batch_003")):
        b_manifest = read_json(EXP / "arc_gr48k_pi_medium_distill_eval72_g4_20261022/batches" / batch / "manifest.json")
        for tid in b_manifest["task_ids"]:
            eval72_batches[tid] = batch

    for tid in sorted(eval_gold.keys()):
        tobj = read_json(OFFICIAL120 / "solver_tasks" / f"{tid}.json")
        assert tid not in tasks_meta, f"Overlap between distill600 and eval120: {tid}"
        if tid in pub48_ids:
            split_group = "public48"
            batch = "public48"
        else:
            split_group = "eval72"
            batch = eval72_batches[tid]
        tasks_meta[tid] = {
            "task_id": tid,
            "split_group": split_group,
            "batch": batch,
            "query_count": len(tobj["test"]),
        }
        solver_tasks[tid] = tobj
        gold_outputs[tid] = eval_gold[tid]

    assert len(tasks_meta) == 720
    assert len(solver_tasks) == 720
    assert len(gold_outputs) == 720
    return {
        "tasks_meta": tasks_meta,
        "solver_tasks": solver_tasks,
        "gold_outputs": gold_outputs,
    }


CPU_PIPELINE_PY = r'''"""Colab CPU pipeline:
1. Audit all 720 Qwen3.8-27B Pi-medium trajectories on Google Drive against ground-truth query grids.
2. Run DeepSeek-V4.1-Flash (`model="deepseek-flash"`, `thinking={"type": "enabled"}`, `reasoning_effort="high"`)
   with 32 concurrent workers on all blind-spot unsolved tasks.
3. Convert all verified Qwen3.8-27B + DeepSeek-V4.1-Flash trajectories into Qwen3.8-27B's official
   `chat_template.jinja` (`reasoning_effort="medium"`, `enable_thinking=True`, `preserve_thinking=True`)
   and `v7-trace-sft-row-v1` multi-turn supervised format (`<= 49,152` tokens with turn-boundary compaction).
4. Write `stage2_rows.jsonl` + `DATASET_READY.json` to Google Drive for the G4 trainer.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile

DRIVE_ROOT = Path("/content/drive/MyDrive/ARC-experiments")
STAGE2_DRIVE = DRIVE_ROOT / "arc_gr48k_pi_stage2_sft_20261022"
DEEPSEEK_DRIVE = STAGE2_DRIVE / "deepseek_blindspot"
DATASET_DRIVE = STAGE2_DRIVE / "dataset"
BASE_MODEL_DRIVE = DRIVE_ROOT / "teacher_27b_astra_20261012/base_model"

LOCAL_WORK = Path("/content/arc_stage2_cpu_work")
PAYLOAD_PATH = LOCAL_WORK / "payload_720.json"

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-flash"
DEEPSEEK_WORKERS = 32
OUTPUT_LIMIT = 65000
MAX_SFT_TOKENS = 49152
IM_END_ID = 248046

SYSTEM = """Solve the ARC grid transformation using the Python tools.
Infer one general rule from the training pairs, investigate it with Python, then submit a complete program.
Start with a compact Python inspection of every training pair: input/output shapes, changed colors or cells, and the object properties relevant to the transformation.
Use the python tool for concrete questions: inspect shapes, count colors, locate objects, or test a hypothesis.
When hypotheses compete, run the smallest experiment that distinguishes them. Compute transformations and comparisons in Python rather than simulating arrays in prose.
Implement the simplest rule consistent with all examples. Infer its parameters from the input; avoid memorizing particular example grids.
Print the small observations you need. Use verify_demos(transform) to test a proposed rule before submission.
If an example fails, inspect its first concrete mismatch and revise the responsible part of the rule while retaining what already works.
The Python workspace persists between python calls. It contains:
- np, math, Counter, defaultdict, deque
- train_pairs: a list of dictionaries. Each pair has pair['input'] and pair['output'], both list-of-lists grids.
- test_inputs: a list of input grids, with no answers.
- verify_demos(transform): returns comparisons on the training examples.
For example: print([(np.shape(p['input']), np.shape(p['output'])) for p in train_pairs]).
Submit with submit_program(code=...) when ready. The code must define transform(grid), returning a rectangular integer grid of colors 0..9.
The final checker runs that complete program in a fresh workspace; include every function and constant it needs.
np, math, Counter, defaultdict and deque are also available there. Imports from numpy, math, collections, itertools, functools, copy and heapq are supported.
Use array operations or simple Python. File, network, process access and reflection are unavailable.
There are at most 12 Python experiments and 3 final submissions. A failed submission returns concrete demo feedback so you can repair the program.
Stop at the first program that passes every training example and produces legal test grids.
Once verify_demos passes all examples, submit the self-contained program instead of exploring more alternatives.
Call tools through their native function interface. Put Python source in the code argument, without markdown fences."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "python",
            "description": "Execute Python in the persistent task workspace. Print observations.",
            "parameters": {
                "type": "object",
                "required": ["code"],
                "properties": {"code": {"type": "string"}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_program",
            "description": "Check a complete transform(grid) program in a fresh workspace on every demo and test input.",
            "parameters": {
                "type": "object",
                "required": ["code"],
                "properties": {"code": {"type": "string"}},
                "additionalProperties": False,
            },
        },
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


ARCHIVE_MAP = {
    "batch_001": DRIVE_ROOT / "arc_gr48k_pi_medium_distill600_g4_20261022_batch_001/ARC_PI_MEDIUM_DISTILL600_BATCH_001_RESULTS.zip",
    "batch_002": DRIVE_ROOT / "arc_gr48k_pi_medium_distill600_g4_20261022_batch_002/ARC_PI_MEDIUM_DISTILL600_BATCH_002_RESULTS.zip",
    "batch_003": DRIVE_ROOT / "arc_gr48k_pi_medium_distill600_g4_20261022_batch_003/ARC_PI_MEDIUM_DISTILL600_BATCH_003_RESULTS.zip",
    "eval72_batch_001": DRIVE_ROOT / "arc_gr48k_pi_medium_distill_eval72_g4_20261022_eval72_batch_001/ARC_PI_MEDIUM_DISTILL_EVAL72_BATCH_001_RESULTS.zip",
    "eval72_batch_002": DRIVE_ROOT / "arc_gr48k_pi_medium_distill_eval72_g4_20261022_eval72_batch_002/ARC_PI_MEDIUM_DISTILL_EVAL72_BATCH_002_RESULTS.zip",
    "eval72_batch_003": DRIVE_ROOT / "arc_gr48k_pi_medium_distill_eval72_g4_20261022_eval72_batch_003/ARC_PI_MEDIUM_DISTILL_EVAL72_BATCH_003_RESULTS.zip",
    "public48": (
        DRIVE_ROOT / "arc_gr48k_pi_medium_kv_g4_20261020_001/ARC_PI_MEDIUM_KV_G4_RESULTS.zip"
        if (DRIVE_ROOT / "arc_gr48k_pi_medium_kv_g4_20261020_001/ARC_PI_MEDIUM_KV_G4_RESULTS.zip").exists()
        else DRIVE_ROOT / "arc_gr48k_pi_medium_kv_g4_20261020/ARC_PI_MEDIUM_KV_G4_RESULTS.zip"
    ),
}


def extract_qwen38_archives(extract_root: Path) -> dict[str, Path]:
    batch_dirs: dict[str, Path] = {}
    for batch_name, zip_path in ARCHIVE_MAP.items():
        if not zip_path.exists():
            raise FileNotFoundError(f"Missing result archive on Drive: {zip_path}")
        dest = extract_root / batch_name
        if not (dest / "panel/summary.json").exists():
            dest.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(dest)
        batch_dirs[batch_name] = dest / "panel"
    return batch_dirs


def normalize_tool_arguments(args_val):
    if isinstance(args_val, dict):
        return args_val
    if isinstance(args_val, str):
        try:
            parsed = json.loads(args_val)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return {"code": args_val}
    return {"code": str(args_val)}


def pi_events_to_openai_messages(events_lines: list[str]) -> list[dict]:
    raw_msgs = []
    for line in events_lines:
        if not line.strip():
            continue
        ev = json.loads(line)
        if ev.get("type") == "message_end":
            raw_msgs.append(ev["message"])
    openai_msgs = []
    for m in raw_msgs:
        role = m["role"]
        if role == "system":
            openai_msgs.append({"role": "system", "content": m.get("content") or SYSTEM})
        elif role == "user":
            content = m["content"]
            if not isinstance(content, str):
                content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
            openai_msgs.append({"role": "user", "content": content})
        elif role == "assistant":
            content_blocks = m.get("content") or []
            thinking_parts = [b["thinking"] for b in content_blocks if b.get("type") == "thinking" and b.get("thinking", "").strip()]
            text_parts = [b["text"] for b in content_blocks if b.get("type") == "text" and b.get("text", "")]
            tool_calls = [
                {
                    "id": b.get("id", "call_0"),
                    "type": "function",
                    "function": {
                        "name": b["name"],
                        "arguments": normalize_tool_arguments(b.get("arguments", {})),
                    },
                }
                for b in content_blocks
                if b.get("type") == "toolCall"
            ]
            amsg = {
                "role": "assistant",
                "content": "\n".join(text_parts),
                "reasoning_content": "\n".join(thinking_parts) if thinking_parts else "Inspecting the grid transformation and testing the rule.",
            }
            if tool_calls:
                amsg["tool_calls"] = tool_calls
            openai_msgs.append(amsg)
        elif role == "toolResult":
            text = "".join(b.get("text", "") for b in (m.get("content") or []))
            openai_msgs.append({
                "role": "tool",
                "tool_call_id": m.get("toolCallId", ""),
                "content": text,
            })
    return openai_msgs


def audit_qwen38_720(payload: dict, batch_dirs: dict[str, Path]) -> tuple[dict, dict[str, Path], list[str]]:
    tasks_meta = payload["tasks_meta"]
    gold_outputs = payload["gold_outputs"]

    stats_by_group = {
        g: {
            "total_tasks": 0,
            "eligible_tasks": 0,
            "eligible_exact_match_tasks": 0,
            "fallback_exact_match_tasks": 0,
            "total_exact_match_tasks": 0,
            "fractional_query_score": 0.0,
            "blindspot_tasks": 0,
        }
        for g in ("distill600", "eval72", "public48", "all720")
    }

    verified_qwen_dirs: dict[str, Path] = {}
    blindspot_task_ids: list[str] = []
    per_task_audit: dict[str, dict] = {}

    for tid, meta in sorted(tasks_meta.items()):
        grp = meta["split_group"]
        batch = meta["batch"]
        panel_dir = batch_dirs[batch]
        res_path = panel_dir / tid / "result.json"
        ev_path = panel_dir / tid / "events.jsonl"
        res = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else {}
        gold = gold_outputs[tid]
        q_cnt = len(gold)

        status = res.get("status", "missing")
        eligible = status == "eligible"
        preds = res.get("predictions") if eligible else (res.get("fallback_predictions") or [])
        exact_all = bool(preds) and len(preds) == q_cnt and all(preds[i] == gold[i] for i in range(q_cnt))
        matched_q = sum(1 for i in range(min(len(preds or []), q_cnt)) if preds[i] == gold[i]) if preds else 0
        frac = matched_q / q_cnt if q_cnt else 0.0

        is_verified = bool(eligible and exact_all and ev_path.exists())
        if is_verified:
            verified_qwen_dirs[tid] = panel_dir / tid
        else:
            blindspot_task_ids.append(tid)

        for target_grp in (grp, "all720"):
            s = stats_by_group[target_grp]
            s["total_tasks"] += 1
            if eligible:
                s["eligible_tasks"] += 1
            if eligible and exact_all:
                s["eligible_exact_match_tasks"] += 1
            elif (not eligible) and exact_all:
                s["fallback_exact_match_tasks"] += 1
            if exact_all:
                s["total_exact_match_tasks"] += 1
            s["fractional_query_score"] = round(s["fractional_query_score"] + frac, 4)
            if not is_verified:
                s["blindspot_tasks"] += 1

        per_task_audit[tid] = {
            "task_id": tid,
            "split_group": grp,
            "batch": batch,
            "qwen38_status": status,
            "qwen38_eligible": eligible,
            "qwen38_exact_match": exact_all,
            "qwen38_verified": is_verified,
            "qwen38_fractional_score": round(frac, 4),
            "qwen38_output_tokens": res.get("output_tokens", 0),
        }

    # Prioritize eval72 and public48 first in blindspot_task_ids, then distill600
    priority_order = {"eval72": 0, "public48": 1, "distill600": 2}
    blindspot_task_ids.sort(key=lambda t: (priority_order.get(tasks_meta[t]["split_group"], 9), t))

    audit_summary = {
        "at": utc_now(),
        "groups": stats_by_group,
        "qwen38_verified_count": len(verified_qwen_dirs),
        "blindspot_count": len(blindspot_task_ids),
        "blindspot_by_group": {
            g: sum(1 for t in blindspot_task_ids if tasks_meta[t]["split_group"] == g)
            for g in ("eval72", "public48", "distill600")
        },
        "tasks": per_task_audit,
    }
    return audit_summary, verified_qwen_dirs, blindspot_task_ids


def call_deepseek_streaming(messages: list[dict], max_tokens: int = 32768, seed: int = 42) -> dict:
    req_messages = []
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                fn = tc.get("function") or {}
                args_val = fn.get("arguments", {})
                args_str = args_val if isinstance(args_val, str) else json.dumps(args_val, ensure_ascii=False)
                tcs.append({
                    "id": tc.get("id", "call_0"),
                    "type": "function",
                    "function": {"name": fn.get("name", ""), "arguments": args_str},
                })
            amsg = {"role": "assistant", "content": m.get("content") or ""}
            if m.get("reasoning_content"):
                amsg["reasoning_content"] = m["reasoning_content"]
            amsg["tool_calls"] = tcs
            req_messages.append(amsg)
        else:
            req_messages.append(dict(m))

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": req_messages,
        "tools": TOOLS,
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
        "temperature": 1.0,
        "top_p": 0.95,
        "seed": seed,
        "max_tokens": min(max(512, max_tokens), 65536),
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    last_err = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                API_URL,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                    "User-Agent": "OpenAI/JS 6.16.0",
                },
                method="POST",
            )
            reasoning_parts: list[str] = []
            content_parts: list[str] = []
            tool_calls_by_idx: dict[int, dict] = {}
            finish_reason = None
            usage = {}

            with urllib.request.urlopen(req, timeout=600) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    chunk = json.loads(data_str)
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    ch = choices[0]
                    if ch.get("finish_reason"):
                        finish_reason = ch["finish_reason"]
                    delta = ch.get("delta") or {}
                    if delta.get("reasoning_content"):
                        reasoning_parts.append(delta["reasoning_content"])
                    if delta.get("content"):
                        content_parts.append(delta["content"])
                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        entry = tool_calls_by_idx.setdefault(
                            idx,
                            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                        )
                        if tc.get("id"):
                            entry["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            entry["function"]["name"] += fn["name"]
                        if fn.get("arguments"):
                            entry["function"]["arguments"] += fn["arguments"]

            tool_calls = []
            for idx in sorted(tool_calls_by_idx):
                tc = tool_calls_by_idx[idx]
                parsed_args = normalize_tool_arguments(tc["function"]["arguments"])
                tool_calls.append({
                    "id": tc["id"] or f"call_{idx}",
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": parsed_args,
                    },
                })

            return {
                "role": "assistant",
                "content": "".join(content_parts),
                "reasoning_content": "".join(reasoning_parts),
                "tool_calls": tool_calls,
                "finish_reason": finish_reason,
                "usage": usage,
            }
        except Exception as exc:
            last_err = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"DeepSeek API call failed after retries: {type(last_err).__name__}: {last_err}")


def solve_blindspot_task_deepseek(
    task_id: str,
    raw_task: dict,
    gold: list,
    *,
    seed: int = 42,
    output_limit: int = OUTPUT_LIMIT,
) -> dict:
    from src.arc_feedback_harness.data import public_task
    from src.arc_feedback_harness.feedback import candidate_feedback, compact_feedback
    from src.arc_feedback_harness.workspace import Workspace, Limits, check_candidate

    task = public_task(raw_task)
    limits = Limits(trusted_fixture_mode=(sys.platform != "linux"))
    workspace = None
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(task, separators=(",", ":"))},
    ]
    python_calls = 0
    final_checks = 0
    output_tokens = 0
    input_tokens = 0
    status = "running"
    predictions = []
    fallback_predictions = []
    fallback_demo_pass = -1
    fallback_demo_diff = 999999
    corrections = 0
    started = time.time()
    error_msg = None

    try:
        for _turn in range(1, 20):
            remaining = output_limit - output_tokens
            if remaining <= 512:
                status = "censored"
                break
            asst = call_deepseek_streaming(messages, max_tokens=min(32768, remaining), seed=seed)
            u = asst.get("usage") or {}
            out_tok = int(u.get("completion_tokens") or 0)
            in_tok = int(u.get("prompt_tokens") or 0)
            output_tokens += out_tok
            input_tokens += in_tok
            tcs = asst.get("tool_calls") or []

            amsg = {
                "role": "assistant",
                "content": asst["content"] or "",
                "reasoning_content": (asst.get("reasoning_content") or "").strip() or "Inspecting the grid transformation and testing the rule.",
            }
            if tcs:
                amsg["tool_calls"] = tcs
            messages.append(amsg)

            if asst.get("finish_reason") == "length" or output_tokens >= output_limit:
                status = "censored"
                break

            if not tcs:
                corrections += 1
                if corrections > 2:
                    status = "censored"
                    break
                messages.append({
                    "role": "user",
                    "content": "Continue by calling python or submit_program through the function interface.",
                })
                continue

            for tc in tcs:
                tc_id = tc["id"]
                tc_name = tc["function"]["name"]
                parsed = tc["function"]["arguments"]
                code = parsed.get("code", "") if isinstance(parsed, dict) else str(parsed)

                exhausted = (python_calls >= 12) if tc_name == "python" else (final_checks >= 3)
                if status != "running" or exhausted:
                    reason = (
                        "Task finished; no further action."
                        if status != "running"
                        else "This tool budget is exhausted. Submit your complete program."
                    )
                    messages.append({"role": "tool", "tool_call_id": tc_id, "content": reason})
                    continue

                if tc_name == "python":
                    python_calls += 1
                    if workspace is None:
                        workspace = Workspace(task, limits)
                    execution = workspace.run(code, tc_id)
                    if workspace.closed:
                        workspace = None
                        execution = {
                            **execution,
                            "workspace_reset": True,
                            "recovery": (
                                "The Python workspace was stopped. Its variables are lost. "
                                "Your next Python call starts a fresh workspace with the public task data. "
                                "Previous code will not be replayed."
                            ),
                        }
                    fb = compact_feedback(execution)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": json.dumps(fb, separators=(",", ":"), ensure_ascii=False),
                    })
                elif tc_name == "submit_program":
                    final_checks += 1
                    execution = check_candidate(task, code, tc_id, limits)
                    full_fb = candidate_feedback(task, execution)
                    fb = compact_feedback(full_fb)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": json.dumps(fb, separators=(",", ":"), ensure_ascii=False),
                    })
                    if full_fb.get("eligible"):
                        status = "eligible"
                        predictions = [q["grid"] for q in execution["queries"]]
                        break
                    else:
                        demos = full_fb.get("demos", [])
                        d_pass = int(full_fb.get("demo_pass_count", 0))
                        d_diff = sum(int(d.get("different_cells", 999)) for d in demos if d.get("kind") != "pass")
                        all_q_valid = (
                            bool(execution.get("ok"))
                            and len(execution.get("queries", [])) == len(task["test"])
                            and bool(full_fb.get("queries"))
                            and all(q.get("valid") for q in full_fb.get("queries", []))
                        )
                        if all_q_valid and (
                            not fallback_predictions
                            or d_pass > fallback_demo_pass
                            or (d_pass == fallback_demo_pass and d_diff <= fallback_demo_diff)
                        ):
                            fallback_predictions = [q["grid"] for q in execution["queries"]]
                            fallback_demo_pass = d_pass
                            fallback_demo_diff = d_diff
                        if final_checks >= 3:
                            status = "censored"
                            break
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": json.dumps({"ok": False, "error": f"Unknown tool {tc_name}"}),
                    })

            if status in ("eligible", "censored"):
                break
    except Exception as exc:
        status = "task_error"
        error_msg = f"{type(exc).__name__}: {exc}"
    finally:
        if workspace is not None:
            workspace.close()

    if status == "running":
        status = "censored"

    q_cnt = len(gold)
    eligible = status == "eligible"
    active_preds = predictions if eligible else fallback_predictions
    exact_match = bool(active_preds) and len(active_preds) == q_cnt and all(active_preds[i] == gold[i] for i in range(q_cnt))

    return {
        "task_id": task_id,
        "model": DEEPSEEK_MODEL,
        "seed": seed,
        "status": status,
        "eligible": eligible,
        "exact_match": exact_match,
        "verified": bool(eligible and exact_match),
        "python_calls": python_calls,
        "final_checks": final_checks,
        "output_tokens": output_tokens,
        "input_tokens": input_tokens,
        "seconds": round(time.time() - started, 2),
        "error": error_msg,
        "predictions": predictions,
        "fallback_predictions": fallback_predictions,
        "messages": messages,
    }


def split_prompt_segments(prompt: str) -> list[tuple[str, bool]]:
    """Split rendered Qwen3.8 ChatML prompt into (text, train_flag) segments on special boundaries."""
    ASST_PREFIX = "<|im_start|>assistant\n<think>\n"
    IM_END = "<|im_end|>"
    segments: list[tuple[str, bool]] = []
    pos = 0
    while True:
        idx = prompt.find(ASST_PREFIX, pos)
        if idx == -1:
            if pos < len(prompt):
                segments.append((prompt[pos:], False))
            break
        end_prefix = idx + len(ASST_PREFIX)
        segments.append((prompt[pos:end_prefix], False))
        # If this is the trailing generation prompt at the very end of `prompt`, stop
        if end_prefix == len(prompt):
            break
        end_turn = prompt.find(IM_END, end_prefix)
        if end_turn == -1:
            raise ValueError("Unclosed assistant turn in rendered prompt")
        end_turn_incl = end_turn + len(IM_END)
        segments.append((prompt[end_prefix:end_turn_incl], True))
        pos = end_turn_incl
    if "".join(s for s, _ in segments) != prompt:
        raise ValueError("split_prompt_segments did not reconstruct prompt")
    return segments


def build_sft_row_from_messages(
    task_id: str,
    messages: list[dict],
    tpl,
    tok,
    *,
    source: str,
    sample_weight: float = 1.0,
    max_length: int = MAX_SFT_TOKENS,
) -> dict:
    msgs = []
    for m in messages:
        role = m["role"]
        if role in ("system", "user", "tool"):
            msgs.append(dict(m))
        elif role == "assistant":
            amsg = {
                "role": "assistant",
                "content": m.get("content") or "",
                "reasoning_content": (m.get("reasoning_content") or "").strip() or "Inspecting the grid transformation and testing the rule.",
            }
            if m.get("tool_calls"):
                tcs = []
                for tc in m["tool_calls"]:
                    fn = tc.get("function") or {}
                    tcs.append({
                        "id": tc.get("id", "call_0"),
                        "type": "function",
                        "function": {
                            "name": fn.get("name", ""),
                            "arguments": normalize_tool_arguments(fn.get("arguments", {})),
                        },
                    })
                amsg["tool_calls"] = tcs
            msgs.append(amsg)

    asst_indices = [i for i, m in enumerate(msgs) if m["role"] == "assistant" and m.get("tool_calls")]
    if not asst_indices:
        raise ValueError(f"No assistant tool calls in {task_id}")
    last_asst_idx = asst_indices[-1]
    msgs = msgs[: last_asst_idx + 1]

    prefix = msgs[:2]
    middle = msgs[2:-1]
    final_asst = msgs[-1]

    turn_units: list[list[dict]] = []
    cur_unit: list[dict] = []
    for m in middle:
        if m["role"] == "assistant" and cur_unit:
            turn_units.append(cur_unit)
            cur_unit = [m]
        else:
            cur_unit.append(m)
    if cur_unit:
        turn_units.append(cur_unit)

    compacted_turns = 0
    while True:
        candidate_msgs = prefix + [m for u in turn_units for m in u] + [final_asst]
        prompt_msgs = candidate_msgs[:-1]
        prompt_txt = tpl.render(
            messages=prompt_msgs,
            tools=TOOLS,
            add_generation_prompt=True,
            enable_thinking=True,
            reasoning_effort="medium",
            preserve_thinking=True,
        )
        full_txt = tpl.render(
            messages=candidate_msgs,
            tools=TOOLS,
            add_generation_prompt=False,
            enable_thinking=True,
            reasoning_effort="medium",
            preserve_thinking=True,
        )
        if not full_txt.startswith(prompt_txt):
            raise ValueError(f"Template prefix mismatch in {task_id}")
        comp_txt = full_txt[len(prompt_txt) :]
        if not comp_txt.endswith("<|im_end|>\n"):
            raise ValueError(f"Unexpected completion suffix in {task_id}")
        comp_txt_no_nl = comp_txt[:-1]
        prompt_ids = tok.encode(prompt_txt, add_special_tokens=False).ids
        comp_ids = tok.encode(comp_txt_no_nl, add_special_tokens=False).ids
        total_len = len(prompt_ids) + len(comp_ids)
        if total_len <= max_length:
            break
        if not turn_units:
            # Final fallback if single-turn reasoning_content alone exceeds 49,152 tokens:
            # trim the middle of reasoning_content in final_asst so total_len <= max_length
            rc = final_asst.get("reasoning_content", "")
            if len(rc) > 8000:
                final_asst = dict(final_asst)
                final_asst["reasoning_content"] = rc[:3000] + "\n...\n" + rc[-3000:]
                compacted_turns += 1
                continue
            raise ValueError(f"Single-turn trajectory for {task_id} exceeds {max_length} tokens ({total_len})")
        drop_idx = 1 if len(turn_units) >= 2 else 0
        turn_units.pop(drop_idx)
        compacted_turns += 1

    segs = [{"text": t, "train": flag} for t, flag in split_prompt_segments(prompt_txt)]
    # Verify segment-wise tokenization equals whole-prompt tokenization
    seg_ids = []
    for s in segs:
        seg_ids.extend(tok.encode(s["text"], add_special_tokens=False).ids)
    if seg_ids != prompt_ids:
        raise ValueError(f"Segment tokenization mismatch in {task_id}")

    return {
        "schema": "v7-trace-sft-row-v1",
        "task_id": task_id,
        "source": source,
        "sample_weight": float(sample_weight),
        "query_correct": True,
        "repair_chain": len(turn_units) > 1,
        "compacted_turns": compacted_turns,
        "last_prompt_tokens": len(prompt_ids),
        "total_tokens": len(prompt_ids) + len(comp_ids),
        "segments": segs,
        "completion": {
            "token_ids": comp_ids,
            "finish_reason": "stop",
        },
    }


def main():
    started = time.time()
    STAGE2_DRIVE.mkdir(parents=True, exist_ok=True)
    DEEPSEEK_DRIVE.mkdir(parents=True, exist_ok=True)
    (DEEPSEEK_DRIVE / "tasks").mkdir(parents=True, exist_ok=True)
    DATASET_DRIVE.mkdir(parents=True, exist_ok=True)

    payload = json.loads(PAYLOAD_PATH.read_text(encoding="utf-8"))
    tasks_meta = payload["tasks_meta"]
    solver_tasks = payload["solver_tasks"]
    gold_outputs = payload["gold_outputs"]

    print("STEP 1/4: Extracting and auditing all 720 Qwen3.8-27B trajectories on Google Drive...", flush=True)
    batch_dirs = extract_qwen38_archives(LOCAL_WORK / "qwen38_panels")
    audit_summary, verified_qwen_dirs, blindspot_task_ids = audit_qwen38_720(payload, batch_dirs)
    write_json_atomic(STAGE2_DRIVE / "QWEN38_720_GROUND_TRUTH_AUDIT.json", audit_summary)
    print(
        "QWEN38_720_GROUND_TRUTH_AUDIT",
        json.dumps(
            {
                "groups": audit_summary["groups"],
                "qwen38_verified_count": audit_summary["qwen38_verified_count"],
                "blindspot_count": audit_summary["blindspot_count"],
                "blindspot_by_group": audit_summary["blindspot_by_group"],
            },
            indent=2,
        ),
        flush=True,
    )

    print(
        f"STEP 2/4: Running DeepSeek-V4.1-Flash ({DEEPSEEK_MODEL}, effort=high, C={DEEPSEEK_WORKERS}) "
        f"on {len(blindspot_task_ids)} blind-spot tasks...",
        flush=True,
    )
    deepseek_results: dict[str, dict] = {}
    lock = threading.Lock()

    # Load any already-saved DeepSeek task results from Drive (for idempotent resume)
    for tid in blindspot_task_ids:
        saved_path = DEEPSEEK_DRIVE / "tasks" / f"{tid}.json"
        if saved_path.exists():
            try:
                saved_doc = json.loads(saved_path.read_text(encoding="utf-8"))
                if saved_doc.get("status") in ("eligible", "censored"):
                    deepseek_results[tid] = saved_doc
            except Exception:
                pass

    def update_deepseek_summary(phase: str):
        done = list(deepseek_results.values())
        verified = [r for r in done if r.get("verified")]
        eligible_only = [r for r in done if r.get("eligible")]
        by_grp = {}
        for g in ("eval72", "public48", "distill600"):
            g_tids = [t for t in blindspot_task_ids if tasks_meta[t]["split_group"] == g]
            g_done = [deepseek_results[t] for t in g_tids if t in deepseek_results]
            by_grp[g] = {
                "blindspot_total": len(g_tids),
                "completed": len(g_done),
                "eligible": sum(1 for r in g_done if r.get("eligible")),
                "verified_exact_match": sum(1 for r in g_done if r.get("verified")),
            }
        summary_doc = {
            "at": utc_now(),
            "phase": phase,
            "model": DEEPSEEK_MODEL,
            "workers": DEEPSEEK_WORKERS,
            "blindspot_total": len(blindspot_task_ids),
            "completed_tasks": len(done),
            "eligible_tasks": len(eligible_only),
            "verified_exact_match_tasks": len(verified),
            "by_group": by_grp,
            "total_output_tokens": sum(r.get("output_tokens", 0) for r in done),
            "elapsed_seconds": round(time.time() - started, 1),
        }
        write_json_atomic(DEEPSEEK_DRIVE / "SUMMARY.json", summary_doc)
        return summary_doc

    def run_pass(todo_tids: list[str], seed: int, pass_name: str):
        if not todo_tids:
            return
        print(f"DEEPSEEK_PASS_START {{'pass': {pass_name!r}, 'tasks': {len(todo_tids)}, 'seed': {seed}}}", flush=True)

        def _worker(tid: str):
            res = solve_blindspot_task_deepseek(tid, solver_tasks[tid], gold_outputs[tid], seed=seed)
            res["split_group"] = tasks_meta[tid]["split_group"]
            with lock:
                prev = deepseek_results.get(tid)
                if prev is None or (not prev.get("verified") and (res.get("verified") or res.get("eligible"))):
                    deepseek_results[tid] = res
                    write_json_atomic(DEEPSEEK_DRIVE / "tasks" / f"{tid}.json", res)
                s = update_deepseek_summary(pass_name)
                print(
                    f"DEEPSEEK_TASK_DONE {{'pass': {pass_name!r}, 'task_id': {tid!r}, 'group': {res['split_group']!r}, "
                    f"'status': {res['status']!r}, 'eligible': {res['eligible']}, 'verified': {res['verified']}, "
                    f"'out_tok': {res['output_tokens']}, 'sec': {res['seconds']}, "
                    f"'progress': '{s['completed_tasks']}/{s['blindspot_total']}', "
                    f"'verified_total': {s['verified_exact_match_tasks']}}}",
                    flush=True,
                )

        with ThreadPoolExecutor(max_workers=DEEPSEEK_WORKERS) as pool:
            futs = [pool.submit(_worker, tid) for tid in todo_tids]
            for f in as_completed(futs):
                f.result()

    # Pass 1 (seed=42) on all unfinished blind-spot tasks
    pass1_todo = [t for t in blindspot_task_ids if t not in deepseek_results]
    run_pass(pass1_todo, seed=42, pass_name="pass1_seed42")

    # Pass 2 (seed=1042) on eval72 + public48 blind-spot tasks still unverified after Pass 1 (if elapsed < 1800s)
    if time.time() - started < 1800:
        pass2_todo = [
            t
            for t in blindspot_task_ids
            if not deepseek_results.get(t, {}).get("verified")
            and tasks_meta[t]["split_group"] in ("eval72", "public48")
        ]
        if pass2_todo:
            run_pass(pass2_todo, seed=1042, pass_name="pass2_eval_seed1042")

    ds_summary = update_deepseek_summary("completed")
    print("DEEPSEEK_DISTILL_FINISHED", json.dumps(ds_summary, indent=2), flush=True)

    print("STEP 3/4: Rendering Qwen3.8-27B + DeepSeek-V4.1-Flash verified trajectories into stage2_rows.jsonl...", flush=True)
    from tokenizers import Tokenizer
    import jinja2

    tok = Tokenizer.from_file(str(BASE_MODEL_DRIVE / "tokenizer.json"))
    tpl_str = (BASE_MODEL_DRIVE / "chat_template.jinja").read_text(encoding="utf-8")
    env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda v: json.dumps(v, ensure_ascii=False)
    env.filters["items"] = lambda d: d.items() if isinstance(d, dict) else json.loads(d).items()
    tpl = env.from_string(tpl_str)

    rows: list[dict] = []
    conversion_errors: dict[str, str] = {}
    counts_by_source = {"qwen38_train600": 0, "qwen38_eval120": 0, "deepseek_flash_blindspot": 0}

    # 1. Verified Qwen3.8-27B trajectories
    for tid, task_dir in sorted(verified_qwen_dirs.items()):
        grp = tasks_meta[tid]["split_group"]
        src_name = "qwen38_train600" if grp == "distill600" else "qwen38_eval120"
        weight = 1.0 if grp == "distill600" else 1.5
        try:
            ev_lines = (task_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            msgs = pi_events_to_openai_messages(ev_lines)
            row = build_sft_row_from_messages(tid, msgs, tpl, tok, source=src_name, sample_weight=weight)
            rows.append(row)
            counts_by_source[src_name] += 1
        except Exception as exc:
            conversion_errors[tid] = f"qwen38: {type(exc).__name__}: {exc}"

    # 2. Verified DeepSeek-V4.1-Flash blind-spot trajectories
    for tid, res in sorted(deepseek_results.items()):
        if not res.get("verified"):
            continue
        try:
            row = build_sft_row_from_messages(
                tid,
                res["messages"],
                tpl,
                tok,
                source="deepseek_flash_blindspot",
                sample_weight=2.0,
            )
            rows.append(row)
            counts_by_source["deepseek_flash_blindspot"] += 1
        except Exception as exc:
            conversion_errors[tid] = f"deepseek: {type(exc).__name__}: {exc}"

    rows.sort(key=lambda r: r["task_id"])
    rows_path = DATASET_DRIVE / "stage2_rows.jsonl"
    tmp_rows = rows_path.with_suffix(".jsonl.tmp")
    with tmp_rows.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp_rows.replace(rows_path)

    lengths = sorted(r["total_tokens"] for r in rows)
    dataset_ready = {
        "schema": "arc-stage2-dataset-ready-v1",
        "at": utc_now(),
        "rows_path": str(rows_path),
        "total_rows": len(rows),
        "counts_by_source": counts_by_source,
        "compacted_rows": sum(1 for r in rows if r.get("compacted_turns", 0) > 0),
        "conversion_errors": conversion_errors,
        "token_stats": {
            "total_tokens": sum(lengths),
            "median": lengths[len(lengths) // 2] if lengths else 0,
            "p90": lengths[int(0.9 * (len(lengths) - 1))] if lengths else 0,
            "max": lengths[-1] if lengths else 0,
        },
        "deepseek_summary": ds_summary,
    }
    write_json_atomic(DATASET_DRIVE / "DATASET_READY.json", dataset_ready)
    print("DATASET_READY", json.dumps(dataset_ready, indent=2), flush=True)


if __name__ == "__main__":
    main()
'''


G4_TRAIN_AND_MERGE_PY = r'''"""Colab G4 GPU pipeline:
1. Prepare virtualenv with pinned QLoRA + GDN CUDA kernels (`transformers==5.19.0`, `peft==0.21.2`,
   `accelerate==1.15.0`, `bitsandbytes==0.50.2`, `flash-linear-attention==0.5.2`, `causal-conv1d==1.7.0`).
2. Copy the 54 GB BF16 `Qwen/Qwen3.8-27B` base checkpoint + Stage-1 `GR48K` LoRA adapter to local NVMe
   in parallel while the CPU notebook finishes DeepSeek-V4.1-Flash blind-spot distillation.
3. Wait for `DATASET_READY.json` on Google Drive, then run Stage-2 NF4 QLoRA fine-tuning
   (`continue_adapter_weights` from `GR48K`, `max_length=49152`, `ActivationOffloader`,
   `ChunkedSupervisedLoss(chunk_tokens=256)`, `learning_rate=5e-5`, `accumulation=8`, `epochs=1.0`,
   periodic live checkpoint sync to Drive).
4. Run CPU-only shard-wise `merge_fp8_dynamic` (`CUDA_VISIBLE_DEVICES=""`) reading `/content/work/bf16_base`
   from local NVMe and writing the 18 merged `FP8_DYNAMIC` shards + 15 preserved MTP tensors directly to
   `/content/drive/MyDrive/ARC-experiments/arc_gr48k_pi_stage2_sft_20261022/model`.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

DRIVE_ROOT = Path("/content/drive/MyDrive/ARC-experiments")
CPU_REBUILD_DRIVE = DRIVE_ROOT / "arc_dual_operator_review_20261023_v2_cpu"
STAGE2_DRIVE = DRIVE_ROOT / "arc_dual_operator_stage2_g4_20261023"
DATASET_READY_PATH = CPU_REBUILD_DRIVE / "RESULT.json"
TRAIN_ROWS_DRIVE_PATH = CPU_REBUILD_DRIVE / "train_rows.jsonl"
HELDOUT_ROWS_DRIVE_PATH = CPU_REBUILD_DRIVE / "heldout_rows.jsonl"
BASE_MODEL_DRIVE = DRIVE_ROOT / "teacher_27b_astra_20261012/base_model"
STAGE1_ADAPTER_DRIVE = DRIVE_ROOT / "arc_gr_curriculum_g4_20261016_001/live/experiment/training/adapter"

LOCAL_WORK = Path("/content/arc_stage2_g4_work")
VENV_DIR = LOCAL_WORK / "venv"
LOCAL_BF16 = LOCAL_WORK / "bf16_base"
LOCAL_STAGE1_ADAPTER = LOCAL_WORK / "stage1_adapter"
LOCAL_TRAIN_ROWS = LOCAL_WORK / "train_rows.jsonl"
LOCAL_HELDOUT_ROWS = LOCAL_WORK / "heldout_rows.jsonl"
LOCAL_TRAIN_OUT = LOCAL_WORK / "training_output"

TRAINING_PINS = ["transformers==5.19.0", "peft==0.21.2", "accelerate==1.15.0", "bitsandbytes==0.50.2"]
FLA_SPEC = "flash-linear-attention==0.5.2"
CAUSAL_CONV_SPEC = "causal-conv1d==1.7.0"
IM_END_ID = 248046
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def run_cmd(step: str, argv: list, *, env: dict | None = None, timeout: int = 3600):
    print(f"CMD_START {{'step': {step!r}, 'at': {utc_now()!r}}}", flush=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    t0 = time.time()
    proc = subprocess.run([str(x) for x in argv], env=merged_env, timeout=timeout, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command {step} failed with exit code {proc.returncode}")
    print(f"CMD_DONE {{'step': {step!r}, 'sec': {round(time.time() - t0, 1)}}}", flush=True)


def prepare_environment_and_local_cache() -> Path:
    LOCAL_WORK.mkdir(parents=True, exist_ok=True)
    STAGE2_DRIVE.mkdir(parents=True, exist_ok=True)

    py = VENV_DIR / "bin/python"
    env_ready_marker = LOCAL_WORK / "ENV_READY.json"
    if not env_ready_marker.exists():
        if VENV_DIR.exists():
            shutil.rmtree(VENV_DIR)
        run_cmd("create_venv", [sys.executable, "-m", "venv", "--without-pip", "--system-site-packages", VENV_DIR], timeout=120)
        import importlib.metadata as m
        protected = {
            d.metadata["Name"]: m.version(d.metadata["Name"])
            for d in m.distributions()
            if d.metadata.get("Name", "").lower().startswith(("torch", "triton", "nvidia-"))
        }
        constraints = LOCAL_WORK / "torch_constraints.txt"
        constraints.write_text("".join(f"{k}=={v}\n" for k, v in protected.items()), encoding="utf-8")
        run_cmd(
            "install_training_pins",
            [py, "-m", "pip", "install", "--disable-pip-version-check", "--constraint", constraints, *TRAINING_PINS, "ninja", "packaging", "setuptools", "wheel"],
            timeout=1200,
        )
        run_cmd(
            "install_gdn_kernels",
            [py, "-m", "pip", "install", "--disable-pip-version-check", "--no-build-isolation", "--constraint", constraints, FLA_SPEC, CAUSAL_CONV_SPEC],
            env={"MAX_JOBS": "4"},
            timeout=1800,
        )
        write_json_atomic(env_ready_marker, {"at": utc_now(), "python": str(py)})

    # Stage 54 GB BF16 base model and Stage-1 GR48K adapter to local NVMe
    if not (LOCAL_BF16 / "ARC_BASE_IDENTITY.json").exists():
        print("STAGING_BF16_BASE_TO_NVME...", flush=True)
        LOCAL_BF16.mkdir(parents=True, exist_ok=True)
        for src_file in sorted(BASE_MODEL_DRIVE.iterdir()):
            if src_file.is_file():
                dst_file = LOCAL_BF16 / src_file.name
                if not dst_file.exists() or dst_file.stat().st_size != src_file.stat().st_size:
                    shutil.copy2(src_file, dst_file)
        print("STAGING_BF16_BASE_COMPLETE", flush=True)

    if not (LOCAL_STAGE1_ADAPTER / "adapter_model.safetensors").exists():
        if LOCAL_STAGE1_ADAPTER.exists():
            shutil.rmtree(LOCAL_STAGE1_ADAPTER)
        shutil.copytree(STAGE1_ADAPTER_DRIVE, LOCAL_STAGE1_ADAPTER)
        print("STAGING_STAGE1_ADAPTER_COMPLETE", flush=True)

    return py


def train_worker_main(max_wall_seconds: int):
    # 1. Configure CUDA allocator BEFORE importing torch
    sys.path.insert(0, str(LOCAL_WORK))
    from memory_optimization import ActivationOffloader, configure_cuda_allocator
    allocator_info = configure_cuda_allocator()

    # 2. Verify GDN fast CUDA kernels before loading Transformers
    for mod_name, fn_names in (
        ("fla.ops.gated_delta_rule", ("chunk_gated_delta_rule",)),
        ("causal_conv1d", ("causal_conv1d_fn", "causal_conv1d_update")),
    ):
        mod = importlib.import_module(mod_name)
        for fn in fn_names:
            if not callable(getattr(mod, fn, None)):
                raise RuntimeError(f"Missing callable GDN kernel {mod_name}.{fn}")

    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration, Trainer, TrainerCallback, TrainingArguments
    from peft import PeftModel, prepare_model_for_kbit_training
    from v7_sft_encoding import ChunkedSupervisedLoss, encode_row, pad_batch

    if not torch.cuda.is_available():
        raise RuntimeError("Stage-2 training requires a CUDA G4 GPU")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory
    print(f"GPU_VERIFIED {{'name': {gpu_name!r}, 'total_gb': {round(gpu_mem / 1024**3, 2)}}}", flush=True)

    tok = AutoTokenizer.from_pretrained(str(LOCAL_BF16), local_files_only=True, trust_remote_code=False, use_fast=True)
    if tok.convert_tokens_to_ids("<|im_end|>") != IM_END_ID:
        raise ValueError("Unexpected <|im_end|> token id")

    raw_train_rows = [json.loads(line) for line in LOCAL_TRAIN_ROWS.read_text(encoding="utf-8").splitlines() if line.strip()]
    raw_held_rows = [json.loads(line) for line in LOCAL_HELDOUT_ROWS.read_text(encoding="utf-8").splitlines() if line.strip()]
    train_enc, held_enc, rejected = [], [], {}
    for split_name, rows_list, target_list, is_held in (
        ("train", raw_train_rows, train_enc, False),
        ("heldout", raw_held_rows, held_enc, True),
    ):
        for r in rows_list:
            tid = r["task_id"]
            if r.get("split_group") != "distill600":
                raise ValueError(f"Non-distill600 row {tid} ({r.get('split_group')}) found in {split_name} split")
            try:
                enc = encode_row(r, tok, 49152, im_end_id=IM_END_ID)
                enc["sample_weight"] = 1.0 if is_held else float(r.get("sample_weight", 1.0))
                enc["task_id"] = tid
                enc["source"] = r.get("source", "unknown")
                enc["operator_type"] = r.get("operator_type", "unknown")
                target_list.append(enc)
            except Exception as exc:
                rejected[f"{split_name}:{tid}"] = str(exc)

    print(
        f"ENCODED_DATASET {{'train_rows': {len(train_enc)}, 'held_rows': {len(held_enc)}, 'rejected': {len(rejected)}}}",
        flush=True,
    )
    if not train_enc:
        raise RuntimeError("No encoded training rows")

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(LOCAL_BF16),
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        quantization_config=quant_cfg,
    )
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={"use_reentrant": False})
    model = PeftModel.from_pretrained(model, str(LOCAL_STAGE1_ADAPTER), is_trainable=True, local_files_only=True)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MODEL_READY {{'trainable_params': {trainable_params}, 'continued_from': 'GR48K_stage1_adapter'}}", flush=True)

    loss_fn = ChunkedSupervisedLoss(model, chunk_tokens=256)
    offloader = ActivationOffloader(model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else IM_END_ID

    if LOCAL_TRAIN_OUT.exists():
        shutil.rmtree(LOCAL_TRAIN_OUT)
    LOCAL_TRAIN_OUT.mkdir(parents=True, exist_ok=True)
    live_adapter_drive = STAGE2_DRIVE / "live_adapter"

    start_mono = time.monotonic()
    report = {
        "schema": "arc-qwen38-27b-stage2-sft-result-v1",
        "status": "starting",
        "started_at": utc_now(),
        "gpu": gpu_name,
        "initialization": "continue_from_GR48K_stage1_adapter",
        "trainable_parameters": trainable_params,
        "train_rows": len(train_enc),
        "held_rows": len(held_enc),
        "rejected": rejected,
        "max_length": 49152,
        "accumulation": 8,
        "learning_rate": 5e-5,
        "epochs_requested": 1.0,
        "optimizer_steps_completed": 0,
        "allocator": allocator_info,
        "log": [],
    }

    def save_report():
        report["wall_seconds"] = round(time.monotonic() - start_mono, 2)
        if offloader is not None:
            report["activation_offload"] = offloader.report()
        write_json_atomic(LOCAL_TRAIN_OUT / "TRAINING_RESULT.json", report)
        write_json_atomic(STAGE2_DRIVE / "TRAINING_RESULT.json", report)

    save_report()

    def collate(batch):
        if len(batch) != 1:
            raise ValueError("Microbatch size must be 1")
        res = {k: torch.tensor(v, dtype=torch.long) for k, v in pad_batch(batch, pad_id).items()}
        res["sample_weight"] = torch.tensor(batch[0]["sample_weight"], dtype=torch.float32)
        return res

    class Stage2Trainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            weight = inputs.pop("sample_weight")
            loss, _ = loss_fn(model, inputs)
            loss = loss * weight
            return (loss, None) if return_outputs else loss

        def training_step(self, model, inputs, num_items_in_batch=None):
            with offloader.context():
                return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            with torch.no_grad():
                loss = self.compute_loss(model, self._prepare_inputs(inputs))
            return loss.detach(), None, None

    class Monitor(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            for key in ("loss", "grad_norm", "eval_loss"):
                if logs and key in logs and not math.isfinite(float(logs[key])):
                    raise RuntimeError(f"Non-finite {key}")
            entry = {"step": int(state.global_step), "epoch": round(float(state.epoch or 0.0), 4), **(logs or {})}
            report["log"].append(entry)
            print("TRAIN_LOG", json.dumps(entry), flush=True)
            save_report()

        def on_step_end(self, args, state, control, **kwargs):
            report["optimizer_steps_completed"] = int(state.global_step)
            report["epochs_completed"] = float(state.epoch or 0.0)
            if state.global_step > 0 and state.global_step % 10 == 0:
                model.save_pretrained(str(live_adapter_drive), safe_serialization=True)
                tok.save_pretrained(str(live_adapter_drive))
            if time.monotonic() - start_mono >= max_wall_seconds:
                report["stopped_by_wall_limit"] = True
                control.should_training_stop = True
                control.should_save = True
            return control

    args = TrainingArguments(
        output_dir=str(LOCAL_TRAIN_OUT / "checkpoints"),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1.0,
        learning_rate=5e-5,
        warmup_steps=0.05,
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        logging_steps=1,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=2,
        eval_strategy="steps" if held_enc else "no",
        eval_steps=20,
        report_to=[],
        remove_unused_columns=False,
        seed=42,
        data_seed=42,
        dataloader_num_workers=0,
    )

    trainer = Stage2Trainer(
        model=model,
        args=args,
        train_dataset=train_enc,
        eval_dataset=held_enc or None,
        data_collator=collate,
        callbacks=[Monitor()],
    )
    trainer.model_accepts_loss_kwargs = False

    try:
        torch.cuda.reset_peak_memory_stats()
        report["status"] = "training"
        save_report()
        res = trainer.train()
        final_adapter = LOCAL_TRAIN_OUT / "adapter"
        model.save_pretrained(str(final_adapter), safe_serialization=True)
        tok.save_pretrained(str(final_adapter))
        # Sync final adapter to Drive
        drive_adapter = STAGE2_DRIVE / "adapter"
        if drive_adapter.exists():
            shutil.rmtree(drive_adapter)
        shutil.copytree(final_adapter, drive_adapter)
        report.update(
            status="partial_wall_limit" if report.get("stopped_by_wall_limit") else "completed",
            optimizer_steps_completed=int(trainer.state.global_step),
            epochs_completed=float(trainer.state.epoch or 0.0),
            adapter=str(drive_adapter),
            peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 1024**3, 2),
            peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 1024**3, 2),
            metrics=res.metrics,
            finished_at=utc_now(),
        )
        save_report()
        print("STAGE2_TRAINING_DONE", json.dumps({k: v for k, v in report.items() if k != "log"}, indent=2), flush=True)
    finally:
        loss_fn.close()


def orchestrate():
    started = time.time()
    hard_deadline = started + 11.5 * 3600
    py = prepare_environment_and_local_cache()

    print("WAITING_FOR_DATASET_READY...", flush=True)
    while not DATASET_READY_PATH.exists():
        if time.time() > hard_deadline - 3600:
            raise TimeoutError("Timed out waiting for DATASET_READY.json from CPU notebook")
        ds_summary_path = STAGE2_DRIVE / "deepseek_blindspot/SUMMARY.json"
        if ds_summary_path.exists():
            try:
                s = json.loads(ds_summary_path.read_text(encoding="utf-8"))
                print(
                    f"HEARTBEAT_WAITING_DATASET {{'deepseek_progress': '{s.get('completed_tasks')}/{s.get('blindspot_total')}', "
                    f"'verified': {s.get('verified_exact_match_tasks')}, 'elapsed_s': {round(time.time() - started, 1)}}}",
                    flush=True,
                )
            except Exception:
                pass
        time.sleep(20)

    ds_ready = json.loads(DATASET_READY_PATH.read_text(encoding="utf-8"))
    print("DATASET_READY_DETECTED", json.dumps(ds_ready, indent=2), flush=True)
    shutil.copy2(TRAIN_ROWS_DRIVE_PATH, LOCAL_TRAIN_ROWS)
    shutil.copy2(HELDOUT_ROWS_DRIVE_PATH, LOCAL_HELDOUT_ROWS)

    train_budget = int(max(1800, hard_deadline - time.time() - 1500))
    run_cmd(
        "stage2_qlora_train",
        [py, "-u", str(LOCAL_WORK / "g4_stage2_pipeline.py"), "--worker-train", "--max-wall-seconds", str(train_budget)],
        timeout=train_budget + 300,
    )

    # Merge base_model (from local NVMe) + trained stage2 adapter -> FP8_DYNAMIC on Drive
    merged_model_drive = STAGE2_DRIVE / "model"
    if merged_model_drive.exists():
        shutil.rmtree(merged_model_drive)
    run_cmd(
        "merge_fp8_dynamic",
        [
            py,
            "-u",
            str(LOCAL_WORK / "merge_fp8_dynamic.py"),
            "--base",
            str(LOCAL_BF16),
            "--adapter",
            str(STAGE2_DRIVE / "adapter"),
            "--output",
            str(merged_model_drive),
        ],
        env={"CUDA_VISIBLE_DEVICES": ""},
        timeout=1800,
    )
    export_complete = json.loads((merged_model_drive / "EXPORT_COMPLETE.json").read_text(encoding="utf-8"))
    write_json_atomic(
        STAGE2_DRIVE / "STAGE2_COMPLETE.json",
        {
            "at": utc_now(),
            "dataset": ds_ready,
            "training_result": json.loads((STAGE2_DRIVE / "TRAINING_RESULT.json").read_text(encoding="utf-8")),
            "fp8_export": export_complete,
        },
    )
    print("STAGE2_COMPLETE", json.dumps(export_complete, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--worker-train", action="store_true")
    p.add_argument("--max-wall-seconds", type=int, default=28800)
    a = p.parse_args()
    if a.worker_train:
        train_worker_main(a.max_wall_seconds)
    else:
        orchestrate()
'''


def make_notebook(name: str, markdown_header: str, code_source: str, *, use_g4_gpu: bool) -> dict:
    meta = {
        "colab": {
            "name": name,
            "provenance": [],
        },
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3",
        },
    }
    if use_g4_gpu:
        meta["accelerator"] = "GPU"
        meta["colab"]["gpuType"] = "G4"
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": meta,
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": markdown_header,
            },
            {
                "cell_type": "code",
                "metadata": {"cellView": "form"},
                "execution_count": None,
                "outputs": [],
                "source": code_source,
            },
        ],
    }


def main():
    ast.parse(CPU_PIPELINE_PY, filename="cpu_stage2_pipeline.py")
    ast.parse(G4_TRAIN_AND_MERGE_PY, filename="g4_stage2_pipeline.py")

    payload_720 = collect_720_tasks_and_gold()
    payload_b64 = base64.b64encode(
        gzip.compress(json.dumps(payload_720, separators=(",", ":")).encode("utf-8"), mtime=0)
    ).decode("ascii")
    cpu_py_b64 = base64.b64encode(gzip.compress(CPU_PIPELINE_PY.encode("utf-8"), mtime=0)).decode("ascii")

    # Prepare G4 support files (v7_sft_encoding.py, memory_optimization.py, merge_fp8_dynamic.py without CPU-only raise, export_map.json)
    merge_src = (FP8_DIR / "merge_fp8_dynamic.py").read_text(encoding="utf-8")
    merge_src = merge_src.replace(
        "    if torch.cuda.is_available():\n        raise RuntimeError('Select a CPU Colab runtime for this preparation notebook')\n",
        "",
    ).replace(
        r"\.lora_([AB])\.weight",
        r"\.lora_([AB])(?:\.default)?\.weight",
    )
    ast.parse(merge_src, filename="merge_fp8_dynamic.py")

    g4_support_files = {
        "g4_stage2_pipeline.py": G4_TRAIN_AND_MERGE_PY,
        "v7_sft_encoding.py": (V7_DIR / "v7_sft_encoding.py").read_text(encoding="utf-8"),
        "memory_optimization.py": (GR_DIR / "memory_optimization.py").read_text(encoding="utf-8"),
        "merge_fp8_dynamic.py": merge_src,
        "export_map.json": (FP8_DIR / "export_map.json").read_text(encoding="utf-8"),
    }
    g4_bundle_b64 = base64.b64encode(
        gzip.compress(json.dumps(g4_support_files, separators=(",", ":")).encode("utf-8"), mtime=0)
    ).decode("ascii")

    cpu_cell = f"""#@title Run Stage-2 720-Task Ground-Truth Audit + DeepSeek-V4.1-Flash Blind-Spot Distillation + SFT Dataset Builder (Colab CPU)
import base64, gzip, json, subprocess, sys, zipfile
from pathlib import Path
from google.colab import drive

drive.mount('/content/drive')

WORK = Path('/content/arc_stage2_cpu_work')
WORK.mkdir(parents=True, exist_ok=True)

DRIVE_ROOT = Path('/content/drive/MyDrive/ARC-experiments')
BASE_ZIP = DRIVE_ROOT / 'ARC_PI_MEDIUM_KV_G4_SOURCE_20261020.zip'
if not (WORK / 'src/arc_feedback_harness').exists():
    with zipfile.ZipFile(BASE_ZIP) as zf:
        for member in zf.namelist():
            if member.startswith('src/'):
                zf.extract(member, WORK)

(WORK / 'payload_720.json').write_bytes(gzip.decompress(base64.b64decode({payload_b64!r})))
(WORK / 'cpu_stage2_pipeline.py').write_bytes(gzip.decompress(base64.b64decode({cpu_py_b64!r})))

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--disable-pip-version-check', 'tokenizers', 'jinja2'], check=True)
env = dict(__import__('os').environ, PYTHONPATH=str(WORK))
proc = subprocess.run([sys.executable, '-u', str(WORK / 'cpu_stage2_pipeline.py')], env=env)
if proc.returncode != 0:
    raise RuntimeError(f'cpu_stage2_pipeline exited with {{proc.returncode}}')
"""
    ast.parse(cpu_cell, filename="cpu_cell.py")

    g4_cell = f"""#@title Run Stage-2 Qwen3.8-27B Multi-Turn QLoRA Fine-Tuning + FP8_DYNAMIC Merge (One Colab G4 GPU)
import base64, gzip, json, subprocess, sys
from pathlib import Path
from google.colab import drive, runtime

drive.mount('/content/drive')

WORK = Path('/content/arc_stage2_g4_work')
WORK.mkdir(parents=True, exist_ok=True)
bundle = json.loads(gzip.decompress(base64.b64decode({g4_bundle_b64!r})).decode('utf-8'))
for name, content in bundle.items():
    (WORK / name).write_text(content, encoding='utf-8')

try:
    proc = subprocess.run([sys.executable, '-u', str(WORK / 'g4_stage2_pipeline.py')])
    if proc.returncode != 0:
        raise RuntimeError(f'g4_stage2_pipeline exited with {{proc.returncode}}')
finally:
    if Path('/content/drive/MyDrive/ARC-experiments/arc_dual_operator_stage2_g4_20261023/STAGE2_COMPLETE.json').exists():
        print('STAGE2_COMPLETE_VERIFIED_RELEASING_G4', flush=True)
        runtime.unassign()
"""
    ast.parse(g4_cell, filename="g4_cell.py")

    delivery = ROOT / "delivery"
    delivery.mkdir(parents=True, exist_ok=True)

    cpu_nb_name = "ARC_STAGE2_DEEPSEEK_V4_1_FLASH_DISTILL_AND_DATASET_CPU_20261022.ipynb"
    g4_nb_name = "ARC_QWEN38_27B_STAGE2_PI_SFT_G4_20261022.ipynb"

    cpu_nb = make_notebook(
        cpu_nb_name,
        (
            "# Stage-2 Qwen3.8-27B 720-Task Ground-Truth Audit + DeepSeek-V4.1-Flash Blind-Spot Distillation (Colab CPU)\n"
            "1. Audits all 720 Qwen3.8-27B Pi-medium trajectories (`600` training + `72` eval72 + `48` public48) against ground-truth query grids.\n"
            "2. Runs `DeepSeek-V4.1-Flash` (`model='deepseek-flash'`, `thinking={'type': 'enabled'}`, `reasoning_effort='high'`, 32 workers) on all blind-spot unsolved tasks.\n"
            "3. Renders all verified Qwen3.8-27B and DeepSeek-V4.1-Flash multi-turn trajectories through Qwen3.8-27B's official `chat_template.jinja` (`<= 49,152` tokens) and writes `stage2_rows.jsonl` + `DATASET_READY.json` to Google Drive."
        ),
        cpu_cell,
        use_g4_gpu=False,
    )
    g4_nb = make_notebook(
        g4_nb_name,
        (
            "# Stage-2 Qwen3.8-27B Multi-Turn QLoRA Fine-Tuning + FP8_DYNAMIC Merge (One Colab G4 GPU)\n"
            "1. Installs pinned QLoRA + GDN CUDA kernels and stages the 54 GB BF16 `Qwen/Qwen3.8-27B` base checkpoint + Stage-1 `GR48K` adapter to local NVMe while the CPU notebook finishes DeepSeek distillation.\n"
            "2. Trains Stage-2 QLoRA (`max_length=49152`, `ActivationOffloader`, `ChunkedSupervisedLoss`, `lr=5e-5`, `accumulation=8`) on `stage2_rows.jsonl`.\n"
            "3. Merges `base_model + stage2_adapter -> FP8_DYNAMIC` (`18` shards, `15` MTP tensors preserved) to `/content/drive/MyDrive/ARC-experiments/arc_gr48k_pi_stage2_sft_20261022/model` and releases the G4."
        ),
        g4_cell,
        use_g4_gpu=True,
    )

    cpu_nb_path = delivery / cpu_nb_name
    g4_nb_path = delivery / g4_nb_name
    write_json(cpu_nb_path, cpu_nb)
    write_json(g4_nb_path, g4_nb)

    packet = {
        "at": datetime.now(timezone.utc).isoformat(),
        "experiment": "arc_gr48k_pi_stage2_sft_20261022",
        "total_tasks_audited": 720,
        "cpu_notebook": {
            "name": cpu_nb_name,
            "path": str(cpu_nb_path.relative_to(ROOT).as_posix()),
            "bytes": cpu_nb_path.stat().st_size,
            "sha256": hashlib.sha256(cpu_nb_path.read_bytes()).hexdigest(),
        },
        "g4_notebook": {
            "name": g4_nb_name,
            "path": str(g4_nb_path.relative_to(ROOT).as_posix()),
            "bytes": g4_nb_path.stat().st_size,
            "sha256": hashlib.sha256(g4_nb_path.read_bytes()).hexdigest(),
        },
    }
    write_json(ROOT / "PACKET.json", packet)
    print(json.dumps(packet, indent=2))


if __name__ == "__main__":
    main()
