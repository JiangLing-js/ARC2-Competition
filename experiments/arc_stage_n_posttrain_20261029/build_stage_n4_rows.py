"""Stage-N4 Zero-Leakage Process-Supervision Re-Masking + Case-Split-Clean RFT + Code-DPO Builder.

Fixes two root-cause process-supervision bugs discovered in `build_stage_n_sft_rows.py:377-395`:
1. Unmasks pre-failure `clean_inspection` Python structural probing turns in `code_repair`
   trajectories (`70` clean Turn-1/Turn-2 inspection turns previously masked out because
   `train_action` required `turn.after_failure` for `code_repair`), while keeping all
   `failed_attempt`, `unverified`, and post-all-pass exploration turns strictly masked (`False`).
2. Bounds supervised `<think>` blocks to `32 <= reasoning_chars <= MAX_SUPERVISED_REASONING_CHARS`
   (`20,000` chars), masking `45` bloated `20k–33k`-char prose-loop `<think>` spans while keeping
   their clean tool-call actions supervised (`train_action = True`).
3. Preserves exact token-ID alignment (`whole == segment_ids`, `len(prompt_ids) == last_prompt_tokens`),
   `trace_candidate_coverage` case-split rejection, and shared-prefix code-only DPO pairs, with
   fail-closed zero overlap against `eval120 | hold_100 | re_arc_holdout_47 | stage2_heldout | dev_hard_24`.
"""
from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path
import sys
import time
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
SRC_ROOT = REPO_ROOT / "src"
for _p in (THIS_DIR, SRC_ROOT, REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from build_stage_n3_dpo_pairs import build_stage_n3_clean_and_dpo  # type: ignore
from build_stage_n_sft_rows import (  # type: ignore
    ASSISTANT_PREFIX,
    MIN_REASONING_CHARS,
    split_prompt_segments_by_operator,
    token_ids,
)
from collect_stage_n_data import load_renderer  # type: ignore
from stage_n_data_contract import base_family_id, load_dev_hard_24_forbidden, read_json, write_json  # type: ignore

MAX_SUPERVISED_REASONING_CHARS = 20_000


def compute_stage_n4_turn_flags(
    tc: dict[str, Any],
    operator_type: str,
    max_reasoning_chars: int = MAX_SUPERVISED_REASONING_CHARS,
) -> tuple[bool, bool]:
    """Return `(train_thinking, train_action)` under Stage-N4 process supervision."""
    kind = str(tc.get("kind") or "")
    after_failure = bool(tc.get("after_failure", False))
    after_all_pass = bool(tc.get("after_all_pass_verify", False))
    reasoning_chars = int(tc.get("reasoning_chars", 0))

    if kind in {"failed_attempt", "unverified"}:
        return False, False
    if after_all_pass and kind in {"clean_inspection", "diagnostic_experiment"}:
        return False, False

    if operator_type == "direct_generate":
        act = (not after_failure) and kind in {"clean_inspection", "direct_solve"}
    elif operator_type == "code_repair":
        # Stage-N4 fix: supervise clean pre-failure Python structural probes (`clean_inspection`)
        # in addition to post-failure clean inspections and final `verified_resolution`.
        act = kind in {"clean_inspection", "verified_resolution"}
    elif operator_type == "near_miss_repair":
        act = kind in {"clean_inspection", "diagnostic_experiment", "verified_near_miss_repair"}
    elif operator_type == "adjudicate":
        act = kind in {
            "clean_inspection",
            "diagnostic_experiment",
            "verified_adjudication",
            "insufficient_evidence_stop",
        }
    else:
        raise ValueError(f"unknown_operator_type:{operator_type}")

    think = bool(act and (MIN_REASONING_CHARS <= reasoning_chars <= max_reasoning_chars))
    return think, bool(act)


def count_prompt_assistant_turns(prompt: str) -> int:
    """Count completed assistant turns in `prompt` (excluding trailing open `<|im_start|>assistant\n`)."""
    count = 0
    pos = 0
    while True:
        idx = prompt.find(ASSISTANT_PREFIX, pos)
        if idx < 0:
            break
        body = idx + len(ASSISTANT_PREFIX)
        if body == len(prompt):
            break
        count += 1
        pos = body
    return count


def remask_stage_n4_row(
    row: dict[str, Any],
    tokenizer: Any,
    max_reasoning_chars: int = MAX_SUPERVISED_REASONING_CHARS,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Losslessly re-mask a V7-trace SFT row's prompt segments with Stage-N4 process supervision."""
    tcs = row.get("turn_classification")
    if not isinstance(tcs, list) or not tcs:
        return row, {"unmasked_actions": 0, "masked_bloated_thinks": 0, "remasked": 0}

    prompt = "".join(s["text"] for s in row["segments"])
    n_prompt_turns = count_prompt_assistant_turns(prompt)
    if n_prompt_turns == 0 or len(tcs) < n_prompt_turns + 1:
        return row, {"unmasked_actions": 0, "masked_bloated_thinks": 0, "remasked": 0}

    op = str(row["operator_type"])
    prompt_tcs = tcs[:n_prompt_turns]
    new_flags: list[tuple[bool, bool]] = []
    unmasked_actions = 0
    masked_bloated_thinks = 0
    updated_tcs: list[dict[str, Any]] = []

    for tc in prompt_tcs:
        old_act = bool(tc.get("train_action", tc.get("train", False)))
        old_think = bool(tc.get("train_thinking", tc.get("train", False)))
        new_think, new_act = compute_stage_n4_turn_flags(tc, op, max_reasoning_chars=max_reasoning_chars)
        if new_act and not old_act:
            unmasked_actions += 1
        if old_think and not new_think:
            masked_bloated_thinks += 1
        new_flags.append((new_think, new_act))
        tc_copy = dict(tc)
        tc_copy["train_thinking"] = new_think
        tc_copy["train_action"] = new_act
        tc_copy["train"] = bool(new_act)
        updated_tcs.append(tc_copy)

    # Preserve terminal completion classification metadata (while updating bloated-think flag for audit)
    for tail_tc in tcs[n_prompt_turns:]:
        tc_copy = dict(tail_tc)
        if int(tc_copy.get("reasoning_chars", 0)) > max_reasoning_chars:
            tc_copy["bloated_final_thinking"] = True
        updated_tcs.append(tc_copy)

    new_segments = split_prompt_segments_by_operator(prompt, new_flags, tokenizer=tokenizer)
    if row.get("last_prompt_tokens") is not None:
        seg_tok_count = sum(len(token_ids(tokenizer, s["text"])) for s in new_segments)
        if seg_tok_count != int(row["last_prompt_tokens"]):
            raise ValueError(
                f"Prompt token count mismatch after Stage-N4 remask on {row.get('task_id')}: "
                f"{seg_tok_count} != {row['last_prompt_tokens']}"
            )

    out = copy.deepcopy(row)
    out["segments"] = new_segments
    out["turn_classification"] = updated_tcs
    return out, {
        "unmasked_actions": unmasked_actions,
        "masked_bloated_thinks": masked_bloated_thinks,
        "remasked": 1,
    }


def build_stage_n4_clean_and_dpo(out_dir: Path | None = None) -> dict[str, Any]:
    t0 = time.time()
    dest_dir = out_dir or (THIS_DIR / "stage_n4_build")
    dest_dir.mkdir(parents=True, exist_ok=True)

    n3_dir = THIS_DIR / "stage_n3_build"
    n3_rows_path = n3_dir / "stage_n3_clean_rows.jsonl"
    if not n3_rows_path.is_file():
        build_stage_n3_clean_and_dpo(n3_dir)

    catalog = read_json(THIS_DIR / "STAGE_N_CATALOG.json")
    distill600_set = set(catalog["pools"]["distill600"])
    stage2_heldout = {t for t in distill600_set if int(t, 16) % 20 == 0}
    dev_hard_24 = set(load_dev_hard_24_forbidden(catalog))
    forbidden = (
        set(catalog["forbidden"]["eval120"])
        | set(catalog["holdout"]["hold_100"])
        | set(catalog["forbidden"]["re_arc_holdout_47"])
        | stage2_heldout
        | dev_hard_24
    )

    _tpl, tokenizer = load_renderer(THIS_DIR / "_tokenizer_cache")
    raw_n3_rows = [
        json.loads(ln)
        for ln in n3_rows_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]

    n4_rows: list[dict[str, Any]] = []
    tot_unmasked = 0
    tot_masked_bloated = 0
    tot_remasked = 0

    for r in raw_n3_rows:
        fam = base_family_id(str(r["task_id"]))
        if fam in forbidden:
            raise ValueError(f"CRITICAL: forbidden family {fam} detected in Stage-N4 input")
        remasked_row, delta = remask_stage_n4_row(r, tokenizer)
        tot_unmasked += delta["unmasked_actions"]
        tot_masked_bloated += delta["masked_bloated_thinks"]
        tot_remasked += delta["remasked"]
        n4_rows.append(remasked_row)

    rows_out = dest_dir / "stage_n4_clean_rows.jsonl"
    with rows_out.open("w", encoding="utf-8") as handle:
        for r in n4_rows:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")

    pure_fams = {base_family_id(str(r["task_id"])) for r in n4_rows if r.get("is_pure_arc2")}
    summary = {
        "schema": "stage-n4-remasked-rft-dpo-v1",
        "built_at": time.time(),
        "wall_seconds": round(time.time() - t0, 2),
        "clean_output_rows": len(n4_rows),
        "clean_pure_arc2_families": len(pure_fams),
        "rows_remasked": tot_remasked,
        "unmasked_code_repair_clean_inspections": tot_unmasked,
        "masked_bloated_intermediate_thinks": tot_masked_bloated,
        "max_supervised_reasoning_chars": MAX_SUPERVISED_REASONING_CHARS,
        "operators": dict(Counter(r["operator_type"] for r in n4_rows)),
        "dpo_pairs_attached": sum(1 for r in n4_rows if isinstance(r.get("dpo_rejected"), dict)),
        "zero_forbidden_overlap_verified": all(base_family_id(str(r["task_id"])) not in forbidden for r in n4_rows),
    }
    write_json(dest_dir / "STAGE_N4_BUILD_SUMMARY.json", summary)
    print("STAGE_N4_BUILD_COMPLETE", json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    build_stage_n4_clean_and_dpo()
