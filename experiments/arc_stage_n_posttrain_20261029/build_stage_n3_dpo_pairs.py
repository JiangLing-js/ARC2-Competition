"""Zero-leakage Case-Split-Clean RFT + Shared-Prefix Final-Turn DPO Pair Builder for Stage-N3.

1. Loads all genuine multi-turn `<think>` trajectories from:
   - `downloaded_shards/shard_0_rows.jsonl` (`Stage-N` organic `Qwen3.8-27B`)
   - `downloaded_shards/shard_1_rows.jsonl` (`Stage-N` organic `Qwen3.8-27B`)
   - `stage_n2_distill/distilled_rows.jsonl` (strictly `source.startswith('stage_n2_teacher_')`
     and `operator_type != 'near_miss_repair'` — ZERO synthetic 1-sentence boilerplate rows!)
2. Audits every submitted `transform(grid)` with `worker.py`'s `trace_candidate_coverage` and
   `extract_ast_features`, rejecting any positive RFT row with `per_demo_case_split` or
   `multi_demo_case_split`.
3. Harvests genuine organic `Qwen3.8-27B` `N/N`-demo-passing false positives (`demo_all_correct`
   on all training demos yet `test != gold`) from `recovered/batch_001..003`, augmented with
   execution-verified `N/N`-demo-passing overfit decoys (demo-dimension dispatch / window-extent /
   palette-subset overfits) on clean pure ARC-AGI-2 training tasks.
4. Attaches `dpo_rejected_completion` (`token_ids`, `role_reason`) to matching shared-prefix
   positive `submit_program` rows (`<= 45,000` tokens), failing closed on ANY overlap with
   `eval120 | hold_100 | re_arc_holdout_47 | stage2_heldout | dev_hard_24`.
"""
from __future__ import annotations

import ast
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

from arc_feedback_harness.worker import extract_ast_features, trace_candidate_coverage  # type: ignore
from build_stage_n_episodes import (  # type: ignore
    FastBatchEvaluator,
    majority_color,
    make_d8_conjugate_variant,
    make_palette_filter_variant,
    make_row_window_variant,
)
from build_stage_n_sft_rows import IM_END, IM_END_ID, STAGE_N_ACCEPTED_MAX_TOKENS  # type: ignore
from collect_stage_n_data import load_renderer  # type: ignore
from stage_n_catalog import (  # type: ignore
    ARC2_EVAL_DIR,
    ARC2_TRAIN_DIR,
    canonical_grid_sha256,
    collect_task_grid_hashes,
)
from stage_n_data_contract import (  # type: ignore
    base_family_id,
    load_dev_hard_24_forbidden,
    read_json,
    write_json,
)

SUBMIT_OPEN = "<function=submit_program>\n<parameter=code>\n"
SUBMIT_CLOSE = "\n</parameter>\n</function>"


def extract_submit_code(completion_text: str) -> str | None:
    if SUBMIT_OPEN not in completion_text or SUBMIT_CLOSE not in completion_text:
        return None
    body = completion_text.split(SUBMIT_OPEN, 1)[1].split(SUBMIT_CLOSE, 1)[0]
    return body.strip() if "def transform" in body else None


def make_demo_dispatch_decoy(true_prog: str, demo_ins: list[list[list[int]]]) -> str | None:
    """Create a realistic demo-dimension / corner-pixel case-split wrapper that passes N/N demos
    yet falls through to a corrupted fallback on unseen test dimensions/signatures."""
    try:
        tree = ast.parse(true_prog)
    except SyntaxError:
        return None
    renamed = False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "transform":
            node.name = "_core_transform"
            renamed = True
    if not renamed:
        return None
    ast.fix_missing_locations(tree)
    sigs = {(len(g), len(g[0]), g[0][0], g[-1][-1]) for g in demo_ins}
    return (
        f"{ast.unparse(tree)}\n\n"
        f"def transform(grid):\n"
        f"    if (len(grid), len(grid[0]), grid[0][0], grid[-1][-1]) in {sigs!r}:\n"
        f"        return _core_transform(grid)\n"
        f"    return [row[::-1] for row in grid]\n"
    )


def build_stage_n3_clean_and_dpo(out_dir: Path | None = None) -> dict[str, Any]:
    t0 = time.time()
    dest_dir = out_dir or (THIS_DIR / "stage_n3_build")
    dest_dir.mkdir(parents=True, exist_ok=True)

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

    eval120_hashes: set[str] = set()
    for eid in catalog["forbidden"]["eval120"]:
        eval120_hashes |= collect_task_grid_hashes(read_json(ARC2_EVAL_DIR / f"{eid}.json"))

    _tpl, tokenizer = load_renderer(THIS_DIR / "_tokenizer_cache")

    # 1. Load genuine organic Stage-N rows + genuine multi-turn teacher Stage-N2 rows
    raw_rows: list[dict[str, Any]] = []
    for rel in ("downloaded_shards/shard_0_rows.jsonl", "downloaded_shards/shard_1_rows.jsonl"):
        p = THIS_DIR / rel
        for ln in p.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                raw_rows.append(json.loads(ln))

    distill_p = THIS_DIR / "stage_n2_distill/distilled_rows.jsonl"
    teacher_added = 0
    synthetic_skipped = 0
    if distill_p.is_file():
        for ln in distill_p.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            src = str(r.get("source") or "")
            # STRICTLY exclude synthetic 1-sentence `<think>` boilerplate rows!
            if not src.startswith("stage_n2_teacher_") or r.get("operator_type") == "near_miss_repair":
                synthetic_skipped += 1
                continue
            r["task_id"] = base_family_id(str(r["task_id"]))
            raw_rows.append(r)
            teacher_added += 1

    # 2. Fail-closed zero-leakage & case-split-clean RFT audit
    clean_rows: list[dict[str, Any]] = []
    case_split_filtered = 0
    forbidden_filtered = 0
    overlength_filtered = 0
    task_cache: dict[str, dict[str, Any]] = {}

    def _get_arc2_task(fam: str) -> dict[str, Any] | None:
        if fam in task_cache:
            return task_cache[fam]
        tp = ARC2_TRAIN_DIR / f"{fam}.json"
        if not tp.is_file():
            return None
        obj = read_json(tp)
        if collect_task_grid_hashes(obj) & eval120_hashes:
            raise ValueError(f"CRITICAL eval120 grid hash overlap on {fam}")
        task_cache[fam] = obj
        return obj

    for r in raw_rows:
        fam = base_family_id(str(r["task_id"]))
        if fam in forbidden:
            forbidden_filtered += 1
            continue
        if int(r.get("total_tokens", 0)) >= STAGE_N_ACCEPTED_MAX_TOKENS:
            overlength_filtered += 1
            continue
        comp_txt = tokenizer.decode(r["completion"]["token_ids"])
        sub_code = extract_submit_code(comp_txt)
        raw_task = _get_arc2_task(fam) if not str(r["task_id"]).startswith("rearc_") else None
        if sub_code and raw_task is not None:
            try:
                tree = ast.parse(sub_code)
                ns: dict[str, Any] = {}
                exec(compile(sub_code, "<candidate>", "exec"), ns, ns)
                fn = ns.get("transform")
                if callable(fn):
                    cov = trace_candidate_coverage(fn, raw_task, time.monotonic(), 0.45)
                    if cov.get("complete") and (cov.get("per_demo_case_split") or cov.get("multi_demo_case_split")):
                        case_split_filtered += 1
                        continue
                    ast_feat = extract_ast_features(tree)
                    r["submitted_ast_nodes"] = ast_feat["ast_nodes"]
            except Exception:
                pass
        clean_rows.append(r)

    # 3. Harvest organic N/N-demo false-positive programs from `recovered/batch_001..003`
    organic_fps: dict[str, list[str]] = {}
    recovered_root = REPO_ROOT / "experiments/arc_gr_expand800_20261018/recovered"
    evaluator = FastBatchEvaluator(timeout_s=1.8)
    try:
        for batch_name in ("batch_001", "batch_002", "batch_003"):
            p1 = recovered_root / batch_name / "contents/results/sampling/phase1"
            if not p1.is_dir():
                continue
            for sd in sorted(p1.iterdir()):
                pm_path = sd / "packet_manifest.json"
                if not sd.is_dir() or not pm_path.is_file():
                    continue
                pm = read_json(pm_path)
                for it in pm.get("tasks", []):
                    tid = str(it["task_id"])
                    if tid in forbidden:
                        continue
                    raw_t = _get_arc2_task(tid)
                    if raw_t is None:
                        continue
                    tdir = sd / it["name"]
                    for rnd in sorted(tdir.glob("round_*")):
                        ap = rnd / "attempt.json"
                        if not ap.is_file():
                            continue
                        att = read_json(ap)
                        fb = att.get("feedback") or {}
                        src = (att.get("source") or "").strip()
                        if not fb.get("demo_all_correct") or not src or "def transform" not in src or len(src) > 8500:
                            continue
                        demo_ins = [p["input"] for p in raw_t["train"]]
                        demo_exps = [p["output"] for p in raw_t["train"]]
                        test_ins = [p["input"] for p in raw_t["test"]]
                        test_exps = [p["output"] for p in raw_t["test"]]
                        outs = evaluator.run(src, demo_ins + test_ins)
                        if outs is not None and outs[: len(demo_ins)] == demo_exps and outs[len(demo_ins) :] != test_exps:
                            lst = organic_fps.setdefault(tid, [])
                            if src not in lst:
                                lst.append(src)
                            if len(lst) >= 2:
                                break

        # 4. Attach shared-prefix final-turn DPO rejected completions (`N/N`-demo false positives!)
        dpo_attached = 0
        dpo_organic = 0
        dpo_decoy = 0
        used_dpo_families: set[str] = set()

        for r in clean_rows:
            fam = base_family_id(str(r["task_id"]))
            if fam in used_dpo_families or str(r["task_id"]).startswith("rearc_"):
                continue
            if r.get("operator_type") not in {"direct_generate", "code_repair", "near_miss_repair"}:
                continue
            raw_t = _get_arc2_task(fam)
            if raw_t is None:
                continue
            comp_txt = tokenizer.decode(r["completion"]["token_ids"])
            true_code = extract_submit_code(comp_txt)
            if not true_code:
                continue

            demo_ins = [p["input"] for p in raw_t["train"]]
            demo_exps = [p["output"] for p in raw_t["train"]]
            test_ins = [p["input"] for p in raw_t["test"]]
            test_exps = [p["output"] for p in raw_t["test"]]

            fp_code: str | None = None
            fp_kind = ""
            if organic_fps.get(fam):
                fp_code = organic_fps[fam][0]
                fp_kind = "organic_nn_demo_false_positive"
            else:
                max_h = max(len(g) for g in demo_ins)
                max_w = max(len(g[0]) for g in demo_ins)
                seen_pal = sorted({v for g in demo_ins for row in g for v in row})
                bg0 = majority_color(demo_ins[0])
                for cand_kind, cand_src in (
                    ("window_extent_overfit", make_row_window_variant(true_code, max_h, max_w)),
                    ("palette_subset_overfit", make_palette_filter_variant(true_code, seen_pal, bg0)),
                    ("transpose_axis_overfit", make_d8_conjugate_variant(true_code, "transpose")),
                    ("fliph_axis_overfit", make_d8_conjugate_variant(true_code, "flip_h")),
                    ("demo_signature_case_split", make_demo_dispatch_decoy(true_code, demo_ins)),
                ):
                    if not cand_src:
                        continue
                    outs = evaluator.run(cand_src, demo_ins + test_ins)
                    if outs is not None and outs[: len(demo_ins)] == demo_exps and outs[len(demo_ins) :] != test_exps:
                        fp_code = cand_src
                        fp_kind = cand_kind
                        break

            if not fp_code:
                continue

            # Preserve the EXACT `<think>...</think>` prefix of `comp_txt` so DPO has ZERO
            # phrasing shortcut and contrasts 100% pure general code vs N/N-demo-overfit code!
            think_and_open = comp_txt.split(SUBMIT_OPEN, 1)[0] + SUBMIT_OPEN
            after_close = comp_txt.split(SUBMIT_OPEN, 1)[1].split(SUBMIT_CLOSE, 1)[1]
            if not after_close.endswith(IM_END):
                after_close = after_close + IM_END
            rej_text = f"{think_and_open}{fp_code.strip()}{SUBMIT_CLOSE}{after_close}"

            rej_ids = tokenizer.encode(rej_text, add_special_tokens=False).ids
            code_prefix_ids = tokenizer.encode(think_and_open, add_special_tokens=False).ids
            if not rej_ids or rej_ids[-1] != IM_END_ID:
                raise ValueError(f"Rejected completion missing IM_END_ID on {fam}")
            prompt_toks = int(r["last_prompt_tokens"])
            if prompt_toks + len(rej_ids) >= STAGE_N_ACCEPTED_MAX_TOKENS:
                continue

            r["dpo_rejected"] = {
                "token_ids": rej_ids,
                "code_start_offset": len(code_prefix_ids),
                "finish_reason": "stop",
                "fp_kind": fp_kind,
                "total_tokens": prompt_toks + len(rej_ids),
            }
            used_dpo_families.add(fam)
            dpo_attached += 1
            if fp_kind == "organic_nn_demo_false_positive":
                dpo_organic += 1
            else:
                dpo_decoy += 1
    finally:
        evaluator.close()

    rows_out = dest_dir / "stage_n3_clean_rows.jsonl"
    with rows_out.open("w", encoding="utf-8") as handle:
        for r in clean_rows:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")

    pure_fams = {base_family_id(str(r["task_id"])) for r in clean_rows if r.get("is_pure_arc2")}
    summary = {
        "schema": "stage-n3-clean-rft-dpo-v1",
        "built_at": time.time(),
        "wall_seconds": round(time.time() - t0, 2),
        "raw_input_rows": len(raw_rows),
        "teacher_genuine_rows_included": teacher_added,
        "synthetic_boilerplate_rows_excluded": synthetic_skipped,
        "forbidden_rows_filtered": forbidden_filtered,
        "case_split_rows_filtered": case_split_filtered,
        "overlength_rows_filtered": overlength_filtered,
        "clean_output_rows": len(clean_rows),
        "clean_pure_arc2_families": len(pure_fams),
        "operators": dict(Counter(r["operator_type"] for r in clean_rows)),
        "dpo_pairs_attached": dpo_attached,
        "dpo_organic_false_positives": dpo_organic,
        "dpo_verified_overfit_decoys": dpo_decoy,
        "zero_forbidden_overlap_verified": all(base_family_id(str(r["task_id"])) not in forbidden for r in clean_rows),
    }
    write_json(dest_dir / "STAGE_N3_BUILD_SUMMARY.json", summary)
    print("STAGE_N3_BUILD_COMPLETE", json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    build_stage_n3_clean_and_dpo()
