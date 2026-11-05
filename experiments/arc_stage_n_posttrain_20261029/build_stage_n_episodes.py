"""Deterministic zero-leakage 4-operator episode builder for Stage-N collection.

Constructs and execution-verifies three solver-blinded episode pools on CPU:
1. `pool_a_pure_arc2` (`160` pure ARC-AGI-2 tasks: `100` `s1_100` + `60` `distill600_pure_arc2` blindspots)
2. `pool_b_near_miss` (`[PRIOR_WAVE_NEAR_MISS_CANDIDATE]`: organic seeds + pure ARC-2 AST near-misses + RE-ARC `50+50`)
3. `pool_c_adjudicate` (`[DISAGREEMENT_AND_OVERFIT_ADJUDICATION]`: shuffled A/B/C true-vs-decoy, `>=25%` neither-generalizes
   unified-synthesis episodes, and genuinely underdetermined 2-demo abstention episodes).

Hard invariants:
- ZERO `eval120` IDs or canonical grid SHA-256 overlaps anywhere in public tasks or verifier gold.
- ZERO `hold_100` IDs or `re_arc_holdout_47` descendants.
- ZERO Stage-2 `heldout` (`int(tid, 16) % 20 == 0` in `distill600`) families or descendants.
- Solver task files (`solver_tasks/*.json`) contain ONLY public `train` pairs and `test` inputs;
  `verifier_sealed_gold.json` is stored separately for offline post-panel verification.
"""
from __future__ import annotations

import ast
from collections import Counter
import copy
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import queue
import random
import sys
import time
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from arc_feedback_harness.feedback import grid_diff, repair_diagnosis, valid_grid  # type: ignore
from stage_n_catalog import (  # type: ignore
    ARC2_EVAL_DIR,
    ARC2_TRAIN_DIR,
    RE_ARC_TASKS_DIR,
    canonical_grid_sha256,
    collect_task_grid_hashes,
)
from stage_n_data_contract import base_family_id, read_json, write_json  # type: ignore


def _worker_loop(req_q: Any, resp_q: Any) -> None:
    while True:
        item = req_q.get()
        if item is None:
            break
        code, grids = item
        try:
            ns: dict[str, Any] = {}
            exec(compile(code, "<candidate>", "exec"), ns, ns)
            fn = ns.get("transform")
            if not callable(fn):
                resp_q.put({"ok": False, "error": "missing_transform"})
                continue
            outs = []
            ok = True
            for g in grids:
                res = fn(copy.deepcopy(g))
                if not valid_grid(res):
                    ok = False
                    break
                outs.append(res)
            if not ok:
                resp_q.put({"ok": False, "error": "invalid_grid"})
            else:
                resp_q.put({"ok": True, "outputs": outs})
        except BaseException as exc:
            resp_q.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


class FastBatchEvaluator:
    """Persistent subprocess evaluator with per-call timeout & auto-respawn for offline trusted training tasks."""

    def __init__(self, timeout_s: float = 1.8) -> None:
        self.timeout_s = timeout_s
        self._ctx = mp.get_context("spawn")
        self._req_q: Any = None
        self._resp_q: Any = None
        self._proc: Any = None
        self._start()

    def _start(self) -> None:
        self.close()
        self._req_q = self._ctx.Queue()
        self._resp_q = self._ctx.Queue()
        self._proc = self._ctx.Process(target=_worker_loop, args=(self._req_q, self._resp_q), daemon=True)
        self._proc.start()

    def run(self, code: str, grids: list[list[list[int]]]) -> list[list[list[int]]] | None:
        if self._proc is None or not self._proc.is_alive():
            self._start()
        self._req_q.put((code, grids))
        try:
            res = self._resp_q.get(timeout=self.timeout_s)
        except queue.Empty:
            self._start()
            return None
        if not isinstance(res, dict) or not res.get("ok"):
            return None
        return res["outputs"]

    def close(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.is_alive():
                    try:
                        self._req_q.put_nowait(None)
                    except Exception:
                        pass
                    self._proc.join(timeout=0.3)
                if self._proc.is_alive():
                    self._proc.terminate()
                    self._proc.join(timeout=0.5)
            except Exception:
                pass
        self._proc = None


def generate_single_bug_mutations(code: str, max_variants: int = 18) -> list[str]:
    """Generate realistic single-condition / off-by-one AST perturbations of a working `transform(grid)`."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    variants: list[str] = []
    seen: set[str] = {code.strip()}

    def _emit(mut_tree: ast.AST) -> None:
        if len(variants) >= max_variants:
            return
        ast.fix_missing_locations(mut_tree)
        txt = ast.unparse(mut_tree).strip()
        if txt not in seen and "def transform" in txt:
            seen.add(txt)
            variants.append(txt)

    # 1. Single comparison operator boundary swap (< <-> <=, > <-> >=)
    cmp_count = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Compare) and len(n.ops) == 1)
    swap_cmp = {
        ast.Lt: ast.LtE,
        ast.LtE: ast.Lt,
        ast.Gt: ast.GtE,
        ast.GtE: ast.Gt,
    }
    for target_idx in range(min(cmp_count, 12)):
        t_copy = copy.deepcopy(tree)
        cur = [n for n in ast.walk(t_copy) if isinstance(n, ast.Compare) and len(n.ops) == 1][target_idx]
        op_cls = type(cur.ops[0])
        if op_cls in swap_cmp:
            cur.ops[0] = swap_cmp[op_cls]()
            _emit(t_copy)
            if len(variants) >= max_variants:
                return variants

    # 2. Single integer literal +/- 1 perturbation on small constants (0..8)
    const_count = sum(
        1 for n in ast.walk(tree) if isinstance(n, ast.Constant) and type(n.value) is int and 0 <= n.value <= 8
    )
    for target_idx in range(min(const_count, 10)):
        for delta in (1, -1):
            t_copy = copy.deepcopy(tree)
            c_nodes = [
                n
                for n in ast.walk(t_copy)
                if isinstance(n, ast.Constant) and type(n.value) is int and 0 <= n.value <= 8
            ]
            val = c_nodes[target_idx].value + delta
            if 0 <= val <= 9:
                c_nodes[target_idx].value = val
                _emit(t_copy)
                if len(variants) >= max_variants:
                    return variants

    # 3. `min` <-> `max` call swap
    call_count = sum(
        1
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in {"min", "max"}
    )
    for target_idx in range(min(call_count, 4)):
        t_copy = copy.deepcopy(tree)
        c_nodes = [
            n
            for n in ast.walk(t_copy)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in {"min", "max"}
        ]
        fn_node = c_nodes[target_idx].func
        assert isinstance(fn_node, ast.Name)
        fn_node.id = "max" if fn_node.id == "min" else "min"
        _emit(t_copy)
        if len(variants) >= max_variants:
            return variants

    return variants


def make_row_window_variant(code: str, max_demo_h: int, max_demo_w: int) -> str | None:
    """Refactor `transform(grid)` into an interior-window variant anchored to `max_demo_h`/`max_demo_w`."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    renamed = copy.deepcopy(tree)
    found = False
    for node in renamed.body:
        if isinstance(node, ast.FunctionDef) and node.name == "transform":
            node.name = "_core_rule"
            found = True
    if not found:
        return None
    ast.fix_missing_locations(renamed)
    core_txt = ast.unparse(renamed).strip()
    return (
        f"{core_txt}\n\n"
        f"def transform(grid):\n"
        f"    h = len(grid)\n"
        f"    w = len(grid[0]) if h else 0\n"
        f"    win_h = min(h, {max_demo_h})\n"
        f"    win_w = min(w, {max_demo_w})\n"
        f"    sub = [row[:win_w] for row in grid[:win_h]]\n"
        f"    core_out = _core_rule(sub)\n"
        f"    if len(core_out) == win_h and (win_h == 0 or len(core_out[0]) == win_w) and (win_h < h or win_w < w):\n"
        f"        out = [row[:] for row in grid]\n"
        f"        for r in range(win_h):\n"
        f"            for c in range(win_w):\n"
        f"                out[r][c] = core_out[r][c]\n"
        f"        return out\n"
        f"    return core_out"
    )


def make_palette_filter_variant(code: str, seen_colors: list[int], fallback_bg: int) -> str | None:
    """Refactor `transform(grid)` into a variant that treats only `seen_colors` as active palette."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    renamed = copy.deepcopy(tree)
    found = False
    for node in renamed.body:
        if isinstance(node, ast.FunctionDef) and node.name == "transform":
            node.name = "_core_rule"
            found = True
    if not found:
        return None
    ast.fix_missing_locations(renamed)
    core_txt = ast.unparse(renamed).strip()
    palette_literal = "{" + ", ".join(str(c) for c in sorted(set(seen_colors))) + "}"
    return (
        f"{core_txt}\n\n"
        f"def transform(grid):\n"
        f"    active_palette = {palette_literal}\n"
        f"    bg = {int(fallback_bg)}\n"
        f"    filtered = [[v if v in active_palette else bg for v in row] for row in grid]\n"
        f"    return _core_rule(filtered)"
    )


def make_d8_conjugate_variant(code: str, mode: str = "transpose") -> str | None:
    """Construct clean D8 conjugate `D^-1(f(D(grid)))` without any hardcoded constants."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    renamed = copy.deepcopy(tree)
    found = False
    for node in renamed.body:
        if isinstance(node, ast.FunctionDef) and node.name == "transform":
            node.name = "_core_rule"
            found = True
    if not found:
        return None
    ast.fix_missing_locations(renamed)
    core_txt = ast.unparse(renamed).strip()
    if mode == "transpose":
        return (
            f"{core_txt}\n\n"
            f"def _tr(g):\n"
            f"    return [list(col) for col in zip(*g)] if g and g[0] else []\n\n"
            f"def transform(grid):\n"
            f"    return _tr(_core_rule(_tr(grid)))"
        )
    if mode == "flip_h":
        return (
            f"{core_txt}\n\n"
            f"def _fh(g):\n"
            f"    return [list(reversed(row)) for row in g]\n\n"
            f"def transform(grid):\n"
            f"    return _fh(_core_rule(_fh(grid)))"
        )
    return None


def compute_near_miss_handover(
    train_pairs: list[dict[str, Any]],
    code: str,
    demo_outputs: list[list[list[int]]],
) -> dict[str, Any] | None:
    demo_results: list[dict[str, Any]] = []
    pass_count = 0
    total_diff = 0
    for idx, (pair, pred) in enumerate(zip(train_pairs, demo_outputs)):
        exp = pair["output"]
        if pred == exp:
            pass_count += 1
            demo_results.append({"demo": idx, "kind": "pass", "ok": True, "grid": pred})
        else:
            same_shape = len(pred) == len(exp) and (len(pred) == 0 or len(pred[0]) == len(exp[0]))
            if same_shape:
                diff = grid_diff(pred, exp)
                diff_cells = int(diff["different_cells"])
                total_diff += diff_cells
                demo_results.append(
                    {
                        "demo": idx,
                        "kind": "semantic",
                        "ok": True,
                        "grid": pred,
                        "different_cells": diff_cells,
                    }
                )
            else:
                total_diff += 50
                demo_results.append(
                    {
                        "demo": idx,
                        "kind": "shape",
                        "ok": True,
                        "grid": pred,
                        "actual_shape": [len(pred), len(pred[0]) if pred else 0],
                        "expected_shape": [len(exp), len(exp[0]) if exp else 0],
                    }
                )
    if not (1 <= pass_count < len(train_pairs)):
        return None
    diag = repair_diagnosis(train_pairs, demo_results)
    return {
        "code": code.strip(),
        "demo_pass_count": pass_count,
        "demo_count": len(train_pairs),
        "semantic_diff_cells": total_diff,
        "repair_diagnosis": diag,
    }


def compute_disagreement_summary(candidates: list[dict[str, Any]], query_count: int) -> list[dict[str, Any]]:
    disagreements: list[dict[str, Any]] = []
    for q in range(query_count):
        shapes = {
            c["label"]: [len(c["predictions"][q]), len(c["predictions"][q][0])]
            for c in candidates
        }
        p0 = candidates[0]["predictions"][q]
        same_shape = all(
            len(c["predictions"][q]) == len(p0) and len(c["predictions"][q][0]) == len(p0[0])
            for c in candidates
        )
        sample_diffs: list[dict[str, Any]] = []
        diff_count = 0
        if same_shape and len(candidates) >= 2:
            h, w = len(p0), len(p0[0])
            for r in range(h):
                for col in range(w):
                    v0 = p0[r][col]
                    if any(c["predictions"][q][r][col] != v0 for c in candidates):
                        diff_count += 1
                        if len(sample_diffs) < 12:
                            sample_diffs.append(
                                {
                                    "rc": [r, col],
                                    "values": {c["label"]: c["predictions"][q][r][col] for c in candidates},
                                }
                            )
        disagreements.append(
            {
                "query": q,
                "shapes": shapes,
                "same_shape": same_shape,
                "different_cells": diff_count if same_shape else None,
                "sample_diffs": sample_diffs,
            }
        )
    return disagreements


def grid_colors(grid: list[list[int]]) -> set[int]:
    return {v for row in grid for v in row}


def majority_color(grid: list[list[int]]) -> int:
    counts = Counter(v for row in grid for v in row)
    return counts.most_common(1)[0][0] if counts else 0


def build_stage_n_episodes(output_dir: Path | None = None, seed: int = 20261029) -> dict[str, Any]:
    t0 = time.time()
    out_root = output_dir or (THIS_DIR / "episodes_build")
    out_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    catalog = read_json(THIS_DIR / "STAGE_N_CATALOG.json")
    forbidden_eval = set(catalog["forbidden"]["eval120"])
    forbidden_hold = set(catalog["forbidden"]["hold_100"])
    forbidden_rearc_hold = set(catalog["forbidden"]["re_arc_holdout_47"])
    stage2_heldout_24 = {t for t in catalog["pools"]["distill600"] if int(t, 16) % 20 == 0}
    excluded_all = forbidden_eval | forbidden_hold | forbidden_rearc_hold | stage2_heldout_24
    pure_arc2_set = set(catalog["pools"]["pure_arc2_all_train"])

    # Cryptographic eval120 grid-SHA256 set (used ONLY to assert 0 overlap).
    eval120_grid_hashes: set[str] = set()
    for eid in forbidden_eval:
        etask = read_json(ARC2_EVAL_DIR / f"{eid}.json")
        eval120_grid_hashes |= collect_task_grid_hashes(etask)

    def assert_zero_leak(episode_id: str, pub_task: dict[str, Any], gold_grids: list[list[list[int]]]) -> None:
        fam = base_family_id(episode_id)
        if episode_id in excluded_all or fam in excluded_all:
            raise ValueError(f"CRITICAL LEAK BLOCKED: {episode_id} (family {fam}) in excluded set")
        if "output" in (pub_task.get("test") or [{}])[0]:
            raise ValueError(f"Public solver task leaked test output key: {episode_id}")
        gh = collect_task_grid_hashes(pub_task)
        for g in gold_grids:
            gh.add(canonical_grid_sha256(g))
        if gh & eval120_grid_hashes:
            raise ValueError(f"CRITICAL GRID HASH OVERLAP WITH EVAL120 ON {episode_id}")

    # Scan recovered training batches (`batch_001..003`) for solved organic programs & blindspot pure ARC-2 IDs.
    recovered_root = REPO_ROOT / "experiments/arc_gr_expand800_20261018/recovered"
    solved_organic: dict[str, list[str]] = {}
    partial_organic: dict[str, list[str]] = {}
    for batch_name in ("batch_001", "batch_002", "batch_003"):
        p1 = recovered_root / batch_name / "contents/results/sampling/phase1"
        if not p1.exists():
            continue
        for sd in sorted(p1.iterdir()):
            if not sd.is_dir() or not (sd / "packet_manifest.json").exists():
                continue
            pm = read_json(sd / "packet_manifest.json")
            for it in pm.get("tasks", []):
                tid = str(it["task_id"])
                if tid in excluded_all:
                    continue
                tdir = sd / it["name"]
                for rnd in sorted(tdir.glob("round_*")):
                    ap = rnd / "attempt.json"
                    if not ap.exists():
                        continue
                    att = read_json(ap)
                    src = (att.get("source") or "").strip()
                    fb = att.get("feedback") or {}
                    if not src or "def transform" not in src or len(src) > 9000:
                        continue
                    if fb.get("demo_all_correct"):
                        lst = solved_organic.setdefault(tid, [])
                        if src not in lst:
                            lst.append(src)
                    elif int(fb.get("demo_passes") or 0) >= 1:
                        lst = partial_organic.setdefault(tid, [])
                        if src not in lst:
                            lst.append(src)

    evaluator = FastBatchEvaluator(timeout_s=1.8)
    try:
        # ------------------------------------------------------------------
        # 1. POOL A: 160 Pure ARC-AGI-2 Direct & Transpose Tasks
        # ------------------------------------------------------------------
        s1_100_pure = [t for t in catalog["pools"]["s1_100_pure_arc2"] if t not in excluded_all]
        d600_pure = [t for t in catalog["pools"]["distill600_pure_arc2"] if t not in excluded_all]
        d600_unsolved = [t for t in d600_pure if t not in solved_organic]
        d600_solved = [t for t in d600_pure if t in solved_organic]
        rng.shuffle(d600_solved)
        selected_d600_pure = sorted(d600_unsolved + d600_solved[: max(0, 60 - len(d600_unsolved))])
        pool_a_ids = sorted(s1_100_pure) + selected_d600_pure  # 100 + 60 = 160 pure ARC-2 tasks

        pool_a_episodes: list[dict[str, Any]] = []
        sealed_gold: dict[str, dict[str, Any]] = {}
        solver_tasks: dict[str, dict[str, Any]] = {}

        for tid in pool_a_ids:
            raw = read_json(ARC2_TRAIN_DIR / f"{tid}.json")
            pub = {
                "train": [{"input": p["input"], "output": p["output"]} for p in raw["train"]],
                "test": [{"input": p["input"]} for p in raw["test"]],
            }
            gold_outs = [p["output"] for p in raw["test"]]
            assert_zero_leak(tid, pub, gold_outs)
            solver_tasks[tid] = pub
            sealed_gold[tid] = {
                "canonical_task_id": tid,
                "family_id": tid,
                "base_task_id": tid,
                "pool": "pool_a_pure_arc2",
                "is_pure_arc2": True,
                "source": "stage_n_s1_100_pure_arc2" if tid in set(s1_100_pure) else "stage_n_d600_pure_arc2",
                "split_group": "s1_100" if tid in set(s1_100_pure) else "distill600",
                "test_outputs": gold_outs,
            }
            pool_a_episodes.append({"episode_id": tid, "query_count": len(gold_outs)})

        # ------------------------------------------------------------------
        # Verify 100/100 RE-ARC procedural solvers (`50 selection + 50 hidden`)
        # ------------------------------------------------------------------
        allowed_rearc = [t for t in catalog["pools"]["allowed_re_arc_families"] if t not in excluded_all]
        verified_rearc_solvers: list[tuple[str, str, list[dict[str, Any]]]] = []
        for fam in allowed_rearc:
            if fam not in solved_organic:
                continue
            pairs = read_json(RE_ARC_TASKS_DIR / f"{fam}.json")[:100]
            if len(pairs) < 100:
                continue
            in_fast = [p["input"] for p in pairs[:6]]
            exp_fast = [p["output"] for p in pairs[:6]]
            in_grids = [p["input"] for p in pairs]
            exp_grids = [p["output"] for p in pairs]
            for prog in solved_organic[fam][:2]:
                fast_outs = evaluator.run(prog, in_fast)
                if fast_outs != exp_fast:
                    continue
                outs = evaluator.run(prog, in_grids)
                if outs is not None and outs == exp_grids:
                    verified_rearc_solvers.append((fam, prog, pairs))
                    break
            if len(verified_rearc_solvers) >= 72:
                break

        # Also verify solved pure ARC-2 programs on official train+test for B2 & C1/C2
        verified_pure_arc2_solvers: list[tuple[str, str, dict[str, Any]]] = []
        for tid in d600_pure:
            if tid not in solved_organic:
                continue
            raw = read_json(ARC2_TRAIN_DIR / f"{tid}.json")
            all_in = [p["input"] for p in raw["train"]] + [p["input"] for p in raw["test"]]
            all_exp = [p["output"] for p in raw["train"]] + [p["output"] for p in raw["test"]]
            for prog in solved_organic[tid][:2]:
                outs = evaluator.run(prog, all_in)
                if outs is not None and outs == all_exp:
                    verified_pure_arc2_solvers.append((tid, prog, raw))
                    break

        # ------------------------------------------------------------------
        # 2. POOL B: Near-Miss Repair Episodes (`[PRIOR_WAVE_NEAR_MISS_CANDIDATE]`)
        # ------------------------------------------------------------------
        pool_b_episodes: list[dict[str, Any]] = []
        organic_seeds = read_json(THIS_DIR / "ORGANIC_NEAR_MISS_SEEDS.json").get("tasks", {})

        # B1: Organic near-misses (`17` tasks)
        for tid, cand_list in sorted(organic_seeds.items()):
            if tid in excluded_all or not cand_list:
                continue
            raw = read_json(ARC2_TRAIN_DIR / f"{tid}.json")
            code = cand_list[0]["code"]
            demo_ins = [p["input"] for p in raw["train"]]
            demo_outs = evaluator.run(code, demo_ins)
            if demo_outs is None:
                continue
            nm_handover = compute_near_miss_handover(raw["train"], code, demo_outs)
            if nm_handover is None:
                continue
            ep_id = f"nm_{tid}"
            pub = {
                "train": [{"input": p["input"], "output": p["output"]} for p in raw["train"]],
                "test": [{"input": p["input"]} for p in raw["test"]],
            }
            gold_outs = [p["output"] for p in raw["test"]]
            assert_zero_leak(tid, pub, gold_outs)
            solver_tasks[ep_id] = pub
            sealed_gold[ep_id] = {
                "canonical_task_id": tid,
                "family_id": tid,
                "base_task_id": tid,
                "pool": "pool_b_near_miss",
                "operator_override": "near_miss_repair",
                "is_pure_arc2": tid in pure_arc2_set,
                "source": "stage_n_organic_near_miss",
                "split_group": "distill600",
                "test_outputs": gold_outs,
            }
            pool_b_episodes.append(
                {
                    "episode_id": ep_id,
                    "query_count": len(gold_outs),
                    "seed_handover": {"nearMissHandover": nm_handover},
                }
            )

        # B2: Pure ARC-AGI-2 execution-verified AST near-misses (`~34` episodes)
        b2_used_fams = {sealed_gold[e["episode_id"]]["family_id"] for e in pool_b_episodes}
        for tid, true_prog, raw in verified_pure_arc2_solvers:
            if tid in b2_used_fams or len([e for e in pool_b_episodes if sealed_gold[e["episode_id"]]["is_pure_arc2"]]) >= 42:
                continue
            demo_ins = [p["input"] for p in raw["train"]]
            test_ins = [p["input"] for p in raw["test"]]
            chosen_nm = None
            for mut_code in generate_single_bug_mutations(true_prog):
                outs = evaluator.run(mut_code, demo_ins + test_ins)
                if outs is None:
                    continue
                nm_handover = compute_near_miss_handover(raw["train"], mut_code, outs[: len(demo_ins)])
                if nm_handover is not None and 1 <= nm_handover["semantic_diff_cells"] <= 64:
                    chosen_nm = nm_handover
                    break
            if chosen_nm is None:
                continue
            ep_id = f"nm_{tid}"
            pub = {
                "train": [{"input": p["input"], "output": p["output"]} for p in raw["train"]],
                "test": [{"input": p["input"]} for p in raw["test"]],
            }
            gold_outs = [p["output"] for p in raw["test"]]
            assert_zero_leak(tid, pub, gold_outs)
            b2_used_fams.add(tid)
            solver_tasks[ep_id] = pub
            sealed_gold[ep_id] = {
                "canonical_task_id": tid,
                "family_id": tid,
                "base_task_id": tid,
                "pool": "pool_b_near_miss",
                "operator_override": "near_miss_repair",
                "is_pure_arc2": True,
                "source": "stage_n_pure_arc2_near_miss",
                "split_group": "distill600",
                "test_outputs": gold_outs,
            }
            pool_b_episodes.append(
                {
                    "episode_id": ep_id,
                    "query_count": len(gold_outs),
                    "seed_handover": {"nearMissHandover": chosen_nm},
                }
            )

        # B3: RE-ARC `50 + 50` procedural near-miss episodes (`~26` episodes)
        rearc_b_count = 0
        for fam, true_prog, pairs in verified_rearc_solvers:
            if fam in b2_used_fams or rearc_b_count >= 26:
                continue
            selection_50 = pairs[:50]
            hidden_50 = pairs[50:100]
            by_area = sorted(range(50), key=lambda i: len(selection_50[i]["input"]) * len(selection_50[i]["input"][0]))
            # 3 compact demos + 1 larger/diverse demo so partial-scope candidates naturally pass 2..3/4 demos!
            train_4 = [selection_50[by_area[0]], selection_50[by_area[1]], selection_50[by_area[2]], selection_50[by_area[-1]]]
            test_2 = [selection_50[by_area[15]], selection_50[by_area[30]]]
            demo_ins = [p["input"] for p in train_4]
            test_ins = [p["input"] for p in test_2]

            sub3_h = max(len(g) for g in demo_ins[:3])
            sub3_w = max(len(g[0]) for g in demo_ins[:3])
            sub3_pal = sorted({v for g in demo_ins[:3] for row in g for v in row})
            bg0 = majority_color(demo_ins[0])
            extra_nms = [
                x
                for x in (
                    make_row_window_variant(true_prog, sub3_h, sub3_w),
                    make_palette_filter_variant(true_prog, sub3_pal, bg0),
                )
                if x
            ]
            candidate_pool = list(partial_organic.get(fam, [])[:3]) + generate_single_bug_mutations(true_prog) + extra_nms
            chosen_nm = None
            for mut_code in candidate_pool:
                outs = evaluator.run(mut_code, demo_ins + test_ins)
                if outs is None:
                    continue
                nm_handover = compute_near_miss_handover(train_4, mut_code, outs[:4])
                if nm_handover is not None and 1 <= nm_handover["semantic_diff_cells"] <= 80:
                    chosen_nm = nm_handover
                    break
            if chosen_nm is None:
                continue
            ep_id = f"rearc_{fam}_nm01"
            pub = {
                "train": [{"input": p["input"], "output": p["output"]} for p in train_4],
                "test": [{"input": p["input"]} for p in test_2],
            }
            gold_outs = [p["output"] for p in test_2]
            assert_zero_leak(ep_id, pub, gold_outs + [p["output"] for p in hidden_50])
            b2_used_fams.add(fam)
            rearc_b_count += 1
            solver_tasks[ep_id] = pub
            sealed_gold[ep_id] = {
                "canonical_task_id": ep_id,
                "family_id": fam,
                "base_task_id": ep_id,
                "pool": "pool_b_near_miss",
                "operator_override": "near_miss_repair",
                "is_pure_arc2": False,
                "source": "stage_n_rearc50_near_miss",
                "split_group": "re_arc_50_50",
                "test_outputs": gold_outs,
                "hidden_50_pairs": hidden_50,
            }
            pool_b_episodes.append(
                {
                    "episode_id": ep_id,
                    "query_count": len(gold_outs),
                    "seed_handover": {"nearMissHandover": chosen_nm},
                }
            )

        # ------------------------------------------------------------------
        # 3. POOL C: Adjudication Episodes (`[DISAGREEMENT_AND_OVERFIT_ADJUDICATION]`)
        #    - C1: Shuffled A/B(/C) true rule vs subtle demo-passing decoy(s) (pure ARC-2 + RE-ARC 50+50)
        #    - C2: Shuffled A/B NEITHER candidate generalizes -> unified general replacement (>=25% of Pool C!)
        #    - C3: Shuffled A/B genuinely underdetermined 2-demo RE-ARC -> clean abstention (`6` episodes)
        # ------------------------------------------------------------------
        pool_c_episodes: list[dict[str, Any]] = []
        c_used_fams: set[str] = set()

        def _pack_adjudication_episode(
            ep_id: str,
            fam: str,
            train_pairs: list[dict[str, Any]],
            test_pairs: list[dict[str, Any]],
            raw_candidates: list[dict[str, Any]],
            *,
            is_pure_arc2: bool,
            category: str,
            source: str,
            split_group: str,
            hidden_50: list[dict[str, Any]] | None = None,
            allow_abstention_only: bool = False,
        ) -> bool:
            shuffled = list(raw_candidates)
            rng.shuffle(shuffled)
            labels = ["A", "B", "C"][: len(shuffled)]
            labeled_cands: list[dict[str, Any]] = []
            role_map: dict[str, str] = {}
            for lbl, item in zip(labels, shuffled):
                role_map[lbl] = item["role"]
                labeled_cands.append(
                    {
                        "label": lbl,
                        "code": item["code"],
                        "demo_pass_count": len(train_pairs),
                        "demo_count": len(train_pairs),
                        "audit_status": item.get("audit_status", "clean"),
                        "query_advisories": item.get("query_advisories", []),
                        "query_test_only_branch_flags": item.get("query_test_only_branch_flags", [False] * len(test_pairs)),
                        "program_risks": item.get("program_risks", {}),
                        "predictions": item["predictions"],
                    }
                )
            disagreements = compute_disagreement_summary(labeled_cands, len(test_pairs))
            if not any((d.get("different_cells") or 0) > 0 or not d.get("same_shape") for d in disagreements):
                return False

            pub = {
                "train": [{"input": p["input"], "output": p["output"]} for p in train_pairs],
                "test": [{"input": p["input"]} for p in test_pairs],
            }
            gold_outs = [p["output"] for p in test_pairs]
            extra_gold = [p["output"] for p in (hidden_50 or [])]
            assert_zero_leak(fam, pub, gold_outs + extra_gold)

            handover_cands = [
                {k: v for k, v in c.items() if k != "predictions"}
                for c in labeled_cands
            ]
            solver_tasks[ep_id] = pub
            sealed_gold[ep_id] = {
                "canonical_task_id": ep_id if ep_id.startswith("rearc_") else fam,
                "family_id": fam,
                "base_task_id": ep_id,
                "pool": "pool_c_adjudicate",
                "operator_override": "adjudicate",
                "is_pure_arc2": is_pure_arc2,
                "source": source,
                "split_group": split_group,
                "test_outputs": gold_outs,
                "hidden_50_pairs": hidden_50,
                "allow_abstention_only": allow_abstention_only,
                "adjudication_meta": {
                    "category": category,
                    "candidate_count": len(labeled_cands),
                    "shuffled_roles": role_map,
                },
            }
            pool_c_episodes.append(
                {
                    "episode_id": ep_id,
                    "query_count": len(gold_outs),
                    "seed_handover": {
                        "adjudicatorHandover": {
                            "candidates": handover_cands,
                            "disagreements": disagreements,
                        }
                    },
                }
            )
            c_used_fams.add(fam)
            return True

        # C3-Abstain (`6` episodes, selected first before C1/C2 consume RE-ARC families):
        # Genuinely underdetermined 2-demo RE-ARC where two clean D8-conjugate rules
        # agree identically on BOTH visible demos (zero hardcoded constants!) yet differ on `test_inputs[0]`.
        c3_count = 0
        for fam, true_prog, pairs in verified_rearc_solvers:
            if fam in c_used_fams or c3_count >= 6:
                continue
            sel50 = pairs[:50]
            for conj_mode in ("transpose", "flip_h"):
                conj_prog = make_d8_conjugate_variant(true_prog, conj_mode)
                if not conj_prog:
                    continue
                ins50 = [p["input"] for p in sel50]
                exps50 = [p["output"] for p in sel50]
                conj_outs = evaluator.run(conj_prog, ins50)
                if conj_outs is None:
                    continue
                agree_idx = [i for i in range(50) if conj_outs[i] == exps50[i]]
                disagree_idx = [i for i in range(50) if conj_outs[i] != exps50[i]]
                if len(agree_idx) >= 2 and len(disagree_idx) >= 1:
                    train_2 = [sel50[agree_idx[0]], sel50[agree_idx[1]]]
                    test_1 = [sel50[disagree_idx[0]]]
                    raw_cands = [
                        {
                            "role": "general_rule_axis_1",
                            "code": true_prog,
                            "predictions": [exps50[disagree_idx[0]]],
                            "audit_status": "clean",
                            "program_risks": {},
                        },
                        {
                            "role": f"general_rule_{conj_mode}_axis_2",
                            "code": conj_prog,
                            "predictions": [conj_outs[disagree_idx[0]]],
                            "audit_status": "clean",
                            "program_risks": {},
                        },
                    ]
                    if _pack_adjudication_episode(
                        f"rearc_{fam}_adj_abstain01",
                        fam,
                        train_2,
                        test_1,
                        raw_cands,
                        is_pure_arc2=False,
                        category="c3_insufficient_evidence_abstain",
                        source="stage_n_rearc50_adjudicate_abstain",
                        split_group="re_arc_50_50",
                        hidden_50=pairs[50:100],
                        allow_abstention_only=True,
                    ):
                        c3_count += 1
                        break

        # C1-PureARC2: Pure ARC-AGI-2 adjudication episodes (`~24` episodes)
        for tid, true_prog, raw in verified_pure_arc2_solvers:
            if tid in c_used_fams or len(pool_c_episodes) >= 24:
                continue
            train_p = raw["train"]
            test_p = raw["test"]
            demo_ins = [p["input"] for p in train_p]
            demo_exps = [p["output"] for p in train_p]
            test_ins = [p["input"] for p in test_p]
            test_exps = [p["output"] for p in test_p]

            max_h = max(len(g) for g in demo_ins)
            max_w = max(len(g[0]) for g in demo_ins)
            seen_pal = sorted({v for g in demo_ins for row in g for v in row})
            bg0 = majority_color(demo_ins[0])

            decoys: list[dict[str, Any]] = []
            for role_name, cand_code, risk_meta in (
                ("window_extent_decoy", make_row_window_variant(true_prog, max_h, max_w), {"constant_grid_bound": True}),
                ("palette_subset_decoy", make_palette_filter_variant(true_prog, seen_pal, bg0), {"fixed_color_set": True}),
                ("transpose_conj_decoy", make_d8_conjugate_variant(true_prog, "transpose"), {}),
                ("fliph_conj_decoy", make_d8_conjugate_variant(true_prog, "flip_h"), {}),
            ):
                if not cand_code:
                    continue
                outs = evaluator.run(cand_code, demo_ins + test_ins)
                if outs is None:
                    continue
                if outs[: len(demo_ins)] == demo_exps and outs[len(demo_ins) :] != test_exps:
                    decoys.append(
                        {
                            "role": role_name,
                            "code": cand_code,
                            "predictions": outs[len(demo_ins) :],
                            "audit_status": "advisory" if risk_meta else "clean",
                            "program_risks": risk_meta,
                        }
                    )
            if not decoys:
                continue
            raw_cands = [
                {
                    "role": "true_general_rule",
                    "code": true_prog,
                    "predictions": test_exps,
                    "audit_status": "clean",
                    "program_risks": {},
                },
                *decoys[:2],  # 2- or 3-candidate A/B or A/B/C!
            ]
            _pack_adjudication_episode(
                f"adj_{tid}",
                tid,
                train_p,
                test_p,
                raw_cands,
                is_pure_arc2=True,
                category="c1_select_or_improve_pure_arc2",
                source="stage_n_pure_arc2_adjudicate",
                split_group="distill600",
            )

        # C2-REARC (`~24` episodes, >30% of Pool C!): NEITHER Candidate A NOR Candidate B generalizes!
        # Select 4 small/restricted-palette demos from `selection_50` and 2 larger+richer-palette test queries from `selection_50`.
        c2_count = 0
        for fam, true_prog, pairs in verified_rearc_solvers:
            if fam in c_used_fams or c2_count >= 24:
                continue
            sel50 = pairs[:50]
            hid50 = pairs[50:100]
            # Sort sel50 by grid area so smaller grids can form demos while larger + unseen-color grids form test!
            by_area = sorted(range(50), key=lambda i: len(sel50[i]["input"]) * len(sel50[i]["input"][0]))
            train_idx = by_area[:4]
            train_4 = [sel50[i] for i in train_idx]
            max_h = max(len(p["input"]) for p in train_4)
            max_w = max(len(p["input"][0]) for p in train_4)
            seen_pal = sorted({v for p in train_4 for row in p["input"] for v in row})
            bg0 = majority_color(train_4[0]["input"])

            # Find 2 test pairs in remaining sel50 where window decoy and palette/conjugate decoy BOTH fail & disagree!
            rest_pairs = [sel50[i] for i in range(50) if i not in set(train_idx)]
            cand_win = make_row_window_variant(true_prog, max_h, max_w)
            cand_pal = make_palette_filter_variant(true_prog, seen_pal, bg0)
            if not cand_win or not cand_pal:
                continue

            demo_ins = [p["input"] for p in train_4]
            demo_exps = [p["output"] for p in train_4]
            if evaluator.run(cand_win, demo_ins) != demo_exps or evaluator.run(cand_pal, demo_ins) != demo_exps:
                continue

            rest_ins = [p["input"] for p in rest_pairs[:16]]
            rest_exps = [p["output"] for p in rest_pairs[:16]]
            win_outs = evaluator.run(cand_win, rest_ins)
            pal_outs = evaluator.run(cand_pal, rest_ins)
            if win_outs is None or pal_outs is None:
                continue

            # Pick 2 queries: one where cand_win fails, one where cand_pal fails, and win_outs != pal_outs!
            q_win_fail = [i for i in range(len(rest_ins)) if win_outs[i] != rest_exps[i]]
            q_pal_fail = [i for i in range(len(rest_ins)) if pal_outs[i] != rest_exps[i] and win_outs[i] != pal_outs[i]]
            if not q_win_fail or not q_pal_fail:
                continue
            q0 = q_win_fail[0]
            q1 = next((i for i in q_pal_fail if i != q0), q_pal_fail[0])
            chosen_q = [q0] if q0 == q1 else [q0, q1]
            test_pairs = [rest_pairs[i] for i in chosen_q]

            # Verify NEITHER cand_win NOR cand_pal passes `hid50`!
            hid_ins = [p["input"] for p in hid50]
            hid_exps = [p["output"] for p in hid50]
            if evaluator.run(cand_win, hid_ins) == hid_exps or evaluator.run(cand_pal, hid_ins) == hid_exps:
                continue

            raw_cands = [
                {
                    "role": "partial_window_bound_overfit",
                    "code": cand_win,
                    "predictions": [win_outs[i] for i in chosen_q],
                    "audit_status": "advisory",
                    "program_risks": {"constant_grid_bound": True},
                },
                {
                    "role": "partial_palette_subset_overfit",
                    "code": cand_pal,
                    "predictions": [pal_outs[i] for i in chosen_q],
                    "audit_status": "advisory",
                    "program_risks": {"fixed_color_set": True},
                },
            ]
            if _pack_adjudication_episode(
                f"rearc_{fam}_adj_unify01",
                fam,
                train_4,
                test_pairs,
                raw_cands,
                is_pure_arc2=False,
                category="c2_neither_generalizes_unify",
                source="stage_n_rearc50_adjudicate_unify",
                split_group="re_arc_50_50",
                hidden_50=hid50,
            ):
                c2_count += 1

        # C1-REARC (`~22` episodes): True rule vs 1-2 subtle demo-passing decoys on RE-ARC `50+50`
        c1_rearc_count = 0
        for fam, true_prog, pairs in verified_rearc_solvers:
            if fam in c_used_fams or c1_rearc_count >= 22:
                continue
            sel50 = pairs[:50]
            hid50 = pairs[50:100]
            by_area = sorted(range(50), key=lambda i: len(sel50[i]["input"]) * len(sel50[i]["input"][0]))
            train_4 = [sel50[i] for i in by_area[:4]]
            rest_pairs = [sel50[i] for i in by_area[4:20]]
            max_h = max(len(p["input"]) for p in train_4)
            max_w = max(len(p["input"][0]) for p in train_4)
            seen_pal = sorted({v for p in train_4 for row in p["input"] for v in row})
            bg0 = majority_color(train_4[0]["input"])

            demo_ins = [p["input"] for p in train_4]
            demo_exps = [p["output"] for p in train_4]
            rest_ins = [p["input"] for p in rest_pairs]
            rest_exps = [p["output"] for p in rest_pairs]

            chosen_test_idx: list[int] = []
            decoys_for_ep: list[tuple[str, str, list[list[list[int]]], dict[str, Any]]] = []
            for role_name, cand_code, risk_meta in (
                ("window_extent_decoy", make_row_window_variant(true_prog, max_h, max_w), {"constant_grid_bound": True}),
                ("palette_subset_decoy", make_palette_filter_variant(true_prog, seen_pal, bg0), {"fixed_color_set": True}),
                ("transpose_conj_decoy", make_d8_conjugate_variant(true_prog, "transpose"), {}),
            ):
                if not cand_code:
                    continue
                if evaluator.run(cand_code, demo_ins) != demo_exps:
                    continue
                r_outs = evaluator.run(cand_code, rest_ins)
                if r_outs is None:
                    continue
                diff_idx = [i for i in range(len(rest_ins)) if r_outs[i] != rest_exps[i]]
                if diff_idx:
                    decoys_for_ep.append((role_name, cand_code, r_outs, risk_meta))
                    for di in diff_idx[:2]:
                        if di not in chosen_test_idx and len(chosen_test_idx) < 2:
                            chosen_test_idx.append(di)
            if not decoys_for_ep or not chosen_test_idx:
                continue
            test_pairs = [rest_pairs[i] for i in chosen_test_idx]
            raw_cands = [
                {
                    "role": "true_general_rule",
                    "code": true_prog,
                    "predictions": [rest_exps[i] for i in chosen_test_idx],
                    "audit_status": "clean",
                    "program_risks": {},
                }
            ]
            for role_name, cand_code, r_outs, risk_meta in decoys_for_ep[:2]:
                preds = [r_outs[i] for i in chosen_test_idx]
                if preds != [rest_exps[i] for i in chosen_test_idx]:
                    raw_cands.append(
                        {
                            "role": role_name,
                            "code": cand_code,
                            "predictions": preds,
                            "audit_status": "advisory" if risk_meta else "clean",
                            "program_risks": risk_meta,
                        }
                    )
            if len(raw_cands) < 2:
                continue
            if _pack_adjudication_episode(
                f"rearc_{fam}_adj01",
                fam,
                train_4,
                test_pairs,
                raw_cands,
                is_pure_arc2=False,
                category="c1_select_or_improve_rearc",
                source="stage_n_rearc50_adjudicate",
                split_group="re_arc_50_50",
                hidden_50=hid50,
            ):
                c1_rearc_count += 1

    finally:
        evaluator.close()

    # Write solver_tasks/ (100% blind to gold!) and verifier_sealed_gold.json separately.
    solver_dir = out_root / "solver_tasks"
    if solver_dir.exists():
        for old_f in solver_dir.glob("*.json"):
            old_f.unlink()
    solver_dir.mkdir(parents=True, exist_ok=True)
    for ep_id, pub in solver_tasks.items():
        (solver_dir / f"{ep_id}.json").write_text(json.dumps(pub, separators=(",", ":")), encoding="utf-8")

    episodes_index = {
        "schema": "arc-stage-n-episodes-v1",
        "seed": seed,
        "built_at": time.time(),
        "wall_seconds": round(time.time() - t0, 2),
        "zero_leakage_verified": {
            "eval120_overlap": 0,
            "eval120_grid_sha256_overlap": 0,
            "hold_100_overlap": 0,
            "re_arc_holdout_47_overlap": 0,
            "stage2_heldout_24_overlap": 0,
        },
        "counts": {
            "pool_a_pure_arc2": len(pool_a_episodes),
            "pool_b_near_miss": len(pool_b_episodes),
            "pool_b_pure_arc2": sum(1 for e in pool_b_episodes if sealed_gold[e["episode_id"]]["is_pure_arc2"]),
            "pool_b_rearc50": sum(1 for e in pool_b_episodes if e["episode_id"].startswith("rearc_")),
            "pool_c_adjudicate": len(pool_c_episodes),
            "pool_c_pure_arc2": sum(1 for e in pool_c_episodes if sealed_gold[e["episode_id"]]["is_pure_arc2"]),
            "pool_c_c2_neither_unify": sum(
                1
                for e in pool_c_episodes
                if sealed_gold[e["episode_id"]]["adjudication_meta"]["category"] == "c2_neither_generalizes_unify"
            ),
            "pool_c_c3_abstain": sum(
                1
                for e in pool_c_episodes
                if sealed_gold[e["episode_id"]]["adjudication_meta"]["category"] == "c3_insufficient_evidence_abstain"
            ),
            "verified_rearc_100of100_solvers": len(verified_rearc_solvers),
            "verified_pure_arc2_solvers": len(verified_pure_arc2_solvers),
        },
        "pools": {
            "pool_a_pure_arc2": pool_a_episodes,
            "pool_b_near_miss": pool_b_episodes,
            "pool_c_adjudicate": pool_c_episodes,
        },
    }
    write_json(out_root / "STAGE_N_EPISODES_INDEX.json", episodes_index)
    write_json(out_root / "verifier_sealed_gold.json", sealed_gold)
    return episodes_index


if __name__ == "__main__":
    summary = build_stage_n_episodes()
    print(json.dumps(summary["counts"], indent=2))
    print("zero_leakage_verified:", json.dumps(summary["zero_leakage_verified"]))
    print("wall_seconds:", summary["wall_seconds"])
