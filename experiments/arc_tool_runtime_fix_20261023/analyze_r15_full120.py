"""Deep forensic analysis of Round-15 (64.00/120) vs Round-14 (60.00/120) and Round-5/7/9/11/12/13."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
R15_DIR = HERE / "g4_round2/r15_final_results"
NVARC = HERE.parent / "nvarc_full_capture_20260928"
gold = json.loads((NVARC / "gold_solutions.json").read_text(encoding="utf-8"))

ARMS = [
    "r15-eval48_results-20261027",
    "r15-eval72-a_results-20261027",
    "r15-eval72-b_results-20261027",
]


def main():
    total_score = 0.0
    total_tasks = 0
    solved_full = []
    partial_rows = []
    zero_rows = []

    for arm in ARMS:
        sub = json.loads((R15_DIR / arm / "submission.json").read_text(encoding="utf-8"))
        summ = json.loads((R15_DIR / arm / "summary.json").read_text(encoding="utf-8"))
        tasks_map = summ.get("tasks", {})
        for tid, res in sorted(tasks_map.items()):
            g_list = gold[tid]
            rows = sub.get(tid, [])
            m = sum(1 for r, g in zip(rows, g_list) if r.get("attempt_1") == g or r.get("attempt_2") == g)
            sc = m / len(g_list)
            total_tasks += 1
            total_score += sc

            w_desc = []
            tot_tok = 0
            for w_key in ("wave1", "wave2", "wave3"):
                w = res.get(w_key)
                if w:
                    tok = w.get("output_tokens", 0) or 0
                    tot_tok += tok
                    w_desc.append(
                        f"{w_key.upper()}:{w.get('status')}(el={int(bool(w.get('eligible')))},"
                        f"src={w.get('answer_source')},pass={w.get('best_demo_pass_count')}/{res.get('demo_count')},"
                        f"diff={w.get('fallback_demo_diff_cells')},tok={tok},stop={w.get('stop_reason')})"
                    )
                else:
                    w_desc.append(f"{w_key.upper()}:SKIP")

            row = {
                "arm": arm.split("_")[0],
                "task_id": tid,
                "score": sc,
                "queries": f"{m}/{len(g_list)}",
                "status": res.get("status"),
                "eligible": bool(res.get("eligible")),
                "source": res.get("answer_source"),
                "attempt_sources": res.get("attempt_sources"),
                "best_pass": f"{res.get('best_demo_pass_count')}/{res.get('demo_count')}",
                "diff_cells": res.get("fallback_demo_diff_cells"),
                "waves": " | ".join(w_desc),
                "total_tokens": tot_tok,
                "pool_len": len(res.get("candidate_pool", [])),
            }
            if sc == 1.0:
                solved_full.append(row)
            elif sc > 0:
                partial_rows.append(row)
            else:
                zero_rows.append(row)

    lines = []
    lines.append(f"# Round-15 Full 120 Forensic Report: {total_score:.2f} / {total_tasks} ({100*total_score/total_tasks:.2f}%)\n")
    lines.append(f"- **Full solves (1.0)**: {len(solved_full)}")
    lines.append(f"- **Partial solves (0.5)**: {len(partial_rows)}")
    lines.append(f"- **Zero solves (0.0)**: {len(zero_rows)}\n")

    lines.append("## 1. 8 Partial-Solve Tasks (0.5)\n")
    for r in partial_rows:
        lines.append(
            f"- `{r['task_id']}` ({r['arm']}): `{r['queries']}` | `el={r['eligible']}` `src={r['source']}` "
            f"`att_src={r['attempt_sources']}` `pass={r['best_pass']}` `tok={r['total_tokens']}` | {r['waves']}"
        )

    z_by_cat = Counter((r["eligible"], r["source"], r["status"]) for r in zero_rows)
    lines.append("\n## 2. 52 Zero-Solve Tasks Breakdown by `(eligible, answer_source, status)`\n")
    for k, v in z_by_cat.most_common():
        lines.append(f"- `{k}`: **{v}** tasks")

    fp_rows = [r for r in zero_rows if r["eligible"]]
    lines.append(f"\n## 3. Zero-Solve Tasks Where Model Verified Demos (`eligible=True`, False Positives / Overfits): {len(fp_rows)} tasks\n")
    for r in fp_rows:
        lines.append(
            f"- `{r['task_id']}` ({r['arm']}): `src={r['source']}` `att_src={r['attempt_sources']}` "
            f"`pool={r['pool_len']}` `tok={r['total_tokens']}` | {r['waves']}"
        )

    unver_rows = [r for r in zero_rows if not r["eligible"]]
    lines.append(f"\n## 4. Zero-Solve Tasks Without Demo-Verified Program (`eligible=False`, Censored / Near-Miss): {len(unver_rows)} tasks\n")
    for r in sorted(unver_rows, key=lambda x: (x["diff_cells"] if x["diff_cells"] is not None else 999999)):
        lines.append(
            f"- `{r['task_id']}` ({r['arm']}): `pass={r['best_pass']}` `diff_cells={r['diff_cells']}` "
            f"`pool={r['pool_len']}` `tok={r['total_tokens']}` | {r['waves']}"
        )

    report_text = "\n".join(lines) + "\n"
    (HERE / "ROUND15_FULL120_FORENSIC_REPORT.md").write_text(report_text, encoding="utf-8")
    print(report_text[:4000])


if __name__ == "__main__":
    main()
