"""Deterministic, offline audit of saved ARC receipts. Never executes candidate code.

Missing required schema fields raise errors. Absent artifacts and telemetry are
explicitly recorded as unavailable, never imputed as zero or as negative evidence.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import random
import statistics
import zipfile
from collections import Counter, defaultdict

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / "experiments/arc_tool_runtime_fix_20261023"
CONFIG_KEYS = (
    "reasoning_effort",
    "output_limit",
    "python_limit",
    "waves",
    "wave_seeds",
    "wave_transforms",
    "wave2_independent",
    "consensus_early_stop",
    "wave2_controls",
    "emergency_controls",
    "wave3_transform",
    "panel_token_budget_per_task",
    "closer_budget_reserve_per_task",
    "scheduling",
    "governor",
)
REQUESTED = [
    "stage2_v3_lm_diag24",
    "stage2_v3_lm_eval48_rest24",
    "stage2_v3_lm_eval72_a",
    "stage2_v3_lm_eval72_b",
    "_merged_stage2_v3_public48",
]
REQUESTED += [f"stage2_v3_lm{i}_public48" for i in range(7, 12)]
REQUESTED += [
    "stage_n_lm8_public48",
    "stage_n_lm9_public48",
    "stage_n2_lm11_public48",
    "stage_n3_lm10_public48",
    "stage_n3_lm11_public48",
]
PAIRS = [
    ("stage_n_lm12", "stage_n3_lm12"),
    ("stage_n_lm4_dev_hard_24", "stage2_v3_lm4_dev_hard_24"),
    ("stage_n2_lm5_dev_hard_24", "stage_n_lm5_dev_hard_24"),
    ("stage_n_lm12", "stage2_v3_lm7"),
    ("stage_n3_lm12", "stage2_v3_lm7"),
    ("stage_n_lm12", "stage2_v3_lm11"),
    ("stage_n3_lm12", "stage2_v3_lm11"),
    ("stage_n_lm12", "stage2_v3_lm10"),
    ("stage_n3_lm12", "stage2_v3_lm10"),
    ("stage_n2_lm5", "stage2_v3_lm7"),
    ("stage_n_lm6", "stage2_v3_lm7"),
]


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def read(path):
    return Path(path).read_bytes()


def load(path):
    return json.loads(read(path))


def required(obj, key, context):
    if key not in obj:
        raise ValueError(f"Missing required field {key}: {context}")
    return obj[key]


def csv_write(out, name, rows):
    rows = rows or [{"status": "unavailable", "reason": "No applicable observations"}]
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with (out / f"{name}.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: canonical(v) if isinstance(v, (dict, list)) else v
                    for k, v in row.items()
                }
            )


def flatten(d, prefix=""):
    result = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            result.update(flatten(v, key))
        else:
            result[key] = v
    return result


def config(summary):
    # Scheduling/governor contain run-state counters, not exclusively config.
    return {
        k: summary[k]
        for k in CONFIG_KEYS
        if k in summary and k not in ("scheduling", "governor")
    }


def diff(a, b):
    aa = flatten(a)
    bb = flatten(b)
    return {
        k: {"baseline": aa.get(k, "UNRECORDED"), "run": bb.get(k, "UNRECORDED")}
        for k in sorted(aa.keys() | bb.keys())
        if aa.get(k) != bb.get(k)
    }


def hits(attempts, gold):
    query_hits = []
    for index, expected in enumerate(gold):
        if index >= len(attempts):
            query_hits.append(False)
            continue
        attempt = attempts[index]
        matched = any(
            attempt.get(key) == expected for key in ("attempt_1", "attempt_2")
        )
        query_hits.append(matched)
    return query_hits


def score(attempts, gold):
    return sum(hits(attempts, gold)) / len(gold)


def stop_reason(wave, skipped):
    raw = str(wave.get("stop_reason", "")) if wave else skipped
    if "panel" in raw and ("budget" in raw or "exhaust" in raw):
        return "panel_budget_exhausted"
    if "low_roi" in raw:
        return "low_roi_guard"
    if not wave:
        return "not_run"
    if wave.get("finalize_reason") == "zero_demo_progress" and not wave.get("eligible"):
        return "zero_demo_finalize"
    if "zero_demo" in raw:
        return "zero_demo_finalize"
    # A finalize alone is not proof of a zero-demo intervention.
    if raw == "finalize_completed" and wave.get("best_demo_pass_count") == 0:
        return "unknown_finalize"
    if "output" in raw or "context_limit" in raw:
        return "output_limit"
    if "error" in raw or wave.get("status") == "error":
        return "error"
    if raw in (
        "first_eligible_program",
        "advisory_review_accepted",
        "duplicate_or_audit_review_accepted",
        "finalize_completed",
    ):
        return "completed"
    return "unknown"


class Receipts:
    def __init__(self, folder):
        self.folder = folder
        self.panel = next(
            (
                p
                for p in (
                    folder / "extracted/panel",
                    folder / "panel",
                    folder / "output/panel",
                )
                if p.is_dir()
            ),
            None,
        )
        self.archive = None
        self.names = []
        if self.panel is None:
            archives = sorted(folder.glob("*.zip.bin"))
            if archives:
                self.archive_path = archives[0]
                self.archive = zipfile.ZipFile(self.archive_path)
                self.names = self.archive.namelist()

    def events(self, tid, wave):
        suffix = f"{tid}/" + (f"wave{wave}/" if wave > 1 else "") + "events.jsonl"
        if self.panel:
            p = self.panel / suffix
            if not p.exists():
                return None
            data = read(p)
        elif self.archive:
            found = [n for n in self.names if n.endswith("/" + suffix)]
            if not found:
                return None
            if len(found) != 1:
                raise ValueError(f"Ambiguous trace {suffix}: {found}")
            data = self.archive.read(found[0])
        else:
            return None
        return [json.loads(line) for line in data.splitlines() if line.strip()]

    def close(self):
        if self.archive:
            self.archive.close()


def risk_features(w):
    result = {}
    pool = w.get("pool_candidates", w.get("candidate_pool", []))
    # Match the W1 selected program; do not pick the riskiest eventual candidate.
    matched = [
        c
        for c in pool
        if c.get("predictions") == w.get("predictions")
        and c.get("verified", c.get("demos_verified", False))
    ]
    candidate = matched[-1] if matched else {}
    for group in ("d8", "cswap"):
        stats = candidate.get(group + "_stats", w.get(group + "_stats", {}))
        for key in ("demo_compatible", "query_agree", "query_disagree"):
            result[f"risk_{group}_{key}"] = stats.get(key)
    for key in (
        "program_risks",
        "query_advisory_flags",
        "query_degenerate_flags",
        "ast_nodes",
    ):
        result["risk_" + key] = candidate.get(key, w.get(key))
    result["risk_output_tokens"] = w.get("output_tokens")
    return result


def normalize(name, folder, split_map, gold, configs, traces):
    from audit_telemetry import parse_events

    summary_path = folder / "summary.json"
    if not summary_path.exists():
        candidates = sorted(folder.rglob("summary.json"))
        if not candidates:
            raise FileNotFoundError(f"No final summary: {folder}")
        summary_path = candidates[0]
    summary = load(summary_path)
    cfg = config(summary)
    configs.append(
        {
            "run": name,
            "source": str(summary_path),
            "resolved_summary_config": cfg,
            "full_cli_status": "unavailable: summary is not a complete CLI receipt",
        }
    )
    sub_path = folder / "submission.json"
    if not sub_path.exists():
        sub_path = folder / "extracted/submission.json"
    submission = load(sub_path) if sub_path.exists() else None
    reader = Receipts(folder)
    rows = []
    for tid, t in required(summary, "tasks", summary_path).items():
        if tid not in split_map:
            continue
        g = required(gold, tid, "gold solutions")
        attempts = (
            required(submission, tid, sub_path)
            if submission is not None
            else required(t, "attempts", tid)
        )
        wave1 = t.get("wave1")
        if wave1 is None and t.get("wave") == 1:
            wave1 = t
        model = (
            "stage2_v3"
            if name in ("R15", "R16", "R17", "LM_baseline")
            else name.split("_lm")[0]
        )
        w1_score = None
        w1_attempt1_score = None
        if wave1 and "attempts" in wave1:
            w1_attempts = wave1["attempts"]
            w1_score = score(w1_attempts, g)
            first_hits = 0
            for index, expected in enumerate(g):
                if (
                    index < len(w1_attempts)
                    and w1_attempts[index].get("attempt_1") == expected
                ):
                    first_hits += 1
            w1_attempt1_score = first_hits / len(g)
        row = dict(
            run=name,
            model=model,
            split="dev_hard_24" if "dev_hard_24" in name else split_map[tid],
            task_id=tid,
            n_test_outputs=len(g),
            score=score(attempts, g),
            score_method="mean query hits in saved attempt_1 OR attempt_2",
            eligible=required(t, "eligible", tid),
            best_demo_pass_count=t.get("best_demo_pass_count"),
            demo_count=t.get("demo_count"),
            submitted=t.get("answer_source") == "submit_program",
            output_tokens=required(t, "output_tokens", tid),
            attempt_2_distinct=sum(
                bool(a.get("attempt_2")) and a.get("attempt_1") != a.get("attempt_2")
                for a in attempts
            ),
            w1_eligible=wave1.get("eligible") if wave1 else None,
            w1_score=w1_score,
            w1_attempt1_score=w1_attempt1_score,
            source=str(summary_path),
            query_hits=hits(attempts, g),
            final_attempts=attempts,
            w1_attempts=wave1.get("attempts") if wave1 else None,
            manifest_index=t.get("manifest_index"),
        )
        row.update(risk_features(wave1 or {}))
        advisory_flags = row["risk_query_advisory_flags"]
        row["w1_query_advisory_count"] = (
            sum(bool(flag) for flag in advisory_flags)
            if advisory_flags is not None else None
        )
        offsets = {
            k: None
            for k in (
                "tok_first_submit",
                "tok_first_partial_pass",
                "tok_first_full_pass",
                "tok_third_stalled_submission",
            )
        }
        failed, cumulative, complete_traces = 0, 0, True
        for w in range(1, 7):
            wave = wave1 if w == 1 else t.get(f"wave{w}")
            skipped = next(
                (
                    x["reason"]
                    for x in summary.get(f"wave{w}_skipped", [])
                    if x["task_id"] == tid
                ),
                "",
            )
            row[f"w{w}_ran"] = wave is not None
            if w == 2:
                row["w2_eligible"] = wave.get("eligible") if wave else None
                row["w2_attempts"] = wave.get("attempts") if wave else None
            row[f"w{w}_tokens"] = (
                required(wave, "output_tokens", f"{tid}/W{w}") if wave else 0
            )
            row[f"w{w}_stop_reason"] = stop_reason(wave, skipped)
            row[f"w{w}_raw_stop_reason"] = wave.get("stop_reason") if wave else skipped
            row[f"w{w}_zero_demo_triggered"] = (
                wave.get("finalize_reason") == "zero_demo_progress" if wave else False
            )
            row[f"w{w}_schedule_index"] = (
                summary.get(f"wave{w}_schedule", []).index(tid)
                if tid in summary.get(f"wave{w}_schedule", [])
                else t.get("manifest_index")
            )
            row[f"w{w}_score"] = (
                score(wave["attempts"], g) if wave and "attempts" in wave else None
            )
            if not wave:
                continue
            events = reader.events(tid, w)
            if events is None:
                complete_traces = False
                traces.append(
                    {
                        "run": name,
                        "task_id": tid,
                        "wave": w,
                        "status": "missing events",
                        "wave_tokens": row[f"w{w}_tokens"],
                    }
                )
            else:
                tel = parse_events(events, row["demo_count"])
                total = tel.get("trace_output_tokens")
                exact = total == row[f"w{w}_tokens"]
                traces.append(
                    {
                        "run": name,
                        "task_id": tid,
                        "wave": w,
                        **tel,
                        "wave_tokens": row[f"w{w}_tokens"],
                        "token_total_matches": exact,
                    }
                )
                if not exact:
                    complete_traces = False
                for k in offsets:
                    if offsets[k] is None and tel.get(k) is not None and exact:
                        offsets[k] = cumulative + tel[k]
                failed += tel.get("n_failed_submissions") or 0
                row[f"w{w}_tok_first_full_pass"] = (
                    tel.get("tok_first_full_pass") if exact else None
                )
                if w == 1:
                    row["w1_trace_exact"] = exact
            cumulative += row[f"w{w}_tokens"]
        # Later-wave receipts cannot establish a first event if earlier traces
        # are missing. Keep exact per-wave offsets separately in telemetry.csv.
        row.update(offsets if complete_traces else {k: None for k in offsets})
        row["n_failed_submissions"] = failed if complete_traces else None
        row["trace_coverage"] = "complete" if complete_traces else "partial_or_missing"
        row["wave_token_sum"] = cumulative
        rows.append(row)
    reader.close()
    return rows, summary


def paired(a, b):
    tids = sorted(a.keys() & b.keys())
    if not tids:
        return {"status": "no overlapping tasks"}
    values = [a[t]["score"] - b[t]["score"] for t in tids]
    rng = random.Random(20261030)
    samples = []
    for _ in range(10000):
        resampled = rng.choices(values, k=len(values))
        samples.append(sum(resampled) / len(resampled))
    samples.sort()
    pos = sum(value > 0 for value in values)
    neg = sum(value < 0 for value in values)
    n = pos + neg
    p = 1.0
    if n:
        tail_count = sum(math.comb(n, index) for index in range(min(pos, neg) + 1))
        p = min(1.0, 2 * tail_count / 2 ** n)
    return {
        "status": "observed",
        "tasks": len(tids),
        "mean_score_difference": statistics.mean(values),
        "task_score_sum_difference": sum(values),
        "ci95_low": samples[249],
        "ci95_high": samples[9749],
        "sign_test_p_two_sided": p,
        "positive": pos,
        "negative": neg,
        "agreement_rate": sum(x == 0 for x in values) / len(values),
    }


def w1_followup_tables(by_run, split_map):
    paired_rows = []
    for run_a, run_b in PAIRS:
        records_a = by_run[run_a]
        records_b = by_run[run_b]
        tokens_a, tokens_b = [], []
        for task_id in sorted(records_a.keys() & records_b.keys()):
            a = records_a[task_id]
            b = records_b[task_id]
            both_eligible = a["w1_eligible"] is True and b["w1_eligible"] is True
            offset_a = a.get("w1_tok_first_full_pass")
            offset_b = b.get("w1_tok_first_full_pass")
            if both_eligible and offset_a is not None and offset_b is not None:
                tokens_a.append(offset_a)
                tokens_b.append(offset_b)
        differences = [b - a for a, b in zip(tokens_a, tokens_b)]
        paired_rows.append({
            "pair": f"{run_a} vs {run_b}",
            "run_a": run_a,
            "run_b": run_b,
            "n_paired_w1": len(differences),
            "median_a_w1_tok": statistics.median(tokens_a) if tokens_a else None,
            "median_b_w1_tok": statistics.median(tokens_b) if tokens_b else None,
            "median_paired_diff_b_minus_a": statistics.median(differences) if differences else None,
            "mean_paired_diff_b_minus_a": statistics.mean(differences) if differences else None,
            "b_faster_count": sum(value < 0 for value in differences),
            "a_faster_count": sum(value > 0 for value in differences),
        })
    curve_rows = []
    for run_id, run_records in by_run.items():
        for split in ("public48", "eval72"):
            records = [r for r in run_records.values() if r["split"] == split]
            if not records:
                continue
            n_tasks = sum(value == split for value in split_map.values())
            tokens = [r["w1_tokens"] for r in records]
            cap_savings = {
                f"w1_tokens_saved_cap_{cap}": sum(max(0, value - cap) for value in tokens)
                for cap in (52000, 58000)
            }
            first_pass_offsets = []
            exact_count = 0
            fallback_count = 0
            for record in records:
                offset = record.get("w1_tok_first_full_pass")
                if offset is not None:
                    exact_count += 1
                elif record["w1_eligible"] is True:
                    offset = record["w1_tokens"]
                    fallback_count += 1
                if offset is not None:
                    first_pass_offsets.append(offset)
            for threshold in (
                10000, 20000, 30000, 34000, 38000, 40000,
                50000, 52000, 58000, 60000,
            ):
                reached = sum(offset <= threshold for offset in first_pass_offsets)
                curve_rows.append({
                    "run_id": run_id,
                    "split": split,
                    "n_tasks": n_tasks,
                    "observed_tasks": len(records),
                    **cap_savings,
                    "w1_eligible_count": sum(r["w1_eligible"] is True for r in records),
                    "mean_w1_tokens_all_tasks": statistics.mean(tokens) if len(records) == n_tasks else None,
                    "median_w1_tokens_all_tasks": statistics.median(tokens) if len(records) == n_tasks else None,
                    "exact_first_pass_count": exact_count,
                    "fallback_first_pass_count": fallback_count,
                    "threshold": threshold,
                    "first_pass_count": reached,
                    "first_pass_pct": 100 * reached / n_tasks,
                })
    return paired_rows, curve_rows


def w2_yield_on_w1_failures(by_run):
    rows = []
    for run_id, run_records in by_run.items():
        splits = sorted({record["split"] for record in run_records.values()})
        for split in splits:
            failures = [
                record for record in run_records.values()
                if record["split"] == split
                and record["w1_eligible"] is False
                and record["w2_ran"] is True
            ]
            tokens = [record["w2_tokens"] for record in failures]
            gains = [
                max(0, record["w2_score"] - record["w1_score"])
                for record in failures
            ]
            became_eligible = sum(record["w2_eligible"] is True for record in failures)
            gain_sum = sum(gains)
            rows.append({
                "run_id": run_id,
                "split": split,
                "n_w1_failed_ran_w2": len(failures),
                "mean_w2_tokens": statistics.mean(tokens) if tokens else None,
                "median_w2_tokens": statistics.median(tokens) if tokens else None,
                "w2_became_eligible_count": became_eligible,
                "w2_became_eligible_pct": 100 * became_eligible / len(failures) if failures else None,
                "w2_positive_score_gain_count": sum(gain > 0 for gain in gains),
                "w2_score_gain_sum": gain_sum,
                "tokens_per_w2_point_gained": sum(tokens) / gain_sum if gain_sum else None,
            })
    return rows


def select_two_slots(attempts_a, eligible_a, attempts_b, eligible_b):
    """Select saved grids without access to gold or correctness metadata."""
    if eligible_a != eligible_b:
        return attempts_a if eligible_a else attempts_b
    selected = []
    for a, b in itertools.zip_longest(attempts_a, attempts_b, fillvalue={}):
        first = a.get("attempt_1", [])
        second = []
        for candidate in (b.get("attempt_1"), a.get("attempt_2"), b.get("attempt_2")):
            if candidate and candidate != first:
                second = candidate
                break
        selected.append({"attempt_1": first, "attempt_2": second})
    return selected


def pooled_three_stage_route(records_a, records_b, public_ids, gold):
    pool_cap = 3744000
    # Concatenate shard manifests in saved source order, as in the budget replay.
    manifest_ids = sorted(
        public_ids,
        key=lambda tid: (
            records_b[tid]["shard_index"],
            records_b[tid]["manifest_index"],
            tid,
        ),
    )
    selected = {tid: records_a[tid]["w1_attempts"] for tid in public_ids}
    pool_used = sum(records_a[tid]["w1_tokens"] for tid in public_ids)
    b_w1_completed = 0
    b_w2_completed = 0
    needs_w2 = []

    # Stage 2 starts only after A W1 has run on every task.
    for tid in manifest_ids:
        a = records_a[tid]
        b = records_b[tid]
        clean_a = a["w1_eligible"] is True and a["w1_query_advisory_count"] == 0
        if clean_a:
            continue
        if pool_used + b["w1_tokens"] > pool_cap:
            pool_used = pool_cap
            break
        pool_used += b["w1_tokens"]
        b_w1_completed += 1
        selected[tid] = select_two_slots(
            a["w1_attempts"], a["w1_eligible"],
            b["w1_attempts"], b["w1_eligible"],
        )
        clean_b = b["w1_eligible"] is True and b["w1_query_advisory_count"] == 0
        if not clean_b and b["w2_ran"]:
            needs_w2.append(tid)

    # Stage 3 retains the completed W1 selection when a W2 endpoint is cut off.
    for tid in needs_w2:
        a = records_a[tid]
        b = records_b[tid]
        if pool_used + b["w2_tokens"] > pool_cap:
            pool_used = pool_cap
            break
        pool_used += b["w2_tokens"]
        b_w2_completed += 1
        selected[tid] = select_two_slots(
            a["w1_attempts"], a["w1_eligible"],
            b["w2_attempts"], b["w2_eligible"],
        )

    return {
        "seq_78k_route": "seq_78k_A_w1_B_w1_B_w2",
        "seq_78k_score_lower": sum(score(selected[tid], gold[tid]) for tid in sorted(public_ids)),
        "seq_78k_tokens_used": pool_used,
        "seq_78k_b_w1_completed": b_w1_completed,
        "seq_78k_b_w2_completed": b_w2_completed,
    }


def non_oracle_pairing(by_run, split_map, gold):
    public_ids = {tid for tid, split in split_map.items() if split == "public48"}
    public_runs = {
        name: records for name, records in by_run.items()
        if public_ids <= records.keys()
        and all(records[tid]["split"] == "public48" for tid in public_ids)
    }
    pairs = []
    for a, b in itertools.combinations(public_runs, 2):
        model_a = public_runs[a][next(iter(public_ids))]["model"]
        model_b = public_runs[b][next(iter(public_ids))]["model"]
        if model_a == model_b:
            continue
        # Stage-N is the first-stage router when paired with stage2_v3.
        if model_a == "stage2_v3":
            a, b = b, a
        pairs.append((a, b))
    results = []
    for run_a, run_b in sorted(pairs):
        row = dict(A=run_a, B=run_b, split="public48", n_tasks=len(public_ids))
        totals = Counter()
        missing_w1 = []
        for tid in sorted(public_ids):
            a, b = public_runs[run_a][tid], public_runs[run_b][tid]
            selected = select_two_slots(
                a["final_attempts"], a["eligible"],
                b["final_attempts"], b["eligible"],
            )
            totals["score_A_full"] += score(a["final_attempts"], gold[tid])
            totals["score_B_full"] += score(b["final_attempts"], gold[tid])
            totals["tokens_A_full"] += a["output_tokens"]
            totals["tokens_B_full"] += b["output_tokens"]
            totals["oracle_union_score"] += sum(
                x or y for x, y in zip(a["query_hits"], b["query_hits"])
            ) / len(gold[tid])
            totals["full_2slot_non_oracle_score"] += score(selected, gold[tid])
            clean_a = a["w1_eligible"] is True and a["w1_query_advisory_count"] == 0
            needed = [a] if clean_a else [a, b]
            if any(r["w1_attempts"] is None or r["w1_eligible"] is None for r in needed):
                missing_w1.append(tid)
                continue
            totals["seq_w1_A_then_B_tokens"] += a["w1_tokens"]
            if clean_a:
                selected = a["w1_attempts"]
                totals["seq_clean_A_stops"] += 1
            else:
                totals["seq_B_tasks"] += 1
                totals["seq_w1_A_then_B_tokens"] += b["w1_tokens"]
                selected = select_two_slots(
                    a["w1_attempts"], a["w1_eligible"],
                    b["w1_attempts"], b["w1_eligible"],
                )
            totals["seq_w1_A_then_B_score"] += score(selected, gold[tid])
        for key in (
            "score_A_full", "tokens_A_full", "score_B_full", "tokens_B_full",
            "oracle_union_score", "full_2slot_non_oracle_score",
            "seq_w1_A_then_B_score", "seq_w1_A_then_B_tokens",
            "seq_clean_A_stops", "seq_B_tasks",
        ):
            row[key] = totals[key]
        row["seq_status"] = "unavailable" if missing_w1 else "exact_saved_W1_replay"
        row["seq_missing_w1_tasks"] = missing_w1
        if missing_w1:
            for key in ("seq_w1_A_then_B_score", "seq_w1_A_then_B_tokens"):
                row[key] = None
        tokens = row["seq_w1_A_then_B_tokens"]
        row["seq_w1_A_then_B_tokens_per_task"] = tokens / len(public_ids) if tokens is not None else None
        row["full_2slot_tokens"] = row["tokens_A_full"] + row["tokens_B_full"]
        if not missing_w1:
            row.update(pooled_three_stage_route(
                public_runs[run_a], public_runs[run_b], public_ids, gold,
            ))
        results.append(row)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runs", nargs="+", help="name=directory; repeat name for disjoint shards"
    )
    ap.add_argument("--split-map", type=Path)
    ap.add_argument(
        "--budget-mode", choices=["pooled", "per_task", "both"], default="both"
    )
    ap.add_argument("--out", type=Path, default=HERE / "audit_out")
    ap.add_argument(
        "--endpoint-scenario-run",
        help="Optional explicitly approximate W1-W3 endpoint budget bounds for this run",
    )
    ap.add_argument(
        "--v17-w2-reservation",
        type=int,
        help="Explicit hypothetical extra W2 token cost for optional endpoint scenarios",
    )
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    metrics = load(OLD / "analysis/round17_metrics_review_20261027/metrics.json")
    gold = load(ROOT / "experiments/nvarc_full_capture_20260928/gold_solutions.json")
    if args.split_map:
        split_map = load(args.split_map)
    else:
        split_map = {}
        for f in sorted((OLD / "g4_round2/r15_final_results").glob("*/summary.json")):
            label = "public48" if "eval48_" in str(f) else "eval72"
            split_map.update({t: label for t in load(f)["tasks"]})
        if Counter(split_map.values()) != {"public48": 48, "eval72": 72}:
            raise ValueError(
                "Cannot derive verified 48/72 split from R15 shard membership"
            )
    for task_id in load(HERE / "DEV_HARD_24_SPLIT.json")["dev_hard_24"]:
        split_map.setdefault(task_id, "dev_hard_24")
        task_path = ROOT / "VARC/raw_data/ARC-AGI-2/data/training" / f"{task_id}.json"
        task = load(task_path)
        gold[task_id] = [query["output"] for query in task["test"]]
    (out / "split_map.json").write_text(
        json.dumps(split_map, indent=2, sort_keys=True), encoding="utf8"
    )
    dl = HERE / "downloaded_evals"
    inventory = []
    for name in sorted(set(REQUESTED) | {p.name for p in dl.iterdir() if p.is_dir()}):
        p = dl / name
        summaries = list(p.rglob("summary.json")) if p.exists() else []
        inventory.append(
            {
                "directory": name,
                "exists": p.exists(),
                "summary_count": len(summaries),
                "traces_json_count": len(list(p.rglob("traces/*.json")))
                if p.exists()
                else 0,
                "status": "available" if summaries else "MISSING FINAL ARTIFACTS",
            }
        )
    run_specs = []
    if args.runs:
        run_specs = [(s.split("=", 1)[0], Path(s.split("=", 1)[1])) for s in args.runs]
    else:
        for r in (15, 16, 17):
            root = OLD / f'g4_round2/r{r}_{"final_results" if r == 15 else "results"}'
            run_specs.extend(
                (f"R{r}", p.parent) for p in sorted(root.glob("*/summary.json"))
            )
        for p in sorted(dl.iterdir()):
            if not p.is_dir() or not list(p.rglob("summary.json")):
                continue
            if p.name.startswith("_merged"):
                if p.name == "_merged_stage2_v3_public48":
                    run_specs.append(("LM_baseline", p))
                continue
            if p.name == "stage2_v3_lm_eval48_rest24":
                continue  # exact subset of merged baseline; audited as alias below
            if p.name in ("stage2_v3_lm_eval72_a", "stage2_v3_lm_eval72_b"):
                run_specs.append(("LM_baseline", p))
                continue
            run_specs.append((p.name.removesuffix("_public48"), p))
    configs, trace_rows, records, summaries, score_receipts = [], [], [], {}, []
    shard_counts = Counter()
    for name, folder in run_specs:
        print(f"Reading {name}: {folder.name}", flush=True)
        rows, summary = normalize(name, folder, split_map, gold, configs, trace_rows)
        for row in rows:
            row["shard_index"] = shard_counts[name]
        shard_counts[name] += 1
        records.extend(rows)
        summaries[str(folder)] = summary
        score_path = folder / "EVALUATION_SCORE.json"
        if score_path.exists():
            receipt = load(score_path)
            computed = sum(r["score"] for r in rows)
            quoted = receipt.get("task_score_sum")
            full_panel = len(rows) == receipt.get("tasks_total")
            score_receipts.append(
                {
                    "run": name,
                    "source": str(score_path),
                    "selected_tasks": len(rows),
                    "receipt_tasks": receipt.get("tasks_total"),
                    "computed_score": computed,
                    "receipt_score": quoted,
                    "match": abs(computed - quoted) < 1e-8
                    if full_panel and quoted is not None
                    else None,
                    "status": "comparable"
                    if full_panel
                    else "excluded or partial split intersection",
                }
            )
    unique = [(r["run"], r["task_id"]) for r in records]
    if len(unique) != len(set(unique)):
        raise ValueError("Duplicate run-task rows; shards must not overlap")
    # Verify the merged public48 alias against its available rest24 source.
    alias = dl / "stage2_v3_lm_eval48_rest24/summary.json"
    merged = dl / "_merged_stage2_v3_public48/summary.json"
    alias_status = "unavailable"
    if alias.exists() and merged.exists():
        aa, bb = load(alias)["tasks"], load(merged)["tasks"]
        alias_status = (
            "exact task-record subset"
            if all(bb.get(k) == v for k, v in aa.items())
            else "MISMATCH"
        )
    tables = {
        "inventory": inventory,
        "telemetry": trace_rows,
        "A2_score_receipts": score_receipts,
    }
    from audit_telemetry import parse_embedded_harness

    historical_launch = {}
    for number in (15, 16, 17):
        launch_path = OLD / f"g4_round2/r{number}_eval48.py"
        if launch_path.exists():
            historical_launch[f"R{number}"] = parse_embedded_harness(
                read(launch_path).decode("utf8")
            )
    reference = next(
        (c["resolved_summary_config"] for c in configs if c["run"] == "R15"), {}
    )
    for c in configs:
        c["diff_against_R15"] = diff(reference, c["resolved_summary_config"])
        c["comparison_label"] = (
            ("confounded: " + ", ".join(c["diff_against_R15"]))
            if c["diff_against_R15"]
            else "summary-config match; full CLI/harness equivalence unproven"
        )
        launch = historical_launch.get(c["run"])
        if launch:
            c["historical_launch_source"] = f'g4_round2/{c["run"].lower()}_eval48.py'
            c["panel_argv_from_launch_source"] = launch["panel_argv"]
            c["launch_harness_sha256"] = launch["harness_sha256"]
            c["launch_diff_against_R15"] = diff(
                historical_launch.get("R15", {}), launch
            )
            c[
                "full_cli_status"
            ] = "launch source literals extracted; runtime expressions and receipt linkage unresolved"
            if c["launch_diff_against_R15"]:
                c[
                    "comparison_label"
                ] = "confounded: summary and/or launch harness differences (see diff columns)"
    tables["A1"] = configs
    by_run = defaultdict(dict)
    for r in records:
        by_run[r["run"]][r["task_id"]] = r
    totals = []
    for name, rr in by_run.items():
        for split in ("public48", "eval72", "dev_hard_24", "all_available"):
            rs = [
                r
                for r in rr.values()
                if split == "all_available" or r["split"] == split
            ]
            if not rs:
                continue
            total = sum(r["score"] for r in rs)
            tokens = sum(r["output_tokens"] for r in rs)
            quoted = (
                metrics["runs"].get(name.removeprefix("R"), {})
                if name in ("R15", "R16")
                else {}
            )
            expected = (
                quoted.get("saved_final_score") if split == "all_available" else None
            )
            expected_tokens = (
                quoted.get("output_tokens") if split == "all_available" else None
            )
            totals.append(
                {
                    "run": name,
                    "split": split,
                    "tasks": len(rs),
                    "score_sum": total,
                    "score_pct": total / len(rs) * 100,
                    "output_tokens": tokens,
                    "query_hits": sum(sum(r["query_hits"]) for r in rs),
                    "query_total": sum(r["n_test_outputs"] for r in rs),
                    "quoted_saved_score": expected,
                    "quoted_output_tokens": expected_tokens,
                    "quoted_match": None
                    if expected is None
                    else abs(total - expected) < 1e-8 and tokens == expected_tokens,
                    "wave_token_mismatches": sum(
                        r["wave_token_sum"] != r["output_tokens"] for r in rs
                    ),
                }
            )
    tables["A2"] = totals
    noise = []
    for a, b in itertools.combinations(sorted(by_run), 2):
        same_model = (
            by_run[a]
            and by_run[b]
            and next(iter(by_run[a].values()))["model"]
            == next(iter(by_run[b].values()))["model"]
        )
        if same_model or {a, b} <= {"R15", "R16", "R17"}:
            for split in ("public48", "eval72"):
                aa = {k: v for k, v in by_run[a].items() if v["split"] == split}
                bb = {k: v for k, v in by_run[b].items() if v["split"] == split}
                if aa.keys() & bb.keys():
                    ca = next(
                        c["resolved_summary_config"] for c in configs if c["run"] == a
                    )
                    cb = next(
                        c["resolved_summary_config"] for c in configs if c["run"] == b
                    )
                    noise.append(
                        {
                            "A": a,
                            "B": b,
                            "split": split,
                            **paired(aa, bb),
                            "config_diff": diff(ca, cb),
                            "noise_floor_valid": False,
                            "reason": "different or incomplete full config/harness receipts; not established repeats",
                        }
                    )
    tables["B1"] = noise
    c1 = []
    r15 = list(by_run.get("R15", {}).values())
    for split in ("public48", "eval72", "all_available"):
        eligible = [
            r
            for r in r15
            if r["eligible"] and (split == "all_available" or r["split"] == split)
        ]
        known = [
            r["tok_first_full_pass"]
            for r in eligible
            if r["tok_first_full_pass"] is not None
        ]
        for threshold in (34000, 38000, 46000):
            c1.append(
                {
                    "split": split,
                    "threshold": threshold,
                    "eventually_eligible": len(eligible),
                    "exact_first_full_pass_available": len(known),
                    "fraction_above": sum(v > threshold for v in known) / len(known)
                    if known
                    else None,
                    "status": "exact subset only"
                    if known
                    else "unavailable: R15 per-call traces missing",
                }
            )
    tables["C1"] = c1
    supplemental = []
    for name, rr in by_run.items():
        eligible = [r for r in rr.values() if r["w1_eligible"]]
        known = [
            r["w1_tok_first_full_pass"]
            for r in eligible
            if r.get("w1_tok_first_full_pass") is not None
        ]
        for threshold in (34000, 38000, 46000):
            supplemental.append(
                {
                    "run": name,
                    "threshold": threshold,
                    "W1_eligible": len(eligible),
                    "exact_offsets": len(known),
                    "fraction_above": sum(v > threshold for v in known) / len(known)
                    if known
                    else None,
                    "median": statistics.median(known) if known else None,
                    "scope": "descriptive W1 subset, not the required clean R15 C1",
                }
            )
    tables["C1_supplemental"] = supplemental
    counts = Counter()
    losses = []
    for r in records:
        for w in range(1, 7):
            counts[(r["run"], r["split"], w, r[f"w{w}_stop_reason"])] += 1
        base = by_run.get("R15", {}).get(r["task_id"])
        if base and base["score"] > r["score"] and r["run"] != "R15":
            losses.append(
                {
                    "run": r["run"],
                    "task_id": r["task_id"],
                    "split": r["split"],
                    "R15_score": base["score"],
                    "run_score": r["score"],
                    "reasons": [r[f"w{w}_stop_reason"] for w in range(1, 7)],
                    "attribution": "observational association; multiple interventions and sampling confounded",
                }
            )
    tables["C2"] = [
        dict(run=k[0], split=k[1], wave=k[2], stop_reason=k[3], tasks=v)
        for k, v in sorted(counts.items())
    ]
    tables["C2_losses"] = losses
    c3 = []
    for name, rr in by_run.items():
        early = [r for r in rr.values() if r["w1_eligible"]]
        tasks_with_skips = 0
        for record in rr.values():
            stop_reasons = [record[f"w{wave}_stop_reason"] for wave in range(2, 7)]
            if any(
                reason in ("panel_budget_exhausted", "low_roi_guard")
                for reason in stop_reasons
            ):
                tasks_with_skips += 1
        c3.append(
            {
                "run": name,
                "W1_eligible_tasks": len(early),
                "W2_tokens_on_W1_eligible": sum(r["w2_tokens"] for r in early),
                "W6_tokens_on_W1_eligible": sum(r["w6_tokens"] for r in early),
                "tasks_with_budget_or_low_roi_skip": tasks_with_skips,
                "causal_claim": "spend and skips observed; starvation causality not identified",
            }
        )
    tables["C3"] = c3
    tables["C4_w2_yield_on_w1_failures"] = w2_yield_on_w1_failures(by_run)
    tables["attempt2"] = []
    for name, run_records in by_run.items():
        query_count = sum(record["n_test_outputs"] for record in run_records.values())
        distinct_attempt2 = sum(
            record["attempt_2_distinct"] for record in run_records.values()
        )
        tables["attempt2"].append(
            {
                "run": name,
                "tasks": len(run_records),
                "queries": query_count,
                "distinct_nonempty_attempt2_queries": distinct_attempt2,
            }
        )
    d1, d2, d3, d4 = [], [], [], []
    for a, b in PAIRS:
        common = sorted(by_run[a].keys() & by_run[b].keys())
        if not common:
            for table in (d1, d2, d3, d4):
                table.append(
                    {"A": a, "B": b, "status": "unavailable: requested pair missing"}
                )
            continue
        config_a = next(c["resolved_summary_config"] for c in configs if c["run"] == a)
        config_b = next(c["resolved_summary_config"] for c in configs if c["run"] == b)
        config_diff = diff(config_a, config_b)
        pair_metadata = {
            "A": a,
            "B": b,
            "config_diff": config_diff,
            "comparison_label": "near-config" if config_diff else "same summary-config",
        }
        d3.append({**pair_metadata, **paired(by_run[a], by_run[b])})
        both = []
        for tid in common:
            ra, rb = by_run[a][tid], by_run[b][tid]
            if ra["score"] == rb["score"] == 1:
                group = "both"
            elif ra["score"] == 1:
                group = "A_only"
            elif rb["score"] == 1:
                group = "B_only"
            else:
                group = "neither"
            loser = None
            if ra["score"] > rb["score"]:
                loser = rb
            elif rb["score"] > ra["score"]:
                loser = ra
            reasons = [loser[f"w{w}_stop_reason"] for w in range(1, 7)] if loser else []
            interaction = any(
                s
                in (
                    "zero_demo_finalize",
                    "panel_budget_exhausted",
                    "low_roi_guard",
                    "output_limit",
                )
                for s in reasons
            )
            attribution = "unknown"
            if interaction:
                attribution = "harness interaction consistent"
            elif loser and all(
                reason in ("completed", "not_run") for reason in reasons
            ):
                attribution = "capability consistent, not causal proof"
            d1.append(
                {
                    **pair_metadata,
                    "task_id": tid,
                    "group": group,
                    "A_score": ra["score"],
                    "B_score": rb["score"],
                    "fractional_difference": ra["score"] - rb["score"],
                    "loser_reasons": reasons,
                    "attribution": attribution,
                }
            )
            if group == "both":
                both.append((ra, rb))
        for side, idx in ((a, 0), (b, 1)):
            ts = [
                r[idx]["tok_first_full_pass"]
                for r in both
                if r[idx]["tok_first_full_pass"] is not None
            ]
            d2.append(
                {
                    **pair_metadata,
                    "side": side,
                    "both_solved": len(both),
                    "exact_offsets": len(ts),
                    "median": statistics.median(ts) if ts else None,
                }
            )
        control_ids = set(common) & by_run["R15"].keys() & by_run["R16"].keys()

        def union_score(first_run, second_run, task_ids):
            total = 0.0
            for task_id in task_ids:
                first = by_run[first_run][task_id]
                second = by_run[second_run][task_id]
                union_hits = sum(
                    first_hit or second_hit
                    for first_hit, second_hit in zip(
                        first["query_hits"], second["query_hits"]
                    )
                )
                total += union_hits / first["n_test_outputs"]
            return total

        d4.append(
            {
                **pair_metadata,
                "pair_tasks": len(common),
                "pair_union": union_score(a, b, common),
                "matched_tasks": len(control_ids),
                "cross_model_union": union_score(a, b, control_ids)
                if control_ids
                else None,
                "R15_R16_union": union_score("R15", "R16", control_ids)
                if control_ids
                else None,
                "control_valid": False,
                "reason": "R15/R16 configs differ"
                if control_ids
                else "no overlapping R15/R16 control tasks",
            }
        )
    tables.update(D1=d1, D2=d2, D3=d3, D4=d4)
    tables["D5_non_oracle_pairing"] = non_oracle_pairing(by_run, split_map, gold)
    tables["D2_paired"], tables["D2_unconditional_curve"] = w1_followup_tables(
        by_run, split_map
    )
    from audit_predicates import analyze

    e = analyze(records)
    for key in ("E1", "E2", "E3", "E4"):
        tables[key] = e[key]
    (out / "frozen_rules.json").write_text(
        json.dumps(e.get("frozen_rules", []), indent=2), encoding="utf8"
    )
    modes = (
        ("pooled", "per_task") if args.budget_mode == "both" else (args.budget_mode,)
    )
    from audit_budget import simulate, w2_skip_policies

    tables["F5_w2_skip_policies"] = []
    for run_id in ("R15", "R16"):
        for split in ("eval72", "all120"):
            scenario_records = [
                {**record, "split": split}
                for record in by_run[run_id].values()
                if record["split"] == split
                or split == "all120" and record["split"] in ("public48", "eval72")
            ]
            for budget in (68750, 78000):
                tables["F5_w2_skip_policies"].extend(
                    w2_skip_policies(scenario_records, trace_rows, budget)
                )

    tables["F1"] = []
    tables["F2"] = []
    primary_rule = {**e["frozen_rules"][0], "w2_token_reservation": 45114}
    for run, split in (("R15", "eval72"), ("R16", "eval72"), ("R15", "public48")):
        scenario_records = [r for r in by_run[run].values() if r["split"] == split]
        for mode in modes:
            for budget in (78000, 68750):
                for table, rule in (("F1", None), ("F2", primary_rule)):
                    result = simulate(scenario_records, trace_rows, mode, budget, rule)
                    tables[table].append(
                        {
                            "run": run,
                            "split": split,
                            "rule_rank": 1 if rule else None,
                            "w2_token_reservation": 45114 if rule else None,
                            **result,
                        }
                    )
    if args.endpoint_scenario_run:
        from audit_budget import simulate

        scenario_records = [
            record
            for record in by_run[args.endpoint_scenario_run].values()
            if record["split"] == "eval72"
        ]
        if not scenario_records:
            raise ValueError("Requested endpoint-scenario run has no records")
        scenario_rows = []
        for mode in modes:
            for budget in (78000, 68750):
                result = simulate(scenario_records, trace_rows, mode, budget)
                scenario_rows.append({"variant": "observed_waves", **result})
                if args.v17_w2_reservation:
                    if not e.get("frozen_rules"):
                        raise ValueError(
                            "No frozen predicate available for V17 scenario"
                        )
                    rule = {
                        **e["frozen_rules"][0],
                        "w2_token_reservation": args.v17_w2_reservation,
                    }
                    scenario_rows.append(
                        {
                            "variant": "hypothetical_W2",
                            **simulate(
                                scenario_records, trace_rows, mode, budget, rule
                            ),
                        }
                    )
        tables["F_endpoint_scenarios"] = scenario_rows
    abort = []
    for r in records:
        if r["tok_third_stalled_submission"] is not None:
            abort.append(
                {
                    "run": r["run"],
                    "task_id": r["task_id"],
                    "split": r["split"],
                    "abort_at": r["tok_third_stalled_submission"],
                    "first_full_pass_after_abort": r["tok_first_full_pass"] is not None
                    and r["tok_first_full_pass"] > r["tok_third_stalled_submission"],
                    "saved_final_score": r["score"],
                    "tokens_after_abort_upper_bound": r["output_tokens"]
                    - r["tok_third_stalled_submission"],
                    "note": "task-cumulative upper bound, not valid saved tokens under rerouting",
                }
            )
    tables["F3"] = abort
    tables["F4"] = [
        {
            "run": r["run"],
            "task_id": r["task_id"],
            "split": r["split"],
            "status": "cannot establish replay validity",
            "reason": "W3 existed after W2; preserved prompt/dependency evidence required before changing W2",
        }
        for r in records
        if r["w2_ran"] and r["w3_ran"]
    ]
    tables["records"] = records
    for name, rows in tables.items():
        csv_write(out, name, rows)
    write_report(out, tables, e, alias_status)
    print(f"Wrote {len(records)} records and {len(tables)} tables to {out}")


def write_report(out, tables, e, alias_status):
    lines = [
        "# R15–R17 and LM saved-artifact audit",
        "",
        "This is a partial-evidence audit, not a completed confirmatory experiment. All available final artifacts were inspected offline. Missing inputs are explicit; blanks in CSV mean unavailable, not zero.",
        "",
        "## Decision rules (thresholds fixed before analysis)",
        "",
        "1. **Zero-demo guillotine: inconclusive.** The pre-agreed threshold is 15% first full demo passes after 34k. R15 per-call trajectories are unavailable; wave totals cannot establish C1. Do not claim that this threshold has been met.",
        "2. **Stage-N superiority: not established.** D1-D4 compare eleven available pairs with paired intervals and summary-config differences. Near-config comparisons remain confounded; same summary config does not prove complete harness equivalence.",
        "3. **Predicate:** Public48 rules are frozen before evaluating R15/R16/R17 eval72. E3 reports unique-task holdout rates and R15 separately; the primary rule is rank 1.",
        "4. **Budget scenarios:** F1/F2 contain serial endpoint bounds from real R15/R16 receipts. F2 assumes 45,114 tokens for each added W2; these bounds do not establish a production V17 improvement. No Colab work was launched.",
        "",
        "## A. Integrity, configs, and recomputed scores",
        "",
        "A1 contains every recorded summary configuration and exact field-by-field differences against the first R15 shard. Historical R15/R16/R17 launch sources additionally provide panel argv literals and harness SHA-256 differences; runtime expressions and linkage to the saved run remain unresolved. LM result archives do not preserve full CLI receipts, and the current eval-suite runner is not treated as historical proof. R15/R16 differ in duplicateExtendTokens (0 versus 12288). R16/R17 use the same panel argv but change candidate-pool, panel and time-governor code; no noise-floor claim is made.",
        "",
        f"Merged public48 versus available rest24 source: **{alias_status}**. The merged copy is counted once for fitting. Other downloaded panels contribute only task IDs in the explicit 48/72 split; dev-only IDs outside it are excluded.",
        "",
        "| Run | Split | Tasks | Score sum | Score % | Output tokens |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in tables["A2"]:
        if r["split"] != "all_available":
            lines.append(
                f"| {r['run']} | {r['split']} | {r['tasks']} | {r['score_sum']:.6f} | {r['score_pct']:.2f} | {r['output_tokens']} |"
            )
    lines += [
        "",
        "A2 compares R15/R16 all-120 saved attempts against the supplied metrics.json saved_final_score and output_tokens. Both match exactly: R15 64/120 at 17,197,095 tokens; R16 59⅓/120 at 16,079,854 tokens. A2_score_receipts separately checks available EVALUATION_SCORE.json totals. It does not confuse current-selector replay scores with saved submissions. Scores are task-normalized fractions; query hit counts are separately included.",
        "",
        "## B. Noise floor",
        "",
        "**No noise floor available.** B1 reports overlapping same-checkpoint pairs and paired bootstrap intervals as descriptive comparisons only. None establishes repeated identical CLI, checkpoint, harness and scheduling conditions. Bootstrap uses 10,000 task resamples, seed 20261030; the exact sign test excludes ties.",
        "",
        "## C. Guillotine and starvation",
        "",
        "C1 explicitly reports missing R15 first-pass offsets. C2 lists every per-wave stop/skip reason and C2_losses lists R15 gains lost by later runs. A zero-demo finalize is not inferred just from a late finalization. Unknown reasons remain unknown. C3 counts W2/W6 spending after W1 eligibility and budget/ROI skips; these associations cannot causally separate guillotine, allocation, model, and sample effects.",
        "",
        "| Run | W2 tokens on W1 eligible | W6 tokens on W1 eligible | Tasks with budget/ROI skips |",
        "|---|---:|---:|---:|",
    ]
    for r in tables["C3"]:
        lines.append(
            f"| {r['run']} | {r['W2_tokens_on_W1_eligible']} | {r['W6_tokens_on_W1_eligible']} | {r['tasks_with_budget_or_low_roi_skip']} |"
        )
    lines += [
        "",
        "The premise that historical W2 ran only on W1-ineligible tasks is contradicted by these receipts. C1_supplemental reports exact W1 timing for other available traces, without substituting those runs for the required R15 test.",
    ]
    exact = sum(r.get("token_total_matches") is True for r in tables["telemetry"])
    mismatch = sum(r.get("token_total_matches") is False for r in tables["telemetry"])
    lines += [
        "",
        "## D. Model comparisons",
        "",
        "D1-D4 contain eleven available pairs with summary-config differences recorded. Pair unions cover all shared tasks; control unions use only the R15/R16 intersection. They are oracle coverage, not achievable selector scores. R15/R16 configs differ.",
        "",
        "## E. Frozen suspicious-eligible rules",
        "",
        "Risk features come from W1 only; the selected W1 prediction is matched to its verified candidate metadata. Final-run gold scores are never used as W1 FP labels. FP means eligible W1 score below 1, including partial success; TP means 1.0. No eval72 outcomes enter rule search.",
        "",
    ]
    lines.extend(e.get("notes", []))
    if e["E3"] and e["E3"][0].get("rule"):
        primary = e["E3"][0]
        lines += [
            "",
            f"Primary frozen rule: `{primary['rule']}`. Public48 training recall is {primary['train_fp_flagged']}/{primary['train_fp_tasks']} ({primary['train_fp_recall']:.1%}); TP flag rate is {primary['train_tp_flagged']}/{primary['train_tp_tasks']} ({primary['train_tp_flag_rate']:.1%}). These are fitted training results, not holdout performance.",
        ]
    if e["E3"] and e["E3"][0].get("rule"):
        for prefix, label in (
            ("eval72_", "R15/R16/R17 eval72 unique tasks"),
            ("R15_eval72_", "R15 eval72"),
        ):
            fp_flagged = primary[prefix + "fp_flagged"]
            fp_tasks = primary[prefix + "fp_tasks"]
            tp_flagged = primary[prefix + "tp_flagged"]
            tp_tasks = primary[prefix + "tp_tasks"]
            lines.append(
                f"{label}: FP recall {fp_flagged}/{fp_tasks}; TP flag rate {tp_flagged}/{tp_tasks}."
            )
    if e["E3"] and e["E3"][0].get("rule"):
        lines.append(
            f"Primary holdout adoption gate passed: {primary['adoption_gate_pass']}. The required FP recall is at least 50%, with TP flag-rate upper bound at most 30%."
        )
    lines += [
        "",
        "| Run | Nonempty distinct attempt_2 queries | Total queries |",
        "|---|---:|---:|",
    ]
    for row in tables["attempt2"]:
        lines.append(
            f"| {row['run']} | {row['distinct_nonempty_attempt2_queries']} | {row['queries']} |"
        )
    lines += [
        "",
        "See E1 for unique-task strata, E2 for univariate sweeps, E3 for the top five frozen one/two-condition rules, and E4 for every eligible false-positive observation. W2 cost estimates are explicitly proxy estimates rather than observed suspicious-eligible counterfactual costs.",
        "",
        "## F. Budget replay and abort",
        "",
        "F1/F2 report both pooled and per_task modes at 78,000 and 68,750 tokens/task. For N=72 the pooled caps are 5,616,000 and 4,950,000 output tokens. Pooled means one split-wide account, not 72 separate accounts. A valid replay must traverse W1 → W2 → W3 in logged scheduling order, account for each completed call, and retain only answers available before cutoff. Concurrent admission and in-flight reservations can differ from serial truncation.",
        "",
        "Shards are concatenated by source path within each wave, retaining local schedule order. The receipts do not establish a global interleaving. Hypothetical W2 uses manifest order with manifest-index tie breaks.",
        "F1 uses observed waves for R15 eval72, R16 eval72, and R15 public48. F2 applies the frozen rank-1 rule and charges a mean W2 reservation of 45,114 tokens for added W2. Scores are task-score-sum lower/upper bounds, not measured counterfactual outcomes. Truncated waves receive [0,1]; added W2 preserves only W1 attempt_1 and invalidates dependent W3 endpoints.",
        "",
        "F3 lists observed third-stall events; tokens after abort are only an upper bound on savings. F4 flags W3-after-W2 dependencies. The optional endpoint scenario is restricted to eval72 for its selected run.",
        "",
        "## Telemetry precision and limitations",
        "",
        f"events.jsonl provides exact per-call output usage and tool execution results. {exact} wave traces reconcile with summary wave totals; {mismatch} do not. telemetry.csv retains per-wave extracted fields and coverage. Missing traces are explicit. Token offsets use output tokens, including reasoning as accounted by the provider, not input tokens or text-length estimates.",
        "",
        "summary.json supplies wave totals, eligibility, final candidates, saved attempts, schedules and skip reasons. It does not supply exact first-submit/first-full-pass offsets. No traces/*.json were present in the inspected downloaded directories. Candidate demo verification is exact where recorded, but missing fields are not converted to zero. A third stall means three submit_program calls without improving best demo count, resetting after improvement; this is not three arbitrary Python calls.",
        "",
        "## Missing inputs and reproducibility",
        "",
    ]
    for r in tables["inventory"]:
        if r["status"] != "available":
            lines.append(f"- `{r['directory']}`: {r['status']}")
    lines += [
        "",
        "R17 public48 has only live named-wave snapshots locally, not a final summary; it is omitted rather than treated as a completed run. Named-wave snapshots may already contain later-wave receipts (as shown by metrics.json).",
        "",
        "Run: `python audit_r15_r17_and_lm.py --budget-mode both`. Supply `--runs name=path ...` and `--split-map path.json` to audit an explicit collection. Missing required schema keys raise errors. No network, model calls, candidate execution, or source modifications occur.",
        "",
        "## Comparison and budget numbers",
        "",
    ]
    lines += [
        "| Pair | Tasks | A minus B score sum | Mean difference 95% CI |",
        "|---|---:|---:|---|",
    ]
    for row in tables["D3"]:
        lines.append(
            f"| {row['A']} vs {row['B']} | {row['tasks']} | {row['task_score_sum_difference']:.3f} | [{row['ci95_low']:.3f}, {row['ci95_high']:.3f}] |"
        )
    lines += [
        "",
        "| Table | Run | Split | Mode | Tokens/task | Used tokens | Score bounds |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for table in ("F1", "F2"):
        for row in tables[table]:
            lines.append(
                f"| {table} | {row['run']} | {row['split']} | {row['mode']} | {row['tokens_per_task']} | {row['total_tokens']} | [{row['score_lower']:.3f}, {row['score_upper']:.3f}] |"
            )
    lines += [
        "", "## Follow-up: paired W1 speed and unconditional first-pass curves", "",
        "D2_paired.csv uses only common task IDs with both W1 eligible and both exact W1 first-pass offsets. Negative B-minus-A means B is faster; final-run correctness is not a filter.",
        "", "| Pair | Paired W1 tasks | Median B minus A tokens | B faster | A faster |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in tables["D2_paired"]:
        difference = row["median_paired_diff_b_minus_a"]
        displayed = f"{difference:,.1f}" if difference is not None else "unavailable"
        lines.append(
            f"| {row['pair']} | {row['n_paired_w1']} | {displayed} | {row['b_faster_count']} | {row['a_faster_count']} |"
        )
    lines += [
        "",
        "D2_unconditional_curve.csv uses all split tasks as the denominator (48 or 72), including W1 failures. Exact first-pass offsets take priority; eligible tasks without exact offsets use total W1 tokens as a conservative timing fallback. All-task W1 spend includes failures. Missing task receipts leave spend statistics blank; observed_tasks and timing-source counts expose coverage.",
        "", "| Run | Split | Observed / all tasks | W1 eligible | First pass by 34k | First pass by 60k |",
        "|---|---|---:|---:|---:|---:|",
    ]
    curve_groups = defaultdict(dict)
    for row in tables["D2_unconditional_curve"]:
        curve_groups[row["run_id"], row["split"]][row["threshold"]] = row
    for (run_id, split), thresholds in curve_groups.items():
        row = thresholds[34000]
        last = thresholds[60000]
        lines.append(
            f"| {run_id} | {split} | {row['observed_tasks']}/{row['n_tasks']} | {row['w1_eligible_count']} | {row['first_pass_count']} ({row['first_pass_pct']:.1f}%) | {last['first_pass_count']} ({last['first_pass_pct']:.1f}%) |"
        )
    lines += [
        "", "## Follow-up: three-policy W2-skip pooled budget replay", "",
        "F5_w2_skip_policies.csv compares P0 recorded, P1 skipping every W1-eligible W2, and P2 skipping only W1-eligible W2 with a known zero advisory count (None is not zero). Historical removed tokens and score changes cover recorded W2 tasks selected for skipping; replay freed tokens count only P0 allocations to those W2 tasks. Extra completions count newly completed task IDs relative to P0, excluding skipped W2.",
        "",
        "For every skipped W2 with recorded positive W3 tokens, bounds include min/max(W1 score, W3 score), even if W3 is not admitted at the cutoff. Interrupted-wave [0,1] uncertainty is retained. These are offline endpoint envelopes, not measured counterfactual results; recorded downstream token costs remain scenario assumptions. All120 pools both splits in the existing serial shard/wave order.",
        "", "| Run | Split | Budget/task | Policy | Replay W2 freed | Extra W2 / W3 | Counterfactual W3 | Score bounds | Delta bounds vs P0 |",
        "|---|---|---:|---|---:|---:|---:|---|---|",
    ]
    for row in tables["F5_w2_skip_policies"]:
        lines.append(
            f"| {row['run_id']} | {row['split']} | {row['budget_per_task']} | {row['policy']} | {row['replay_w2_tokens_freed']} | {row['extra_w2_tasks_completed']} / {row['extra_w3_tasks_completed']} | {row['counterfactual_w3_tasks']} | [{row['lower_score']:.3f}, {row['upper_score']:.3f}] | [{row['delta_lower_vs_P0']:+.3f}, {row['delta_upper_vs_P0']:+.3f}] |"
        )
    lines += [
        "", "## D5. Non-oracle two-slot pairing and sequential W1 routing", "",
        "Selection sees only saved grids and eligibility; the sequential gate also uses known-zero W1 advisory count. Gold is used only after selection for scoring. If exactly one run is eligible, both its slots are retained; otherwise A's first slot is followed by the first distinct nonempty grid from B1, A2, B2. No cleaner-first ranking is added to the specified rule.",
        "",
        "Scores are sums of per-task query-average hits (divide by 48 for accuracy). Oracle union may use four grids per query. Sequential results replay complete recorded W1 endpoints and costs, not a new 52k/58k cap or a live pooled execution. Missing required W1 receipts produce blank sequential totals.",
        "",
        "| A | B | A full | B full | Oracle union | Full two-slot | Sequential W1 | Sequential tokens | Tokens/task |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in tables["D5_non_oracle_pairing"]:
        values = []
        for key in ("score_A_full", "score_B_full", "oracle_union_score", "full_2slot_non_oracle_score", "seq_w1_A_then_B_score", "seq_w1_A_then_B_tokens", "seq_w1_A_then_B_tokens_per_task"):
            value = row[key]
            values.append(f"{value:,.3f}" if value is not None else "unavailable")
        lines.append(f"| {row['A']} | {row['B']} | " + " | ".join(values) + " |")
    lines += [
        "", "## Claude Round 2: exact offline additions", "",
        "D2_unconditional_curve.csv now includes 52,000 and 58,000 thresholds for every available public48/eval72 run. The w1_tokens_saved_cap_52000 and w1_tokens_saved_cap_58000 columns sum max(0, W1 tokens - cap) over observed receipts, including failures; observed_tasks exposes partial coverage. Savings are arithmetic spend reductions, not evidence that capped answers retain their scores.",
        "",
        "C4_w2_yield_on_w1_failures.csv measures only W1-ineligible tasks that actually ran W2. Eligibility and positive score gains use the saved W2 endpoint, excluding later waves. Tokens per point uses all W2 tokens in this subset divided by summed positive task-score gains; a zero denominator is blank.",
        "",
        "D5 adds seq_78k_A_w1_B_w1_B_w2 with a 3,744,000-token public48 pool: all A W1 first, then B W1 for tasks not clean-eligible under A, then recorded B W2 for tasks still not clean-eligible under B. Each stage follows B manifest order, concatenating shards in source order. A clean stop requires eligibility and known-zero advisory count. Interrupted endpoints consume the remaining pool and retain the prior candidate; absent W2 receipts provide no continuation. Completed W2 candidates merge with A W1 through the same non-oracle selector. These are saved-endpoint lower bounds, not live results.",
        "",
        "| A | B | 78k score lower | Tokens used | B W1 completed | B W2 completed |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in tables["D5_non_oracle_pairing"]:
        if "seq_78k_score_lower" in row:
            lines.append(
                f"| {row['A']} | {row['B']} | {row['seq_78k_score_lower']:.3f} | {row['seq_78k_tokens_used']:,} | {row['seq_78k_b_w1_completed']} | {row['seq_78k_b_w2_completed']} |"
            )
    lines.extend(["", "## Table index", ""])
    for name in tables:
        lines.append(f"- [{name}.csv]({name}.csv) ({len(tables[name])} rows)")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
