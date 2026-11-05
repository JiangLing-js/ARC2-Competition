"""CPU-only protocol, grouping and self-scored pooled KGMon selection."""
from __future__ import annotations

from collections import defaultdict
import math

from selection_helpers import grid_key

PROTOCOL = "nvarc_equal_budget_v1"


def pass_configs(arm, started_at, work_deadline):
    if arm not in {"baseline", "split"} or work_deadline <= started_at:
        raise ValueError("Invalid arm or work interval")
    common = {"min_probability": .2, "score_views": 8, "learning_rate": 5e-5,
              "scoring_reserve_seconds": 60.0}
    first = {**common, "pass": "baseline" if arm == "baseline" else "split_a",
             "training_rows": 128 if arm == "baseline" else 64,
             "generation_views": 16 if arm == "baseline" else 8,
             "initialization_seed": 42, "training_seed": 42,
             "augmentation_seed": 1, "generation_seed": 2,
             "deadline": work_deadline if arm == "baseline" else (started_at + work_deadline) / 2}
    configs = [first] if arm == "baseline" else [first, {**common, "pass": "split_b", "training_rows": 64, "generation_views": 8,
                    "initialization_seed": 137, "training_seed": 137,
                    "augmentation_seed": 17, "generation_seed": 29, "deadline": work_deadline}]
    for config in configs:
        config["generation_deadline"] = config["deadline"] - config["scoring_reserve_seconds"]
    return configs


def orientation_batches(keys, query_ids, expected_views):
    mapping = defaultdict(list)
    for key in sorted(keys):
        query_id = int(key.split(".")[0].rsplit("_", 1)[1])
        if query_id in query_ids:
            mapping[query_id].append(key)
    if set(mapping) != set(query_ids):
        raise ValueError("Missing query augmentation group")
    if any(len(group) != expected_views for group in mapping.values()):
        raise ValueError(f"Expected {expected_views} views per query")
    if expected_views == 16:
        # Exact original non-transpose groups, then transpose groups.
        return [[key for offset in offsets for key in group[offset:offset + 2]]
                for phase in (((0, 4), (2, 6)), ((8, 12), (10, 14)))
                for group in mapping.values() for offsets in phase]
    if expected_views == 8:
        return [[group[index] for index in indexes]
                for group in mapping.values() for indexes in ((0, 2, 5, 7), (1, 3, 4, 6))]
    raise ValueError("Only 8-view and 16-view B4 protocols are supported")


def rank_kgmon(records):
    groups, seen = {}, set()
    for record in records:
        key = grid_key(record["solution"])
        identity = record["record_id"]
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError("Missing or duplicate candidate record identity")
        seen.add(identity)
        scores = record["score_aug"]
        if len(scores) != 8 or any(isinstance(x, bool) or not isinstance(x, (float, int))
                                  or not math.isfinite(x) for x in scores):
            raise ValueError("Candidate needs eight finite self scores")
        groups.setdefault(key, []).append(record)
    ranked = []
    for key, rows in groups.items():
        mean = sum(sum(row["score_aug"]) / 8 for row in rows) / len(rows)
        ranked.append({"grid": [list(r) for r in key], "count": len(rows),
                       "mean_self_nll": mean, "kgmon": len(rows) - mean,
                       "source_passes": sorted({r["pass"] for r in rows}),
                       "source_record_ids": [r["record_id"] for r in rows]})
    # Stable ordering deliberately matches source KGMon's first-observed tie break.
    return sorted(ranked, key=lambda row: row["kgmon"], reverse=True)


def choose_pooled(arm, first_records, second_records):
    first_ranked = rank_kgmon(first_records)
    second_ranked = rank_kgmon(second_records)
    pooled = rank_kgmon([*first_records, *second_records])
    first_top2 = [r["grid"] for r in first_ranked[:2]]
    if arm == "baseline" and second_records:
        raise ValueError("Baseline cannot contain a second adapter's records")
    if arm not in {"baseline", "split"}:
        raise ValueError("Unknown arm")
    selected = first_top2[:1] if first_top2 else []
    for row in pooled:
        if row["grid"] not in selected:
            selected.append(row["grid"])
        if len(selected) == 2:
            break
    selected += [[[0]]] * (2 - len(selected))
    return {"selected_top2": selected,
            "pooled": {"ranked": pooled, "unprotected_top2": [r["grid"] for r in pooled[:2]]},
            "selection_audit": {
                "rule": "original_kgmon" if arm == "baseline" else "pooled_self_kgmon_lock_split_a_top1",
                "top1_locked": bool(first_top2),
                "lock_scope": "full_control" if arm == "baseline" else "split_A_not_full_control",
                "common_calibration": False,
                "a_second_discarded": len(first_top2) == 2 and first_top2[1] not in selected,
                "second_pass_has_candidates": bool(second_ranked),
                "incomplete_pass_candidates_eligible": True}}
