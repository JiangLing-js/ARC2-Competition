"""Gold-free, dependency-free selection rules for the frozen V2 pilot."""
from __future__ import annotations

import math
from numbers import Integral, Real


def grid_key(grid):
    if hasattr(grid, "tolist"):
        grid = grid.tolist()
    if not isinstance(grid, (list, tuple)) or not 1 <= len(grid) <= 30:
        raise ValueError("Grid must have 1..30 rows")
    result, width = [], None
    for row in grid:
        if not isinstance(row, (list, tuple)) or not 1 <= len(row) <= 30:
            raise ValueError("Grid must have 1..30 columns")
        width = len(row) if width is None else width
        if len(row) != width:
            raise ValueError("Ragged grid")
        if any(isinstance(x, bool) or not isinstance(x, Integral) or not 0 <= x <= 9 for x in row):
            raise ValueError("Grid colors must be integers 0..9")
        result.append(tuple(int(x) for x in row))
    return tuple(result)


def _grid(key):
    return [list(row) for row in key]


def _record_grid(record):
    return record.get("grid", record.get("solution"))


def uncertainty_gate(primary_records, top1_grid, completed_views, support_threshold=5, distinct_threshold=8):
    """Record the proposed exploratory gate; support counts distinct view IDs."""
    if isinstance(completed_views, bool) or not isinstance(completed_views, Integral) or completed_views < 0:
        raise ValueError("completed_views must be a nonnegative integer")
    top = None if top1_grid is None else grid_key(top1_grid)
    distinct, supporting = set(), set()
    for record in primary_records:
        key = grid_key(_record_grid(record))
        distinct.add(key)
        if key == top:
            if not record.get("view_id"):
                raise ValueError("view_id is required for unique-view support")
            supporting.add(str(record["view_id"]))
    support = len(supporting)
    reasons = []
    if top is None:
        reasons.append("no_primary_candidate")
    if support <= support_threshold:
        reasons.append("low_top1_view_support")
    if len(distinct) >= distinct_threshold:
        reasons.append("many_distinct_grids")
    return {"eligible": bool(reasons), "reasons": reasons,
            "top1_support_views": support, "distinct_grids": len(distinct),
            "completed_views": int(completed_views),
            "top1_support_fraction": support / completed_views if completed_views else None,
            "support_threshold": support_threshold, "distinct_threshold": distinct_threshold,
            "exploratory_thresholds": True}


def build_common_shortlist(primary_ranked_grids, retry_records, max_new=12):
    """Preserve the supplied primary shortlist, then append first-seen new grids.

    Caller supplies the primary shortlist (normally six, always including top2).
    A retry grid already in that shortlist does not consume the new-grid cap.
    Order is a deterministic tie break, favoring the original primary ranking.
    """
    if isinstance(max_new, bool) or not isinstance(max_new, Integral) or max_new < 0:
        raise ValueError("max_new must be a nonnegative integer")
    keys = dict.fromkeys(grid_key(grid) for grid in primary_ranked_grids)
    added = 0
    for record in retry_records:
        key = grid_key(_record_grid(record))
        if key not in keys and added < max_new:
            keys[key] = None
            added += 1
    return [_grid(key) for key in keys]


def select_locked_top1(primary_top2, common_scores, expected_views=8):
    """Use only one common adapter's complete, identical-view NLL scores.

    Runtime must guarantee the common adapter and view identities, not just the
    score count. Any incomplete/nonfinite/duplicate entry preserves the baseline.
    A nonempty primary top1 is locked. When primary is empty choose best two.
    """
    if not isinstance(expected_views, Integral) or isinstance(expected_views, bool) or expected_views < 1:
        raise ValueError("expected_views must be positive")
    primary = list(dict.fromkeys(grid_key(grid) for grid in primary_top2[:2]))
    baseline = [_grid(key) for key in primary]
    baseline += [[[0]]] * (2 - len(baseline))
    audit = {"rule": "locked_primary_top1_common_retry_mean_nll",
             "expected_views": expected_views, "common_scoring_complete": False,
             "top1_locked": bool(primary), "fallback_reason": None, "ranked": []}

    def fallback(reason):
        audit["fallback_reason"] = reason
        return {"attempt_1": baseline[0], "attempt_2": baseline[1], "audit": audit}

    if not common_scores:
        return fallback("no_common_scores")
    ranked, seen = [], set()
    for index, record in enumerate(common_scores):
        key = grid_key(_record_grid(record))
        scores = record.get("nlls")
        if key in seen:
            return fallback("duplicate_common_candidate")
        seen.add(key)
        if not isinstance(scores, (list, tuple)) or len(scores) != expected_views:
            return fallback("incomplete_common_scores")
        if any(isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(float(x)) for x in scores):
            return fallback("nonfinite_common_scores")
        nll = sum(float(x) for x in scores) / expected_views
        ranked.append((nll, index, key, [float(x) for x in scores]))
    if not set(primary).issubset(seen):
        return fallback("primary_top2_missing_from_common_scores")
    ranked.sort(key=lambda item: (item[0], item[1]))
    audit["common_scoring_complete"] = True
    audit["ranked"] = [{"grid": _grid(key), "mean_nll": nll, "nlls": scores}
                       for nll, _, key, scores in ranked]
    chosen = primary[:1]
    for _, _, key, _ in ranked:
        if key not in chosen:
            chosen.append(key)
        if len(chosen) == 2:
            break
    output = [_grid(key) for key in chosen]
    output += [[[0]]] * (2 - len(output))
    audit["changed_second_attempt"] = output[1] != baseline[1]
    return {"attempt_1": output[0], "attempt_2": output[1], "audit": audit}
