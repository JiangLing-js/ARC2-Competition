"""ARC-AGI challenge and solution JSON loading helpers."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

from .grid_utils import validate_rectangular_grid


def load_arc_challenges(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load an ARC-AGI challenges file as ``task_id -> task``."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("ARC challenges top-level JSON must be a dict.")
    for task_id, task in data.items():
        if not isinstance(task, dict) or "train" not in task or "test" not in task:
            raise ValueError(f"Challenge task {task_id!r} must contain train and test.")
    return data


def load_arc_solutions(path: str | Path) -> dict[str, list[list[list[int]]]]:
    """Load an ARC-AGI solutions file as ``task_id -> output grids``."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("ARC solutions top-level JSON must be a dict.")
    for task_id, outputs in data.items():
        if not isinstance(outputs, list):
            raise ValueError(f"Solutions for task {task_id!r} must be a list.")
    return data


def attach_solutions_to_challenges(
    challenges: dict[str, dict[str, Any]],
    solutions: dict[str, list[list[list[int]]]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return a deep-copied challenge dict with optional test outputs attached."""
    tasks = copy.deepcopy(challenges)
    if solutions is None:
        return tasks

    for task_id, task in tasks.items():
        if task_id not in solutions:
            continue
        task_solutions = solutions[task_id]
        if len(task_solutions) != len(task.get("test", [])):
            raise ValueError(
                f"Task {task_id!r} has {len(task.get('test', []))} test inputs but "
                f"{len(task_solutions)} solutions."
            )
        for test_item, output_grid in zip(task["test"], task_solutions):
            test_item["output"] = copy.deepcopy(output_grid)
    return tasks


def split_task_ids(
    task_ids: list[str],
    val_fraction: float = 0.1,
    seed: int = 0,
    max_train_tasks: int | None = None,
    max_val_tasks: int | None = None,
) -> tuple[list[str], list[str]]:
    """Deterministically split task ids into train and validation ids."""
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1).")
    shuffled = list(task_ids)
    random.Random(seed).shuffle(shuffled)
    val_count = int(len(shuffled) * val_fraction)
    val_ids = shuffled[:val_count]
    train_ids = shuffled[val_count:]
    if max_train_tasks is not None:
        train_ids = train_ids[:max_train_tasks]
    if max_val_tasks is not None:
        val_ids = val_ids[:max_val_tasks]
    if set(train_ids) & set(val_ids):
        raise ValueError("Train and validation task ids overlap.")
    return train_ids, val_ids


def grid_shape(grid: list[list[int]]) -> tuple[int, int]:
    """Validate an ARC grid and return ``(height, width)``."""
    normalized = validate_rectangular_grid(grid)
    return len(normalized), len(normalized[0])
