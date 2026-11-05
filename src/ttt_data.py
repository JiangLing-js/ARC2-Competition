"""Construct leave-one-out test-time-training samples for ARC tasks."""

from __future__ import annotations

import copy
import json
from itertools import islice
from pathlib import Path
from typing import Any, Iterable

from .transforms import TRANSFORMS, apply_transform


Grid = list[list[int]]
'''
ARC 的训练/测试样本里，每个 样本对 包含 "input" 和 "output" 两个 grid。
ArcPair 就是这样的字典：
'''
ArcPair = dict[str, Grid]
ArcTask = dict[str, list[ArcPair]]
'''Any 表示内容可能是任意类型'''
TttSample = dict[str, Any]


def load_arc_task(path: str | Path) -> Any:
    """Load an ARC json file.

    The returned value is the parsed json. For single-task files this is usually
    a dict with ``train`` and ``test`` keys. ARC-AGI bundle files may instead be
    a dict mapping task ids to task dicts; use ``iter_arc_tasks`` to iterate
    those files uniformly.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _is_task(obj: Any) -> bool:
    return isinstance(obj, dict) and isinstance(obj.get("train"), list) and isinstance(obj.get("test"), list)


def iter_arc_tasks(problem_dir: str | Path) -> Iterable[tuple[str, ArcTask]]:
    """Yield ``(task_id, task)`` pairs from all json files in ``problem_dir``.

    Supports both single-task json files and ARC-AGI bundle files whose top-level
    object maps task ids to task dicts. Files are yielded in lexicographic order.
    """
    for path in sorted(Path(problem_dir).glob("*.json")):
        data = load_arc_task(path)
        if _is_task(data):
            '''stem从文件路径 task_file 里取出不带后缀的文件名'''
            yield path.stem, data
        elif isinstance(data, dict):
            for task_id, task in data.items():
                if _is_task(task):
                    yield str(task_id), task


def build_leave_one_out_samples(task: ArcTask) -> list[TttSample]:
    """Build unaugmented leave-one-out samples from a task's training pairs.

    Each training pair becomes the held-out query/target. All other training
    pairs become context. True ARC test outputs are intentionally ignored.
    """
    return build_augmented_dttt(task, transform_names=["identity"])


def _transform_pair(pair: ArcPair, transform_name: str) -> ArcPair:
    return {
        "input": apply_transform(pair["input"], transform_name),
        "output": apply_transform(pair["output"], transform_name),
    }


# leave-one-out samples
def _make_sample(task: ArcTask, heldout_index: int, transform_name: str, reverse_context: bool = False) -> TttSample:
    train_pairs = task.get("train", [])
    heldout = train_pairs[heldout_index]
    context_pairs = [
        _transform_pair(pair, transform_name)
        for idx, pair in enumerate(train_pairs)
        if idx != heldout_index
    ]
    if reverse_context:
        context_pairs = context_pairs[::-1]

    return {
        "context_pairs": context_pairs,
        "query": apply_transform(heldout["input"], transform_name),
        "target": apply_transform(heldout["output"], transform_name),
        "heldout_index": heldout_index,
        "transform": transform_name,
    }


def build_augmented_dttt(
    task: ArcTask,
    transform_names: list[str] | None = None,
    include_permutations: bool = False,
    max_samples: int | None = None,
) -> list[TttSample]:
    """Build augmented leave-one-out D_TTT samples.

    Args:
        task: ARC task dict containing ``train`` and ``test`` lists.
        transform_names: Transform names applied to context, query, and target.
            Defaults to all transforms in ``src.transforms``.
        include_permutations: If true, add a simple reversed-context variant for
            samples whose context contains more than one pair.
        max_samples: Optional cap on the number of returned samples.

    Returns:
        A list of sample dicts with ``context_pairs``, ``query``, ``target``,
        ``heldout_index``, and ``transform`` fields.
    """
    names = transform_names or list(TRANSFORMS)
    samples: list[TttSample] = []

    for transform_name in names:
        if transform_name not in TRANSFORMS:
            raise ValueError(f"Unknown transform: {transform_name}")

    train_pairs = task.get("train", [])
    for transform_name in names:
        for heldout_index in range(len(train_pairs)):
            samples.append(_make_sample(task, heldout_index, transform_name))
            if max_samples is not None and len(samples) >= max_samples:
                return samples

            if include_permutations and len(train_pairs) > 2:
                samples.append(_make_sample(task, heldout_index, transform_name, reverse_context=True))
                if max_samples is not None and len(samples) >= max_samples:
                    return samples

    return list(islice(samples, max_samples)) if max_samples is not None else samples


def clone_task(task: ArcTask) -> ArcTask:
    """Return a deep copy of an ARC task."""
    return copy.deepcopy(task)
