"""Non-destructive preparation of clean, normalized ARC-AGI datasets.

The ARC-AGI-1 repository stores one JSON object per task and includes the
official test output inside that object.  The training code in this repository
uses aggregate ``task_id -> challenge`` files instead.  This module converts
the former to the latter while physically separating solutions, audits overlap
with the locally available ARC-AGI-2 aggregate files, and creates frozen
challenge-only manifests.

No source file is moved, modified, or deleted.  Every generated file is
write-once: rerunning with identical inputs is a no-op, while a different
payload at the same path is rejected.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .arc_agi_io import load_arc_challenges, load_arc_solutions
from .grid_utils import validate_rectangular_grid
from .vit5_clean_task_split import (
    SPLIT_SCHEMA_VERSION,
    create_split_manifest,
    save_frozen_split_manifest,
    sha256_file,
    split_manifest_sha256,
    validate_split_manifest,
)


PREPARATION_SCHEMA_VERSION = 1


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_train_payload(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return train pairs in a canonical order independent of demo ordering."""
    train = task.get("train")
    if not isinstance(train, list) or not train:
        raise ValueError("An ARC task must contain at least one train pair.")
    rendered_pairs: list[tuple[bytes, dict[str, Any]]] = []
    for pair in train:
        if not isinstance(pair, Mapping) or "input" not in pair or "output" not in pair:
            raise ValueError("Every ARC train pair must contain input and output grids.")
        normalized = {
            "input": copy.deepcopy(pair["input"]),
            "output": copy.deepcopy(pair["output"]),
        }
        _validate_grid(normalized["input"])
        _validate_grid(normalized["output"])
        rendered_pairs.append((_canonical_json_bytes(normalized), normalized))
    rendered_pairs.sort(key=lambda item: item[0])
    return [pair for _, pair in rendered_pairs]


def canonical_train_sha256(task: Mapping[str, Any]) -> str:
    """Hash only ``task.train``, invariant to demonstration ordering."""
    return hashlib.sha256(_canonical_json_bytes(canonical_train_payload(task))).hexdigest()


def train_pairs_same_shape(task: Mapping[str, Any]) -> bool:
    """Whether every visible train input/output pair has the same rectangle."""
    train = task.get("train")
    if not isinstance(train, list) or not train:
        raise ValueError("An ARC task must contain at least one train pair.")
    for pair in train:
        if _grid_shape(pair["input"]) != _grid_shape(pair["output"]):
            return False
    return True


def _validate_grid(grid: Any) -> list[list[int]]:
    normalized = validate_rectangular_grid(grid)
    for row in normalized:
        for value in row:
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 9:
                raise ValueError("ARC grid cells must be integer colors in [0, 9].")
    return normalized


def _grid_shape(grid: Any) -> tuple[int, int]:
    normalized = _validate_grid(grid)
    return len(normalized), len(normalized[0])


def load_arc1_task_directory(
    directory: str | Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[list[list[int]]]]]:
    """Aggregate ARC-AGI-1 per-task files and separate their test solutions."""
    source = Path(directory)
    if not source.is_dir():
        raise FileNotFoundError(f"ARC-AGI-1 task directory does not exist: {source}")
    task_paths = sorted(source.glob("*.json"))
    if not task_paths:
        raise ValueError(f"ARC-AGI-1 task directory contains no JSON files: {source}")

    challenges: dict[str, dict[str, Any]] = {}
    solutions: dict[str, list[list[list[int]]]] = {}
    for task_path in task_paths:
        task_id = task_path.stem
        if task_id in challenges:
            raise ValueError(f"Duplicate ARC-AGI-1 task ID: {task_id}")
        with task_path.open("r", encoding="utf-8") as handle:
            task = json.load(handle)
        if not isinstance(task, Mapping):
            raise ValueError(f"ARC task {task_id!r} must be a JSON object.")
        train = task.get("train")
        test = task.get("test")
        if not isinstance(train, list) or not train:
            raise ValueError(f"ARC task {task_id!r} must contain train pairs.")
        if not isinstance(test, list) or not test:
            raise ValueError(f"ARC task {task_id!r} must contain test pairs.")

        normalized_train: list[dict[str, Any]] = []
        for pair in train:
            if not isinstance(pair, Mapping) or "input" not in pair or "output" not in pair:
                raise ValueError(f"ARC task {task_id!r} has an invalid train pair.")
            _validate_grid(pair["input"])
            _validate_grid(pair["output"])
            normalized_train.append(
                {"input": copy.deepcopy(pair["input"]), "output": copy.deepcopy(pair["output"])}
            )

        challenge_test: list[dict[str, Any]] = []
        task_solutions: list[list[list[int]]] = []
        for pair in test:
            if not isinstance(pair, Mapping) or "input" not in pair or "output" not in pair:
                raise ValueError(
                    f"ARC-AGI-1 source task {task_id!r} must include each test output "
                    "so it can be separated safely."
                )
            _validate_grid(pair["input"])
            _validate_grid(pair["output"])
            challenge_test.append({"input": copy.deepcopy(pair["input"])})
            task_solutions.append(copy.deepcopy(pair["output"]))

        challenges[task_id] = {"train": normalized_train, "test": challenge_test}
        solutions[task_id] = task_solutions
    return challenges, solutions


def _validate_challenge_solutions(
    challenges: Mapping[str, Mapping[str, Any]],
    solutions: Mapping[str, Sequence[Any]],
    label: str,
) -> None:
    if set(challenges) != set(solutions):
        missing = sorted(set(challenges) - set(solutions))
        extra = sorted(set(solutions) - set(challenges))
        raise ValueError(
            f"{label} challenge/solution task IDs differ; missing={missing[:5]}, "
            f"extra={extra[:5]}."
        )
    for task_id, task in challenges.items():
        test = task.get("test", [])
        if len(test) != len(solutions[task_id]):
            raise ValueError(
                f"{label} task {task_id!r} has {len(test)} test inputs but "
                f"{len(solutions[task_id])} solutions."
            )
        for item in test:
            if "output" in item:
                raise ValueError(f"Challenge-only task {task_id!r} leaks a test output.")


def _render_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _write_frozen_text(path: str | Path, rendered: str) -> None:
    """Atomically create ``path`` or accept an identical existing file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(
                f"Refusing to replace non-identical prepared dataset file: {target}. "
                "Use a new output directory."
            )
        return

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # Linking a complete temporary file makes target creation atomic and,
            # unlike os.replace(), can never overwrite an existing frozen file.
            os.link(temporary, target)
        except FileExistsError:
            if target.read_text(encoding="utf-8") != rendered:
                raise FileExistsError(
                    f"Refusing to replace concurrently-created dataset file: {target}."
                )
    finally:
        temporary.unlink(missing_ok=True)


def _write_frozen_json(path: str | Path, value: Any) -> None:
    _write_frozen_text(path, _render_json(value))


def _directory_sha256(path: str | Path) -> str:
    root = Path(path)
    digest = hashlib.sha256()
    files = sorted(item for item in root.rglob("*") if item.is_file())
    for item in files:
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def _overlap_report(
    reference: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    reference_ids = set(reference)
    reference_hash_to_ids: dict[str, list[str]] = {}
    for task_id, task in reference.items():
        reference_hash_to_ids.setdefault(canonical_train_sha256(task), []).append(task_id)

    id_overlap = sorted(reference_ids & set(candidate))
    content_matches: list[dict[str, Any]] = []
    for task_id, task in sorted(candidate.items()):
        digest = canonical_train_sha256(task)
        matching = sorted(reference_hash_to_ids.get(digest, []))
        if matching:
            content_matches.append(
                {
                    "candidate_task_id": task_id,
                    "reference_task_ids": matching,
                    "canonical_train_sha256": digest,
                }
            )
    return {
        "reference_task_count": len(reference),
        "candidate_task_count": len(candidate),
        "task_id_overlap_count": len(id_overlap),
        "task_id_overlap": id_overlap,
        "train_content_overlap_count": len(content_matches),
        "train_content_matches": content_matches,
    }


def _build_clean_global_pool(
    arc1_training: Mapping[str, Mapping[str, Any]],
    arc2_training: Mapping[str, Mapping[str, Any]],
    arc1_evaluation: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    evaluation_ids = set(arc1_evaluation)
    evaluation_hashes = {
        canonical_train_sha256(task) for task in arc1_evaluation.values()
    }
    pool: dict[str, dict[str, Any]] = {}
    seen_hash_to_pool_id: dict[str, str] = {}
    provenance: dict[str, dict[str, str]] = {}
    excluded: list[dict[str, str]] = []
    renamed: list[dict[str, str]] = []

    for source_name, source in (
        ("arc_agi_1_training", arc1_training),
        ("arc_agi_2_training", arc2_training),
    ):
        for source_task_id, task in sorted(source.items()):
            digest = canonical_train_sha256(task)
            if source_task_id in evaluation_ids:
                excluded.append(
                    {
                        "source": source_name,
                        "source_task_id": source_task_id,
                        "reason": "arc_agi_1_evaluation_task_id_overlap",
                        "canonical_train_sha256": digest,
                    }
                )
                continue
            if digest in evaluation_hashes:
                excluded.append(
                    {
                        "source": source_name,
                        "source_task_id": source_task_id,
                        "reason": "arc_agi_1_evaluation_train_content_overlap",
                        "canonical_train_sha256": digest,
                    }
                )
                continue
            if digest in seen_hash_to_pool_id:
                excluded.append(
                    {
                        "source": source_name,
                        "source_task_id": source_task_id,
                        "reason": "duplicate_train_content_in_clean_pool",
                        "canonical_train_sha256": digest,
                        "kept_pool_task_id": seen_hash_to_pool_id[digest],
                    }
                )
                continue

            pool_task_id = source_task_id
            if pool_task_id in pool:
                base = f"arc2__{source_task_id}"
                pool_task_id = base
                index = 2
                while pool_task_id in pool:
                    pool_task_id = f"{base}__{index}"
                    index += 1
                renamed.append(
                    {
                        "source": source_name,
                        "source_task_id": source_task_id,
                        "pool_task_id": pool_task_id,
                    }
                )
            pool[pool_task_id] = copy.deepcopy(dict(task))
            seen_hash_to_pool_id[digest] = pool_task_id
            provenance[pool_task_id] = {
                "source": source_name,
                "source_task_id": source_task_id,
                "canonical_train_sha256": digest,
            }

    exclusion_counts = Counter(item["reason"] for item in excluded)
    manifest = {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "selection_source": "challenge_json_only",
        "solutions_used_for_selection": False,
        "arc_agi_1_evaluation_excluded_by_task_id_or_train_content": True,
        "source_task_count": {
            "arc_agi_1_training": len(arc1_training),
            "arc_agi_2_training": len(arc2_training),
        },
        "clean_pool_task_count": len(pool),
        "excluded_count_by_reason": dict(sorted(exclusion_counts.items())),
        "excluded": excluded,
        "renamed_for_task_id_collision": renamed,
        "task_provenance": provenance,
    }
    return pool, manifest


def _same_shape_ids(tasks: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return sorted(task_id for task_id, task in tasks.items() if train_pairs_same_shape(task))


def qualification_split_counts(task_count: int) -> tuple[int, int, int]:
    """Return a deterministic approximately 80/10/10 three-way split."""
    if task_count < 3:
        raise ValueError("A three-way qualification split requires at least three tasks.")
    global_train = max(1, round(task_count * 0.8))
    development = max(1, round(task_count * 0.1))
    final_heldout = task_count - global_train - development
    if final_heldout < 1:
        global_train -= 1 - final_heldout
        final_heldout = 1
    return global_train, development, final_heldout


def _file_record(path: Path, base: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(base).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def prepare_arc_datasets(
    arc1_data_dir: str | Path,
    arc2_problems_dir: str | Path,
    output_dir: str | Path,
    *,
    split_seed: int = 0,
    global_train_count: int = 320,
    development_count: int = 40,
    final_heldout_count: int = 40,
) -> dict[str, Any]:
    """Prepare normalized datasets, overlap audits, and frozen split manifests."""
    arc1_root = Path(arc1_data_dir).resolve()
    arc2_root = Path(arc2_problems_dir).resolve()
    output = Path(output_dir).resolve()

    arc1_training, arc1_training_solutions = load_arc1_task_directory(
        arc1_root / "training"
    )
    arc1_evaluation, arc1_evaluation_solutions = load_arc1_task_directory(
        arc1_root / "evaluation"
    )
    _validate_challenge_solutions(
        arc1_training, arc1_training_solutions, "ARC-AGI-1 training"
    )
    _validate_challenge_solutions(
        arc1_evaluation, arc1_evaluation_solutions, "ARC-AGI-1 evaluation"
    )

    arc2_challenge_paths = {
        split: arc2_root / f"arc-agi_{split}_challenges.json"
        for split in ("training", "evaluation", "test")
    }
    for split, path in arc2_challenge_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing ARC-AGI-2 {split} challenges: {path}")
    arc2_challenges = {
        split: load_arc_challenges(path) for split, path in arc2_challenge_paths.items()
    }
    arc2_solutions: dict[str, dict[str, Any]] = {}
    for split in ("training", "evaluation"):
        path = arc2_root / f"arc-agi_{split}_solutions.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing ARC-AGI-2 {split} solutions: {path}")
        arc2_solutions[split] = load_arc_solutions(path)
        _validate_challenge_solutions(
            arc2_challenges[split], arc2_solutions[split], f"ARC-AGI-2 {split}"
        )

    normalized = output / "normalized"
    sealed = output / "sealed"
    manifests = output / "manifests"
    clean = output / "clean"
    paths: dict[str, Path] = {
        "arc1_training_challenges": normalized
        / "arc_agi_1"
        / "training_challenges.json",
        "arc1_evaluation_challenges": normalized
        / "arc_agi_1"
        / "evaluation_challenges.json",
        "arc1_training_solutions": sealed / "arc_agi_1" / "training_solutions.json",
        "arc1_evaluation_solutions": sealed
        / "arc_agi_1"
        / "evaluation_solutions.json",
    }
    _write_frozen_json(paths["arc1_training_challenges"], arc1_training)
    _write_frozen_json(paths["arc1_evaluation_challenges"], arc1_evaluation)
    _write_frozen_json(paths["arc1_training_solutions"], arc1_training_solutions)
    _write_frozen_json(paths["arc1_evaluation_solutions"], arc1_evaluation_solutions)

    for split, challenges in arc2_challenges.items():
        key = f"arc2_{split}_challenges"
        paths[key] = normalized / "arc_agi_2" / f"{split}_challenges.json"
        _write_frozen_json(paths[key], challenges)
    for split, solutions in arc2_solutions.items():
        key = f"arc2_{split}_solutions"
        paths[key] = sealed / "arc_agi_2" / f"{split}_solutions.json"
        _write_frozen_json(paths[key], solutions)

    overlap_audit = {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "reference": "arc_agi_1_evaluation",
        "comparison_semantics": {
            "task_id": "exact string equality",
            "train_content": "SHA256 of train input/output pairs after demo-order canonicalization",
            "test_inputs_used": False,
            "solutions_used": False,
        },
        "arc_agi_1_training": _overlap_report(arc1_evaluation, arc1_training),
        "arc_agi_2_training": _overlap_report(
            arc1_evaluation, arc2_challenges["training"]
        ),
        "arc_agi_2_evaluation": _overlap_report(
            arc1_evaluation, arc2_challenges["evaluation"]
        ),
        "arc_agi_2_test": _overlap_report(arc1_evaluation, arc2_challenges["test"]),
    }
    paths["overlap_audit"] = manifests / "arc_agi_1_evaluation_overlap_audit.json"
    _write_frozen_json(paths["overlap_audit"], overlap_audit)

    clean_pool, pool_manifest = _build_clean_global_pool(
        arc1_training, arc2_challenges["training"], arc1_evaluation
    )
    paths["clean_global_pool"] = clean / "arc_agi_1_eval_clean_global_challenges.json"
    paths["clean_global_pool_manifest"] = (
        manifests / "arc_agi_1_eval_clean_global_pool_manifest.json"
    )
    _write_frozen_json(paths["clean_global_pool"], clean_pool)
    _write_frozen_json(paths["clean_global_pool_manifest"], pool_manifest)

    split_path = manifests / "arc_agi_1_training_320_40_40_split.json"
    split_manifest = create_split_manifest(
        arc1_training,
        paths["arc1_training_challenges"],
        global_train_count,
        development_count,
        final_heldout_count,
        seed=split_seed,
    )
    save_frozen_split_manifest(split_path, split_manifest)
    paths["arc1_training_split"] = split_path

    evaluation_hashes = {
        canonical_train_sha256(task) for task in arc1_evaluation.values()
    }
    quarantined_by_split: dict[str, list[str]] = {}
    eligible_by_split: dict[str, list[str]] = {}
    for split_name in (
        "global_train_task_ids",
        "development_task_ids",
        "final_heldout_task_ids",
    ):
        quarantined = sorted(
            task_id
            for task_id in split_manifest[split_name]
            if task_id in arc1_evaluation
            or canonical_train_sha256(arc1_training[task_id]) in evaluation_hashes
        )
        quarantined_by_split[split_name] = quarantined
        eligible_by_split[split_name] = sorted(
            set(split_manifest[split_name]) - set(quarantined)
        )
    clean_split_view = {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "base_split_manifest": split_path.name,
        "base_split_manifest_sha256": sha256_file(split_path),
        "selection_source": "challenge_json_only",
        "solutions_used_for_selection": False,
        "official_evaluation_overlap_quarantined": True,
        "eligible_task_ids_by_split": eligible_by_split,
        "eligible_task_count_by_split": {
            name: len(values) for name, values in eligible_by_split.items()
        },
        "quarantined_task_ids_by_split": quarantined_by_split,
        "quarantined_task_count": sum(map(len, quarantined_by_split.values())),
    }
    paths["arc1_training_clean_split_view"] = (
        manifests / "arc_agi_1_training_320_40_40_clean_view.json"
    )
    _write_frozen_json(paths["arc1_training_clean_split_view"], clean_split_view)

    # Produce a directly runnable qualification dataset.  Membership is a pure
    # filtering of the frozen 320/40/40 split: no task is redistributed between
    # splits after quarantine or same-shape qualification.
    same_shape_arc1_training_ids = set(_same_shape_ids(arc1_training))
    runnable_ids_by_split = {
        name: [
            task_id
            for task_id in split_manifest[name]
            if task_id in set(eligible_by_split[name])
            and task_id in same_shape_arc1_training_ids
        ]
        for name in (
            "global_train_task_ids",
            "development_task_ids",
            "final_heldout_task_ids",
        )
    }
    runnable_task_ids = set().union(
        *(set(values) for values in runnable_ids_by_split.values())
    )
    runnable_challenges = {
        task_id: copy.deepcopy(arc1_training[task_id])
        for task_id in sorted(runnable_task_ids)
    }
    paths["arc1_training_same_shape_challenges"] = (
        clean / "arc_agi_1_training_same_shape_challenges.json"
    )
    _write_frozen_json(
        paths["arc1_training_same_shape_challenges"], runnable_challenges
    )
    runnable_split_manifest: dict[str, Any] = {
        "split_schema_version": SPLIT_SCHEMA_VERSION,
        "selection_source": "challenge_json_only",
        "solutions_used_for_selection": False,
        "seed": int(split_seed),
        **runnable_ids_by_split,
        "challenges_file_sha256": sha256_file(
            paths["arc1_training_same_shape_challenges"]
        ),
        "task_count_by_split": {
            name: len(values) for name, values in runnable_ids_by_split.items()
        },
        "overlap_count": 0,
        "missing_task_count": 0,
        "unselected_task_count": 0,
        "derivation": "base_320_40_40_membership_then_eval_overlap_quarantine_then_same_shape_filter",
        "base_split_manifest_sha256": sha256_file(split_path),
        "test_output_shape_used_for_selection": False,
    }
    runnable_split_manifest["split_manifest_sha256"] = split_manifest_sha256(
        runnable_split_manifest
    )
    validate_split_manifest(
        runnable_split_manifest,
        available_task_ids=set(runnable_challenges),
        expected_challenges_sha256=runnable_split_manifest["challenges_file_sha256"],
    )
    paths["arc1_training_same_shape_derived_split"] = (
        manifests / "arc_agi_1_training_same_shape_derived_320_40_40_split.json"
    )
    save_frozen_split_manifest(
        paths["arc1_training_same_shape_derived_split"], runnable_split_manifest
    )

    qualification_counts = qualification_split_counts(len(runnable_challenges))
    qualification_split_manifest = create_split_manifest(
        runnable_challenges,
        paths["arc1_training_same_shape_challenges"],
        *qualification_counts,
        seed=split_seed,
    )
    qualification_split_manifest["protocol"] = (
        "same_shape_filter_first_then_approximately_80_10_10"
    )
    qualification_split_manifest["official_evaluation_overlap_quarantined"] = True
    qualification_split_manifest["split_manifest_sha256"] = split_manifest_sha256(
        qualification_split_manifest
    )
    validate_split_manifest(
        qualification_split_manifest,
        available_task_ids=set(runnable_challenges),
        expected_challenges_sha256=qualification_split_manifest[
            "challenges_file_sha256"
        ],
    )
    paths["arc1_training_same_shape_qualification_split"] = (
        manifests / "arc_agi_1_training_same_shape_qualification_split.json"
    )
    save_frozen_split_manifest(
        paths["arc1_training_same_shape_qualification_split"],
        qualification_split_manifest,
    )

    arc1_evaluation_same_shape_ids = _same_shape_ids(arc1_evaluation)
    arc1_evaluation_same_shape = {
        task_id: copy.deepcopy(arc1_evaluation[task_id])
        for task_id in arc1_evaluation_same_shape_ids
    }
    paths["arc1_evaluation_same_shape_challenges"] = (
        clean / "arc_agi_1_evaluation_same_shape_challenges.json"
    )
    _write_frozen_json(
        paths["arc1_evaluation_same_shape_challenges"], arc1_evaluation_same_shape
    )
    evaluation_same_shape_manifest = {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "selection_source": "challenge_json_only",
        "solutions_used_for_selection": False,
        "test_output_shape_used_for_selection": False,
        "definition": (
            "Every visible task.train pair has identical input and output "
            "height/width; no official test output is inspected."
        ),
        "challenge_file": paths[
            "arc1_evaluation_same_shape_challenges"
        ].relative_to(output).as_posix(),
        "challenges_file_sha256": sha256_file(
            paths["arc1_evaluation_same_shape_challenges"]
        ),
        "task_count": len(arc1_evaluation_same_shape),
        "task_ids": arc1_evaluation_same_shape_ids,
    }
    paths["arc1_evaluation_same_shape_manifest"] = (
        manifests / "arc_agi_1_evaluation_same_shape_manifest.json"
    )
    _write_frozen_json(
        paths["arc1_evaluation_same_shape_manifest"],
        evaluation_same_shape_manifest,
    )
    evaluation_final_split: dict[str, Any] = {
        "split_schema_version": SPLIT_SCHEMA_VERSION,
        "selection_source": "challenge_json_only",
        "solutions_used_for_selection": False,
        "seed": 0,
        "global_train_task_ids": [],
        "development_task_ids": [],
        "final_heldout_task_ids": arc1_evaluation_same_shape_ids,
        "challenges_file_sha256": sha256_file(
            paths["arc1_evaluation_same_shape_challenges"]
        ),
        "task_count_by_split": {
            "global_train_task_ids": 0,
            "development_task_ids": 0,
            "final_heldout_task_ids": len(arc1_evaluation_same_shape_ids),
        },
        "overlap_count": 0,
        "missing_task_count": 0,
        "unselected_task_count": 0,
        "protocol": "official_arc_agi_1_evaluation_same_shape_final_only",
        "test_output_shape_used_for_selection": False,
    }
    evaluation_final_split["split_manifest_sha256"] = split_manifest_sha256(
        evaluation_final_split
    )
    validate_split_manifest(
        evaluation_final_split,
        available_task_ids=set(arc1_evaluation_same_shape),
        expected_challenges_sha256=evaluation_final_split["challenges_file_sha256"],
    )
    paths["arc1_evaluation_same_shape_final_split"] = (
        manifests / "arc_agi_1_evaluation_same_shape_final_split.json"
    )
    save_frozen_split_manifest(
        paths["arc1_evaluation_same_shape_final_split"], evaluation_final_split
    )

    dataset_groups: dict[str, Mapping[str, Mapping[str, Any]]] = {
        "arc_agi_1_training": arc1_training,
        "arc_agi_1_evaluation": arc1_evaluation,
        "arc_agi_2_training": arc2_challenges["training"],
        "arc_agi_2_evaluation": arc2_challenges["evaluation"],
        "arc_agi_2_test": arc2_challenges["test"],
        "arc_agi_1_eval_clean_global_pool": clean_pool,
    }
    same_shape_manifest = {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "selection_source": "challenge_json_only",
        "solutions_used_for_selection": False,
        "definition": (
            "A task is included iff every visible task.train pair has identical "
            "input and output height/width. Test output shape is never inspected."
        ),
        "same_shape_task_ids_by_dataset": {
            name: _same_shape_ids(tasks) for name, tasks in dataset_groups.items()
        },
        "same_shape_task_count_by_dataset": {
            name: len(_same_shape_ids(tasks)) for name, tasks in dataset_groups.items()
        },
    }
    arc1_training_same_shape = set(
        same_shape_manifest["same_shape_task_ids_by_dataset"]["arc_agi_1_training"]
    )
    same_shape_manifest["same_shape_eligible_task_ids_by_split"] = {
        name: sorted(set(values) & arc1_training_same_shape)
        for name, values in eligible_by_split.items()
    }
    same_shape_manifest["same_shape_eligible_task_count_by_split"] = {
        name: len(values)
        for name, values in same_shape_manifest[
            "same_shape_eligible_task_ids_by_split"
        ].items()
    }
    paths["same_shape_manifest"] = manifests / "train_pair_same_shape_manifest.json"
    _write_frozen_json(paths["same_shape_manifest"], same_shape_manifest)

    source_catalog = {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "sources_are_read_only": True,
        "arc_agi_1_data_directory": str(arc1_root),
        "arc_agi_1_data_directory_sha256": _directory_sha256(arc1_root),
        "arc_agi_2_problems_directory": str(arc2_root),
        "arc_agi_2_source_files": {
            path.name: sha256_file(path)
            for path in sorted(arc2_root.glob("arc-agi_*.json"))
        },
    }
    paths["source_catalog"] = manifests / "source_catalog.json"
    _write_frozen_json(paths["source_catalog"], source_catalog)

    preparation_manifest = {
        "preparation_schema_version": PREPARATION_SCHEMA_VERSION,
        "non_destructive": True,
        "source_files_moved": False,
        "frozen_write_policy": "identical_rerun_only",
        "task_counts": {
            "arc_agi_1_training": len(arc1_training),
            "arc_agi_1_evaluation": len(arc1_evaluation),
            "arc_agi_2_training": len(arc2_challenges["training"]),
            "arc_agi_2_evaluation": len(arc2_challenges["evaluation"]),
            "arc_agi_2_test": len(arc2_challenges["test"]),
            "arc_agi_1_eval_clean_global_pool": len(clean_pool),
        },
        "generated_files": {
            key: _file_record(path, output) for key, path in sorted(paths.items())
        },
    }
    preparation_manifest["preparation_manifest_sha256"] = hashlib.sha256(
        _canonical_json_bytes(preparation_manifest)
    ).hexdigest()
    preparation_path = manifests / "preparation_manifest.json"
    _write_frozen_json(preparation_path, preparation_manifest)
    return preparation_manifest
