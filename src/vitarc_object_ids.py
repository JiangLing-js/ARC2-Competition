"""Object id extraction and batching utilities for future ViTARC OPE."""

from __future__ import annotations

from collections import deque
from typing import Any

import torch

from .grid_utils import validate_rectangular_grid
from .vitarc_position_ids import is_grid_token


Grid = list[list[int]]


def _connected_components(
    grid: Grid,
    *,
    excluded_color: int | None,
    connectivity: int,
) -> dict[str, Any]:
    """Assign deterministic IDs to same-color components in a real HxW grid."""
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8.")

    normalized = validate_rectangular_grid(grid)
    height, width = len(normalized), len(normalized[0])
    object_grid = [[0 for _ in range(width)] for _ in range(height)]
    visited = [[False for _ in range(width)] for _ in range(height)]
    if connectivity == 4:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    objects: list[dict[str, Any]] = []
    next_object_id = 1
    for row in range(height):
        for col in range(width):
            if visited[row][col]:
                continue
            visited[row][col] = True
            color = normalized[row][col]
            if excluded_color is not None and color == excluded_color:
                continue

            cells: list[tuple[int, int]] = []
            queue: deque[tuple[int, int]] = deque([(row, col)])
            object_grid[row][col] = next_object_id
            while queue:
                current_row, current_col = queue.popleft()
                cells.append((current_row, current_col))
                for delta_row, delta_col in directions:
                    next_row = current_row + delta_row
                    next_col = current_col + delta_col
                    if not (0 <= next_row < height and 0 <= next_col < width):
                        continue
                    if visited[next_row][next_col]:
                        continue
                    if normalized[next_row][next_col] != color:
                        continue
                    visited[next_row][next_col] = True
                    object_grid[next_row][next_col] = next_object_id
                    queue.append((next_row, next_col))

            min_row = min(cell[0] for cell in cells)
            min_col = min(cell[1] for cell in cells)
            max_row = max(cell[0] for cell in cells)
            max_col = max(cell[1] for cell in cells)
            objects.append(
                {
                    "object_id": next_object_id,
                    "color": color,
                    "size": len(cells),
                    "bbox": [min_row, min_col, max_row, max_col],
                    "cells": cells,
                }
            )
            next_object_id += 1

    return {
        "object_grid": object_grid,
        "num_objects": len(objects),
        "objects": objects,
    }


def connected_components_object_ids(
    grid: Grid,
    background_color: int = 0,
    connectivity: int = 4,
) -> dict[str, Any]:
    """Encoder-side object IDs, preserving the historical background-color rule."""
    return _connected_components(
        grid,
        excluded_color=int(background_color),
        connectivity=connectivity,
    )


def all_color_connected_component_ids(
    grid: Grid,
    connectivity: int = 4,
) -> dict[str, Any]:
    """Target-side grouping in which every ARC color, including 0, participates.

    The input must be the real rectangular target rather than a padded canvas.
    IDs are grouping labels only and have no embedding or ordinal semantics.
    """
    return _connected_components(
        grid,
        excluded_color=None,
        connectivity=connectivity,
    )


def pad_object_grid_to_vitarc_layout(
    object_grid: list[list[int]],
    grid_rows: int = 33,
    row_len: int = 34,
    content_cols: int = 32,
) -> list[int]:
    """Flatten object ids into one ViTARC grid body layout without outer tokens."""
    normalized = validate_rectangular_grid(object_grid)
    height, width = len(normalized), len(normalized[0])
    if grid_rows < 2:
        raise ValueError("grid_rows must be at least 2.")
    if row_len != content_cols + 2:
        raise ValueError("row_len must equal content_cols + 2.")
    if height > grid_rows - 1:
        raise ValueError(f"Object grid height {height} exceeds max content rows {grid_rows - 1}.")
    if width > content_cols:
        raise ValueError(f"Object grid width {width} exceeds content_cols {content_cols}.")

    rows: list[list[int]] = []
    for row in normalized:
        rows.append(
            list(row)
            + [0]
            + [0] * (content_cols - width)
            + [0]
        )

    rows.append([0] * row_len)
    while len(rows) < grid_rows:
        rows.append([0] * row_len)

    flat = [value for row in rows for value in row]
    expected_length = grid_rows * row_len
    if len(flat) != expected_length:
        raise ValueError(f"Object id body length {len(flat)} != expected {expected_length}.")
    return flat


def build_object_ids_for_grid_sequence(
    grid: Grid,
    background_color: int = 0,
    connectivity: int = 4,
    grid_rows: int = 33,
    row_len: int = 34,
    content_cols: int = 32,
) -> dict[str, Any]:
    """Build object ids for one complete ``<s> grid_body </s>`` sequence."""
    component_result = connected_components_object_ids(
        grid,
        background_color=background_color,
        connectivity=connectivity,
    )
    body = pad_object_grid_to_vitarc_layout(
        component_result["object_grid"],
        grid_rows=grid_rows,
        row_len=row_len,
        content_cols=content_cols,
    )
    object_ids = [0, *body, 0]
    object_mask = [1 if value > 0 else 0 for value in object_ids]
    return {
        "object_ids": object_ids,
        "object_mask": object_mask,
        "num_objects": component_result["num_objects"],
        "objects": component_result["objects"],
    }


def extract_original_grid_from_prepared_grid(prepared_grid: dict) -> Grid:
    """Recover the original ARC grid from a prepared padded-grid dict."""
    for key in ("padded_grid", "original_size"):
        if key not in prepared_grid:
            raise KeyError(f"Prepared grid is missing required key: {key}")
    padded_grid = prepared_grid["padded_grid"]
    height, width = prepared_grid["original_size"]
    return [list(row[:width]) for row in padded_grid[:height]]


def _prepared_input_grids_in_formatter_order(prepared_sample: dict) -> list[Grid]:
    grids: list[Grid] = []
    for pair in prepared_sample.get("context_pairs", []):
        grids.append(extract_original_grid_from_prepared_grid(pair["input"]))
        grids.append(extract_original_grid_from_prepared_grid(pair["output"]))
    grids.append(extract_original_grid_from_prepared_grid(prepared_sample["query"]))
    return grids


def build_input_object_ids_for_prepared_sample(
    prepared_sample: dict,
    formatted_sample: dict,
    background_color: int = 0,
    connectivity: int = 4,
    grid_rows: int = 33,
    row_len: int = 34,
    content_cols: int = 32,
    strict: bool = True,
) -> dict[str, Any]:
    """Build object ids aligned with ``formatted_sample['input_tokens']``."""
    if "input_tokens" not in formatted_sample:
        raise KeyError("formatted_sample is missing input_tokens.")

    grids = _prepared_input_grids_in_formatter_order(prepared_sample)
    segment_results = [
        build_object_ids_for_grid_sequence(
            grid,
            background_color=background_color,
            connectivity=connectivity,
            grid_rows=grid_rows,
            row_len=row_len,
            content_cols=content_cols,
        )
        for grid in grids
    ]
    segment_bodies = [
        result["object_ids"][1:-1]
        for result in segment_results
    ]

    input_object_ids: list[int] = []
    input_object_mask: list[int] = []
    segment_index = 0
    body_index = 0
    for token_index, token in enumerate(formatted_sample["input_tokens"]):
        if is_grid_token(token):
            if strict and segment_index >= len(segment_bodies):
                raise ValueError(f"Unexpected grid token at input token index {token_index}.")
            if segment_index >= len(segment_bodies):
                value = 0
            else:
                body = segment_bodies[segment_index]
                if strict and body_index >= len(body):
                    raise ValueError(
                        f"Too many grid tokens for segment {segment_index} at input token index {token_index}."
                    )
                value = body[body_index] if body_index < len(body) else 0
                body_index += 1
                if body_index == len(body):
                    segment_index += 1
                    body_index = 0
            input_object_ids.append(value)
            input_object_mask.append(1 if value > 0 else 0)
        else:
            input_object_ids.append(0)
            input_object_mask.append(0)

    if strict:
        if len(input_object_ids) != len(formatted_sample["input_tokens"]):
            raise ValueError(
                f"Object id length {len(input_object_ids)} != input token length "
                f"{len(formatted_sample['input_tokens'])}."
            )
        if segment_index != len(segment_bodies) or body_index != 0:
            raise ValueError(
                "Not all prepared grid segments were consumed while aligning object ids: "
                f"consumed {segment_index} of {len(segment_bodies)}, body_index={body_index}."
            )

    return {
        "input_object_ids": input_object_ids,
        "input_object_mask": input_object_mask,
        "num_input_grid_segments": len(segment_results),
        "num_objects_per_segment": [result["num_objects"] for result in segment_results],
        "objects_per_segment": [result["objects"] for result in segment_results],
    }


def pad_object_id_sequences(
    sequences: list[list[int]],
    pad_value: int = 0,
    target_length: int | None = None,
) -> list[list[int]]:
    """Pad object id or mask sequences to a shared length."""
    if not sequences:
        return []

    effective_target = max(len(sequence) for sequence in sequences) if target_length is None else target_length
    padded: list[list[int]] = []
    for sequence in sequences:
        if len(sequence) > effective_target:
            raise ValueError(
                f"Object id sequence length {len(sequence)} exceeds target_length {effective_target}."
            )
        padded.append(list(sequence) + [pad_value] * (effective_target - len(sequence)))
    return padded


def build_object_id_batch_for_samples(
    prepared_samples: list[dict],
    formatted_samples: list[dict],
    input_target_length: int | None = None,
    background_color: int = 0,
    connectivity: int = 4,
    grid_rows: int = 33,
    row_len: int = 34,
    content_cols: int = 32,
    strict: bool = True,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Build batched object id tensors aligned with encoder ``input_ids``."""
    if len(prepared_samples) != len(formatted_samples):
        raise ValueError("prepared_samples and formatted_samples must have the same length.")
    if not prepared_samples:
        raise ValueError("prepared_samples must contain at least one sample.")

    aligned_samples = [
        build_input_object_ids_for_prepared_sample(
            prepared_sample,
            formatted_sample,
            background_color=background_color,
            connectivity=connectivity,
            grid_rows=grid_rows,
            row_len=row_len,
            content_cols=content_cols,
            strict=strict,
        )
        for prepared_sample, formatted_sample in zip(prepared_samples, formatted_samples)
    ]

    object_ids = pad_object_id_sequences(
        [sample["input_object_ids"] for sample in aligned_samples],
        pad_value=0,
        target_length=input_target_length,
    )
    object_masks = pad_object_id_sequences(
        [sample["input_object_mask"] for sample in aligned_samples],
        pad_value=0,
        target_length=input_target_length,
    )
    target_device = torch.device(device) if device is not None else None

    return {
        "input_object_ids": torch.tensor(object_ids, dtype=torch.long, device=target_device),
        "input_object_mask": torch.tensor(object_masks, dtype=torch.bool, device=target_device),
        "object_info": {
            "background_color": background_color,
            "connectivity": connectivity,
            "grid_rows": grid_rows,
            "row_len": row_len,
            "content_cols": content_cols,
            "num_input_grid_segments": [
                sample["num_input_grid_segments"] for sample in aligned_samples
            ],
            "num_objects_per_sample": [
                sum(sample["num_objects_per_segment"]) for sample in aligned_samples
            ],
            "num_objects_per_segment": [
                sample["num_objects_per_segment"] for sample in aligned_samples
            ],
        },
    }


def augment_batch_with_object_ids(
    batch: dict,
    prepared_samples: list[dict],
    formatted_samples: list[dict],
    background_color: int = 0,
    connectivity: int = 4,
    grid_rows: int = 33,
    row_len: int = 34,
    content_cols: int = 32,
    strict: bool = True,
    device: str | torch.device | None = None,
) -> dict:
    """Return a shallow batch copy with encoder-side object id tensors added."""
    if "input_ids" not in batch:
        raise KeyError("batch must contain input_ids.")
    target_device = device
    if target_device is None and hasattr(batch["input_ids"], "device"):
        target_device = batch["input_ids"].device

    object_batch = build_object_id_batch_for_samples(
        prepared_samples=prepared_samples,
        formatted_samples=formatted_samples,
        input_target_length=int(batch["input_ids"].shape[1]),
        background_color=background_color,
        connectivity=connectivity,
        grid_rows=grid_rows,
        row_len=row_len,
        content_cols=content_cols,
        strict=strict,
        device=target_device,
    )
    augmented = dict(batch)
    augmented.update(object_batch)
    return augmented
