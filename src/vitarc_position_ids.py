"""Row/column position id utilities for ViTARC-style token sequences."""

from __future__ import annotations

from typing import Any

import torch


GRID_EXTRA_TOKENS = {
    "<arc_pad>",
    "<arc_nl>",
    "<arc_endxgrid>",
    "<arc_endygrid>",
    "<arc_endxygrid>",
}


def is_grid_token(token: str) -> bool:
    """Return True for tokens that belong to a ViTARC 2D grid body."""
    return token in GRID_EXTRA_TOKENS or token in {f"<arc_{idx}>" for idx in range(10)}


def build_position_ids_for_token_sequence(
    tokens: list[str],
    grid_rows: int = 33,
    row_len: int = 34,
    start_token: str = "<s>",
    end_token: str = "</s>",
    sep_token: str = "<sep>",
    pad_token: str = "<pad>",
    unk_token: str = "<unk>",
    strict: bool = True,
) -> dict[str, Any]:
    """Build row/column ids aligned to a possibly multi-grid token sequence.

    Each complete grid body contributes ``grid_rows * row_len`` positioned
    tokens. Positions reset for every grid body. Non-grid tokens use the special
    row/column ids and have ``position_mask=0``.

    The parser accepts ViTARC single-grid segments (``<s> grid </s>``) and the
    project-local D_TTT prompt wrapper (``<s> grid <sep> grid ... </s>``), where
    ``<sep>`` may appear between already-complete grid bodies.
    """
    if grid_rows < 1:
        raise ValueError("grid_rows must be at least 1.")
    if row_len < 1:
        raise ValueError("row_len must be at least 1.")

    special_row_id = grid_rows
    special_col_id = row_len
    grid_area = grid_rows * row_len
    row_ids: list[int] = []
    col_ids: list[int] = []
    position_mask: list[int] = []
    inside_grid_segment = False
    current_grid_index = 0
    grid_token_count = 0
    num_grid_segments = 0

    def emit_special() -> None:
        row_ids.append(special_row_id)
        col_ids.append(special_col_id)
        position_mask.append(0)

    def finish_grid_segment(reason: str) -> None:
        nonlocal current_grid_index, num_grid_segments
        if strict and current_grid_index != grid_area:
            raise ValueError(
                f"Incomplete grid before {reason}: got {current_grid_index} grid tokens, "
                f"expected {grid_area}."
            )
        if current_grid_index == grid_area:
            num_grid_segments += 1
        current_grid_index = 0

    specials = {start_token, end_token, sep_token, pad_token, unk_token}

    for token in tokens:
        if not inside_grid_segment:
            if token == start_token:
                emit_special()
                inside_grid_segment = True
                current_grid_index = 0
            elif token in specials:
                emit_special()
            elif is_grid_token(token):
                if strict:
                    raise ValueError(f"Grid token {token!r} appeared outside a grid segment.")
                emit_special()
            else:
                if strict:
                    raise ValueError(f"Unknown non-grid token {token!r}.")
                emit_special()
            continue

        if token == end_token:
            finish_grid_segment(end_token)
            emit_special()
            inside_grid_segment = False
            continue

        if token == sep_token:
            if current_grid_index == grid_area:
                finish_grid_segment(sep_token)
            elif current_grid_index != 0 and strict:
                raise ValueError(
                    f"Incomplete grid before {sep_token}: got {current_grid_index} "
                    f"grid tokens, expected {grid_area}."
                )
            else:
                current_grid_index = 0
            emit_special()
            inside_grid_segment = True
            continue

        if token == start_token:
            if strict:
                raise ValueError(f"Unexpected start token {start_token!r} inside a grid segment.")
            emit_special()
            current_grid_index = 0
            continue

        if is_grid_token(token):
            if strict and current_grid_index >= grid_area:
                raise ValueError(
                    f"Too many grid tokens in segment: expected {grid_area}, "
                    f"encountered at least {current_grid_index + 1}."
                )
            if current_grid_index < grid_area:
                row_ids.append(current_grid_index // row_len)
                col_ids.append(current_grid_index % row_len)
                position_mask.append(1)
                current_grid_index += 1
                grid_token_count += 1
            else:
                emit_special()
            continue

        if strict:
            raise ValueError(f"Non-grid token {token!r} appeared inside a grid segment.")
        emit_special()

    if strict and inside_grid_segment:
        raise ValueError(
            f"Unclosed grid segment: got {current_grid_index} grid tokens before sequence end."
        )

    return {
        "row_ids": row_ids,
        "col_ids": col_ids,
        "position_mask": position_mask,
        "special_row_id": special_row_id,
        "special_col_id": special_col_id,
        "grid_token_count": grid_token_count,
        "num_grid_segments": num_grid_segments,
    }


def pad_position_id_sequences(
    sequences: list[list[int]],
    pad_value: int,
    target_length: int | None = None,
) -> list[list[int]]:
    """Pad position id sequences to a shared length."""
    if not sequences:
        return []

    effective_target = max(len(sequence) for sequence in sequences) if target_length is None else target_length
    padded: list[list[int]] = []
    for sequence in sequences:
        if len(sequence) > effective_target:
            raise ValueError(
                f"Position id sequence length {len(sequence)} exceeds target_length {effective_target}."
            )
        padded.append(list(sequence) + [pad_value] * (effective_target - len(sequence)))
    return padded


def build_position_batch_for_formatted_samples(
    formatted_samples: list[dict],
    input_target_length: int | None = None,
    label_target_length: int | None = None,
    grid_rows: int = 33,
    row_len: int = 34,
    strict: bool = True,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Build batched row/column position tensors for formatted samples."""
    if not formatted_samples:
        raise ValueError("formatted_samples must contain at least one sample.")

    input_positions = []
    label_positions = []
    for index, sample in enumerate(formatted_samples):
        if "input_tokens" not in sample or "label_tokens" not in sample:
            raise KeyError(f"Formatted sample {index} must contain input_tokens and label_tokens.")
        input_positions.append(
            build_position_ids_for_token_sequence(
                sample["input_tokens"],
                grid_rows=grid_rows,
                row_len=row_len,
                strict=strict,
            )
        )
        label_positions.append(
            build_position_ids_for_token_sequence(
                sample["label_tokens"],
                grid_rows=grid_rows,
                row_len=row_len,
                strict=strict,
            )
        )

    special_row_id = grid_rows
    special_col_id = row_len
    input_row_ids = pad_position_id_sequences(
        [pos["row_ids"] for pos in input_positions],
        pad_value=special_row_id,
        target_length=input_target_length,
    )
    input_col_ids = pad_position_id_sequences(
        [pos["col_ids"] for pos in input_positions],
        pad_value=special_col_id,
        target_length=input_target_length,
    )
    input_masks = pad_position_id_sequences(
        [pos["position_mask"] for pos in input_positions],
        pad_value=0,
        target_length=input_target_length,
    )
    label_row_ids = pad_position_id_sequences(
        [pos["row_ids"] for pos in label_positions],
        pad_value=special_row_id,
        target_length=label_target_length,
    )
    label_col_ids = pad_position_id_sequences(
        [pos["col_ids"] for pos in label_positions],
        pad_value=special_col_id,
        target_length=label_target_length,
    )
    label_masks = pad_position_id_sequences(
        [pos["position_mask"] for pos in label_positions],
        pad_value=0,
        target_length=label_target_length,
    )

    target_device = torch.device(device) if device is not None else None

    def long_tensor(values: list[list[int]]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.long, device=target_device)

    def bool_tensor(values: list[list[int]]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.bool, device=target_device)

    return {
        "input_row_ids": long_tensor(input_row_ids),
        "input_col_ids": long_tensor(input_col_ids),
        "input_position_mask": bool_tensor(input_masks),
        "label_row_ids": long_tensor(label_row_ids),
        "label_col_ids": long_tensor(label_col_ids),
        "label_position_mask": bool_tensor(label_masks),
        "position_info": {
            "grid_rows": grid_rows,
            "row_len": row_len,
            "special_row_id": special_row_id,
            "special_col_id": special_col_id,
            "num_input_grid_segments": [pos["num_grid_segments"] for pos in input_positions],
            "num_label_grid_segments": [pos["num_grid_segments"] for pos in label_positions],
            "input_grid_token_counts": [pos["grid_token_count"] for pos in input_positions],
            "label_grid_token_counts": [pos["grid_token_count"] for pos in label_positions],
        },
    }


def augment_batch_with_position_ids(
    batch: dict,
    formatted_samples: list[dict],
    grid_rows: int = 33,
    row_len: int = 34,
    strict: bool = True,
    device: str | torch.device | None = None,
) -> dict:
    """Return a shallow batch copy with ViTARC row/column tensors added."""
    if "input_ids" not in batch or "labels" not in batch:
        raise KeyError("batch must contain input_ids and labels.")

    input_target_length = int(batch["input_ids"].shape[1])
    label_target_length = int(batch["labels"].shape[1])
    target_device = device
    if target_device is None and hasattr(batch["input_ids"], "device"):
        target_device = batch["input_ids"].device

    position_batch = build_position_batch_for_formatted_samples(
        formatted_samples,
        input_target_length=input_target_length,
        label_target_length=label_target_length,
        grid_rows=grid_rows,
        row_len=row_len,
        strict=strict,
        device=target_device,
    )
    augmented = dict(batch)
    augmented.update(position_batch)
    return augmented
