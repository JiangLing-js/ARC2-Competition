"""Grid validation and padding helpers for ARC-style visual inputs."""

from __future__ import annotations

from typing import Iterable


Grid = list[list[int]]


def _to_list_grid(grid: Iterable[Iterable[int]]) -> Grid:
    if hasattr(grid, "tolist"):
        grid = grid.tolist()

    if not isinstance(grid, list):
        grid = list(grid)

    copied: Grid = []
    for row in grid:
        if not isinstance(row, list):
            row = list(row)
        copied.append(list(row))
    return copied


def validate_rectangular_grid(grid: Iterable[Iterable[int]]) -> Grid:
    """Validate that ``grid`` is a rectangular 2D grid and return a copied list grid.

    Args:
        grid: ARC grid as ``list[list[int]]`` or a numpy array-like object.

    Raises:
        ValueError: If the grid is empty, not 2D, or has rows of unequal length.
    """
    copied = _to_list_grid(grid)
    if not copied:
        raise ValueError("ARC grid must have at least one row.")

    expected_width = len(copied[0])
    if expected_width == 0:
        raise ValueError("ARC grid rows must have at least one column.")

    for row in copied:
        if len(row) != expected_width:
            raise ValueError("ARC grid must be rectangular.")

    return copied


def grid_size(grid: Iterable[Iterable[int]]) -> tuple[int, int]:
    """Return ``(height, width)`` for a validated rectangular ARC grid."""
    copied = validate_rectangular_grid(grid)
    return len(copied), len(copied[0])


def pad_grid(
    grid: Iterable[Iterable[int]],
    max_h: int = 30,
    max_w: int = 30,
    pad_value: int = 0,
) -> tuple[Grid, Grid, tuple[int, int]]:
    """Pad an ARC grid to a fixed size.

    Args:
        grid: ARC grid as ``list[list[int]]`` or a numpy array-like object.
        max_h: Output height.
        max_w: Output width.
        pad_value: Value used outside the real grid area.

    Returns:
        ``(padded_grid, mask, original_size)`` where ``mask`` is 1 in the real
        grid area and 0 in the padded area.

    Raises:
        ValueError: If ``grid`` is not rectangular or exceeds ``max_h``/``max_w``.
    """
    copied = validate_rectangular_grid(grid)
    h, w = len(copied), len(copied[0])
    if h > max_h or w > max_w:
        raise ValueError(f"Grid size {(h, w)} exceeds max size {(max_h, max_w)}.")

    padded_grid = [[pad_value for _ in range(max_w)] for _ in range(max_h)]
    mask = [[0 for _ in range(max_w)] for _ in range(max_h)]

    for r, row in enumerate(copied):
        for c, value in enumerate(row):
            padded_grid[r][c] = value
            mask[r][c] = 1

    return padded_grid, mask, (h, w)
