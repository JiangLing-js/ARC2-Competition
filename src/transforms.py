"""Basic reversible transforms for ARC grids.

All public transform functions accept an ARC grid represented as either a
``list[list[int]]`` or a ``numpy.ndarray``-like object with ``tolist()``. Inputs
are validated as rectangular grids, are never modified in place, and outputs are
always returned as plain ``list[list[int]]`` values.
"""

from __future__ import annotations

from typing import Callable, Iterable


Grid = list[list[int]]


def _to_list_grid(grid: Iterable[Iterable[int]]) -> Grid:
    """Return a copied ``list[list[int]]`` after validating rectangular shape."""
    if hasattr(grid, "tolist"):
        grid = grid.tolist()

    if not isinstance(grid, list):
        grid = list(grid)

    copied: Grid = []
    expected_width: int | None = None

    for row in grid:
        if not isinstance(row, list):
            row = list(row)
        if expected_width is None:
            expected_width = len(row)
        elif len(row) != expected_width:
            raise ValueError("ARC grid must be rectangular.")
        copied.append(list(row))

    return copied


def identity(grid: Iterable[Iterable[int]]) -> Grid:
    """Return an unchanged copy of ``grid``."""
    return _to_list_grid(grid)


def rotate90(grid: Iterable[Iterable[int]]) -> Grid:
    """Rotate ``grid`` 90 degrees clockwise."""
    normalized = _to_list_grid(grid)
    if not normalized:
        return []
    # zip(*normalized[::-1])这里的 * 是拆包 zip 会按列配对
    return [list(row) for row in zip(*normalized[::-1])]


def rotate180(grid: Iterable[Iterable[int]]) -> Grid:
    """Rotate ``grid`` 180 degrees."""
    normalized = _to_list_grid(grid)
    return [row[::-1] for row in normalized[::-1]]


def rotate270(grid: Iterable[Iterable[int]]) -> Grid:
    """Rotate ``grid`` 270 degrees clockwise, equivalent to 90 degrees counter-clockwise."""
    normalized = _to_list_grid(grid)
    if not normalized:
        return []
    return [list(row) for row in zip(*normalized)][::-1]


def flip_horizontal(grid: Iterable[Iterable[int]]) -> Grid:
    """Mirror ``grid`` left-to-right."""
    normalized = _to_list_grid(grid)
    return [row[::-1] for row in normalized]


def flip_vertical(grid: Iterable[Iterable[int]]) -> Grid:
    """Mirror ``grid`` top-to-bottom."""
    normalized = _to_list_grid(grid)
    return normalized[::-1]


def transpose(grid: Iterable[Iterable[int]]) -> Grid:
    """Swap rows and columns of ``grid`` across the main diagonal."""
    normalized = _to_list_grid(grid)
    if not normalized:
        return []
    return [list(row) for row in zip(*normalized)]


TRANSFORMS: dict[str, Callable[[Iterable[Iterable[int]]], Grid]] = {
    "identity": identity,
    "rotate90": rotate90,
    "rotate180": rotate180,
    "rotate270": rotate270,
    "flip_horizontal": flip_horizontal,
    "flip_vertical": flip_vertical,
    "transpose": transpose,
}


INVERSE_TRANSFORMS: dict[str, str] = {
    "identity": "identity",
    "rotate90": "rotate270",
    "rotate180": "rotate180",
    "rotate270": "rotate90",
    "flip_horizontal": "flip_horizontal",
    "flip_vertical": "flip_vertical",
    "transpose": "transpose",
}


def apply_transform(grid: Iterable[Iterable[int]], name: str) -> Grid:
    """Apply the named transform to ``grid`` and return a new list grid.

    Args:
        grid: Rectangular ARC grid as ``list[list[int]]`` or ``numpy.ndarray``.
        name: Transform name from ``TRANSFORMS``.

    Raises:
        ValueError: If ``name`` is unknown or ``grid`` is not rectangular.
    """
    try:
        transform = TRANSFORMS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown transform: {name}") from exc
    return transform(grid)


def inverse_transform_name(name: str) -> str:
    """Return the transform name that reverses ``name``.

    Raises:
        ValueError: If ``name`` is not a supported transform.
    """
    try:
        return INVERSE_TRANSFORMS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown transform: {name}") from exc


def invert_grid(grid: Iterable[Iterable[int]], name: str) -> Grid:
    """Invert a grid that has already been transformed by ``name``."""
    return apply_transform(grid, inverse_transform_name(name))
