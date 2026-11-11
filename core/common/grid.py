"""Shared microwell-grid utilities."""
from typing import Dict, List, Tuple

Point = Tuple[float, float]


def assign_grid_positions(well_coords: List[Point], radius: float) -> Tuple[List[Point], Dict[Point, Tuple[int, int]]]:
    """Order wells row-by-row (top to bottom, left to right) and assign 1-based (row, col).

    Two consecutive wells (sorted by y) belong to the same row when their
    y distance is below ``1.5 * radius``.

    Returns ``(ordered_coords, {coord: (row, col)})``.
    """
    sorted_by_y = sorted(well_coords, key=lambda p: p[1])
    rows: List[List[Point]] = []
    if sorted_by_y:
        current_row = [sorted_by_y[0]]
        for p in sorted_by_y[1:]:
            if p[1] - current_row[-1][1] < radius * 1.5:
                current_row.append(p)
            else:
                rows.append(current_row)
                current_row = [p]
        rows.append(current_row)

    ordered: List[Point] = []
    positions: Dict[Point, Tuple[int, int]] = {}
    for row_idx, row in enumerate(rows, start=1):
        for col_idx, p in enumerate(sorted(row, key=lambda q: q[0]), start=1):
            ordered.append(p)
            positions[p] = (row_idx, col_idx)
    return ordered, positions


def coverage_percent(covered: int, total: int) -> float:
    return (covered / total) * 100.0 if total > 0 else 0.0
