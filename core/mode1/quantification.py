import cv2

from core.common.grid import assign_grid_positions, coverage_percent
from utils.helpers import get_physical_grid_params

DEFAULT_RADIUS = 15.5


def _box_touches_circle(box, wx, wy, radius):
    x1, y1, x2, y2 = box[:4]
    closest_x = max(x1, min(wx, x2))
    closest_y = max(y1, min(wy, y2))
    return ((wx - closest_x) ** 2 + (wy - closest_y) ** 2) ** 0.5 <= radius


def compute_well_statuses(well_coords, cell_boxes, w_crop=None):
    """Determine, for each well, whether any cell bounding box overlaps it.

    Returns ``(well_statuses, coverage_percent, effective_radius)``.
    """
    if not well_coords:
        return [], 0.0, DEFAULT_RADIUS

    _, effective_radius, _ = get_physical_grid_params(well_coords=well_coords, w_crop=w_crop)
    ordered, positions = assign_grid_positions(well_coords, effective_radius)

    well_statuses = []
    for idx, (wx, wy) in enumerate(ordered, start=1):
        has_cell = any(_box_touches_circle(box, wx, wy, effective_radius) for box in cell_boxes)
        row_idx, col_idx = positions[(wx, wy)]
        well_statuses.append({
            'id': idx,
            'row': row_idx,
            'col': col_idx,
            'coord': (int(wx), int(wy)),
            'has_cell': has_cell,
        })

    covered = sum(1 for w in well_statuses if w['has_cell'])
    return well_statuses, coverage_percent(covered, len(well_coords)), effective_radius


def draw_wells(img, well_statuses, radius):
    """Draw well circles in place: green if the well contains a cell, red otherwise."""
    radius_int = max(1, int(round(radius)))
    for well in well_statuses:
        color = (0, 255, 0) if well['has_cell'] else (0, 0, 255)
        cv2.circle(img, well['coord'], radius_int, color, 1)
    return img


def calculate_coverage_and_draw(img, well_coords, cell_boxes, radius=None, w_crop=None):
    """Backward-compatible wrapper: compute statuses and draw them on ``img``.

    Returns ``(img, coverage_percent, well_statuses, effective_radius)``.
    """
    if not well_coords:
        return img, 0.0, [], DEFAULT_RADIUS
    well_statuses, cov, eff_radius = compute_well_statuses(well_coords, cell_boxes, w_crop)
    draw_wells(img, well_statuses, eff_radius)
    return img, cov, well_statuses, eff_radius
