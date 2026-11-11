import cv2
import numpy as np

import config
from core.common.grid import assign_grid_positions, coverage_percent
from utils.helpers import calculate_scale_from_crop, draw_scale_bar

GRID_COLORS = {
    "a549": (0, 255, 0),
    "nk": (0, 0, 255),
    "empty": (160, 160, 160),
    "red_all": (0, 0, 255)
}

CLASS_COLORS = {
    "a549": {"fill": [0, 180, 0], "border": (0, 255, 0), "dot": (255, 255, 255)},
    "nk": {"fill": [0, 0, 180], "border": (0, 0, 255), "dot": (255, 255, 255)},
    "undefine": {"fill": [0, 180, 180], "border": (0, 255, 255), "dot": (255, 255, 255)}
}

GRID_MODES = ["Full grid", "Full red grid", "Only a549", "Only nk", "Hide"]
MASK_MODES = ["All labels", "a549", "nk", "undefine", "Hide"]
HIDE = "Hide"


def quantify_multilabel_wells(well_coords, cell_boxes, w_crop=None):
    """Assign cell labels to wells (cell box centre within 0.9 * radius of the well centre).

    Returns ``(well_statuses, coverage_percent, radius_px, px_per_um)``.
    Coverage counts wells containing a549 and/or nk cells.
    """
    if not well_coords:
        return [], 0.0, 15.5, config.BASE_PX_PER_UM

    ratio_new, radius_px, _ = calculate_scale_from_crop(w_crop=w_crop, well_coords=well_coords)
    ordered, positions = assign_grid_positions(well_coords, radius_px)
    max_dist = radius_px * config.MODE2_CENTER_DISTANCE_FACTOR

    well_statuses = []
    covered_count = 0

    for idx, (wx, wy) in enumerate(ordered, start=1):
        has_a549 = has_nk = has_undefine = False

        for box in cell_boxes:
            x1, y1, x2, y2, label_name = box[:5]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            center_dist = ((wx - cx) ** 2 + (wy - cy) ** 2) ** 0.5

            # Distance from well center to closest point on cell bounding box
            closest_x = max(x1, min(wx, x2))
            closest_y = max(y1, min(wy, y2))
            distance = ((wx - closest_x) ** 2 + (wy - closest_y) ** 2) ** 0.5

            # Associate if cell bounding box touches/intersects the well circle (distance <= radius_px)
            # or cell center is within well (center_dist <= radius_px * 1.15)
            if distance <= radius_px or center_dist <= radius_px * 1.15:
                if label_name == "a549":
                    has_a549 = True
                elif label_name == "nk":
                    has_nk = True
                else:
                    has_undefine = True

        if has_a549 or has_nk:
            covered_count += 1

        row_idx, col_idx = positions[(wx, wy)]
        well_statuses.append({
            "id": idx,
            "row": row_idx,
            "col": col_idx,
            "coord": (int(wx), int(wy)),
            "has_a549": has_a549,
            "has_nk": has_nk,
            "has_undefine": has_undefine,
            "has_cell": has_a549 or has_nk or has_undefine
        })

    return well_statuses, coverage_percent(covered_count, len(well_coords)), radius_px, ratio_new


def render_mode2_canvas(base_img, seg_data, well_statuses, radius, ratio_new=None,
                        grid_mode="Full grid", mask_mode="All labels", active_masks=None,
                        show_scale_bar=False):
    canvas = base_img.copy()

    # Determine which mask classes are actively enabled
    if active_masks is not None:
        if isinstance(active_masks, (list, tuple, set)):
            active_labels = [lbl for lbl in config.MODE2_CLASS_NAMES if lbl in active_masks]
        else:
            active_labels = []
    elif mask_mode is not None:
        if isinstance(mask_mode, (list, tuple, set)):
            active_labels = [lbl for lbl in config.MODE2_CLASS_NAMES if lbl in mask_mode]
        elif mask_mode == "All labels":
            active_labels = list(config.MODE2_CLASS_NAMES)
        elif mask_mode in config.MODE2_CLASS_NAMES:
            active_labels = [mask_mode]
        else:
            active_labels = []
    else:
        active_labels = list(config.MODE2_CLASS_NAMES)

    if active_labels:
        colored_mask = np.zeros_like(canvas)
        for lbl in active_labels:
            mask_orig = seg_data.get("masks", {}).get(lbl)
            if mask_orig is not None and np.any(mask_orig > 0):
                colored_mask[mask_orig > 0] = CLASS_COLORS[lbl]["fill"]

        canvas = cv2.addWeighted(canvas, 1.0, colored_mask, 0.3, 0)

        for lbl in active_labels:
            contours = seg_data.get("contours", {}).get(lbl, [])
            if contours:
                cv2.drawContours(canvas, contours, -1, CLASS_COLORS[lbl]["border"], 1)

            peaks = seg_data.get("peaks", {}).get(lbl, [])
            for (cx, cy) in peaks:
                cv2.circle(canvas, (cx, cy), radius=1, color=CLASS_COLORS[lbl]["dot"], thickness=-1)

    if grid_mode != HIDE:
        radius_int = max(2, int(round(radius)))

        for well in well_statuses:
            wx, wy = well["coord"]
            has_a549 = well["has_a549"]
            has_nk = well["has_nk"]

            if grid_mode == "Full grid":
                if has_a549 and has_nk:
                    cv2.circle(canvas, (wx, wy), radius_int, GRID_COLORS["a549"], 2)
                    cv2.circle(canvas, (wx, wy), max(1, radius_int - 2), GRID_COLORS["nk"], 2)
                elif has_a549:
                    cv2.circle(canvas, (wx, wy), radius_int, GRID_COLORS["a549"], 1)
                elif has_nk:
                    cv2.circle(canvas, (wx, wy), radius_int, GRID_COLORS["nk"], 1)
                else:
                    cv2.circle(canvas, (wx, wy), radius_int, GRID_COLORS["empty"], 1)

            elif grid_mode == "Full red grid":
                cv2.circle(canvas, (wx, wy), radius_int, GRID_COLORS["red_all"], 1)

            elif grid_mode == "Only a549":
                if has_a549:
                    cv2.circle(canvas, (wx, wy), radius_int, GRID_COLORS["a549"], 1)

            elif grid_mode == "Only nk":
                if has_nk:
                    cv2.circle(canvas, (wx, wy), radius_int, GRID_COLORS["nk"], 1)

    if show_scale_bar:
        actual_ratio = ratio_new if ratio_new is not None else (radius / config.WELL_RADIUS_UM)
        canvas = draw_scale_bar(canvas, well_statuses, actual_ratio)

    return canvas
