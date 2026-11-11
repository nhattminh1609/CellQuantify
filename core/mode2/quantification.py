import cv2
import numpy as np
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

def quantify_multilabel_wells(well_coords, cell_boxes, w_crop=None):
    total_wells = len(well_coords)
    well_statuses = []

    if total_wells == 0:
        return [], 0.0, 15.5, 0.775

    ratio_new, radius_px, _ = calculate_scale_from_crop(w_crop=w_crop, well_coords=well_coords)

    sorted_by_y = sorted(well_coords, key=lambda p: p[1])
    rows = []
    if sorted_by_y:
        current_row = [sorted_by_y[0]]
        for p in sorted_by_y[1:]:
            if p[1] - current_row[-1][1] < radius_px * 1.5:
                current_row.append(p)
            else:
                rows.append(current_row)
                current_row = [p]
        rows.append(current_row)

    final_well_coords = []
    well_grid_info = {}

    for row_idx, row in enumerate(rows, start=1):
        sorted_row = sorted(row, key=lambda p: p[0])
        for col_idx, p in enumerate(sorted_row, start=1):
            final_well_coords.append(p)
            well_grid_info[p] = (row_idx, col_idx)

    covered_count = 0

    for idx, (wx, wy) in enumerate(final_well_coords, start=1):
        has_a549 = False
        has_nk = False
        has_undefine = False

        for box in cell_boxes:
            x1, y1, x2, y2, label_name = box[:5]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            center_dist = ((wx - cx) ** 2 + (wy - cy) ** 2) ** 0.5
            if center_dist <= radius_px * 0.90:
                if label_name == "a549":
                    has_a549 = True
                elif label_name == "nk":
                    has_nk = True
                else:
                    has_undefine = True

        has_any = has_a549 or has_nk or has_undefine
        if has_a549 or has_nk:
            covered_count += 1

        row_idx, col_idx = well_grid_info.get((wx, wy), ((idx // 10) + 1, (idx % 10) + 1))
        well_statuses.append({
            "id": idx,
            "row": row_idx,
            "col": col_idx,
            "coord": (int(wx), int(wy)),
            "has_a549": has_a549,
            "has_nk": has_nk,
            "has_undefine": has_undefine,
            "has_cell": has_any
        })

    coverage_percent = (covered_count / total_wells) * 100.0 if total_wells > 0 else 0.0
    return well_statuses, coverage_percent, radius_px, ratio_new

def render_mode2_canvas(base_img, seg_data, well_statuses, radius, ratio_new=None, grid_mode="Full grid", mask_mode="All labels", show_scale_bar=False):
    canvas = base_img.copy()

    if mask_mode != "Hide":
        colored_mask = np.zeros_like(canvas)
        active_labels = []

        if mask_mode == "All labels":
            active_labels = ["a549", "nk", "undefine"]
        elif mask_mode == "a549":
            active_labels = ["a549"]
        elif mask_mode == "nk":
            active_labels = ["nk"]
        elif mask_mode == "undefine":
            active_labels = ["undefine"]

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

    if grid_mode != "Hide grid":
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
        actual_ratio = ratio_new if ratio_new is not None else (radius / 20.0)
        canvas = draw_scale_bar(canvas, well_statuses, actual_ratio)

    return canvas