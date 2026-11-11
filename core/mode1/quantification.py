import cv2
from utils.helpers import get_physical_grid_params

def calculate_coverage_and_draw(img, well_coords, cell_boxes, radius=None, w_crop=None):
    total_wells = len(well_coords)
    well_statuses = []

    if total_wells == 0:
        return img, 0.0, [], 15.5

    pitch, phys_radius, _ = get_physical_grid_params(well_coords=well_coords, w_crop=w_crop)
    effective_radius = phys_radius

    sorted_by_y = sorted(well_coords, key=lambda p: p[1])
    rows = []
    if sorted_by_y:
        current_row = [sorted_by_y[0]]
        for p in sorted_by_y[1:]:
            if p[1] - current_row[-1][1] < effective_radius * 1.5:
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
    radius_int = max(1, int(round(effective_radius)))

    for idx, (wx, wy) in enumerate(final_well_coords, start=1):
        has_cell = False

        for box in cell_boxes:
            x1, y1, x2, y2 = box[:4]
            closest_x = max(x1, min(wx, x2))
            closest_y = max(y1, min(wy, y2))
            distance = ((wx - closest_x) ** 2 + (wy - closest_y) ** 2) ** 0.5

            if distance <= effective_radius:
                has_cell = True
                break

        row_idx, col_idx = well_grid_info.get((wx, wy), ((idx // 10) + 1, (idx % 10) + 1))
        well_statuses.append({
            'id': idx,
            'row': row_idx,
            'col': col_idx,
            'coord': (int(wx), int(wy)),
            'has_cell': has_cell
        })

        color = (0, 255, 0) if has_cell else (0, 0, 255)
        if has_cell:
            covered_count += 1

        cv2.circle(img, (int(wx), int(wy)), radius_int, color, 1)

    coverage_percent = (covered_count / total_wells) * 100.0 if total_wells > 0 else 0.0
    return img, coverage_percent, well_statuses, effective_radius