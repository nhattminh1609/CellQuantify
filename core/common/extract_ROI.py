from typing import List, NamedTuple, Optional, Tuple

import cv2
import numpy as np

import config


class CropResult(NamedTuple):
    """Result of :func:`crop_well_image`.

    Always returned with every field populated, whether or not the grid was found.
    """
    cropped_img: np.ndarray                    # fluorescence crop (CROP_OUTPUT_SIZE square)
    yellow_clean: np.ndarray                   # bright-field frame (deskewed if grid found)
    yellow_grid: np.ndarray                    # bright-field frame with detected grid drawn
    well_coords: List[Tuple[float, float]]     # well centres in cropped_img coordinates
    radius: float                              # well radius in cropped_img pixels
    grid_found: bool


def filter_interstitial_noise(circles_list, min_distance=config.NOISE_MIN_DISTANCE):
    valid_circles = []
    for circle in circles_list:
        if not valid_circles or np.min(np.hypot(np.array(valid_circles)[:, 0] - circle[0],
                                                np.array(valid_circles)[:, 1] - circle[1])) >= min_distance:
            valid_circles.append(circle)
    if not valid_circles:
        return np.empty((0, 3), dtype=np.float32)
    return np.array(valid_circles)


def group_coordinates(coords, tolerance=15, min_points=2, max_gap=config.GROUP_MAX_GAP):
    if len(coords) == 0:
        return []
    coords = np.sort(coords)
    groups = np.split(coords, np.where(np.diff(coords) > tolerance)[0] + 1)
    group_means = [np.mean(g) for g in groups if len(g) >= min_points]
    if len(group_means) <= 1:
        return group_means
    final_means = []
    for i in range(len(group_means)):
        dist_prev = (group_means[i] - group_means[i - 1]) if i > 0 else float('inf')
        dist_next = (group_means[i + 1] - group_means[i]) if i < len(group_means) - 1 else float('inf')
        if min(dist_prev, dist_next) <= max_gap:
            final_means.append(group_means[i])
    return final_means


def estimate_skew_angle(coords, tolerance=config.SKEW_ROW_TOLERANCE):
    """Estimate the grid rotation (degrees) from detected well centres.

    Points are grouped into rows by their y coordinate; a line is fitted to each
    row with >= 3 points and the median slope angle is returned.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or len(coords) < 3:
        return 0.0
    # Sort by y BEFORE splitting so the split indices refer to the same ordering.
    coords = coords[np.argsort(coords[:, 1])]
    rows = np.split(coords, np.where(np.diff(coords[:, 1]) > tolerance)[0] + 1)
    angles = []
    for row in rows:
        mean_y = np.mean(row[:, 1])
        row_points = coords[np.abs(coords[:, 1] - mean_y) <= tolerance]
        if len(row_points) >= 3 and np.ptp(row_points[:, 0]) > 0:
            slope, _ = np.polyfit(row_points[:, 0], row_points[:, 1], 1)
            angles.append(np.arctan(slope))
    if not angles:
        return 0.0
    return float(np.degrees(np.median(angles)))


def extract_perfect_grid(coords, raw_coords, expected=config.GRID_SIZE):
    if len(coords) < 2:
        return list(coords)
    coords = np.sort(coords)
    valid_diffs = np.diff(coords)[np.diff(coords) > config.GRID_MIN_STEP]
    if len(valid_diffs) == 0:
        return list(coords)
    base_step = np.median(valid_diffs)
    best_grid, max_score = [], -1
    for anchor in coords:
        for anchor_idx in range(expected):
            for step_adj in np.linspace(-4.0, 4.0, 9):
                current_step = base_step + step_adj
                theoretical_grid = [anchor - anchor_idx * current_step + i * current_step for i in range(expected)]
                tolerance = current_step * 0.35
                score = 0
                for raw_val in raw_coords:
                    min_dist = min([abs(raw_val - t_val) for t_val in theoretical_grid])
                    if min_dist < tolerance:
                        score += (1.0 - (min_dist / tolerance))
                if score > max_score:
                    max_score, best_grid = score, theoretical_grid
    return best_grid


def _generate_nominal_grid(output_size=config.CROP_OUTPUT_SIZE, grid_size=config.GRID_SIZE):
    """Generate a centered 10x10 microwell grid inside the cropped frame."""
    span = 480.0
    start = (output_size - span) / 2.0
    step = span / float(grid_size - 1)
    coords = []
    for r in range(grid_size):
        for c in range(grid_size):
            coords.append((float(start + c * step), float(start + r * step)))
    return coords


def _fallback_result(img_green, img_yellow) -> CropResult:
    """Result used when the microwell grid cannot be accurately fitted."""
    base = img_green if img_green is not None else img_yellow
    h_img, w_img = base.shape[:2]
    size = config.CROP_OUTPUT_SIZE

    # Crop the central region where the microwell chip is located so cell scale is preserved
    x1 = max(0, min(config.ROI_MIN_X, w_img - 100))
    x2 = min(w_img, max(config.ROI_MAX_X, x1 + 100))
    y1 = max(0, min(config.ROI_MIN_Y, h_img - 100))
    y2 = min(h_img, max(config.ROI_MAX_Y, y1 + 100))

    chip_crop = base[y1:y2, x1:x2]
    if chip_crop.size > 0:
        cropped = cv2.resize(chip_crop, (size, size))
    else:
        cropped = cv2.resize(base, (size, size))

    yellow = img_yellow.copy() if img_yellow is not None else base.copy()
    nominal_wells = _generate_nominal_grid(size, config.GRID_SIZE)
    return CropResult(cropped, yellow, yellow.copy(), nominal_wells, float(config.DEFAULT_WELL_RADIUS_PX), False)


def _detect_circles(img_yellow) -> np.ndarray:
    gray_yellow = cv2.cvtColor(img_yellow, cv2.COLOR_RGB2GRAY)
    normalized_img = cv2.normalize(gray_yellow, None, -50, 230, cv2.NORM_MINMAX)
    enhanced_img = np.clip(cv2.createCLAHE(7.0, (8, 8)).apply(cv2.medianBlur(normalized_img, 5)).astype(np.int16) + 30,
                           0, 255).astype(np.uint8)

    # Pass 1: standard parameters
    circles = cv2.HoughCircles(enhanced_img, cv2.HOUGH_GRADIENT, config.HOUGH_DP, config.HOUGH_MIN_DIST,
                               param1=config.HOUGH_PARAM1, param2=config.HOUGH_PARAM2,
                               minRadius=config.HOUGH_MIN_RADIUS, maxRadius=config.HOUGH_MAX_RADIUS)

    # Pass 2: relaxed threshold if pass 1 found very few circles
    if circles is None or circles.shape[1] < 20:
        circles_relaxed = cv2.HoughCircles(enhanced_img, cv2.HOUGH_GRADIENT, config.HOUGH_DP, config.HOUGH_MIN_DIST - 5,
                                           param1=config.HOUGH_PARAM1, param2=11,
                                           minRadius=max(10, config.HOUGH_MIN_RADIUS - 3),
                                           maxRadius=config.HOUGH_MAX_RADIUS + 4)
        if circles_relaxed is not None and (circles is None or circles_relaxed.shape[1] > circles.shape[1]):
            circles = circles_relaxed

    if circles is None:
        return np.empty((0, 3), dtype=np.float32)

    circles_list = circles[0, :].astype(np.float32)
    mask_x = (circles_list[:, 0] > config.ROI_MIN_X) & (circles_list[:, 0] < config.ROI_MAX_X)
    mask_y = (circles_list[:, 1] > config.ROI_MIN_Y) & (circles_list[:, 1] < config.ROI_MAX_Y)
    filtered = circles_list[mask_x & mask_y]

    # If ROI filter leaves too few circles but image has plenty, use all circles
    if len(filtered) < 15 and len(circles_list) >= 15:
        return circles_list
    return filtered


def crop_well_image(img_green, img_yellow, offset_x=0, offset_y=0) -> CropResult:
    """Locate the 10x10 microwell grid on the bright-field image, deskew both
    channels and crop/resize the fluorescence image around the grid.

    Never raises for "grid not found" situations; check ``result.grid_found``.
    """
    if img_green is None and img_yellow is None:
        raise ValueError("Both input images are None")
    if img_yellow is None or img_green is None:
        return _fallback_result(img_green, img_yellow)

    roi_circles = _detect_circles(img_yellow)
    clean_circles = filter_interstitial_noise(roi_circles)
    if len(clean_circles) == 0:
        return _fallback_result(img_green, img_yellow)

    # --- Deskew -----------------------------------------------------------
    angle_deg = estimate_skew_angle(clean_circles[:, :2])
    h_img, w_img = img_green.shape[:2]
    center_pt = (w_img // 2, h_img // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center_pt, angle_deg, 1.0)
    img_green = cv2.warpAffine(img_green, rotation_matrix, (w_img, h_img), flags=cv2.INTER_LINEAR)
    img_yellow = cv2.warpAffine(img_yellow, rotation_matrix, (w_img, h_img), flags=cv2.INTER_LINEAR)

    coords_homogeneous = np.hstack([clean_circles[:, :2], np.ones((len(clean_circles), 1))])
    clean_circles = clean_circles.copy()
    clean_circles[:, :2] = rotation_matrix.dot(coords_homogeneous.T).T

    # --- Grid fitting -------------------------------------------------------
    valid_cols = group_coordinates(clean_circles[:, 0], config.GROUP_TOLERANCE, config.GROUP_MIN_POINTS)
    valid_rows = group_coordinates(clean_circles[:, 1], config.GROUP_TOLERANCE, config.GROUP_MIN_POINTS)
    if not valid_cols or not valid_rows:
        return _fallback_result(img_green, img_yellow)

    full_cols = extract_perfect_grid(valid_cols, clean_circles[:, 0])
    full_rows = extract_perfect_grid(valid_rows, clean_circles[:, 1])

    avg_radius = int(np.mean(clean_circles[:, 2])) + 2
    padding = avg_radius + config.CROP_PADDING_EXTRA

    crop_x_s = max(0, int(full_cols[0] - padding))
    crop_y_s = max(0, int(full_rows[0] - padding))
    crop_x_e = min(w_img, int(full_cols[-1] + padding))
    crop_y_e = min(h_img, int(full_rows[-1] + padding))
    if crop_x_e <= crop_x_s or crop_y_e <= crop_y_s:
        return _fallback_result(img_green, img_yellow)

    img_yellow_clean_full = img_yellow.copy()
    img_yellow_grid_full = img_yellow.copy()
    for r in full_rows:
        for c in full_cols:
            cv2.circle(img_yellow_grid_full, (int(c), int(r)), avg_radius, 255, 2)

    cropped_img = img_green[crop_y_s:crop_y_e, crop_x_s:crop_x_e]
    orig_h, orig_w = cropped_img.shape[:2]
    size = config.CROP_OUTPUT_SIZE
    cropped_img = cv2.resize(cropped_img, (size, size))
    scale_x = float(size) / orig_w
    scale_y = float(size) / orig_h
    scaled_radius = avg_radius * ((scale_x + scale_y) / 2.0)

    local_well_coords = []
    for r in full_rows:
        for c in full_cols:
            local_x = (c - crop_x_s + offset_x) * scale_x
            local_y = (r - crop_y_s + offset_y) * scale_y
            local_well_coords.append((float(local_x), float(local_y)))

    return CropResult(cropped_img, img_yellow_clean_full, img_yellow_grid_full,
                      local_well_coords, float(scaled_radius), True)
