import cv2
import numpy as np

def filter_interstitial_noise(circles_list, min_distance=45):
    valid_circles = []
    for circle in circles_list:
        if not valid_circles or np.min(np.hypot(np.array(valid_circles)[:, 0] - circle[0],
                                                np.array(valid_circles)[:, 1] - circle[1])) >= min_distance:
            valid_circles.append(circle)
    return np.array(valid_circles) if valid_circles else np.array([])

def group_coordinates(coords, tolerance=15, min_points=2, max_gap=65):
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

def estimate_skew_angle(coords, tolerance=20):
    if len(coords) < 3:
        return 0.0
    y_coords = np.sort(coords[:, 1])
    rows = np.split(coords, np.where(np.diff(y_coords) > tolerance)[0] + 1)
    angles = []
    for row in rows:
        mean_y = np.mean(row[:, 1])
        row_points = coords[np.abs(coords[:, 1] - mean_y) <= tolerance]
        if len(row_points) >= 3:
            slope, _ = np.polyfit(row_points[:, 0], row_points[:, 1], 1)
            angles.append(np.arctan(slope))
    if not angles:
        return 0.0
    median_angle_rad = np.median(angles)
    return np.degrees(median_angle_rad)

def extract_perfect_grid(coords, raw_coords, expected=10):
    if len(coords) < 2:
        return coords
    coords = np.sort(coords)
    valid_diffs = np.diff(coords)[np.diff(coords) > 40]
    if len(valid_diffs) == 0:
        return coords
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

def crop_well_image(img_green, img_yellow, offset_x=0, offset_y=0):
    cropped_img = img_yellow.copy() if img_yellow is not None else img_green.copy()
    local_well_coords = []
    avg_radius = 18

    if img_yellow is None:
        return cropped_img, local_well_coords, avg_radius

    gray_yellow = cv2.cvtColor(img_yellow, cv2.COLOR_RGB2GRAY)
    normalized_img = cv2.normalize(gray_yellow, None, -50, 230, cv2.NORM_MINMAX)
    enhanced_img = np.clip(cv2.createCLAHE(7.0, (8, 8)).apply(cv2.medianBlur(normalized_img, 5)).astype(np.int16) + 30,
                           0, 255).astype(np.uint8)

    circles = cv2.HoughCircles(enhanced_img, cv2.HOUGH_GRADIENT, 1.2, 35, param1=55, param2=15, minRadius=15,
                               maxRadius=20)
    mask = np.zeros_like(enhanced_img)

    if circles is not None:
        circles_list = circles[0, :].astype(np.float32)
        MIN_X, MAX_X = 250, 1050
        MIN_Y, MAX_Y = 150, 900
        mask_x = (circles_list[:, 0] > MIN_X) & (circles_list[:, 0] < MAX_X)
        mask_y = (circles_list[:, 1] > MIN_Y) & (circles_list[:, 1] < MAX_Y)
        roi_circles = circles_list[mask_x & mask_y]

        clean_circles = filter_interstitial_noise(roi_circles, min_distance=45)

        if len(clean_circles) > 0:
            angle_deg = estimate_skew_angle(clean_circles, tolerance=20)
            h_img, w_img = img_green.shape[:2]
            center_pt = (w_img // 2, h_img // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center_pt, angle_deg, 1.0)
            img_green = cv2.warpAffine(img_green, rotation_matrix, (w_img, h_img), flags=cv2.INTER_LINEAR)
            img_yellow = cv2.warpAffine(img_yellow, rotation_matrix, (w_img, h_img), flags=cv2.INTER_LINEAR)

            coords_xy = clean_circles[:, :2]
            ones = np.ones((len(coords_xy), 1))
            coords_homogeneous = np.hstack([coords_xy, ones])
            rotated_coords = rotation_matrix.dot(coords_homogeneous.T).T
            clean_circles[:, :2] = rotated_coords

        for x, y, r in clean_circles:
            if r >= 10:
                cv2.circle(mask, (int(x), int(y)), int(r), (255, 255, 255), -1)

        valid_cols = group_coordinates(clean_circles[:, 0], 10, 3)
        valid_rows = group_coordinates(clean_circles[:, 1], 10, 3)

        if valid_cols and valid_rows:
            full_cols = extract_perfect_grid(valid_cols, clean_circles[:, 0], 10)
            full_rows = extract_perfect_grid(valid_rows, clean_circles[:, 1], 10)

            avg_radius = int(np.mean(clean_circles[:, 2])) + 2
            padding = avg_radius + 15

            img_yellow_clean_full = img_yellow.copy()
            img_yellow_grid_full = img_yellow.copy()

            for r in full_rows:
                for c in full_cols:
                    cv2.circle(img_yellow_grid_full, (int(c), int(r)), avg_radius, 255, 2)

            h_img, w_img = img_green.shape[:2]
            crop_x_s = max(0, int(full_cols[0] - padding))
            crop_y_s = max(0, int(full_rows[0] - padding))
            crop_x_e = min(w_img, int(full_cols[-1] + padding))
            crop_y_e = min(h_img, int(full_rows[-1] + padding))

            cropped_img = img_green[crop_y_s:crop_y_e, crop_x_s:crop_x_e]

            orig_h, orig_w = cropped_img.shape[:2]
            if orig_h > 0 and orig_w > 0:
                cropped_img = cv2.resize(cropped_img, (640, 640))
                scale_x = 640.0 / orig_w
                scale_y = 640.0 / orig_h
                scaled_radius = avg_radius * ((scale_x + scale_y) / 2.0)
            else:
                scale_x = 1.0
                scale_y = 1.0
                scaled_radius = avg_radius

            for r in full_rows:
                for c in full_cols:
                    local_x = (c - crop_x_s + offset_x) * scale_x
                    local_y = (r - crop_y_s + offset_y) * scale_y
                    local_well_coords.append((local_x, local_y))

            avg_radius = scaled_radius

        return cropped_img, img_yellow_clean_full, img_yellow_grid_full, local_well_coords, avg_radius
