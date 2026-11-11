"""Shared building blocks for the single- and multi-label segmentation pipelines."""
from typing import List, Tuple

import cv2
import numpy as np

import config


def sigmoid(x):
    """Numerically stable logistic function (no overflow warnings)."""
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def preprocess(img_bgr, input_w, input_h):
    """Return (resized RGB uint8 image, NCHW float32 tensor in [0, 1])."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (input_w, input_h))

    # In fluorescence microscopy, unspecific autofluorescence creates a non-zero
    # pedestal baseline (~15-40 intensity) across the whole sensor frame. If left
    # uncorrected, neural models saturate on the entire background.
    # We estimate the dark background floor per channel using the 5th percentile
    # and gently pull the pedestal floor down near 0 so cells pop out cleanly.
    img_clean = img_resized.astype(np.float32)
    for c in range(3):
        floor = np.percentile(img_clean[:, :, c], 5)
        if floor > 12.0:
            sub = min(30.0, float(floor - 4.0))
            img_clean[:, :, c] = np.maximum(0.0, img_clean[:, :, c] - sub)

    img_clean = np.clip(img_clean, 0, 255).astype(np.uint8)

    # If the fluorescence image has low dynamic range (e.g. max intensity < 70),
    # gently boost intensity for tensor inference so dim cells can be detected
    max_val = float(np.max(img_clean))
    if 0 < max_val < 70.0:
        scale = min(2.5, 120.0 / max_val)
        img_for_tensor = np.clip(img_clean.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    else:
        img_for_tensor = img_clean

    tensor = np.transpose(img_for_tensor.astype(np.float32) / 255.0, [2, 0, 1])[np.newaxis, ...]
    return img_resized, tensor


def filter_components(binary_mask, min_area, max_area=config.MAX_COMPONENT_AREA,
                      max_dimension=config.MAX_COMPONENT_DIMENSION):
    """Remove noise components smaller than ``min_area``, abnormally large artifacts,
    or broad border-framing noise lines, while preserving genuine cells even when
    they touch image borders or well boundaries.

    Vectorised: single lookup-table indexing instead of one full-image comparison per component.

    Returns ``(clean_mask, labels, stats, centroids, valid_ids)``.
    """
    h, w = binary_mask.shape[:2]
    clean_binary = binary_mask.copy()

    # Clear 1-pixel frame border noise lines ONLY if an edge row/col is heavily populated (> 35 px along edge)
    # This severs thin edge lines without erasing real cells touching the border
    for edge in [clean_binary[0, :], clean_binary[-1, :], clean_binary[:, 0], clean_binary[:, -1]]:
        if np.count_nonzero(edge) > 35:
            edge[:] = 0

    num_components, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_binary, connectivity=8)
    if num_components <= 1:
        return np.zeros_like(binary_mask), labels, stats, centroids, np.empty((0,), dtype=int)

    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False  # background

    # Identify border-spanning artifacts vs compact cells touching an edge:
    touch_left = stats[:, cv2.CC_STAT_LEFT] <= 1
    touch_right = stats[:, cv2.CC_STAT_LEFT] + stats[:, cv2.CC_STAT_WIDTH] >= w - 1
    touch_top = stats[:, cv2.CC_STAT_TOP] <= 1
    touch_bottom = stats[:, cv2.CC_STAT_TOP] + stats[:, cv2.CC_STAT_HEIGHT] >= h - 1

    # Component spans all the way across opposite edges of the frame -> definitely border noise
    spans_opposite = (touch_left & touch_right) | (touch_top & touch_bottom)

    # Broad border artifact (> 70 px along the edge; real cells are ~5-25 px)
    long_edge_artifact = (
        (touch_top & (stats[:, cv2.CC_STAT_WIDTH] > 70))
        | (touch_bottom & (stats[:, cv2.CC_STAT_WIDTH] > 70))
        | (touch_left & (stats[:, cv2.CC_STAT_HEIGHT] > 70))
        | (touch_right & (stats[:, cv2.CC_STAT_HEIGHT] > 70))
    )

    # Very thin border noise lines along edges (> 30 px long, <= 2 px thin)
    thin_edge_line = (
        ((touch_top | touch_bottom) & (stats[:, cv2.CC_STAT_HEIGHT] <= 2) & (stats[:, cv2.CC_STAT_WIDTH] > 30))
        | ((touch_left | touch_right) & (stats[:, cv2.CC_STAT_WIDTH] <= 2) & (stats[:, cv2.CC_STAT_HEIGHT] > 30))
    )

    is_artifact = spans_opposite | long_edge_artifact | thin_edge_line
    keep[is_artifact] = False

    # Discard components that are both touching a border and excessively large (> max_dimension)
    if max_dimension is not None:
        border_and_large = (touch_left | touch_right | touch_top | touch_bottom) & (
            (stats[:, cv2.CC_STAT_WIDTH] > max_dimension) | (stats[:, cv2.CC_STAT_HEIGHT] > max_dimension)
        )
        keep[border_and_large] = False

    # Discard frame-sized artifacts (area > 20000 px) but keep legitimate dense internal cell colonies
    if max_area is not None:
        keep[stats[:, cv2.CC_STAT_AREA] > max_area * 4] = False

    clean_mask = np.where(keep[labels], 255, 0).astype(np.uint8)
    valid_ids = np.flatnonzero(keep)
    return clean_mask, labels, stats, centroids, valid_ids


def scale_contours(clean_mask, scale_x, scale_y) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """Find external contours on ``clean_mask`` and rescale them to the original image.

    Returns ``(scaled_contours, bounding_boxes)`` with boxes as ``(x1, y1, x2, y2)``.
    """
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scale = np.array([scale_x, scale_y], dtype=np.float32)
    scaled_contours, boxes = [], []
    for cnt in contours:
        scaled_cnt = (cnt.astype(np.float32) * scale).astype(np.int32)
        scaled_contours.append(scaled_cnt)
        x, y, w, h = cv2.boundingRect(scaled_cnt)
        boxes.append((x, y, x + w, y + h))
    return scaled_contours, boxes


def find_cell_centers(labels, stats, centroids, valid_ids, img_resized_rgb, scale_x, scale_y,
                      single_cell_threshold=config.SINGLE_CELL_AREA_THRESHOLD) -> List[Tuple[int, int]]:
    """Return one (x, y) centre per detected cell, in original-image coordinates.

    Small components count as a single cell. Larger components (clusters) are
    split by searching for local intensity maxima inside the component.
    """
    input_h, input_w = labels.shape[:2]
    clahe = cv2.createCLAHE(config.CLUSTER_CLAHE_CLIP, config.CLUSTER_CLAHE_TILE)
    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)
    centers = []

    def component_center(i):
        return int(round(centroids[i][0] * scale_x)), int(round(centroids[i][1] * scale_y))

    for i in valid_ids:
        if stats[i, cv2.CC_STAT_AREA] < single_cell_threshold:
            centers.append(component_center(i))
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        x1 = max(0, x - config.CLUSTER_PAD_LEFT_TOP)
        y1 = max(0, y - config.CLUSTER_PAD_LEFT_TOP)
        x2 = min(input_w, x + w + config.CLUSTER_PAD_RIGHT)
        y2 = min(input_h, y + h + config.CLUSTER_PAD_BOTTOM)

        local_cluster_mask = labels[y1:y2, x1:x2] == i
        if not np.any(local_cluster_mask):
            continue

        local_patch = img_resized_rgb[y1:y2, x1:x2]
        # Preserve red/green peak intensities equally without 70% red attenuation from cv2.COLOR_RGB2GRAY
        local_gray = np.max(local_patch, axis=2)
        local_gray = cv2.normalize(local_gray, None, 0, 255, cv2.NORM_MINMAX)
        local_gray = np.clip(clahe.apply(local_gray).astype(np.int16) + config.CLUSTER_INTENSITY_OFFSET,
                             0, 255).astype(np.uint8)

        local_max = cv2.dilate(local_gray, kernel3)
        peaks_mask = (local_gray == local_max) & (local_gray > config.CLUSTER_MIN_PEAK_INTENSITY) & local_cluster_mask
        peaks_uint8 = cv2.dilate(peaks_mask.astype(np.uint8) * 255, kernel5)

        num_peaks, _, _, peak_centroids = cv2.connectedComponentsWithStats(peaks_uint8, connectivity=8)

        if num_peaks <= 2:
            centers.append(component_center(i))
            continue

        for p_id in range(1, num_peaks):
            local_cx, local_cy = peak_centroids[p_id]
            py, px = int(round(local_cy)), int(round(local_cx))
            if py >= local_cluster_mask.shape[0] or px >= local_cluster_mask.shape[1] or not local_cluster_mask[py, px]:
                continue
            centers.append((int(round((x1 + local_cx) * scale_x)), int(round((y1 + local_cy) * scale_y))))

    return centers
