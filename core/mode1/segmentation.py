import cv2
import numpy as np

import config
from core.common.segmentation_utils import (preprocess, sigmoid, filter_components,
                                            scale_contours, find_cell_centers)


class SegmentationProcessor:
    def __init__(self, conf_threshold=config.MODE1_CONF_THRESHOLD,
                 min_component_area=config.MODE1_MIN_COMPONENT_AREA):
        self.conf_threshold = conf_threshold
        self.min_component_area = min_component_area

    def process(self, img, model):
        """Run single-class segmentation.

        Returns ``(overlay_bgr, total_cells, cell_boxes)`` where boxes are
        ``(x1, y1, x2, y2)`` in ``img`` coordinates.
        """
        input_w, input_h = model.get_input_size()
        orig_h, orig_w = img.shape[:2]

        img_resized, img_tensor = preprocess(img, input_w, input_h)
        raw_output = model.inference(img_tensor)

        prob_mask = sigmoid(raw_output).squeeze()
        binary_mask = (prob_mask > self.conf_threshold).astype(np.uint8) * 255

        scale_x = orig_w / input_w
        scale_y = orig_h / input_h

        clean_mask, labels, stats, centroids, valid_ids = filter_components(binary_mask, self.min_component_area)

        overlay = img.copy()
        mask_orig = cv2.resize(clean_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        colored_mask = np.zeros_like(overlay)
        colored_mask[mask_orig > 0] = [0, 0, 255]
        overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.3, 0)

        scaled_contours, cell_boxes = scale_contours(clean_mask, scale_x, scale_y)
        cv2.drawContours(overlay, scaled_contours, -1, (0, 255, 0), 1)

        centers = find_cell_centers(labels, stats, centroids, valid_ids, img_resized, scale_x, scale_y)
        for cx, cy in centers:
            cv2.circle(overlay, (cx, cy), radius=1, color=(0, 0, 255), thickness=-1)

        return overlay, len(centers), cell_boxes
