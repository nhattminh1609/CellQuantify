import cv2
import numpy as np

import config
from core.common.segmentation_utils import (preprocess, sigmoid, filter_components,
                                            scale_contours, find_cell_centers)

CLASS_NAMES = config.MODE2_CLASS_NAMES


class MultiLabelSegmentationProcessor:
    def __init__(self, conf_threshold=None,
                 min_component_area=None):
        self.conf_threshold = conf_threshold
        self.min_component_area = min_component_area

    def process(self, img, model):
        """Run multi-label segmentation (one probability map per class).

        Returns a ``seg_data`` dict with keys ``masks``, ``contours``, ``peaks``,
        ``cell_boxes`` (``(x1, y1, x2, y2, label)``) and ``class_counts``.
        """
        input_w, input_h = model.get_input_size()
        orig_h, orig_w = img.shape[:2]

        img_resized, img_tensor = preprocess(img, input_w, input_h)
        raw_output = model.inference(img_tensor)

        prob_maps = sigmoid(raw_output).squeeze()
        if prob_maps.ndim == 2:
            prob_maps = np.expand_dims(prob_maps, axis=0)

        scale_x = orig_w / input_w
        scale_y = orig_h / input_h

        class_masks_orig = {}
        class_contours = {}
        class_peaks = {}
        class_counts = {name: 0 for name in CLASS_NAMES}
        cell_boxes = []

        num_channels = min(len(CLASS_NAMES), prob_maps.shape[0])

        for c_idx in range(num_channels):
            label_name = CLASS_NAMES[c_idx]

            # Determine confidence threshold for this class
            if isinstance(self.conf_threshold, (int, float)):
                th = float(self.conf_threshold)
            elif isinstance(self.conf_threshold, dict):
                th = float(self.conf_threshold.get(label_name, config.MODE2_CONF_THRESHOLD))
            else:
                th = float(config.MODE2_CLASS_CONF_THRESHOLDS.get(label_name, config.MODE2_CONF_THRESHOLD))

            # Binary mask generation with competitive dominance for undefined label
            if label_name == "undefine" and num_channels >= 3:
                # Competitive dominance rule: An undefined / ambiguous cell is only
                # recognized where its probability is not dominated by definitive A549 or NK cells.
                # This prevents spurious overlapping masks from corrupting genuine cell overlays.
                is_active = (prob_maps[c_idx] > th) & (prob_maps[c_idx] >= prob_maps[0]) & (prob_maps[c_idx] >= prob_maps[1])
            else:
                is_active = (prob_maps[c_idx] > th)

            binary_mask = is_active.astype(np.uint8) * 255

            # Safeguard: if background saturation makes the raw mask occupy > 40% of the entire frame,
            # adaptively raise the threshold to the top 15% quantile of that class to prevent wipeout by filter_components.
            if np.mean(binary_mask > 0) > 0.40:
                adaptive_th = max(th, float(np.percentile(prob_maps[c_idx], 85)))
                if label_name == "undefine" and num_channels >= 3:
                    is_active = (prob_maps[c_idx] > adaptive_th) & (prob_maps[c_idx] >= prob_maps[0]) & (prob_maps[c_idx] >= prob_maps[1])
                else:
                    is_active = (prob_maps[c_idx] > adaptive_th)
                binary_mask = is_active.astype(np.uint8) * 255

            # Determine minimum component area for this class
            if isinstance(self.min_component_area, (int, float)):
                min_area = int(self.min_component_area)
            elif isinstance(self.min_component_area, dict):
                min_area = int(self.min_component_area.get(label_name, config.MODE2_MIN_COMPONENT_AREA))
            else:
                min_area = int(config.MODE2_CLASS_MIN_AREAS.get(label_name, config.MODE2_MIN_COMPONENT_AREA))

            clean_mask, labels, stats, centroids, valid_ids = filter_components(binary_mask, min_area)
            class_masks_orig[label_name] = cv2.resize(clean_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            scaled_contours, boxes = scale_contours(clean_mask, scale_x, scale_y)
            class_contours[label_name] = scaled_contours
            cell_boxes.extend(box + (label_name,) for box in boxes)

            peaks = find_cell_centers(labels, stats, centroids, valid_ids, img_resized, scale_x, scale_y)
            class_peaks[label_name] = peaks
            class_counts[label_name] = len(peaks)

        return {
            "masks": class_masks_orig,
            "contours": class_contours,
            "peaks": class_peaks,
            "cell_boxes": cell_boxes,
            "class_counts": class_counts,
        }
