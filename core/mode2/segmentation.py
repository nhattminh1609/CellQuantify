import cv2
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

CLASS_NAMES = ["a549", "nk", "undefine"]

CLASS_COLORS = {
    "a549": {"fill": [0, 180, 0], "border": (0, 255, 0), "dot": (0, 255, 0)},
    "nk": {"fill": [0, 0, 180], "border": (0, 0, 255), "dot": (0, 0, 255)},
    "undefine": {"fill": [0, 180, 180], "border": (0, 255, 255), "dot": (0, 255, 255)}
}

class MultiLabelSegmentationProcessor:
    def __init__(self, conf_threshold=0.8):
        self.conf_threshold = conf_threshold

    def process(self, img, model):
        if hasattr(model, "get_input_size"):
            input_w, input_h = model.get_input_size()
        else:
            meta = model.session.get_inputs()[0]
            input_h = meta.shape[2] if isinstance(meta.shape[2], int) else 640
            input_w = meta.shape[3] if isinstance(meta.shape[3], int) else 640

        orig_h, orig_w = img.shape[:2]

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (input_w, input_h))
        img_normalized = img_resized.astype(np.float32) / 255.0

        img_tensor = np.transpose(img_normalized, [2, 0, 1])
        img_tensor = np.expand_dims(img_tensor, axis=0)

        if hasattr(model, "inference"):
            raw_output = model.inference(img_tensor)
        else:
            input_name = model.session.get_inputs()[0].name
            raw_output = model.session.run(None, {input_name: img_tensor})[0]

        prob_maps = sigmoid(raw_output).squeeze()
        if prob_maps.ndim == 2:
            prob_maps = np.expand_dims(prob_maps, axis=0)

        scale_x = orig_w / input_w
        scale_y = orig_h / input_h
        SINGLE_CELL_THRESHOLD = 100

        class_masks_orig = {}
        class_contours = {}
        class_peaks = {}
        class_counts = {name: 0 for name in CLASS_NAMES}
        cell_boxes = []

        num_channels = min(len(CLASS_NAMES), prob_maps.shape[0])

        for c_idx in range(num_channels):
            label_name = CLASS_NAMES[c_idx]
            ch_prob = prob_maps[c_idx]
            binary_mask = (ch_prob > self.conf_threshold).astype(np.uint8) * 255

            num_components, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
            clean_binary_mask = np.zeros_like(binary_mask)
            valid_components = []

            for i in range(1, num_components):
                if stats[i, cv2.CC_STAT_AREA] >= 30:
                    clean_binary_mask[labels == i] = 255
                    valid_components.append(i)

            mask_orig = cv2.resize(clean_binary_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            class_masks_orig[label_name] = mask_orig

            contours, _ = cv2.findContours(clean_binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            scaled_contours = []
            for cnt in contours:
                scaled_cnt = np.zeros_like(cnt, dtype=np.float32)
                scaled_cnt[:, 0, 0] = cnt[:, 0, 0] * scale_x
                scaled_cnt[:, 0, 1] = cnt[:, 0, 1] * scale_y
                scaled_cnt = scaled_cnt.astype(np.int32)
                scaled_contours.append(scaled_cnt)
                x, y, w, h = cv2.boundingRect(scaled_cnt)
                cell_boxes.append((x, y, x + w, y + h, label_name))

            class_contours[label_name] = scaled_contours
            peaks_list = []

            for i in valid_components:
                area = stats[i, cv2.CC_STAT_AREA]

                if area < SINGLE_CELL_THRESHOLD:
                    cx = int(round(centroids[i][0] * scale_x))
                    cy = int(round(centroids[i][1] * scale_y))
                    peaks_list.append((cx, cy))
                    class_counts[label_name] += 1
                    continue

                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]

                pad = 2
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(input_w, x + w + 10)
                y2 = min(input_h, y + h + 5)

                local_cluster_mask = labels[y1:y2, x1:x2] == i
                if not np.any(local_cluster_mask):
                    continue

                local_img_crop = img_resized[y1:y2, x1:x2]
                local_gray = cv2.cvtColor(local_img_crop, cv2.COLOR_RGB2GRAY)
                local_gray_smooth = cv2.normalize(local_gray, None, 0, 255, cv2.NORM_MINMAX)
                local_gray_smooth = np.clip(
                    cv2.createCLAHE(4.5, (4, 4)).apply(local_gray_smooth).astype(np.int16) - 10, 0, 255
                ).astype(np.uint8)

                local_max = cv2.dilate(local_gray_smooth, np.ones((3, 3), np.uint8))
                peaks_mask = (local_gray_smooth == local_max) & (local_gray_smooth > 2) & local_cluster_mask
                peaks_uint8 = peaks_mask.astype(np.uint8) * 255
                peaks_uint8 = cv2.dilate(peaks_uint8, np.ones((5, 5), np.uint8))

                num_peaks, peak_labels, peak_stats, peak_centroids = cv2.connectedComponentsWithStats(
                    peaks_uint8, connectivity=8
                )

                if num_peaks <= 2:
                    cx = int(round(centroids[i][0] * scale_x))
                    cy = int(round(centroids[i][1] * scale_y))
                    peaks_list.append((cx, cy))
                    class_counts[label_name] += 1
                    continue

                for p_id in range(1, num_peaks):
                    local_cx, local_cy = peak_centroids[p_id]
                    py, px = int(round(local_cy)), int(round(local_cx))
                    if (
                        py >= local_cluster_mask.shape[0]
                        or px >= local_cluster_mask.shape[1]
                        or not local_cluster_mask[py, px]
                    ):
                        continue

                    global_cx = int(round((x1 + local_cx) * scale_x))
                    global_cy = int(round((y1 + local_cy) * scale_y))
                    peaks_list.append((global_cx, global_cy))
                    class_counts[label_name] += 1

            class_peaks[label_name] = peaks_list

        seg_data = {
            "masks": class_masks_orig,
            "contours": class_contours,
            "peaks": class_peaks,
            "cell_boxes": cell_boxes,
            "class_counts": class_counts
        }

        return seg_data