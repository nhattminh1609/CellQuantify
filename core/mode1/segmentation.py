import cv2
import numpy as np
# import time


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class SegmentationProcessor:
    def __init__(self, conf_threshold=0.8):
        self.conf_threshold = conf_threshold

    def process(self, img, model):
        input_w, input_h = model.get_input_size()
        orig_h, orig_w = img.shape[:2]

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (input_w, input_h))
        img_normalized = img_resized.astype(np.float32) / 255.0

        img_tensor = np.transpose(img_normalized, [2, 0, 1])
        img_tensor = np.expand_dims(img_tensor, axis=0)

        # start = time.time()

        raw_output = model.inference(img_tensor)

        prob_mask = sigmoid(raw_output).squeeze()
        binary_mask = (prob_mask > self.conf_threshold).astype(np.uint8) * 255

        # print('s2:', round((time.time() - start) * 1000, 3))
        # start = time.time()

        overlay = img.copy()
        cell_boxes = []
        total_cells = 0

        scale_x = orig_w / input_w
        scale_y = orig_h / input_h

        num_components, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        SINGLE_CELL_THRESHOLD = 100

        clean_binary_mask = np.zeros_like(binary_mask)
        valid_components = []

        for i in range(1, num_components):
            if stats[i, cv2.CC_STAT_AREA] >= 5:
                clean_binary_mask[labels == i] = 255
                valid_components.append(i)

        mask_orig = cv2.resize(clean_binary_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        colored_mask = np.zeros_like(overlay)
        colored_mask[mask_orig > 0] = [0, 0, 255]
        overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.3, 0)

        contours, _ = cv2.findContours(clean_binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        scaled_contours = []

        for cnt in contours:
            scaled_cnt = np.zeros_like(cnt, dtype=np.float32)
            scaled_cnt[:, 0, 0] = cnt[:, 0, 0] * scale_x
            scaled_cnt[:, 0, 1] = cnt[:, 0, 1] * scale_y
            scaled_cnt = scaled_cnt.astype(np.int32)
            scaled_contours.append(scaled_cnt)
            x, y, w, h = cv2.boundingRect(scaled_cnt)
            cell_boxes.append((x, y, x + w, y + h))

        cv2.drawContours(overlay, scaled_contours, -1, (0, 255, 0), 1)

        for i in valid_components:
            area = stats[i, cv2.CC_STAT_AREA]

            if area < SINGLE_CELL_THRESHOLD:
                cx = int(round(centroids[i][0] * scale_x))
                cy = int(round(centroids[i][1] * scale_y))
                cv2.circle(overlay, (cx, cy), radius=1, color=(0, 0, 255), thickness=-1)
                total_cells += 1
                continue

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            pad = 2
            x1 = max(0, x-pad)
            y1 = max(0, y-pad)
            x2 = min(input_w, x+w+10)
            y2 = min(input_h, y+h+5)

            local_cluster_mask = labels[y1:y2, x1:x2] == i
            if not np.any(local_cluster_mask):
                continue

            local_img_crop = img_resized[y1:y2, x1:x2]
            local_gray = cv2.cvtColor(local_img_crop, cv2.COLOR_RGB2GRAY)
            local_gray_smooth = cv2.normalize(local_gray, None, 0, 255, cv2.NORM_MINMAX)
            local_gray_smooth = np.clip(cv2.createCLAHE(4.5, (4, 4)).apply(local_gray_smooth).astype(np.int16) - 10, 0, 255).astype(np.uint8)

            local_max = cv2.dilate(local_gray_smooth, np.ones((3, 3), np.uint8))
            peaks_mask = (local_gray_smooth == local_max) & (local_gray_smooth > 2) & local_cluster_mask
            peaks_uint8 = peaks_mask.astype(np.uint8) * 255
            peaks_uint8 = cv2.dilate(peaks_uint8, np.ones((5, 5), np.uint8))

            num_peaks, peak_labels, peak_stats, peak_centroids = cv2.connectedComponentsWithStats(peaks_uint8, connectivity=8)

            if num_peaks <= 2:
                cx = int(round(centroids[i][0] * scale_x))
                cy = int(round(centroids[i][1] * scale_y))
                cv2.circle(overlay, (cx, cy), radius=1, color=(0, 0, 255), thickness=-1)
                total_cells += 1
                continue

            for p_id in range(1, num_peaks):
                local_cx, local_cy = peak_centroids[p_id]

                py, px = int(round(local_cy)), int(round(local_cx))
                if py >= local_cluster_mask.shape[0] or px >= local_cluster_mask.shape[1] or not local_cluster_mask[py, px]:
                    continue

                global_cx = int(round((x1 + local_cx) * scale_x))
                global_cy = int(round((y1 + local_cy) * scale_y))

                cv2.circle(overlay, (global_cx, global_cy), radius=1, color=(0, 0, 255), thickness=-1)
                total_cells += 1

        # print('s3:', round((time.time() - start) * 1000, 3))

        return overlay, total_cells, cell_boxes
