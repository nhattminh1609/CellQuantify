"""Pure (Qt-free) image/scale helpers. Qt conversions live in ui/common/qt_utils.py."""
import cv2

import config


def calculate_scale_from_crop(w_crop=None, well_coords=None):
    """Return ``(px_per_um, well_radius_px, well_pitch_px)`` for the cropped image.

    Priority: explicit crop width -> span of detected wells -> nominal calibration.
    """
    orig_ratio = config.BASE_PX_PER_UM

    def from_ratio(ratio):
        return ratio, config.WELL_RADIUS_UM * ratio, config.WELL_PITCH_UM * ratio

    if w_crop is not None and w_crop > 0:
        return from_ratio(orig_ratio * (float(config.CROP_OUTPUT_SIZE) / float(w_crop)))

    if well_coords and len(well_coords) >= 4:
        xs = [p[0] for p in well_coords]
        ys = [p[1] for p in well_coords]
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        if span > 0:
            return from_ratio(span / config.GRID_SPAN_UM)

    return from_ratio(orig_ratio)


def get_physical_grid_params(well_coords=None, w_crop=None):
    ratio, radius, pitch = calculate_scale_from_crop(w_crop, well_coords)
    return pitch, radius, ratio


def draw_scale_bar(canvas, well_statuses, scale_param, bar_um=config.SCALE_BAR_UM):
    """Draw a scale bar below the bottom-right wells.

    ``scale_param`` is interpreted as a well radius in px when > 3.0, otherwise
    as a px/µm ratio (kept for backward compatibility).
    """
    img = canvas.copy()
    h, w = img.shape[:2]

    if scale_param > 3.0:
        ratio = scale_param / config.WELL_RADIUS_UM
        radius_approx = scale_param
    else:
        ratio = scale_param
        radius_approx = config.WELL_RADIUS_UM * ratio

    bar_len_px = int(round(bar_um * ratio))

    w9 = None
    w10 = None
    if well_statuses:
        for well in well_statuses:
            r = well.get('row')
            c = well.get('col')
            if r == 10 and c == 9:
                w9 = well
            elif r == 10 and c == 10:
                w10 = well

        if w9 is None or w10 is None:
            sorted_by_y = sorted(well_statuses, key=lambda item: item['coord'][1])
            bottom_wells = sorted_by_y[-10:] if len(sorted_by_y) >= 10 else sorted_by_y
            sorted_bottom = sorted(bottom_wells, key=lambda item: item['coord'][0])
            if len(sorted_bottom) >= 2:
                w9 = sorted_bottom[-2]
                w10 = sorted_bottom[-1]

    if w9 and w10:
        c9_x, c9_y = w9['coord']
        c10_x, c10_y = w10['coord']
        center_x = (c9_x + c10_x) / 2.0
        x1 = int(round(center_x - bar_len_px / 2.0))
        x2 = x1 + bar_len_px
        y_bottom = max(c9_y, c10_y) + radius_approx
        y_bar = int(round(y_bottom + 12))
    else:
        x2 = w - 25
        x1 = x2 - bar_len_px
        y_bar = h - 25

    y_bar = min(h - 18, max(25, y_bar))
    if x1 < 10:
        x1 = 10
        x2 = x1 + bar_len_px
    if x2 > w - 10:
        x2 = w - 10
        x1 = x2 - bar_len_px

    bar_thickness = 3
    # cv2.rectangle(img, (x1 - 1, y_bar - 1), (x2 + 1, y_bar + bar_thickness + 1), (0, 0, 0), -1)
    cv2.rectangle(img, (x1, y_bar), (x2, y_bar + bar_thickness), (0, 0, 255), -1)

    tick_h = 3
    cv2.line(img, (x1-1, y_bar - tick_h), (x1-1, y_bar + bar_thickness + tick_h), (0, 0, 255), 1)
    cv2.line(img, (x2+1, y_bar - tick_h), (x2+1, y_bar + bar_thickness + tick_h), (0, 0, 255), 1)

    label = f"{int(bar_um)} µm"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.05
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
    tx = int(round((x1 + x2) / 2.0 - tw / 2.0))
    ty = y_bar - 10
    if ty - th < 5:
        ty = y_bar + bar_thickness + th + 6

    cv2.putText(img, label, (tx, ty), font, font_scale, (0, 0, 255), 1, cv2.LINE_AA)

    return img