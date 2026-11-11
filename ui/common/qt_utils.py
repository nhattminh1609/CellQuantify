"""Qt <-> OpenCV conversion helpers."""
import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap


def cv2_to_qpixmap(cv_img):
    """Convert a BGR (or grayscale) numpy image to a QPixmap (deep copy)."""
    if cv_img is None:
        return QPixmap()
    if cv_img.ndim == 2:
        img = np.ascontiguousarray(cv_img)
        h, w = img.shape
        q_img = QImage(img.data, w, h, img.strides[0], QImage.Format_Grayscale8)
    else:
        img = np.ascontiguousarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        h, w = img.shape[:2]
        q_img = QImage(img.data, w, h, img.strides[0], QImage.Format_RGB888)
    # QPixmap.fromImage copies the pixel data, so `img` may be freed afterwards.
    return QPixmap.fromImage(q_img)
