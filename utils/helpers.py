import cv2
from PySide6.QtGui import QPixmap, QImage


def cv2_to_qpixmap(cv_img):
    if len(cv_img.shape) == 3:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = cv_img.shape
        q_img = QImage(cv_img.data, w, h, ch * w, QImage.Format_RGB888)
    else:
        h, w = cv_img.shape
        q_img = QImage(cv_img.data, w, h, w, QImage.Format_Grayscale8)
    return QPixmap.fromImage(q_img)
