"""Background batch-processing thread shared by both analysis modes."""
import logging
import os

import cv2
from PySide6.QtCore import QThread, Signal

from core.common.extract_ROI import CropResult, crop_well_image

logger = logging.getLogger("CellCounter.Processing")

GRID_NOT_FOUND_MSG = "Microwell grid not detected"


class BaseProcessingThread(QThread):
    """Reads image pairs, crops the well grid and delegates analysis to
    :meth:`process_item`. Supports cooperative cancellation through
    ``requestInterruption()`` (never use ``terminate()``).
    """
    item_processed = Signal(str, object)      # filename, payload dict
    image_progress = Signal(str, int, str)    # filename, percent, status
    batch_progress = Signal(int)              # number of images handled
    error_occurred = Signal(str, str)         # filename, message
    warning_occurred = Signal(str, str)       # filename, message
    all_done = Signal()                       # not named `finished` to avoid shadowing QThread.finished

    def __init__(self, detector, seg_processor, fluo_dir, bright_dir, image_files, parent=None):
        super().__init__(parent)
        self.detector = detector
        self.seg_processor = seg_processor
        self.fluo_dir = fluo_dir
        self.bright_dir = bright_dir
        self.image_files = list(image_files)

    def process_item(self, img_name: str, crop: CropResult) -> dict:
        """Analyse one cropped sample and return a payload for the GUI thread."""
        raise NotImplementedError

    def run(self):
        for idx, img_name in enumerate(self.image_files):
            if self.isInterruptionRequested():
                break
            try:
                self.image_progress.emit(img_name, 10, "Reading images...")
                img_green = cv2.imread(os.path.join(self.fluo_dir, img_name))
                img_yellow = cv2.imread(os.path.join(self.bright_dir, img_name))

                if img_green is None or img_yellow is None:
                    self.error_occurred.emit(img_name, "File corrupted or missing")
                    continue

                self.image_progress.emit(img_name, 30, "Aligning and cropping...")
                crop = crop_well_image(img_green, img_yellow)
                if not crop.grid_found:
                    logger.warning("%s: %s", img_name, GRID_NOT_FOUND_MSG)
                    self.warning_occurred.emit(img_name, GRID_NOT_FOUND_MSG)

                if self.isInterruptionRequested():
                    break

                self.image_progress.emit(img_name, 60, "Inference...")
                payload = self.process_item(img_name, crop)
                payload.setdefault("grid_found", crop.grid_found)

                self.image_progress.emit(img_name, 100, "Done")
                self.item_processed.emit(img_name, payload)

            except Exception as e:
                logger.error("Failed processing %s: %s", img_name, e, exc_info=True)
                self.error_occurred.emit(img_name, f"Error: {e}")

            finally:
                self.batch_progress.emit(idx + 1)

        self.all_done.emit()
