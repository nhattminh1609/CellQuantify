import logging

import config
from core.mode2.quantification import quantify_multilabel_wells, render_mode2_canvas
from core.mode2.segmentation import MultiLabelSegmentationProcessor
from ui.common.base_window import BaseAnalysisWindow
from ui.common.processing import BaseProcessingThread
from ui.common.qt_utils import cv2_to_qpixmap
from ui.mode2.components import MultiLabelDetailDialog, MultiLabelSampleCard

logger = logging.getLogger("CellCounter.MultiLabel")


class MultiLabelProcessingThread(BaseProcessingThread):
    def process_item(self, img_name, crop):
        cropped_img = crop.cropped_img
        seg_data = self.seg_processor.process(cropped_img, self.detector)

        self.image_progress.emit(img_name, 90, "Quantifying cells...")
        well_statuses, cov_percent, phys_radius, ratio_new = quantify_multilabel_wells(
            crop.well_coords, seg_data["cell_boxes"])

        preview_img = render_mode2_canvas(
            cropped_img, seg_data, well_statuses, phys_radius, ratio_new=ratio_new,
            grid_mode="Full grid", mask_mode="All labels", show_scale_bar=False
        )

        return {
            "preview_img": preview_img,
            "cropped_img": cropped_img,
            "seg_data": seg_data,
            "wells": well_statuses,
            "class_counts": seg_data["class_counts"],
            "coverage": cov_percent,
            "avg_radius": float(phys_radius),
            "ratio_new": float(ratio_new),
            "yellow_grid": crop.yellow_grid,
            "yellow_clean": crop.yellow_clean,
        }


class MultiLabelWindow(BaseAnalysisWindow):
    WINDOW_TITLE = "Cell Counter - Multi-Label"
    MODEL_FILE = config.MODE2_MODEL_FILE
    GRID_SPACING = 20

    def create_processor(self):
        return MultiLabelSegmentationProcessor()

    def create_thread(self, fluo_dir, bright_dir, image_files):
        return MultiLabelProcessingThread(self.detector, self.seg_processor, fluo_dir, bright_dir, image_files)

    def create_card(self, filename):
        return MultiLabelSampleCard(filename)

    def on_item_processed(self, filename, payload):
        # The preview is only needed for the thumbnail; don't keep it in memory.
        preview_pixmap = cv2_to_qpixmap(payload.pop("preview_img"))
        self.image_data_store[filename] = payload
        card = self.grid_widgets.get(filename)
        if card is not None:
            card.set_result(preview_pixmap, payload["class_counts"], payload["coverage"])

    def create_dialog(self, filename, data):
        return MultiLabelDetailDialog(
            cropped_img=data["cropped_img"],
            seg_data=data["seg_data"],
            well_statuses=data["wells"],
            avg_radius=data["avg_radius"],
            ratio_new=data.get("ratio_new"),
            yellow_grid=data["yellow_grid"],
            yellow_clean=data["yellow_clean"],
            filename=filename,
            parent=None
        )
