from PySide6.QtWidgets import QCheckBox

import config
from ui.common.detail_dialog import BaseDetailDialog
from ui.common.sample_card import BaseSampleCard
from utils.helpers import draw_scale_bar


class SingleLabelDetailDialog(BaseDetailDialog):
    def __init__(self, raw_images, yellow_grid, yellow_clean, filename="", well_statuses=None,
                 avg_radius=config.WELL_RADIUS_UM * config.BASE_PX_PER_UM, parent=None):
        super().__init__(filename, yellow_grid, yellow_clean, parent)
        self.raw_images = raw_images
        self.well_statuses = well_statuses or []
        self.avg_radius = avg_radius
        self.build_ui()

    def build_fluorescence_controls(self, layout):
        self.cb_grid = QCheckBox("Show Grid Overlay")
        self.cb_grid.setChecked(True)
        self.cb_cells = QCheckBox("Show Overlay Cell Prediction")
        self.cb_cells.setChecked(True)
        self.cb_scale = QCheckBox("Show Scale Bar")
        self.cb_scale.setChecked(False)

        for cb in (self.cb_grid, self.cb_cells, self.cb_scale):
            cb.toggled.connect(self.update_fluorescence_view)
            layout.addWidget(cb)

    def render_fluorescence(self):
        key = {
            (True, True): 'both',
            (True, False): 'grid',
            (False, True): 'cells',
            (False, False): 'clean',
        }[(self.cb_grid.isChecked(), self.cb_cells.isChecked())]
        base_bgr = self.raw_images[key]

        if self.cb_scale.isChecked():
            return draw_scale_bar(base_bgr, self.well_statuses, self.avg_radius)
        return base_bgr


class SingleLabelSampleCard(BaseSampleCard):
    def __init__(self, filename, parent=None):
        super().__init__(filename, selectable=True, parent=parent)

    def set_result(self, preview_pixmap, cell_count, coverage_percent):
        self.set_finished(
            preview_pixmap,
            f"{self.filename}\nCell count: {cell_count} | Coverage: {int(coverage_percent)}%"
        )
