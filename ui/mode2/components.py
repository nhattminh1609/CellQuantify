from PySide6.QtWidgets import QCheckBox, QComboBox, QLabel

import config
from core.mode2.quantification import GRID_MODES, render_mode2_canvas
from ui.common.detail_dialog import BaseDetailDialog
from ui.common.sample_card import BaseSampleCard


class MultiLabelDetailDialog(BaseDetailDialog):
    def __init__(self, cropped_img, seg_data=None, well_statuses=None, avg_radius=None,
                 yellow_grid=None, yellow_clean=None, filename="", ratio_new=None, parent=None):
        super().__init__(filename, yellow_grid, yellow_clean, parent)
        self.cropped_img = cropped_img
        self.seg_data = seg_data if seg_data is not None else {}
        self.well_statuses = well_statuses if well_statuses is not None else []
        self.avg_radius = avg_radius if avg_radius is not None else config.WELL_RADIUS_UM * config.BASE_PX_PER_UM
        self.ratio_new = ratio_new if ratio_new is not None else (self.avg_radius / config.WELL_RADIUS_UM)
        self.build_ui()

    def build_fluorescence_controls(self, layout):
        self.combo_grid = QComboBox()
        self.combo_grid.addItems(GRID_MODES)
        self.combo_grid.currentTextChanged.connect(self.update_fluorescence_view)

        # Multi-select mask checkboxes to enable arbitrary mix-and-match overlays
        self.cb_mask_a549 = QCheckBox("A549")
        self.cb_mask_a549.setChecked(True)
        self.cb_mask_a549.setStyleSheet("QCheckBox { font-weight: bold; color: #15803d; }")
        self.cb_mask_a549.toggled.connect(self.update_fluorescence_view)

        self.cb_mask_nk = QCheckBox("NK")
        self.cb_mask_nk.setChecked(True)
        self.cb_mask_nk.setStyleSheet("QCheckBox { font-weight: bold; color: #b91c1c; }")
        self.cb_mask_nk.toggled.connect(self.update_fluorescence_view)

        self.cb_mask_undef = QCheckBox("Undefined")
        self.cb_mask_undef.setChecked(True)
        self.cb_mask_undef.setStyleSheet("QCheckBox { font-weight: bold; color: #0284c7; }")
        self.cb_mask_undef.toggled.connect(self.update_fluorescence_view)

        self.cb_scale = QCheckBox("Scale Bar")
        self.cb_scale.setChecked(False)
        self.cb_scale.toggled.connect(self.update_fluorescence_view)

        layout.addWidget(QLabel("Grid Mode:"))
        layout.addWidget(self.combo_grid)
        layout.addSpacing(15)

        layout.addWidget(QLabel("Masks:"))
        layout.addWidget(self.cb_mask_a549)
        layout.addWidget(self.cb_mask_nk)
        layout.addWidget(self.cb_mask_undef)
        layout.addSpacing(15)

        layout.addWidget(self.cb_scale)

    def get_active_masks(self):
        active = []
        if self.cb_mask_a549.isChecked():
            active.append("a549")
        if self.cb_mask_nk.isChecked():
            active.append("nk")
        if self.cb_mask_undef.isChecked():
            active.append("undefine")
        return active

    def render_fluorescence(self):
        return render_mode2_canvas(
            self.cropped_img,
            self.seg_data,
            self.well_statuses,
            self.avg_radius,
            ratio_new=self.ratio_new,
            grid_mode=self.combo_grid.currentText(),
            active_masks=self.get_active_masks(),
            show_scale_bar=self.cb_scale.isChecked()
        )


class MultiLabelSampleCard(BaseSampleCard):
    def __init__(self, filename, parent=None):
        super().__init__(filename, selectable=False, parent=parent)

    def set_result(self, preview_pixmap, class_counts, coverage_percent):
        self.set_finished(
            preview_pixmap,
            f"{self.filename} (Cov: {int(coverage_percent)}%)\n"
            f"a549: {class_counts.get('a549', 0)} | nk: {class_counts.get('nk', 0)} | "
            f"undef: {class_counts.get('undefine', 0)}"
        )
