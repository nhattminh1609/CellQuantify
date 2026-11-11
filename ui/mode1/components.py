import os
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                               QCheckBox, QWidget, QTabWidget, QFrame,
                               QStackedWidget, QProgressBar, QSizePolicy, QPushButton,
                               QFileDialog, QMessageBox)
from PySide6.QtGui import QCursor
from PySide6.QtCore import Qt, Signal

from ui.common.theme import GLOBAL_STYLESHEET, SAMPLE_CARD_DEFAULT_STYLE, SAMPLE_CARD_SELECTED_STYLE
from ui.common.zoomable_viewer import ZoomableViewer
from utils.helpers import cv2_to_qpixmap, draw_scale_bar


class SingleLabelDetailDialog(QDialog):
    def __init__(self, raw_images, yellow_grid, yellow_clean, filename="", well_statuses=None, avg_radius=15.5,
                 parent=None):
        super().__init__(parent)
        self.filename = filename
        self.setWindowTitle(f"Detail Viewer - {filename}")
        self.setWindowFlags(
            Qt.Window | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint
        )
        self.resize(1400, 900)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setStyleSheet(GLOBAL_STYLESHEET)

        self.raw_images = raw_images
        self.yellow_grid = yellow_grid
        self.yellow_clean = yellow_clean
        self.well_statuses = well_statuses or []
        self.avg_radius = avg_radius

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)

        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self.on_tab_changed)
        layout.addWidget(self.tabs)

        green_tab = QWidget()
        green_layout = QVBoxLayout(green_tab)

        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(15)

        self.cb_grid = QCheckBox("Show Grid Overlay")
        self.cb_grid.setChecked(True)
        self.cb_grid.toggled.connect(self.update_green_view)

        self.cb_cells = QCheckBox("Show Overlay Cell Prediction")
        self.cb_cells.setChecked(True)
        self.cb_cells.toggled.connect(self.update_green_view)

        self.cb_scale = QCheckBox("Show Scale Bar")
        self.cb_scale.setChecked(False)
        self.cb_scale.toggled.connect(self.update_green_view)

        self.btn_export = QPushButton("Export Image")
        self.btn_export.setObjectName("BtnPrimary")
        self.btn_export.clicked.connect(self.export_current_image)

        controls_layout.addWidget(self.cb_grid)
        controls_layout.addWidget(self.cb_cells)
        controls_layout.addWidget(self.cb_scale)
        controls_layout.addStretch()
        controls_layout.addWidget(self.btn_export)

        self.green_viewer = ZoomableViewer()
        green_layout.addLayout(controls_layout)
        green_layout.addWidget(self.green_viewer)
        self.tabs.addTab(green_tab, "Fluorescence Channel")

        yellow_tab = QWidget()
        yellow_layout = QVBoxLayout(yellow_tab)

        yellow_top_layout = QHBoxLayout()
        self.cb_yellow = QCheckBox("Show Detected Grid")
        self.cb_yellow.setChecked(True)
        self.cb_yellow.toggled.connect(self.update_yellow_view)

        self.btn_export_yellow = QPushButton("Export Image")
        self.btn_export_yellow.setObjectName("BtnPrimary")
        self.btn_export_yellow.clicked.connect(self.export_current_image)

        yellow_top_layout.addWidget(self.cb_yellow)
        yellow_top_layout.addStretch()
        yellow_top_layout.addWidget(self.btn_export_yellow)

        self.yellow_viewer = ZoomableViewer()
        yellow_layout.addLayout(yellow_top_layout)
        yellow_layout.addWidget(self.yellow_viewer)
        self.tabs.addTab(yellow_tab, "Bright Channel")

        self.update_green_view()
        self.yellow_viewer.set_image(cv2_to_qpixmap(self.yellow_grid))

    def showEvent(self, event):
        super().showEvent(event)
        self.green_viewer.fitInView(self.green_viewer.scene.sceneRect(), Qt.KeepAspectRatio)

    def on_tab_changed(self, index):
        if index == 0:
            self.green_viewer.fitInView(self.green_viewer.scene.sceneRect(), Qt.KeepAspectRatio)
        elif index == 1:
            self.yellow_viewer.fitInView(self.yellow_viewer.scene.sceneRect(), Qt.KeepAspectRatio)

    def update_green_view(self):
        show_grid = self.cb_grid.isChecked()
        show_cells = self.cb_cells.isChecked()

        if show_grid and show_cells:
            base_bgr = self.raw_images['both']
        elif show_grid and not show_cells:
            base_bgr = self.raw_images['grid']
        elif not show_grid and show_cells:
            base_bgr = self.raw_images['cells']
        else:
            base_bgr = self.raw_images['clean']

        if self.cb_scale.isChecked():
            rendered = draw_scale_bar(base_bgr, self.well_statuses, self.avg_radius)
        else:
            rendered = base_bgr

        self.green_viewer.set_image(cv2_to_qpixmap(rendered))

    def update_yellow_view(self, checked):
        bgr = self.yellow_grid if checked else self.yellow_clean
        self.yellow_viewer.set_image(cv2_to_qpixmap(bgr))

    def export_current_image(self):
        current_tab_idx = self.tabs.currentIndex()
        viewer = self.green_viewer if current_tab_idx == 0 else self.yellow_viewer
        channel_name = "fluorescence" if current_tab_idx == 0 else "bright"

        pixmap = viewer.pixmap_item.pixmap()
        if pixmap.isNull():
            QMessageBox.warning(self, "Export Warning", "No image currently available to export.")
            return

        base_name = os.path.splitext(self.filename)[0] if self.filename else "sample"
        default_filename = f"{base_name}_{channel_name}.png"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Image",
            default_filename,
            "PNG Image (*.png);;TIFF Image (*.tif *.tiff);;JPEG Image (*.jpg *.jpeg)"
        )
        if not file_path:
            return

        qimage = pixmap.toImage()
        dpm = int(round(300.0 / 0.0254))
        qimage.setDotsPerMeterX(dpm)
        qimage.setDotsPerMeterY(dpm)

        if qimage.save(file_path):
            QMessageBox.information(
                self,
                "Export Successful",
                f"Image exported successfully:\n{file_path}"
            )
        else:
            QMessageBox.critical(self, "Export Error", f"Failed to save image to:\n{file_path}")


class SingleLabelSampleCard(QFrame):
    clicked = Signal(str)
    selection_changed = Signal(object, bool)

    def __init__(self, filename, parent=None):
        super().__init__(parent)
        self.filename = filename
        self.preview_pixmap = None

        self.setFixedHeight(360)
        self.setFrameShape(QFrame.StyledPanel)
        self.update_style(False)

        self.stacked_widget = QStackedWidget(self)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.stacked_widget)

        self.loading_page = QWidget()
        loading_layout = QVBoxLayout(self.loading_page)

        self.lbl_loading_name = QLabel(f"Processing File:\n{filename}")
        self.lbl_loading_name.setAlignment(Qt.AlignCenter)
        self.lbl_loading_name.setStyleSheet("font-weight: bold; font-size: 14px; border: none;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(18)

        self.lbl_loading_status = QLabel("Pending...")
        self.lbl_loading_status.setAlignment(Qt.AlignCenter)
        self.lbl_loading_status.setStyleSheet("color: #7F8C8D; font-size: 12px; font-style: italic; border: none;")

        loading_layout.addStretch()
        loading_layout.addWidget(self.lbl_loading_name)
        loading_layout.addWidget(self.progress_bar)
        loading_layout.addWidget(self.lbl_loading_status)
        loading_layout.addStretch()
        self.stacked_widget.addWidget(self.loading_page)

        self.result_page = QWidget()
        result_layout = QVBoxLayout(self.result_page)
        result_layout.setContentsMargins(10, 8, 10, 8)

        top_layout = QHBoxLayout()
        self.checkbox = QCheckBox()
        self.checkbox.toggled.connect(self.on_checkbox_toggled)
        top_layout.addWidget(self.checkbox)
        top_layout.addStretch()

        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setFixedHeight(45)
        self.info_label.setStyleSheet("font-weight: bold; font-size: 13px; border: none;")

        result_layout.addLayout(top_layout)
        result_layout.addWidget(self.img_label)
        result_layout.addWidget(self.info_label)

        self.stacked_widget.addWidget(self.result_page)
        self.stacked_widget.setCurrentIndex(0)

    def update_progress(self, percent, status):
        self.progress_bar.setValue(percent)
        self.lbl_loading_status.setText(status)

    def set_finished(self, preview_pixmap, cell_count, coverage_percent):
        self.preview_pixmap = preview_pixmap
        self.info_label.setText(f"{self.filename}\nCell count: {cell_count} | Coverage: {int(coverage_percent)}%")
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.stacked_widget.setCurrentIndex(1)

        if self.preview_pixmap:
            scaled = self.preview_pixmap.scaled(self.img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.img_label.setPixmap(scaled)

    def on_checkbox_toggled(self, checked):
        self.update_style(checked)
        self.selection_changed.emit(self, checked)

    def update_style(self, checked):
        self.setStyleSheet(SAMPLE_CARD_SELECTED_STYLE if checked else SAMPLE_CARD_DEFAULT_STYLE)

    def resizeEvent(self, event):
        if event: super().resizeEvent(event)
        if self.stacked_widget.currentIndex() == 1 and self.preview_pixmap:
            scaled = self.preview_pixmap.scaled(self.img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.img_label.setPixmap(scaled)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.stacked_widget.currentIndex() == 1:
            self.clicked.emit(self.filename)