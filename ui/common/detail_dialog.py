"""Detail viewer dialog shared by both modes: a fluorescence tab with
mode-specific controls and a bright-field tab, both exportable."""
import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QDialog, QFileDialog, QHBoxLayout, QMessageBox,
                               QPushButton, QTabWidget, QVBoxLayout, QWidget)

from ui.common.qt_utils import cv2_to_qpixmap
from ui.common.theme import GLOBAL_STYLESHEET
from ui.common.zoomable_viewer import ZoomableViewer

EXPORT_DPI = 300
FLUO_TAB, BRIGHT_TAB = 0, 1


class BaseDetailDialog(QDialog):
    """Subclasses must set their own state, then call :meth:`build_ui`.

    Hooks:
      * ``build_fluorescence_controls(layout)`` — add widgets to the control bar;
        connect them to :meth:`update_fluorescence_view`.
      * ``render_fluorescence()`` — return the BGR image to display.
    """

    def __init__(self, filename, yellow_grid, yellow_clean, parent=None):
        super().__init__(parent)
        self.filename = filename
        self.yellow_grid = yellow_grid
        self.yellow_clean = yellow_clean

        self.setWindowTitle(f"Detail Viewer - {filename}")
        self.setWindowFlags(
            Qt.Window | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint
        )
        self.resize(1400, 900)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setStyleSheet(GLOBAL_STYLESHEET)

    # ------------------------------------------------------------------ hooks
    def build_fluorescence_controls(self, layout):
        raise NotImplementedError

    def render_fluorescence(self):
        raise NotImplementedError

    # ------------------------------------------------------------------ build
    def build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Fluorescence tab
        green_tab = QWidget()
        green_layout = QVBoxLayout(green_tab)
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(15)
        self.build_fluorescence_controls(controls_layout)
        controls_layout.addStretch()
        controls_layout.addWidget(self._make_export_button())

        self.green_viewer = ZoomableViewer()
        green_layout.addLayout(controls_layout)
        green_layout.addWidget(self.green_viewer)
        self.tabs.addTab(green_tab, "Fluorescence Channel")

        # Bright-field tab
        yellow_tab = QWidget()
        yellow_layout = QVBoxLayout(yellow_tab)
        yellow_top_layout = QHBoxLayout()
        self.cb_yellow = QCheckBox("Show Detected Grid")
        self.cb_yellow.setChecked(True)
        self.cb_yellow.toggled.connect(self.update_yellow_view)
        yellow_top_layout.addWidget(self.cb_yellow)
        yellow_top_layout.addStretch()
        yellow_top_layout.addWidget(self._make_export_button())

        self.yellow_viewer = ZoomableViewer()
        yellow_layout.addLayout(yellow_top_layout)
        yellow_layout.addWidget(self.yellow_viewer)
        self.tabs.addTab(yellow_tab, "Bright Channel")

        self.tabs.currentChanged.connect(self.on_tab_changed)
        self.update_fluorescence_view()
        self.update_yellow_view(True)

    def _make_export_button(self):
        btn = QPushButton("Export Image")
        btn.setObjectName("BtnPrimary")
        btn.clicked.connect(self.export_current_image)
        return btn

    # ----------------------------------------------------------------- views
    def update_fluorescence_view(self, *_):
        self.green_viewer.set_image(cv2_to_qpixmap(self.render_fluorescence()))

    def update_yellow_view(self, checked):
        bgr = self.yellow_grid if checked else self.yellow_clean
        self.yellow_viewer.set_image(cv2_to_qpixmap(bgr))

    def _fit(self, viewer):
        viewer.fitInView(viewer.scene.sceneRect(), Qt.KeepAspectRatio)

    def showEvent(self, event):
        super().showEvent(event)
        self._fit(self.green_viewer)

    def on_tab_changed(self, index):
        self._fit(self.green_viewer if index == FLUO_TAB else self.yellow_viewer)

    # ---------------------------------------------------------------- export
    def export_current_image(self):
        is_fluo = self.tabs.currentIndex() == FLUO_TAB
        viewer = self.green_viewer if is_fluo else self.yellow_viewer
        channel_name = "fluorescence" if is_fluo else "bright"

        pixmap = viewer.pixmap_item.pixmap()
        if pixmap.isNull():
            QMessageBox.warning(self, "Export Warning", "No image currently available to export.")
            return

        base_name = os.path.splitext(self.filename)[0] if self.filename else "sample"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Image",
            f"{base_name}_{channel_name}.png",
            "PNG Image (*.png);;TIFF Image (*.tif *.tiff);;JPEG Image (*.jpg *.jpeg)"
        )
        if not file_path:
            return

        qimage = pixmap.toImage()
        dpm = int(round(EXPORT_DPI / 0.0254))
        qimage.setDotsPerMeterX(dpm)
        qimage.setDotsPerMeterY(dpm)

        if qimage.save(file_path):
            QMessageBox.information(self, "Export Successful", f"Image exported successfully:\n{file_path}")
        else:
            QMessageBox.critical(self, "Export Error", f"Failed to save image to:\n{file_path}")
