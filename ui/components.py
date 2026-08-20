from PySide6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QFrame,
                               QSizePolicy, QHBoxLayout, QCheckBox, QStackedWidget,
                               QProgressBar, QWidget, QGraphicsView,
                               QGraphicsScene, QGraphicsPixmapItem, QTabWidget)
from PySide6.QtGui import QCursor, QPainter
from PySide6.QtCore import Qt, Signal


class ZoomableViewer(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

    def set_image(self, pixmap):
        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(self.pixmap_item.boundingRect())

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            zoom_factor = 1.15
        else:
            zoom_factor = 1 / 1.15
        self.scale(zoom_factor, zoom_factor)


class ImageDialog(QDialog):
    def __init__(self, green_overlay, green_clean, yellow_grid, yellow_clean, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detail Viewer")
        self.setWindowFlags(
            Qt.Window | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint)
        self.resize(1400, 900)

        self.setStyleSheet("""
                    QDialog {
                        background-color: #F8F9FA;
                    }
                    QTabWidget::pane {
                        border: 1px solid #DCDDE1;
                        background-color: #FFFFFF;
                        border-radius: 4px;
                    }
                    QTabBar::tab {
                        background-color: #F1F2F6;
                        border: 1px solid #DCDDE1;
                        padding: 8px 0px;
                        margin-right: 2px;
                        border-top-left-radius: 4px;
                        border-top-right-radius: 4px;
                        color: #576574;
                        width: 320px;
                    }
                    QTabBar::tab:selected {
                        background-color: #FFFFFF;
                        border-bottom-color: #FFFFFF;
                        color: #2980B9;
                        font-weight: bold;
                    }
                    QTabBar::tab:hover {
                        background-color: #E2E6EA;
                    }
                    QCheckBox {
                        color: #2F3640;
                        padding: 5px;
                        spacing: 8px;
                    }
                    QCheckBox::indicator {
                        width: 18px;
                        height: 18px;
                    }
                """)

        self.green_overlay = green_overlay
        self.green_clean = green_clean
        self.yellow_grid = yellow_grid
        self.yellow_clean = yellow_clean

        layout = QVBoxLayout(self)

        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self.on_tab_changed)
        layout.addWidget(self.tabs)

        green_tab = QWidget()
        green_layout = QVBoxLayout(green_tab)

        self.cb_green = QCheckBox("Show Mask Overlay")
        self.cb_green.setChecked(True)
        self.cb_green.toggled.connect(self.update_green_view)

        self.green_viewer = ZoomableViewer()

        green_layout.addWidget(self.cb_green)
        green_layout.addWidget(self.green_viewer)
        self.tabs.addTab(green_tab, "Fluorescence Channel")

        yellow_tab = QWidget()
        yellow_layout = QVBoxLayout(yellow_tab)

        self.cb_yellow = QCheckBox("Show Detected Grid")
        self.cb_yellow.setChecked(True)
        self.cb_yellow.toggled.connect(self.update_yellow_view)

        self.yellow_viewer = ZoomableViewer()

        yellow_layout.addWidget(self.cb_yellow)
        yellow_layout.addWidget(self.yellow_viewer)
        self.tabs.addTab(yellow_tab, "Bright Channel")

        self.green_viewer.set_image(self.green_overlay)
        self.yellow_viewer.set_image(self.yellow_grid)

    def showEvent(self, event):
        super().showEvent(event)
        self.green_viewer.fitInView(self.green_viewer.scene.sceneRect(), Qt.KeepAspectRatio)

    def on_tab_changed(self, index):
        if index == 0:
            self.green_viewer.fitInView(self.green_viewer.scene.sceneRect(), Qt.KeepAspectRatio)
        elif index == 1:
            self.yellow_viewer.fitInView(self.yellow_viewer.scene.sceneRect(), Qt.KeepAspectRatio)

    def update_green_view(self, checked):
        pixmap = self.green_overlay if checked else self.green_clean
        self.green_viewer.set_image(pixmap)

    def update_yellow_view(self, checked):
        pixmap = self.yellow_grid if checked else self.yellow_clean
        self.yellow_viewer.set_image(pixmap)


class ImageItemWidget(QFrame):
    clicked = Signal(object, object, object, object)
    selection_changed = Signal(object, bool)

    def __init__(self, filename, parent=None):
        super().__init__(parent)
        self.filename = filename
        self.green_overlay = None
        self.green_clean = None
        self.yellow_grid = None
        self.yellow_clean = None

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
        self.lbl_loading_name.setStyleSheet("font-weight: bold; color: #34495E; font-size: 14px;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(20)

        self.lbl_loading_status = QLabel("Pending...")
        self.lbl_loading_status.setAlignment(Qt.AlignCenter)
        self.lbl_loading_status.setStyleSheet("color: #7F8C8D; font-size: 12px; font-style: italic;")

        loading_layout.addStretch()
        loading_layout.addWidget(self.lbl_loading_name)
        loading_layout.addWidget(self.progress_bar)
        loading_layout.addWidget(self.lbl_loading_status)
        loading_layout.addStretch()

        self.stacked_widget.addWidget(self.loading_page)

        self.result_page = QWidget()
        result_layout = QVBoxLayout(self.result_page)
        result_layout.setContentsMargins(10, 5, 10, 5)

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
        self.info_label.setStyleSheet("border: none; font-weight: bold; color: #2C3E50; font-size: 14px;")

        result_layout.addLayout(top_layout)
        result_layout.addWidget(self.img_label)
        result_layout.addWidget(self.info_label)

        self.stacked_widget.addWidget(self.result_page)
        self.stacked_widget.setCurrentIndex(0)

    def update_progress(self, percent, status):
        self.progress_bar.setValue(percent)
        self.lbl_loading_status.setText(status)

    def set_finished(self, green_overlay, green_clean, yellow_grid, yellow_clean, cell_count, coverage_percent):
        self.green_overlay = green_overlay
        self.green_clean = green_clean
        self.yellow_grid = yellow_grid
        self.yellow_clean = yellow_clean

        self.info_label.setText(f"{self.filename}\nCell count: {cell_count} | Coverage: {int(coverage_percent)}%")
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.stacked_widget.setCurrentIndex(1)

        if self.green_overlay:
            scaled = self.green_overlay.scaled(self.img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.img_label.setPixmap(scaled)

    def on_checkbox_toggled(self, checked):
        self.update_style(checked)
        self.selection_changed.emit(self, checked)

    def update_style(self, checked):
        if checked:
            self.setStyleSheet(
                "ImageItemWidget { border: 2px solid #3498DB; border-radius: 8px; background: #EBF5FB; }")
        else:
            self.setStyleSheet("""
                ImageItemWidget { border: 1px solid #ccc; border-radius: 8px; background: white; }
                ImageItemWidget:hover { border: 2px solid #27AE60; }
            """)

    def resizeEvent(self, event):
        if event:
            super().resizeEvent(event)
        if self.stacked_widget.currentIndex() == 1 and self.green_overlay:
            scaled = self.green_overlay.scaled(self.img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.img_label.setPixmap(scaled)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.stacked_widget.currentIndex() == 1:
            self.clicked.emit(self.green_overlay, self.green_clean, self.yellow_grid, self.yellow_clean)