"""Main analysis window shared by both modes."""
import logging
import os

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QMainWindow,
                               QMessageBox, QProgressBar, QPushButton, QScrollArea, QVBoxLayout, QWidget)

import config
from core.common.onnx_inference import ONNXModel
from ui.common.theme import GLOBAL_STYLESHEET
from ui.common.window_manager import show_window
from utils.files import find_channel_folders, list_images

logger = logging.getLogger("CellCounter.UI")

STATUS_ERROR_STYLE = "color: #C0392B; font-weight: bold; border: none;"
STATUS_OK_STYLE = "color: #27AE60; font-weight: bold; border: none;"


class BaseAnalysisWindow(QMainWindow):
    """Subclasses configure the class attributes and implement the hooks:

    * ``create_processor()``                       -> segmentation processor
    * ``create_thread(fluo_dir, bright_dir, files)`` -> BaseProcessingThread
    * ``create_card(filename)``                    -> BaseSampleCard
    * ``on_item_processed(filename, payload)``     -> store data / update card
    * ``create_dialog(filename, data)``            -> BaseDetailDialog
    * optional ``init_state()``, ``build_side_panel()`` and ``on_reset()``
    """
    WINDOW_TITLE = "Cell Counter"
    MODEL_FILE = ""
    GRID_SPACING = 15

    def __init__(self):
        super().__init__()
        self.setWindowTitle(self.WINDOW_TITLE)
        self.resize(1600, 900)
        self.setStyleSheet(GLOBAL_STYLESHEET)

        self.detector = None
        self.seg_processor = None
        self.worker_thread = None
        self.image_data_store = {}
        self.grid_widgets = {}
        self.open_dialogs = []
        self.settings = QSettings(config.SETTINGS_ORG, config.SETTINGS_APP)

        self.init_state()
        self.init_ui()
        self.auto_load_model()

    # ------------------------------------------------------------------ hooks
    def init_state(self):
        """Initialise subclass-specific attributes (called before init_ui)."""

    def create_processor(self):
        raise NotImplementedError

    def create_thread(self, fluo_dir, bright_dir, image_files):
        raise NotImplementedError

    def create_card(self, filename):
        raise NotImplementedError

    def on_item_processed(self, filename, payload):
        raise NotImplementedError

    def create_dialog(self, filename, data):
        raise NotImplementedError

    def build_side_panel(self):
        return None

    def on_reset(self):
        pass

    # --------------------------------------------------------------------- UI
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        self.control_panel = QFrame()
        self.control_panel.setObjectName("ControlPanel")
        control_layout = QHBoxLayout(self.control_panel)
        control_layout.setContentsMargins(12, 10, 12, 10)

        self.btn_load_folder = QPushButton("Load Images and Process")
        self.btn_load_folder.setObjectName("BtnPrimary")
        self.btn_switch_mode = QPushButton("Switch Mode")
        self.btn_switch_mode.setObjectName("BtnSecondary")
        self.btn_reset = QPushButton("Reset All")
        self.btn_reset.setObjectName("BtnDanger")

        self.lbl_status = QLabel("Status: Initializing...")
        self.lbl_status.setStyleSheet("color: #7F8C8D; font-style: italic; border: none;")

        self.progress_bar_batch = QProgressBar()
        self.progress_bar_batch.setFixedWidth(200)
        self.progress_bar_batch.setFixedHeight(18)
        self.progress_bar_batch.hide()

        control_layout.addWidget(self.btn_load_folder)
        control_layout.addWidget(self.lbl_status)
        control_layout.addStretch()
        control_layout.addWidget(self.progress_bar_batch)
        control_layout.addWidget(self.btn_switch_mode)
        control_layout.addWidget(self.btn_reset)

        self.viewer_panel = QFrame()
        self.viewer_panel.setObjectName("ViewerPanel")
        viewer_layout = QVBoxLayout(self.viewer_panel)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("border: none; background: transparent;")

        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(self.GRID_SPACING)
        for i in range(config.GRID_COLUMNS):
            self.grid_layout.setColumnStretch(i, 1)
        self.scroll_area.setWidget(self.grid_container)

        self.lbl_placeholder = QLabel("No data available. Please load image folder.")
        self.lbl_placeholder.setAlignment(Qt.AlignCenter)
        self.lbl_placeholder.setStyleSheet("font-size: 18px; color: #7F8C8D; border: none;")

        viewer_layout.addWidget(self.lbl_placeholder)
        viewer_layout.addWidget(self.scroll_area)
        self.scroll_area.hide()

        left_layout.addWidget(self.control_panel)
        left_layout.addWidget(self.viewer_panel, stretch=1)
        main_layout.addWidget(left_widget, stretch=1)

        side_panel = self.build_side_panel()
        if side_panel is not None:
            main_layout.addWidget(side_panel)

        self.btn_load_folder.clicked.connect(self.start_processing)
        self.btn_switch_mode.clicked.connect(self.switch_mode_action)
        self.btn_reset.clicked.connect(self.reset_ui)

    def auto_load_model(self):
        model_path = os.path.join(config.MODELS_DIR, self.MODEL_FILE)
        if not os.path.exists(model_path):
            self.lbl_status.setText(f"Error: models/{self.MODEL_FILE} missing")
            self.lbl_status.setStyleSheet(STATUS_ERROR_STYLE)
            self.btn_load_folder.setEnabled(False)
            return

        try:
            self.detector = ONNXModel(model_path)
            self.seg_processor = self.create_processor()
            self.lbl_status.setText("Model Loaded - System Ready")
            self.lbl_status.setStyleSheet(STATUS_OK_STYLE)
        except Exception as e:
            logger.exception("Model initialisation failed")
            QMessageBox.critical(self, "Model Error", f"Cannot load model: {e}")
            self.btn_load_folder.setEnabled(False)

    # ------------------------------------------------------------- processing
    def start_processing(self):
        last_dir = self.settings.value("last_folder", "")
        root_folder = QFileDialog.getExistingDirectory(self, "Select Image Folder", last_dir)
        if not root_folder:
            return
        self.settings.setValue("last_folder", root_folder)

        folders = find_channel_folders(root_folder)
        if folders is None:
            QMessageBox.critical(self, "Error", "Valid pair folders not found.")
            return
        fluo_dir, bright_dir = folders

        image_files = list_images(fluo_dir)
        if not image_files:
            QMessageBox.warning(self, "No Images", f"No images found in:\n{fluo_dir}")
            return

        self.stop_worker()
        self.reset_ui()
        self.btn_load_folder.setEnabled(False)
        self.btn_reset.setEnabled(False)

        self.progress_bar_batch.setRange(0, len(image_files))
        self.progress_bar_batch.setValue(0)
        self.progress_bar_batch.setFormat("%v/%m (%p%)")
        self.progress_bar_batch.show()

        self.lbl_placeholder.hide()
        self.scroll_area.show()

        for idx, filename in enumerate(image_files):
            card = self.create_card(filename)
            card.clicked.connect(self.open_detail_dialog)
            self.grid_layout.addWidget(card, idx // config.GRID_COLUMNS, idx % config.GRID_COLUMNS)
            self.grid_widgets[filename] = card

        thread = self.create_thread(fluo_dir, bright_dir, image_files)
        thread.item_processed.connect(self._handle_item_processed)
        thread.image_progress.connect(self.update_image_progress)
        thread.batch_progress.connect(self.progress_bar_batch.setValue)
        thread.error_occurred.connect(self.handle_thread_error)
        thread.warning_occurred.connect(self.handle_thread_warning)
        thread.all_done.connect(self.on_processing_finished)
        self.worker_thread = thread
        thread.start()

    def _handle_item_processed(self, filename, payload):
        self.on_item_processed(filename, payload)
        card = self.grid_widgets.get(filename)
        if card is not None and not payload.get("grid_found", True):
            card.set_warning("Grid not detected")

    def update_image_progress(self, filename, percent, status):
        card = self.grid_widgets.get(filename)
        if card is not None:
            card.update_progress(percent, status)

    def handle_thread_error(self, filename, err_msg):
        card = self.grid_widgets.get(filename)
        if card is not None:
            card.mark_failed(err_msg)

    def handle_thread_warning(self, filename, message):
        logger.warning("%s: %s", filename, message)

    def on_processing_finished(self):
        self.btn_load_folder.setEnabled(self.detector is not None)
        self.btn_reset.setEnabled(True)
        self.progress_bar_batch.hide()

    def stop_worker(self):
        """Cooperatively stop the worker thread and wait for it to exit."""
        thread = self.worker_thread
        if thread is None:
            return
        self.worker_thread = None
        if thread.isRunning():
            thread.blockSignals(True)       # drop late results aimed at a closing window
            thread.requestInterruption()
            thread.wait()                   # at most one in-flight image

    # ---------------------------------------------------------------- dialogs
    def open_detail_dialog(self, filename):
        data = self.image_data_store.get(filename)
        if data is None:
            return
        dialog = self.create_dialog(filename, data)
        self.open_dialogs.append(dialog)
        dialog.finished.connect(lambda *_: self.open_dialogs.remove(dialog) if dialog in self.open_dialogs else None)
        dialog.show()

    def close_dialogs(self):
        for dialog in list(self.open_dialogs):
            dialog.close()
        self.open_dialogs.clear()

    # ------------------------------------------------------------- lifecycle
    def switch_mode_action(self):
        from ui.launcher import LauncherWindow  # local import: avoid circular import
        show_window(LauncherWindow())          # closes this window -> closeEvent

    def reset_ui(self):
        self.close_dialogs()
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        self.image_data_store.clear()
        self.grid_widgets.clear()
        self.on_reset()
        self.lbl_placeholder.show()
        self.scroll_area.hide()

    def closeEvent(self, event):
        self.stop_worker()
        self.close_dialogs()
        self.image_data_store.clear()
        super().closeEvent(event)
