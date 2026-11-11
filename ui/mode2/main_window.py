import os
import re
import cv2
import numpy as np
import logging

from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QGridLayout, QPushButton, QLabel,
                               QFileDialog, QScrollArea, QFrame, QMessageBox, QProgressBar)
from PySide6.QtCore import Qt, QThread, Signal, QSettings

from ui.common.theme import GLOBAL_STYLESHEET
from ui.mode2.components import MultiLabelSampleCard, MultiLabelDetailDialog
from core.common.extract_ROI import crop_well_image
from core.common.onnx_inference import ONNXModel
from core.mode2.segmentation import MultiLabelSegmentationProcessor
from core.mode2.quantification import quantify_multilabel_wells, render_mode2_canvas
from utils.helpers import cv2_to_qpixmap

logger = logging.getLogger("CellCounter.MultiLabel")

class MultiLabelProcessingThread(QThread):
    item_processed = Signal(str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, object, object, float, float, float)
    image_progress = Signal(str, int, str)
    batch_progress = Signal(int)
    error_occurred = Signal(str, str)
    finished = Signal()

    def __init__(self, detector, seg_processor, folder_1, folder_2, image_files):
        super().__init__()
        self.detector = detector
        self.seg_processor = seg_processor
        self.folder_1 = folder_1
        self.folder_2 = folder_2
        self.image_files = image_files

    def run(self):
        for idx, img_name in enumerate(self.image_files):
            try:
                self.image_progress.emit(img_name, 10, "Reading images...")
                green_full_path = os.path.join(self.folder_1, img_name)
                yellow_full_path = os.path.join(self.folder_2, img_name)

                img_green = cv2.imread(green_full_path)
                img_yellow = cv2.imread(yellow_full_path)

                if img_green is None or img_yellow is None:
                    self.error_occurred.emit(img_name, "File corrupted or missing")
                    continue

                self.image_progress.emit(img_name, 30, "Aligning and cropping...")
                cropped_img, full_yellow_clean, full_yellow_grid, local_well_coords, detected_r = crop_well_image(img_green, img_yellow)

                self.image_progress.emit(img_name, 60, "Inference...")
                seg_data = self.seg_processor.process(cropped_img, self.detector)

                self.image_progress.emit(img_name, 90, "Quantifying cells...")
                well_statuses, cov_percent, phys_radius, ratio_new = quantify_multilabel_wells(local_well_coords, seg_data["cell_boxes"])

                preview_img = render_mode2_canvas(
                    cropped_img, seg_data, well_statuses, phys_radius, ratio_new=ratio_new,
                    grid_mode="Full grid", mask_mode="All labels", show_scale_bar=False
                )

                self.image_progress.emit(img_name, 100, "Done")
                self.item_processed.emit(
                    img_name, preview_img, cropped_img, full_yellow_grid, full_yellow_clean,
                    seg_data, (well_statuses, seg_data["class_counts"]), cov_percent, float(phys_radius), float(ratio_new)
                )

            except Exception as e:
                logger.error(f"Failed processing {img_name}: {str(e)}", exc_info=True)
                self.error_occurred.emit(img_name, f"Error: {str(e)}")

            finally:
                self.batch_progress.emit(idx + 1)

        self.finished.emit()


class MultiLabelWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cell Counter - Multi-Label")
        self.resize(1600, 900)
        self.setStyleSheet(GLOBAL_STYLESHEET)

        self.detector = None
        self.seg_processor = None
        self.image_data_store = {}
        self.grid_widgets = {}
        self.open_dialogs = []
        self.settings = QSettings("BioLab", "CellCounter")

        self.init_ui()
        self.auto_load_model()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(12)

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
        self.grid_layout.setSpacing(20)
        for i in range(4):
            self.grid_layout.setColumnStretch(i, 1)
        self.scroll_area.setWidget(self.grid_container)

        self.lbl_placeholder = QLabel("No data available. Please load image folder.")
        self.lbl_placeholder.setAlignment(Qt.AlignCenter)
        self.lbl_placeholder.setStyleSheet("font-size: 18px; color: #7F8C8D; border: none;")

        viewer_layout.addWidget(self.lbl_placeholder)
        viewer_layout.addWidget(self.scroll_area)
        self.scroll_area.hide()

        main_layout.addWidget(self.control_panel)
        main_layout.addWidget(self.viewer_panel, stretch=1)

        self.btn_load_folder.clicked.connect(self.start_processing)
        self.btn_switch_mode.clicked.connect(self.switch_mode_action)
        self.btn_reset.clicked.connect(self.reset_ui)

    def auto_load_model(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(base_dir, "models", "mode2_multi_label.onnx")

        if not os.path.exists(model_path):
            self.lbl_status.setText("Error: models/mode2_multi_label.onnx missing")
            self.lbl_status.setStyleSheet("color: #C0392B; font-weight: bold; border: none;")
            self.btn_load_folder.setEnabled(False)
            return

        try:
            self.detector = ONNXModel(model_path)
            self.seg_processor = MultiLabelSegmentationProcessor(conf_threshold=0.3)
            self.lbl_status.setText(f"Model Loaded - System Ready")
            self.lbl_status.setStyleSheet("color: #27AE60; font-weight: bold; border: none;")
        except Exception as e:
            QMessageBox.critical(self, "Model Error", f"Failed to initialize ONNX session: {str(e)}")
            self.btn_load_folder.setEnabled(False)

    def start_processing(self):
        last_dir = self.settings.value("last_folder", "")
        root_folder = QFileDialog.getExistingDirectory(self, "Select Image Folder", last_dir)
        if not root_folder: return

        self.settings.setValue("last_folder", root_folder)

        folder_1 = os.path.join(root_folder, "green_background")
        folder_2 = os.path.join(root_folder, "yellow_background")

        if not (os.path.exists(folder_1) and os.path.exists(folder_2)):
            folder_1 = os.path.join(root_folder, "bright_background")
            folder_2 = os.path.join(root_folder, "dark_background")

        if not (os.path.exists(folder_1) and os.path.exists(folder_2)):
            QMessageBox.critical(self, "Error", "Valid pair folders not found.")
            return

        self.reset_ui()
        self.btn_load_folder.setEnabled(False)
        self.btn_reset.setEnabled(False)

        image_files = [f for f in os.listdir(folder_1) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort(key=lambda s: [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)])

        total_images = len(image_files)
        self.progress_bar_batch.setRange(0, total_images)
        self.progress_bar_batch.setValue(0)
        self.progress_bar_batch.setFormat("%v/%m (%p%)")
        self.progress_bar_batch.show()

        self.lbl_placeholder.hide()
        self.scroll_area.show()

        for idx, filename in enumerate(image_files):
            widget = MultiLabelSampleCard(filename)
            widget.clicked.connect(self.open_detail_dialog)

            row = idx // 4
            col = idx % 4
            self.grid_layout.addWidget(widget, row, col)
            self.grid_widgets[filename] = widget

        self.worker_thread = MultiLabelProcessingThread(self.detector, self.seg_processor, folder_1, folder_2, image_files)
        self.worker_thread.item_processed.connect(self.finish_image_processing)
        self.worker_thread.image_progress.connect(self.update_image_progress)
        self.worker_thread.batch_progress.connect(self.progress_bar_batch.setValue)
        self.worker_thread.error_occurred.connect(self.handle_thread_error)
        self.worker_thread.finished.connect(self.on_processing_finished)
        self.worker_thread.start()

    def update_image_progress(self, filename, percent, status):
        if filename in self.grid_widgets:
            self.grid_widgets[filename].update_progress(percent, status)

    def finish_image_processing(self, filename, preview_img, cropped_img, yellow_grid, yellow_clean, seg_data, quant_tuple, cov_percent, avg_radius, ratio_new):
        well_statuses, class_counts = quant_tuple
        preview_pixmap = cv2_to_qpixmap(preview_img)

        self.image_data_store[filename] = {
            "cropped_img": cropped_img,
            "seg_data": seg_data,
            "wells": well_statuses,
            "class_counts": class_counts,
            "coverage": cov_percent,
            "avg_radius": avg_radius,
            "ratio_new": ratio_new,
            "yellow_grid": yellow_grid,
            "yellow_clean": yellow_clean
        }

        if filename in self.grid_widgets:
            self.grid_widgets[filename].set_finished(preview_pixmap, class_counts, cov_percent)

    def open_detail_dialog(self, filename):
        if filename not in self.image_data_store: return
        data = self.image_data_store[filename]
        dialog = MultiLabelDetailDialog(
            cropped_img=data["cropped_img"],
            seg_data=data["seg_data"],
            well_statuses=data["wells"],
            avg_radius=data["avg_radius"],
            ratio_new=data.get("ratio_new", None),
            yellow_grid=data["yellow_grid"],
            yellow_clean=data["yellow_clean"],
            filename=filename,
            parent=None
        )
        self.open_dialogs.append(dialog)
        dialog.finished.connect(lambda: self.open_dialogs.remove(dialog) if dialog in self.open_dialogs else None)
        dialog.show()

    def handle_thread_error(self, filename, err_msg):
        if filename in self.grid_widgets:
            self.grid_widgets[filename].update_progress(100, f"FAILED: {err_msg}")
            self.grid_widgets[filename].lbl_loading_status.setStyleSheet("color: #C0392B; font-weight: bold;")

    def on_processing_finished(self):
        self.btn_load_folder.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self.progress_bar_batch.hide()

    def switch_mode_action(self):
        if hasattr(self, 'worker_thread') and self.worker_thread.isRunning():
            self.worker_thread.terminate()
            self.worker_thread.wait()

        for dialog in self.open_dialogs:
            dialog.close()
        self.open_dialogs.clear()

        from ui.launcher import LauncherWindow
        self.launcher = LauncherWindow()
        self.launcher.show()
        self.close()

    def reset_ui(self):
        for dialog in self.open_dialogs:
            dialog.close()
        self.open_dialogs.clear()

        for i in reversed(range(self.grid_layout.count())):
            if self.grid_layout.itemAt(i).widget():
                self.grid_layout.itemAt(i).widget().setParent(None)

        self.image_data_store.clear()
        self.grid_widgets.clear()
        self.lbl_placeholder.show()
        self.scroll_area.hide()