import os
import re
import cv2
import numpy as np
import logging

from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QGridLayout, QPushButton, QLabel,
                               QFileDialog, QScrollArea, QFrame, QMessageBox, QTextEdit, QProgressBar)
from PySide6.QtCore import Qt, QThread, Signal

from ui.components import ImageItemWidget, ImageDialog
from core.extract_ROI import crop_well_image
from core.onnx_inference import ONNXModel
from core.segmentation import SegmentationProcessor
from core.quantification import calculate_coverage_and_draw
from utils.helpers import cv2_to_qpixmap

logger = logging.getLogger("CellCounter.UI")

class ProcessingThread(QThread):
    item_processed = Signal(str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float, list)
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

                self.image_progress.emit(img_name, 30, "ROI extraction...")
                cropped_img, full_yellow_clean, full_yellow_grid, local_well_coords, avg_radius = crop_well_image(
                    img_green, img_yellow)

                self.image_progress.emit(img_name, 60, "Segmentation...")
                result_img, total_c, cell_boxes = self.seg_processor.process(cropped_img, self.detector)

                self.image_progress.emit(img_name, 90, "Quantifying cells...")
                final_img, cov_percent, well_statuses = calculate_coverage_and_draw(result_img, local_well_coords,
                                                                                    cell_boxes, avg_radius)

                self.image_progress.emit(img_name, 100, "Done")
                self.item_processed.emit(img_name, final_img, cropped_img, full_yellow_grid, full_yellow_clean, total_c,
                                         cov_percent, well_statuses)

            except Exception as e:
                logger.error(f"Failed processing {img_name}: {str(e)}", exc_info=True)
                self.error_occurred.emit(img_name, f"Error: {str(e)}")

            finally:
                self.batch_progress.emit(idx + 1)

        self.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cell Counter")
        self.resize(1600, 900)
        self.detector = None
        self.seg_processor = None

        self.image_data_store = {}
        self.selected_images = []
        self.grid_widgets = {}

        self.init_ui()
        self.auto_load_model()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self.control_panel = QFrame()
        self.control_panel.setStyleSheet("QFrame { border: 1px solid #ccc; background: #fdfdfd; }")
        control_layout = QHBoxLayout(self.control_panel)

        self.btn_load_folder = QPushButton("Load Images and Process")
        self.btn_reset = QPushButton("Reset All")

        btn_style = "QPushButton { padding: 10px 20px; font-weight: bold; min-width: 150px; }"
        self.btn_load_folder.setStyleSheet(btn_style + "background-color: #3498DB; color: white;")
        self.btn_reset.setStyleSheet(btn_style + "background-color: #f2dede; color: #a94442;")

        self.lbl_status = QLabel("Status: Initializing...")
        self.lbl_status.setStyleSheet("border: none; color: #7f8c8d; font-style: italic;")

        self.progress_bar_batch = QProgressBar()
        self.progress_bar_batch.setFixedWidth(200)
        self.progress_bar_batch.setAlignment(Qt.AlignCenter)
        self.progress_bar_batch.hide()

        control_layout.addWidget(self.btn_load_folder)
        control_layout.addWidget(self.lbl_status)
        control_layout.addStretch()
        control_layout.addWidget(self.progress_bar_batch)
        control_layout.addWidget(self.btn_reset)

        self.viewer_panel = QFrame()
        viewer_layout = QVBoxLayout(self.viewer_panel)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)

        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(20)
        for i in range(4):
            self.grid_layout.setColumnStretch(i, 1)
        self.scroll_area.setWidget(self.grid_container)

        self.lbl_placeholder = QLabel("No data available. Please select image folder.")
        self.lbl_placeholder.setAlignment(Qt.AlignCenter)
        self.lbl_placeholder.setStyleSheet("font-size: 20px; color: #bbb;")

        viewer_layout.addWidget(self.lbl_placeholder)
        viewer_layout.addWidget(self.scroll_area)
        self.scroll_area.hide()

        left_layout.addWidget(self.control_panel)
        left_layout.addWidget(self.viewer_panel, stretch=1)

        self.stats_panel = QFrame()
        self.stats_panel.setFixedWidth(300)
        self.stats_panel.setStyleSheet("border: 1px solid #ccc; background: #fff;")
        stats_layout = QVBoxLayout(self.stats_panel)

        title_stats = QLabel("Statistical Comparison")
        title_stats.setStyleSheet(
            "border: none; font-size: 20px; font-weight: bold; color: #2C3E50; margin-bottom: 10px;")

        self.txt_stats = QTextEdit()
        self.txt_stats.setReadOnly(True)
        self.txt_stats.setStyleSheet("border: none; font-size: 14px; color: #34495E; background: transparent;")
        self.txt_stats.setHtml("<p>Select exactly 2 images (checkboxes) to view the comparison report.</p>")

        stats_layout.addWidget(title_stats)
        stats_layout.addWidget(self.txt_stats)

        main_layout.addWidget(left_widget, stretch=1)
        main_layout.addWidget(self.stats_panel)

        self.btn_load_folder.clicked.connect(self.start_processing)
        self.btn_reset.clicked.connect(self.reset_ui)

    def auto_load_model(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "models", "LRDAUNet.onnx")

        if not os.path.exists(model_path):
            self.lbl_status.setText(f"Error: {os.path.basename(model_path)} not found")
            self.lbl_status.setStyleSheet("color: red; font-weight: bold;")
            self.btn_load_folder.setEnabled(False)
            logger.critical(f"Model file missing at {model_path}")
            return

        try:
            self.detector = ONNXModel(model_path)
            self.seg_processor = SegmentationProcessor(conf_threshold=0.8)

            self.lbl_status.setText("   System Ready | Model Loaded")
            self.lbl_status.setStyleSheet("border: none; color: #27AE60; font-weight: bold; font-size: 16px;")

        except Exception as e:
            QMessageBox.critical(self, "Initialization Error", f"Cannot load model: {str(e)}")
            self.btn_load_folder.setEnabled(False)

    def start_processing(self):
        root_folder = QFileDialog.getExistingDirectory(self, "Select the root folder containing the image pairs")
        if not root_folder: return

        green_path = os.path.join(root_folder, "green_background")
        yellow_path = os.path.join(root_folder, "yellow_background")
        bright_path = os.path.join(root_folder, "bright_background")
        dark_path = os.path.join(root_folder, "dark_background")

        if os.path.exists(green_path) and os.path.exists(yellow_path):
            folder_1, folder_2 = green_path, yellow_path
        elif os.path.exists(bright_path) and os.path.exists(dark_path):
            folder_1, folder_2 = bright_path, dark_path
        else:
            QMessageBox.critical(self, "Error", "Valid folder pair not found.")
            logger.warning(f"Invalid folder structure selected: {root_folder}")
            return

        self.reset_ui()

        self.btn_load_folder.setEnabled(False)
        self.btn_reset.setEnabled(False)

        image_files = [f for f in os.listdir(folder_1) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort(
            key=lambda s: [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)])

        total_images = len(image_files)
        self.progress_bar_batch.setRange(0, total_images)
        self.progress_bar_batch.setValue(0)
        self.progress_bar_batch.setFormat("%v/%m (%p%)")
        self.progress_bar_batch.show()

        self.lbl_placeholder.hide()
        self.scroll_area.show()

        for idx, filename in enumerate(image_files):
            widget = ImageItemWidget(filename)
            widget.clicked.connect(self.show_clean_image)
            widget.selection_changed.connect(self.handle_image_selection)

            row = idx // 4
            col = idx % 4
            self.grid_layout.addWidget(widget, row, col)
            self.grid_widgets[filename] = widget

        logger.info(f"Starting batch processing of {total_images} files from {folder_1}")

        self.worker_thread = ProcessingThread(self.detector, self.seg_processor, folder_1, folder_2, image_files)
        self.worker_thread.item_processed.connect(self.finish_image_processing)
        self.worker_thread.image_progress.connect(self.update_image_progress)
        self.worker_thread.batch_progress.connect(self.progress_bar_batch.setValue)
        self.worker_thread.error_occurred.connect(self.handle_thread_error)
        self.worker_thread.finished.connect(self.on_processing_finished)
        self.worker_thread.start()

    def update_image_progress(self, filename, percent, status):
        if filename in self.grid_widgets:
            self.grid_widgets[filename].update_progress(percent, status)

    def finish_image_processing(self, filename, result_img, clean_img, yellow_grid, yellow_clean, total_c, cov_percent,
                                well_statuses):
        self.image_data_store[filename] = {
            'count': total_c,
            'coverage': cov_percent,
            'wells': well_statuses
        }

        result_pixmap = cv2_to_qpixmap(result_img)
        clean_pixmap = cv2_to_qpixmap(clean_img)
        yellow_grid_pixmap = cv2_to_qpixmap(yellow_grid)
        yellow_clean_pixmap = cv2_to_qpixmap(yellow_clean)

        if filename in self.grid_widgets:
            self.grid_widgets[filename].set_finished(result_pixmap, clean_pixmap, yellow_grid_pixmap,
                                                     yellow_clean_pixmap, total_c, cov_percent)

    def handle_thread_error(self, filename, err_msg):
        if filename in self.grid_widgets:
            self.grid_widgets[filename].update_progress(100, f"FAILED: {err_msg}")
            self.grid_widgets[filename].lbl_loading_status.setStyleSheet("color: #E74C3C; font-weight: bold;")

    def handle_image_selection(self, widget, is_checked):
        if is_checked:
            if len(self.selected_images) >= 2:
                widget.checkbox.blockSignals(True)
                widget.checkbox.setChecked(False)
                widget.checkbox.blockSignals(False)
                widget.update_style(False)
                QMessageBox.warning(self, "Limit Reached", "Maximum of 2 samples can be selected for comparison.")
                return
            self.selected_images.append(widget.filename)
        else:
            if widget.filename in self.selected_images:
                self.selected_images.remove(widget.filename)

        self.update_comparison_stats()

    def update_comparison_stats(self):
        if len(self.selected_images) == 0:
            self.txt_stats.setHtml("<p>Select exactly 2 images (checkboxes) to view the comparison report.</p>")
            return
        elif len(self.selected_images) == 1:
            self.txt_stats.setHtml(
                f"<p>Currently selected: <b>{self.selected_images[0]}</b><br>Please select 1 more sample to compare.</p>")
            return

        self.selected_images.sort()
        img1, img2 = self.selected_images[0], self.selected_images[1]
        d1, d2 = self.image_data_store[img1], self.image_data_store[img2]

        count_diff = d2['count'] - d1['count']
        cov_diff = d2['coverage'] - d1['coverage']

        lost_wells = []
        gained_wells = []

        for w1 in d1['wells']:
            c1 = w1['coord']
            matched_w2 = None
            for w2 in d2['wells']:
                c2 = w2['coord']
                dist = ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
                if dist <= 15:
                    matched_w2 = w2
                    break

            if matched_w2:
                if w1['has_cell'] and not matched_w2['has_cell']:
                    lost_wells.append(w1)
                elif not w1['has_cell'] and matched_w2['has_cell']:
                    gained_wells.append(w1)

        def build_well_list(wells):
            if not wells:
                return "<p style='color:#7f8c8d;'><i>None detected.</i></p>"
            items = [f"<li>Well {w['id']} (Row {w['row']}, Col {w['col']})</li>" for w in wells]
            return "<ul>" + "".join(items) + "</ul>"

        html = f"""
        <p><b>Sample A:</b> {img1}<br>
        <i>Cell count: {d1['count']} | Coverage: {d1['coverage']:.2f}%</i></p>
        <p><b>Sample B:</b> {img2}<br>
        <i>Cell count: {d2['count']} | Coverage: {d2['coverage']:.2f}%</i></p>
        <hr>
        <h3 style="color:#C0392B;">Difference Index (B - A)</h3>
        <ul>
            <li>Cell count: <b>{count_diff:+d}</b></li>
            <li>Coverage: <b>{cov_diff:+.2f}%</b></li>
        </ul>
        <hr>
        <h3 style="color:#2980B9;">Cell Tracking Analysis</h3>
        <p style="margin-bottom:0;"><b>Lost in Sample B</b><br>
        <span style="font-size:12px; color:#555;">(Present in A, missing in B: <b>{len(lost_wells)}</b> wells)</span></p>
        {build_well_list(lost_wells)}
        <p style="margin-bottom:0; margin-top:10px;"><b>Gained in Sample B</b><br>
        <span style="font-size:12px; color:#555;">(Missing in A, present in B: <b>{len(gained_wells)}</b> wells)</span></p>
        {build_well_list(gained_wells)}
        """
        self.txt_stats.setHtml(html)

    def on_processing_finished(self):
        self.btn_load_folder.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self.progress_bar_batch.hide()
        logger.info("Batch processing completed.")

    def show_clean_image(self, green_overlay, green_clean, yellow_grid, yellow_clean):
        dialog = ImageDialog(green_overlay, green_clean, yellow_grid, yellow_clean, self)
        dialog.exec()

    def reset_ui(self):
        self.clear_grid()
        self.image_data_store.clear()
        self.selected_images.clear()
        self.grid_widgets.clear()
        self.txt_stats.setHtml("<p>Select exactly 2 images (checkboxes) to view the comparison report.</p>")
        self.lbl_placeholder.show()
        self.scroll_area.hide()

    def clear_grid(self):
        for i in reversed(range(self.grid_layout.count())):
            if self.grid_layout.itemAt(i).widget():
                self.grid_layout.itemAt(i).widget().setParent(None)