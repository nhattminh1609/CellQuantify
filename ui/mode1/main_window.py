import os
import re
import cv2
import numpy as np
import logging

from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QFileDialog,
                               QHBoxLayout, QGridLayout, QPushButton, QLabel,
                               QScrollArea, QFrame, QMessageBox, QTextEdit,
                               QProgressBar)
from PySide6.QtCore import Qt, QThread, Signal, QSettings

from ui.common.theme import GLOBAL_STYLESHEET
from ui.mode1.components import SingleLabelSampleCard, SingleLabelDetailDialog
from core.common.extract_ROI import crop_well_image
from core.common.onnx_inference import ONNXModel
from core.mode1.segmentation import SegmentationProcessor
from core.mode1.quantification import calculate_coverage_and_draw
from utils.helpers import cv2_to_qpixmap

logger = logging.getLogger("CellCounter.SingleLabel")

class SingleLabelProcessingThread(QThread):
    item_processed = Signal(str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float, list, float)
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
                result_img, total_c, cell_boxes = self.seg_processor.process(cropped_img.copy(), self.detector)

                self.image_progress.emit(img_name, 90, "Quantifying cells...")
                final_img, cov_percent, well_statuses, phys_radius = calculate_coverage_and_draw(result_img.copy(), local_well_coords, cell_boxes)
                grid_img, _, _, _ = calculate_coverage_and_draw(cropped_img.copy(), local_well_coords, cell_boxes)

                self.image_progress.emit(img_name, 100, "Done")
                self.item_processed.emit(img_name, final_img, grid_img, result_img, cropped_img, full_yellow_grid, full_yellow_clean, total_c, cov_percent, well_statuses, phys_radius)

            except Exception as e:
                logger.error(f"Failed processing {img_name}: {str(e)}", exc_info=True)
                self.error_occurred.emit(img_name, f"Error: {str(e)}")

            finally:
                self.batch_progress.emit(idx + 1)

        self.finished.emit()


class SingleLabelWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cell Counter - Single-Label Quantification")
        self.resize(1600, 900)
        self.setStyleSheet(GLOBAL_STYLESHEET)

        self.detector = None
        self.seg_processor = None
        self.image_data_store = {}
        self.selected_images = []
        self.grid_widgets = {}
        self.open_dialogs = []
        self.settings = QSettings("BioLab", "CellCounter")

        self.init_ui()
        self.auto_load_model()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

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
        self.grid_layout.setSpacing(15)
        for i in range(4):
            self.grid_layout.setColumnStretch(i, 1)
        self.scroll_area.setWidget(self.grid_container)

        self.lbl_placeholder = QLabel("No data available. Please load image folder.")
        self.lbl_placeholder.setAlignment(Qt.AlignCenter)
        self.lbl_placeholder.setStyleSheet("font-size: 18px; color: #BDC3C7; border: none;")

        viewer_layout.addWidget(self.lbl_placeholder)
        viewer_layout.addWidget(self.scroll_area)
        self.scroll_area.hide()

        left_layout.addWidget(self.control_panel)
        left_layout.addWidget(self.viewer_panel, stretch=1)

        self.stats_panel = QFrame()
        self.stats_panel.setObjectName("StatsPanel")
        self.stats_panel.setFixedWidth(300)
        stats_layout = QVBoxLayout(self.stats_panel)

        title_stats = QLabel("Statistical Comparison")
        title_stats.setStyleSheet("font-size: 18px; font-weight: bold; border: none; margin-bottom: 8px;")

        self.txt_stats = QTextEdit()
        self.txt_stats.setReadOnly(True)
        self.txt_stats.setHtml("<p>Select exactly 2 samples to view comparison.</p>")

        stats_layout.addWidget(title_stats)
        stats_layout.addWidget(self.txt_stats)

        main_layout.addWidget(left_widget, stretch=1)
        main_layout.addWidget(self.stats_panel)

        self.btn_load_folder.clicked.connect(self.start_processing)
        self.btn_switch_mode.clicked.connect(self.switch_mode_action)
        self.btn_reset.clicked.connect(self.reset_ui)

    def auto_load_model(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(base_dir, "models", "mode1_model.onnx")

        if not os.path.exists(model_path):
            self.lbl_status.setText("Error: models/mode1_model.onnx missing")
            self.lbl_status.setStyleSheet("color: #C0392B; font-weight: bold; border: none;")
            self.btn_load_folder.setEnabled(False)
            return

        try:
            self.detector = ONNXModel(model_path)
            self.seg_processor = SegmentationProcessor(conf_threshold=0.8)
            self.lbl_status.setText(f"Model Loaded - System Ready")
            self.lbl_status.setStyleSheet("color: #27AE60; font-weight: bold; border: none;")
        except Exception as e:
            QMessageBox.critical(self, "Model Error", f"Cannot load model: {str(e)}")
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
            widget = SingleLabelSampleCard(filename)
            widget.clicked.connect(self.show_clean_image)
            widget.selection_changed.connect(self.handle_image_selection)

            row = idx // 4
            col = idx % 4
            self.grid_layout.addWidget(widget, row, col)
            self.grid_widgets[filename] = widget

        self.worker_thread = SingleLabelProcessingThread(self.detector, self.seg_processor, folder_1, folder_2, image_files)
        self.worker_thread.item_processed.connect(self.finish_image_processing)
        self.worker_thread.image_progress.connect(self.update_image_progress)
        self.worker_thread.batch_progress.connect(self.progress_bar_batch.setValue)
        self.worker_thread.error_occurred.connect(self.handle_thread_error)
        self.worker_thread.finished.connect(self.on_processing_finished)
        self.worker_thread.start()

    def update_image_progress(self, filename, percent, status):
        if filename in self.grid_widgets:
            self.grid_widgets[filename].update_progress(percent, status)

    def finish_image_processing(self, filename, final_img, grid_img, result_img, cropped_img, yellow_grid, yellow_clean, total_c, cov_percent, well_statuses, phys_radius):
        self.image_data_store[filename] = {
            'count': total_c,
            'coverage': cov_percent,
            'wells': well_statuses,
            'avg_radius': phys_radius,
            'raw_images': {
                'both': final_img,
                'grid': grid_img,
                'cells': result_img,
                'clean': cropped_img
            },
            'yellow_grid': yellow_grid,
            'yellow_clean': yellow_clean
        }

        if filename in self.grid_widgets:
            self.grid_widgets[filename].set_finished(
                cv2_to_qpixmap(final_img), total_c, cov_percent
            )

    def handle_thread_error(self, filename, err_msg):
        if filename in self.grid_widgets:
            self.grid_widgets[filename].update_progress(100, f"FAILED: {err_msg}")
            self.grid_widgets[filename].lbl_loading_status.setStyleSheet("color: #C0392B; font-weight: bold;")

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
        if len(self.selected_images) < 2:
            msg = "<p>Select exactly 2 samples to view comparison.</p>"
            if len(self.selected_images) == 1:
                msg = f"<p>Selected: <b>{self.selected_images[0]}</b><br>Select 1 more sample to compare.</p>"
            self.txt_stats.setHtml(msg)
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
                dist = ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)**0.5
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
        <p style="margin-bottom:0;"><b>Lost in Sample B</b> (<b>{len(lost_wells)}</b> wells)</p>
        {build_well_list(lost_wells)}
        <p style="margin-bottom:0; margin-top:8px;"><b>Gained in Sample B</b> (<b>{len(gained_wells)}</b> wells)</p>
        {build_well_list(gained_wells)}
        """
        self.txt_stats.setHtml(html)

    def on_processing_finished(self):
        self.btn_load_folder.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self.progress_bar_batch.hide()

    def show_clean_image(self, filename):
        if filename not in self.image_data_store: return
        data = self.image_data_store[filename]
        dialog = SingleLabelDetailDialog(
            raw_images=data['raw_images'],
            yellow_grid=data['yellow_grid'],
            yellow_clean=data['yellow_clean'],
            filename=filename,
            well_statuses=data['wells'],
            avg_radius=data['avg_radius'],
            parent=None
        )
        self.open_dialogs.append(dialog)
        dialog.finished.connect(lambda: self.open_dialogs.remove(dialog) if dialog in self.open_dialogs else None)
        dialog.show()

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
        self.selected_images.clear()
        self.grid_widgets.clear()
        self.txt_stats.setHtml("<p>Select exactly 2 samples to view comparison.</p>")
        self.lbl_placeholder.show()
        self.scroll_area.hide()
