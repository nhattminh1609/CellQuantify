import logging

from PySide6.QtWidgets import QFrame, QLabel, QMessageBox, QTextEdit, QVBoxLayout

import config
from core.mode1.quantification import compute_well_statuses, draw_wells
from core.mode1.segmentation import SegmentationProcessor
from ui.common.base_window import BaseAnalysisWindow
from ui.common.processing import BaseProcessingThread
from ui.common.qt_utils import cv2_to_qpixmap
from ui.mode1.components import SingleLabelDetailDialog, SingleLabelSampleCard

logger = logging.getLogger("CellCounter.SingleLabel")

EMPTY_STATS_HTML = "<p>Select exactly 2 samples to view comparison.</p>"


class SingleLabelProcessingThread(BaseProcessingThread):
    def process_item(self, img_name, crop):
        cropped_img = crop.cropped_img
        result_img, total_c, cell_boxes = self.seg_processor.process(cropped_img.copy(), self.detector)

        self.image_progress.emit(img_name, 90, "Quantifying cells...")
        well_statuses, cov_percent, phys_radius = compute_well_statuses(crop.well_coords, cell_boxes)

        return {
            'count': total_c,
            'coverage': cov_percent,
            'wells': well_statuses,
            'avg_radius': phys_radius,
            'raw_images': {
                'both': draw_wells(result_img.copy(), well_statuses, phys_radius),
                'grid': draw_wells(cropped_img.copy(), well_statuses, phys_radius),
                'cells': result_img,
                'clean': cropped_img,
            },
            'yellow_grid': crop.yellow_grid,
            'yellow_clean': crop.yellow_clean,
        }


class SingleLabelWindow(BaseAnalysisWindow):
    WINDOW_TITLE = "Cell Counter - Single-Label Quantification"
    MODEL_FILE = config.MODE1_MODEL_FILE
    GRID_SPACING = 15

    def init_state(self):
        self.selected_images = []

    # ------------------------------------------------------------------ hooks
    def create_processor(self):
        return SegmentationProcessor()

    def create_thread(self, fluo_dir, bright_dir, image_files):
        return SingleLabelProcessingThread(self.detector, self.seg_processor, fluo_dir, bright_dir, image_files)

    def create_card(self, filename):
        card = SingleLabelSampleCard(filename)
        card.selection_changed.connect(self.handle_image_selection)
        return card

    def on_item_processed(self, filename, payload):
        self.image_data_store[filename] = payload
        card = self.grid_widgets.get(filename)
        if card is not None:
            card.set_result(cv2_to_qpixmap(payload['raw_images']['both']), payload['count'], payload['coverage'])

    def create_dialog(self, filename, data):
        return SingleLabelDetailDialog(
            raw_images=data['raw_images'],
            yellow_grid=data['yellow_grid'],
            yellow_clean=data['yellow_clean'],
            filename=filename,
            well_statuses=data['wells'],
            avg_radius=data['avg_radius'],
            parent=None
        )

    def build_side_panel(self):
        self.stats_panel = QFrame()
        self.stats_panel.setObjectName("StatsPanel")
        self.stats_panel.setFixedWidth(300)
        stats_layout = QVBoxLayout(self.stats_panel)

        title_stats = QLabel("Statistical Comparison")
        title_stats.setStyleSheet("font-size: 18px; font-weight: bold; border: none; margin-bottom: 8px;")

        self.txt_stats = QTextEdit()
        self.txt_stats.setReadOnly(True)
        self.txt_stats.setHtml(EMPTY_STATS_HTML)

        stats_layout.addWidget(title_stats)
        stats_layout.addWidget(self.txt_stats)
        return self.stats_panel

    def on_reset(self):
        self.selected_images.clear()
        self.txt_stats.setHtml(EMPTY_STATS_HTML)

    # ------------------------------------------------------------- comparison
    def handle_image_selection(self, widget, is_checked):
        if is_checked:
            if len(self.selected_images) >= 2:
                widget.set_checked_silently(False)
                QMessageBox.warning(self, "Limit Reached", "Maximum of 2 samples can be selected for comparison.")
                return
            self.selected_images.append(widget.filename)
        elif widget.filename in self.selected_images:
            self.selected_images.remove(widget.filename)

        self.update_comparison_stats()

    def update_comparison_stats(self):
        if len(self.selected_images) < 2:
            msg = EMPTY_STATS_HTML
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
                dist = ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
                if dist <= config.COMPARE_WELL_MATCH_DIST:
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
