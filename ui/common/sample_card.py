"""Thumbnail card shown in the batch grid (loading page -> result page)."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (QCheckBox, QFrame, QHBoxLayout, QLabel, QProgressBar,
                               QSizePolicy, QStackedWidget, QVBoxLayout, QWidget)

from ui.common.theme import SAMPLE_CARD_DEFAULT_STYLE, SAMPLE_CARD_SELECTED_STYLE

LOADING_PAGE, RESULT_PAGE = 0, 1


class BaseSampleCard(QFrame):
    clicked = Signal(str)
    selection_changed = Signal(object, bool)   # (card, checked) — only when selectable

    def __init__(self, filename, selectable=False, parent=None):
        super().__init__(parent)
        self.filename = filename
        self.selectable = selectable
        self.preview_pixmap = None
        self.checkbox = None

        self.setFixedHeight(360)
        self.setFrameShape(QFrame.StyledPanel)
        self.update_style(False)

        self.stacked_widget = QStackedWidget(self)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.stacked_widget)

        self._build_loading_page()
        self._build_result_page()
        self.stacked_widget.setCurrentIndex(LOADING_PAGE)

    # ------------------------------------------------------------------ build
    def _build_loading_page(self):
        self.loading_page = QWidget()
        loading_layout = QVBoxLayout(self.loading_page)

        self.lbl_loading_name = QLabel(f"Processing File:\n{self.filename}")
        self.lbl_loading_name.setAlignment(Qt.AlignCenter)
        self.lbl_loading_name.setStyleSheet("font-weight: bold; font-size: 14px; border: none;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(18)

        self.lbl_loading_status = QLabel("Pending...")
        self.lbl_loading_status.setAlignment(Qt.AlignCenter)
        self.lbl_loading_status.setWordWrap(True)
        self.lbl_loading_status.setStyleSheet("color: #7F8C8D; font-size: 12px; font-style: italic; border: none;")

        loading_layout.addStretch()
        loading_layout.addWidget(self.lbl_loading_name)
        loading_layout.addWidget(self.progress_bar)
        loading_layout.addWidget(self.lbl_loading_status)
        loading_layout.addStretch()
        self.stacked_widget.addWidget(self.loading_page)

    def _build_result_page(self):
        self.result_page = QWidget()
        result_layout = QVBoxLayout(self.result_page)
        result_layout.setContentsMargins(10, 8, 10, 8)

        top_layout = QHBoxLayout()
        if self.selectable:
            self.checkbox = QCheckBox()
            self.checkbox.toggled.connect(self._on_checkbox_toggled)
            top_layout.addWidget(self.checkbox)
        top_layout.addStretch()

        self.lbl_warning = QLabel()
        self.lbl_warning.setStyleSheet("color: #C0392B; font-size: 12px; font-weight: bold; border: none;")
        self.lbl_warning.hide()
        top_layout.addWidget(self.lbl_warning)

        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.img_label.setStyleSheet("border: none;")

        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setFixedHeight(45)
        self.info_label.setStyleSheet("font-weight: bold; font-size: 13px; border: none;")

        result_layout.addLayout(top_layout)
        result_layout.addWidget(self.img_label)
        result_layout.addWidget(self.info_label)
        self.stacked_widget.addWidget(self.result_page)

    # ------------------------------------------------------------------ state
    def update_progress(self, percent, status):
        self.progress_bar.setValue(percent)
        self.lbl_loading_status.setText(status)

    def mark_failed(self, message):
        self.update_progress(100, f"FAILED: {message}")
        self.lbl_loading_status.setStyleSheet("color: #C0392B; font-weight: bold; border: none;")

    def set_warning(self, message):
        self.lbl_warning.setText(f"\u26a0 {message}")
        self.lbl_warning.setToolTip(message)
        self.lbl_warning.show()

    def set_finished(self, preview_pixmap, info_text):
        self.preview_pixmap = preview_pixmap
        self.info_label.setText(info_text)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.stacked_widget.setCurrentIndex(RESULT_PAGE)
        self._refresh_preview()

    def set_checked_silently(self, checked):
        """Change the checkbox state without emitting ``selection_changed``."""
        if self.checkbox is None:
            return
        self.checkbox.blockSignals(True)
        self.checkbox.setChecked(checked)
        self.checkbox.blockSignals(False)
        self.update_style(checked)

    def update_style(self, checked):
        self.setStyleSheet(SAMPLE_CARD_SELECTED_STYLE if checked else SAMPLE_CARD_DEFAULT_STYLE)

    # ---------------------------------------------------------------- events
    def _on_checkbox_toggled(self, checked):
        self.update_style(checked)
        self.selection_changed.emit(self, checked)

    def _refresh_preview(self):
        if self.stacked_widget.currentIndex() == RESULT_PAGE and self.preview_pixmap:
            scaled = self.preview_pixmap.scaled(self.img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.img_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_preview()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.stacked_widget.currentIndex() == RESULT_PAGE:
            self.clicked.emit(self.filename)
        super().mousePressEvent(event)
