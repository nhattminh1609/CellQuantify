from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QListWidget, QListWidgetItem, QTextEdit, QPushButton, QFrame)
from PySide6.QtCore import Qt, QSize, QTimer

from ui.common.theme import GLOBAL_STYLESHEET
from ui.common.window_manager import show_window
from ui.mode1.main_window import SingleLabelWindow
from ui.mode2.main_window import MultiLabelWindow

LAUNCHER_LIST_STYLE = """
    QListWidget#LauncherModeList {
        border: 1px solid #D0D0D0;
        border-radius: 6px;
        background-color: #FFFFFF;
        outline: none;
        padding: 4px;
    }
    QListWidget#LauncherModeList::item {
        padding: 10px 14px;
        border-radius: 5px;
        font-size: 13px;
        font-weight: 500;
        color: #212529;
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        margin: 2px 2px;
    }
    QListWidget#LauncherModeList::item:hover {
        background-color: #F1F5F9;
        border-color: #94A3B8;
    }
    QListWidget#LauncherModeList::item:selected {
        background-color: #E0F2FE;
        border: 1.5px solid #0284C7;
        color: #0369A1;
        font-weight: bold;
    }
"""


class LauncherWindow(QWidget):
    def __init__(self):
        super().__init__()
        self._is_launching = False
        self.setWindowTitle("Cell Counter - Select Working Mode")
        self.setMinimumSize(800, 520)
        self.resize(820, 560)
        self.setStyleSheet(GLOBAL_STYLESHEET + LAUNCHER_LIST_STYLE)

        self.modes_info = {
            "Mode 1: Single-Label Quantification": (
                "<h3>Single-Label Cell Counting and Coverage Microwell</h3>"
                "<p><b>Application:</b> Homogeneous monoculture assays.</p>"
                "<ul>"
                "<li><b>Grid Detection:</b> Standard 10x10 microfluidic well array alignment.</li>"
                "<li><b>Inference:</b> Single-class segmentation model.</li>"
                "<li><b>Metrics:</b> Cell count, coverage microwell, and comparative delta analysis.</li>"
                "</ul>"
            ),
            "Mode 2: Multi-Label Analysis (a549/nk/undefine)": (
                "<h3>Multi-Label Co-Culture Classification and Spatial Filtering</h3>"
                "<p><b>Application:</b> Immune-cancer interaction assays.</p>"
                "<ul>"
                "<li><b>Grid Detection:</b> Standard 10x10 microfluidic well array alignment.</li>"
                "<li><b>Inference:</b> Multi-class probability maps.</li>"
                "<li><b>Inspection:</b> Dynamic multi-checkbox mask mixing (A549, NK, Undefined) and grid overlays.</li>"
                "</ul>"
            )
        }

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(16)

        header_label = QLabel("Select Experimental Analysis Mode")
        header_label.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 4px;")
        main_layout.addWidget(header_label)

        # Dynamic mode list with auto-calculated height and generous spacing
        self.mode_list = QListWidget()
        self.mode_list.setObjectName("LauncherModeList")
        self.mode_list.setSpacing(2)

        item_height = 48
        for mode_name in self.modes_info.keys():
            item = QListWidgetItem(mode_name)
            item.setSizeHint(QSize(0, item_height))
            self.mode_list.addItem(item)

        # Auto-fit height to content so items never overlap and future modes fit cleanly
        num_modes = len(self.modes_info)
        total_list_height = num_modes * item_height + (num_modes - 1) * 6 + 16
        self.mode_list.setFixedHeight(min(260, max(114, total_list_height)))

        desc_frame = QFrame()
        desc_frame.setObjectName("StatsPanel")
        desc_layout = QVBoxLayout(desc_frame)
        self.desc_box = QTextEdit()
        self.desc_box.setReadOnly(True)
        desc_layout.addWidget(self.desc_box)

        main_layout.addWidget(self.mode_list)
        main_layout.addWidget(desc_frame, stretch=1)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.btn_launch = QPushButton("Launch Analysis")
        self.btn_launch.setObjectName("BtnPrimary")
        self.btn_launch.setCursor(Qt.PointingHandCursor)
        btn_layout.addWidget(self.btn_launch)

        main_layout.addLayout(btn_layout)

        self.mode_list.currentRowChanged.connect(self.update_description)
        self.btn_launch.clicked.connect(self.launch_mode)
        self.mode_list.itemDoubleClicked.connect(self.launch_mode)
        self.mode_list.setCurrentRow(0)

    def update_description(self, row):
        if row < 0:
            return
        mode_name = self.mode_list.item(row).text()
        self.desc_box.setHtml(self.modes_info.get(mode_name, ""))

    def launch_mode(self, item=None):
        if getattr(self, "_is_launching", False):
            return
        self._is_launching = True

        if item is not None and hasattr(self.mode_list, "row"):
            row = self.mode_list.row(item)
        else:
            row = self.mode_list.currentRow()

        window_classes = {0: SingleLabelWindow, 1: MultiLabelWindow}
        window_cls = window_classes.get(row)
        if window_cls is None:
            self._is_launching = False
            return

        # Defer window instantiation and display to the next event loop cycle.
        # This allows Qt C++ QListWidget::mouseDoubleClickEvent to completely unwind
        # its call stack before LauncherWindow is closed and destroyed.
        QTimer.singleShot(0, lambda: self._safe_launch(window_cls))

    def _safe_launch(self, window_cls):
        try:
            show_window(window_cls())
        except Exception:
            self._is_launching = False
            raise