from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QListWidget, QListWidgetItem, QTextEdit, QPushButton, QFrame)
from PySide6.QtCore import Qt

from ui.common.theme import GLOBAL_STYLESHEET
from ui.mode1.main_window import SingleLabelWindow
from ui.mode2.main_window import MultiLabelWindow

class LauncherWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cell Counter - Select Working Mode")
        self.setFixedSize(780, 500)
        self.setStyleSheet(GLOBAL_STYLESHEET)
        self.active_window = None

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
                "<li><b>Inspection:</b> Dynamic ListBox filtering for grid boundaries and segmentation masks.</li>"
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

        self.mode_list = QListWidget()
        self.mode_list.setFixedHeight(110)
        for mode_name in self.modes_info.keys():
            self.mode_list.addItem(QListWidgetItem(mode_name))

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
        row = self.mode_list.currentRow()
        if row == 0:
            self.active_window = SingleLabelWindow()
        elif row == 1:
            self.active_window = MultiLabelWindow()

        if self.active_window:
            self.active_window.show()
            self.close()