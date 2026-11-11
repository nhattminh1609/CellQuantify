import os
import struct
import zlib
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

CHECK_ICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkbox_checked.png").replace("\\", "/")

def _ensure_check_icon():
    if os.path.exists(CHECK_ICON_PATH):
        return

    width, height = 16, 16
    fg = (255, 255, 255, 255)
    bg = (0, 0, 0, 0)

    check_pixels = {
        (3, 7), (3, 8),
        (4, 8), (4, 9),
        (5, 9), (5, 10),
        (6, 10), (6, 11),
        (7, 9), (7, 10),
        (8, 8), (8, 9),
        (9, 7), (9, 8),
        (10, 6), (10, 7),
        (11, 5), (11, 6),
        (12, 4), (12, 5),
    }

    raw_data = bytearray()
    for y in range(height):
        raw_data.append(0)
        for x in range(width):
            raw_data.extend(fg if (x, y) in check_pixels else bg)

    def make_chunk(tag, data):
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)

    ihdr = struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0)
    png_bytes = (
        b'\x89PNG\r\n\x1a\n'
        + make_chunk(b'IHDR', ihdr)
        + make_chunk(b'IDAT', zlib.compress(bytes(raw_data)))
        + make_chunk(b'IEND', b'')
    )

    with open(CHECK_ICON_PATH, 'wb') as f:
        f.write(png_bytes)

_ensure_check_icon()

THEME_COLORS = {
    "primary": "#0078D7",
    "primary_hover": "#1084D9",
    "danger": "#A80000",
    "border": "#D0D0D0",
    "surface": "#FFFFFF",
    "background": "#F4F6F9",
    "text": "#212529",
    "text_muted": "#606060"
}

SCROLLBAR_STYLE = """
    /* Vertical scrollbar */
    QScrollBar:vertical {
        border: none;
        background: #F1F1F1;
        width: 8px;
        margin: 0px;
        border-radius: 4px;
    }
    QScrollBar::handle:vertical {
        background: #C1C1C1;
        min-height: 25px;
        border-radius: 4px;
    }
    QScrollBar::handle:vertical:hover {
        background: #A8A8A8;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
        border: none;
        background: none;
    }
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
        background: none;
    }

    /* Horizontal scrollbar */
    QScrollBar:horizontal {
        border: none;
        background: #F1F1F1;
        height: 8px;
        margin: 0px;
        border-radius: 4px;
    }
    QScrollBar::handle:horizontal {
        background: #C1C1C1;
        min-width: 25px;
        border-radius: 4px;
    }
    QScrollBar::handle:horizontal:hover {
        background: #A8A8A8;
    }
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        width: 0px;
        border: none;
        background: none;
    }
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
        background: none;
    }
"""

CHECKBOX_STYLE = f"""
    QCheckBox {{
        background-color: transparent;
        color: #212529;
    }}
    QCheckBox::indicator {{
        width: 17px;
        height: 17px;
        border: 1px solid #606060;
        border-radius: 3px;
        background-color: #FFFFFF;
        margin: 1px;
    }}
    QCheckBox::indicator:hover {{
        border: 1px solid #0078D7;
    }}
    QCheckBox::indicator:checked {{
        border: 1px solid #0078D7;
        background-color: #0078D7;
        image: url("{CHECK_ICON_PATH}");
    }}
"""

GLOBAL_STYLESHEET = f"""
    QMainWindow, QDialog {{
        background-color: #F4F6F9;
        color: #212529;
    }}

    QWidget {{
        color: #212529;
    }}

    QLabel {{
        color: #212529;
        background-color: transparent;
    }}

    QFrame#ControlPanel, QFrame#StatsPanel, QFrame#ViewerPanel {{
        background-color: #FFFFFF;
        border: 1px solid #D0D0D0;
        border-radius: 4px;
    }}

    QPushButton#BtnPrimary {{
        background-color: #0078D7;
        color: #FFFFFF;
        border: 1px solid #005A9E;
        padding: 6px 14px;
        border-radius: 3px;
        font-weight: bold;
    }}
    QPushButton#BtnPrimary:hover {{
        background-color: #1084D9;
    }}
    QPushButton#BtnPrimary:disabled {{
        background-color: #C0C0C0;
        border-color: #B0B0B0;
        color: #FFFFFF;
    }}

    QPushButton#BtnSecondary {{
        background-color: #F0F0F0;
        color: #212529;
        border: 1px solid #ADADAD;
        padding: 6px 14px;
        border-radius: 3px;
    }}
    QPushButton#BtnSecondary:hover {{
        background-color: #E5F1FB;
        border-color: #0078D7;
    }}

    QPushButton#BtnDanger {{
        background-color: #FDF3F2;
        color: #A80000;
        border: 1px solid #E0A8A8;
        padding: 6px 14px;
        border-radius: 3px;
    }}
    QPushButton#BtnDanger:hover {{
        background-color: #F9D6D5;
    }}

    QComboBox {{
        background-color: #FFFFFF;
        color: #212529;
        border: 1px solid #D0D0D0;
        border-radius: 3px;
        padding: 4px 8px;
        min-height: 22px;
    }}
    QComboBox:hover {{
        border-color: #0078D7;
    }}
    QComboBox::drop-down {{
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 20px;
        border-left: 1px solid #D0D0D0;
    }}
    QComboBox QAbstractItemView {{
        background-color: #FFFFFF;
        color: #212529;
        selection-background-color: #E5F1FB;
        selection-color: #000000;
        border: 1px solid #D0D0D0;
    }}

    QTabWidget::pane {{
        border: 1px solid #D0D0D0;
        background-color: #FFFFFF;
        top: -1px;
    }}
    QTabBar::tab {{
        background-color: #E9ECEF;
        color: #495057;
        border: 1px solid #D0D0D0;
        padding: 7px 18px;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        margin-right: 2px;
    }}
    QTabBar::tab:selected {{
        background-color: #FFFFFF;
        color: #0078D7;
        border-bottom-color: #FFFFFF;
        font-weight: bold;
    }}
    QTabBar::tab:hover:!selected {{
        background-color: #DEE2E6;
    }}

    QProgressBar {{
        border: 1px solid #D0D0D0;
        border-radius: 3px;
        background-color: #E9ECEF;
        text-align: center;
        color: #212529;
        font-weight: bold;
    }}
    QProgressBar::chunk {{
        background-color: #0078D7;
        border-radius: 2px;
    }}

    QListWidget {{
        border: 1px solid #D0D0D0;
        border-radius: 3px;
        background-color: #FFFFFF;
        color: #212529;
    }}
    QListWidget::item {{
        padding: 8px 12px;
        color: #212529;
    }}
    QListWidget::item:selected {{
        background-color: #E5F1FB;
        color: #000000;
    }}

    QTextEdit {{
        border: none;
        background-color: #FFFFFF;
        color: #212529;
    }}

    QStackedWidget {{
        border: none;
        background: transparent;
    }}

    QToolTip {{
        background-color: #FFFFFF;
        color: #212529;
        border: 1px solid #D0D0D0;
        padding: 4px;
    }}

    {SCROLLBAR_STYLE}
    {CHECKBOX_STYLE}
"""

SAMPLE_CARD_DEFAULT_STYLE = f"""
    QFrame {{
        border: 1px solid #D0D0D0;
        border-radius: 4px;
        background-color: #FFFFFF;
    }}
    QFrame:hover {{
        border: 1px solid #0078D7;
    }}
    QStackedWidget {{
        border: none;
        background: transparent;
    }}
    {CHECKBOX_STYLE}
"""

SAMPLE_CARD_SELECTED_STYLE = f"""
    QFrame {{
        border: 1px solid #0078D7;
        border-radius: 4px;
        background-color: #F2F7FC;
    }}
    QStackedWidget {{
        border: none;
        background: transparent;
    }}
    {CHECKBOX_STYLE}
"""

def apply_light_theme(app: QApplication):
    """Enforce a strict, high-contrast Light Theme across all platforms (ignoring OS dark mode)."""
    app.setStyle("Fusion")

    palette = QPalette()
    # Main window and surfaces
    palette.setColor(QPalette.Window, QColor(245, 246, 248))
    palette.setColor(QPalette.WindowText, QColor(33, 37, 41))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(240, 242, 245))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ToolTipText, QColor(33, 37, 41))
    palette.setColor(QPalette.Text, QColor(33, 37, 41))
    palette.setColor(QPalette.Button, QColor(240, 240, 240))
    palette.setColor(QPalette.ButtonText, QColor(33, 37, 41))
    palette.setColor(QPalette.BrightText, QColor(215, 58, 73))
    palette.setColor(QPalette.Link, QColor(0, 120, 215))
    palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))

    # Disabled element states
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(140, 140, 140))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(140, 140, 140))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(140, 140, 140))
    palette.setColor(QPalette.Disabled, QPalette.Highlight, QColor(210, 210, 210))
    palette.setColor(QPalette.Disabled, QPalette.HighlightedText, QColor(140, 140, 140))

    app.setPalette(palette)
    app.setStyleSheet(GLOBAL_STYLESHEET)
