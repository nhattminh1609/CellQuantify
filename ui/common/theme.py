import os
import struct
import zlib

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
    "text": "#000000",
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

    QPushButton#BtnSecondary {{
        background-color: #F0F0F0;
        color: #000000;
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

    QListWidget {{
        border: 1px solid #D0D0D0;
        border-radius: 3px;
        background-color: #FFFFFF;
    }}
    QListWidget::item {{
        padding: 8px 12px;
    }}
    QListWidget::item:selected {{
        background-color: #E5F1FB;
        color: #000000;
    }}

    QTextEdit {{
        border: none;
        background-color: transparent;
    }}

    QStackedWidget {{
        border: none;
        background: transparent;
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