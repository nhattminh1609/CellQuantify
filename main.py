import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont
from ui.launcher import LauncherWindow
from ui.common.window_manager import show_window
from ui.common.theme import apply_light_theme
from utils.logger import setup_logger

def main():
    setup_logger()
    app = QApplication(sys.argv)
    apply_light_theme(app)
    default_font = QFont("Segoe UI", 11)
    app.setFont(default_font)

    show_window(LauncherWindow())
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
