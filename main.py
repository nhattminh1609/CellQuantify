import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont
from ui.launcher import LauncherWindow
from utils.logger import setup_logger

def main():
    setup_logger()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    default_font = QFont("Roboto", 13)
    app.setFont(default_font)

    launcher = LauncherWindow()
    launcher.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()