"""Keeps exactly one top-level window alive at a time.

Previously each window held a reference to the next one (launcher -> analysis
window -> new launcher -> ...), so closed windows and all their image data were
never freed. Now only this module holds the current window; the previous one is
closed and, thanks to ``WA_DeleteOnClose``, destroyed by Qt.
"""
from PySide6.QtCore import QTimer, Qt

_current_window = None


def show_window(window):
    """Show ``window`` and safely close the previously shown top-level window."""
    global _current_window
    previous = _current_window
    _current_window = window
    window.setAttribute(Qt.WA_DeleteOnClose)
    window.show()

    if previous is not None and previous is not window:
        def _close_previous(p):
            try:
                p.close()
            except RuntimeError:
                pass  # underlying C++ object already deleted (e.g. user closed it)

        # Defer previous window closure to the event loop so ongoing event handlers
        # (e.g. mouse clicks or double clicks) completely finish unwinding before destruction.
        QTimer.singleShot(0, lambda: _close_previous(previous))


def current_window():
    return _current_window
