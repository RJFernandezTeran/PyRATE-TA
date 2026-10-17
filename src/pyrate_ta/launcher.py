"""PyRATE-TA startup launcher.

Ensures fonts and shared settings are verified, hides the host command terminal
on Windows, and launches the PyRATE-TA graphical user interface.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Launch the PyRATE-TA GUI with terminal suppression."""
    if getattr(sys, "frozen", False):
        import multiprocessing

        multiprocessing.freeze_support()
    args = list(argv) if argv is not None else sys.argv[1:]
    if getattr(sys, "frozen", False) and args and args[0] in ("-m", "-c"):
        return 2
    try:
        from pymorgan.fonts import ensure_fonts_installed
        from pymorgan.gui.console import hide_console, set_windows_app_id

        set_windows_app_id("PyRATE-TA.GUI")
        hide_console()
        ensure_fonts_installed(quiet=True)
    except Exception:
        pass

    from pyrate_ta.gui.app import main as gui_main

    return gui_main(argv)


if __name__ == "__main__":
    sys.exit(main())
