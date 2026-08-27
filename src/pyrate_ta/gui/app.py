"""Application entry point for the PyRATE-TA GUI.

Neither ``pyrate`` nor ``pymorgan`` is imported at module level: the splash
screen has to reach the screen before the pipelines (scipy, matplotlib, the
loader registries) are paid for.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from ..log import get_logger

logger = get_logger(__name__)

_START_TIME = time.perf_counter()

# Pin matplotlib (and our widgets) to the PyQt6 Qt binding.
os.environ.setdefault("QT_API", "pyqt6")


def _load_pymorgan_settings():
    """Activate PyMORGAN's ``settings.toml`` -- the aesthetics half.

    PyRATE-TA loads only its own file, so without this PyMORGAN's settings stayed
    at their dataclass defaults and the two applications drew the same data
    differently: the ``profile`` field selects the ``.mplstyle``, and with it the
    font stack and every font size, so a user who had chosen a profile or a
    ``font_scale`` in PyMORGAN saw neither honoured here.

    Returns the file that was read, or ``None``.
    """
    from .settings_app import pymorgan_settings_path

    cfg = pymorgan_settings_path()
    if cfg is None:
        logger.debug("no PyMORGAN settings.toml found; its defaults are used")
        return None
    try:
        import pymorgan as pm

        pm.load_settings(cfg)
    except Exception:
        logger.warning("Could not read %s; PyMORGAN's defaults are used.", cfg, exc_info=True)
        return None
    logger.info("Aesthetics from %s", cfg)
    return cfg


def _report_missing_fonts() -> list[str]:
    """Say so, once and actionably, when the preferred fonts are unavailable.

    The font stack is PyMORGAN's (``Helvetica, TeX Gyre Heros, Arial``, from its
    ``.mplstyle``) and PyRATE-TA applies it unchanged. What PyRATE-TA does *not*
    inherit is the installation: fonts are registered with the Matplotlib of one
    environment, so a PyRATE-TA virtual environment that has never run the
    installer falls back silently to DejaVu Sans -- which is why a figure can
    come out looking unlike PyMORGAN's while both claim the same style.

    Returns the preferred fonts that *are* available, so a caller can test this.
    """
    try:
        from pymorgan.fonts import PREFERRED_FONTS, available_preferred_fonts

        found = available_preferred_fonts()
    except Exception:  # pragma: no cover - PyMORGAN is a hard dependency
        logger.debug("could not query the preferred fonts", exc_info=True)
        return []
    if not found:
        logger.warning(
            "None of the preferred fonts %s are available to Matplotlib, so plots fall "
            "back to DejaVu Sans. Run 'uv run pymorgan-install-fonts' in this environment "
            "to install them (PyRATE-TA uses PyMORGAN's fonts and ships no second installer).",
            list(PREFERRED_FONTS),
        )
    else:
        logger.debug("preferred fonts available: %s", ", ".join(found))
    return found


def main(argv: list[str] | None = None) -> int:
    """Create the QApplication, show the main window and run the event loop."""
    t_start = _START_TIME
    args = list(argv) if argv is not None else list(sys.argv)
    cli_dark = "--dark" in args
    if cli_dark:
        args = [a for a in args if a != "--dark"]
    if sys.platform.startswith("win") and "-platform" not in args:
        args += ["-platform", "windows:darkmode=0"]
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("PyRATE-TA.GUI")
        except Exception:
            logger.debug("Could not set the Windows AppUserModelID.", exc_info=True)

    from PyQt6.QtGui import QIcon
    from PyQt6.QtWidgets import QApplication, QStyleFactory

    app = QApplication.instance() or QApplication(args)

    icon = Path(__file__).with_name("icons") / "pirate_ship.png"
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))

    # PyRATE-TA keeps its own settings.toml (analysis + interface); the theme
    # *resolution* is PyMORGAN's, so both applications look identical.
    from pymorgan.gui.app import _resolve_theme

    from ..settings import ensure_settings_file, get_settings, load_settings

    try:
        load_settings(ensure_settings_file())
    except Exception:
        logger.warning(
            "Could not read settings.toml; the built-in defaults are used.", exc_info=True
        )
    style_name, dark = _resolve_theme(
        QStyleFactory, theme=get_settings().gui_theme, force_dark=cli_dark
    )
    app.setStyle(style_name)
    if app.style().objectName().lower() == "fusion":
        if dark:
            from pymorgan.gui.theme import dark_palette

            app.setPalette(dark_palette())
        else:
            fusion = QStyleFactory.create("Fusion")
            if fusion is not None:
                app.setPalette(fusion.standardPalette())

    from ..cite import citation_notice

    # Console banner: what to cite when results from this software are
    # published. Logged, not printed, so a host application keeps control.
    citation_notice()
    _load_pymorgan_settings()
    _report_missing_fonts()

    from .main_window import MainWindow

    window = MainWindow()
    window.show()

    logger.info("PyRATE-TA GUI loaded in %.2f s", time.perf_counter() - t_start)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
