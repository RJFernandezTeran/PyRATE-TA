"""Entry point for the standalone PyRATE-TA settings editor (``pyrate-ta-settings``).

Edits the analysis-only settings in ``settings.toml``. Presentation settings
live in PyMORGAN and are edited with ``pymorgan-settings``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_API", "pyqt6")


def settings_path() -> Path:
    """``settings.toml`` in the working directory, else the project root."""
    from pyrate_ta.settings import settings_path as _sp

    return _sp()


def pymorgan_settings_path() -> Path | None:
    """PyMORGAN's own ``settings.toml``, or ``None`` if it has none."""
    try:
        import pymorgan as pm

        p = pm.settings_path()
        return p if p.is_file() else None
    except Exception:  # pragma: no cover - PyMORGAN is a hard dependency
        return None


def main(argv: list[str] | None = None) -> int:
    args = list(argv) if argv is not None else list(sys.argv)

    from PyQt6.QtGui import QIcon
    from PyQt6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QMessageBox, QVBoxLayout

    app = QApplication.instance() or QApplication(args)

    import pyrate_ta as pr

    from .settings_panel import SettingsPanel

    cfg = settings_path()
    if cfg.is_file():
        pr.load_settings(cfg)

    dlg = QDialog()
    dlg.setWindowTitle("PyRATE-TA Settings Editor")
    dlg.resize(600, 500)
    icon = Path(__file__).with_name("icons") / "pirate_ship.png"
    if icon.exists():
        dlg.setWindowIcon(QIcon(str(icon)))

    layout = QVBoxLayout(dlg)
    layout.addWidget(SettingsPanel(dlg))

    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Close
    )
    buttons.button(QDialogButtonBox.StandardButton.Save).setText("Save permanently")

    def _save():
        try:
            pr.save_settings(cfg)
        except Exception as exc:
            QMessageBox.critical(dlg, "Save failed", str(exc))
            return
        QMessageBox.information(dlg, "Settings saved", f"Settings written to {cfg}")

    buttons.accepted.connect(_save)
    buttons.rejected.connect(dlg.reject)
    layout.addWidget(buttons)

    dlg.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
