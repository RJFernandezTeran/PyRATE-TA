"""The live parameter window: what the optimiser is doing, while it does it.

A dialog holding the progress status, the live parameter plot, and a cancel button.
It is opened when ``Settings.fit_monitor_every`` is non-zero and closed when the fit ends.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_API", "pyqt6")

from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ..log import get_logger

logger = get_logger(__name__)


class ParameterMonitor(QDialog):
    """A live fitting dialog containing evaluation status, parameter bar plot, and a cancel button.

    Parameters
    ----------
    parent : QWidget
        Owner window.
    names : sequence of str
        Parameter names, in vector order.
    fixed : sequence of bool, optional
        Which are held fixed; they are drawn greyed.
    every : int
        Redraw every ``every`` evaluations.
    scale : str
        Bar scale, from ``Settings.fit_monitor_scale``.
    """

    def __init__(self, parent, names, *, fixed=None, every: int = 1, scale: str = "symlog"):
        super().__init__(parent)
        self.setWindowTitle("PyRATE-TA - Fit Monitor")
        self.setModal(False)
        self._canceled = False
        self._names = [str(n) for n in names]
        self._fixed = list(fixed) if fixed is not None else None
        self._every = max(1, int(every))
        self._scale = str(scale)
        self._history: list[list[float]] = []

        from pymorgan.gui.widgetplot import WidgetPlot

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.label = QLabel("Fitting...", self)
        self.label.setWordWrap(True)
        self.label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.label)

        self.plot = WidgetPlot(self, show_toolbar=False)
        layout.addWidget(self.plot)
        # Use existing single subplot on WidgetPlot canvas (do not add a duplicate subplot)
        self._ax = self.plot.ax
        self.resize(max(400, 110 * len(self._names)), 360)

        button_box = QHBoxLayout()
        button_box.addStretch()
        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self._on_cancel)
        button_box.addWidget(self.cancel_button)
        layout.addLayout(button_box)

        self.update_parameters(0, [float("nan")] * len(self._names), float("nan"), force=True)

    def _on_cancel(self):
        self._canceled = True
        self.cancel_button.setEnabled(False)
        self.label.setText("Cancelling fit...")

    def wasCanceled(self) -> bool:
        return self._canceled

    def setLabelText(self, text: str):
        self.label.setText(text)

    # ------------------------------------------------------------------ #
    #                              Updating                              #
    # ------------------------------------------------------------------ #
    def update_parameters(self, n_eval: int, params, cost: float, *, force: bool = False) -> bool:
        """Redraw if this evaluation is due; returns whether it drew."""
        self._history.append([float(v) for v in params])
        if not force and (n_eval % self._every):
            return False

        from ..plot.monitor import plot_parameter_bars

        try:
            plot_parameter_bars(
                params,
                self._names,
                ax=self._ax,
                fixed=self._fixed,
                scale=self._scale,
                title=f"evaluation {n_eval}" + ("" if cost != cost else f" | cost {cost:.6g}"),
            )
            self._style()
            self.plot.canvas.draw_idle()
            QApplication.processEvents()
        except Exception:
            logger.debug("parameter monitor update failed", exc_info=True)
            return False
        return True

    def history(self):
        """Every parameter vector seen, as ``[n_eval, n_params]``."""
        import numpy as np

        return np.asarray(self._history, dtype=float)

    def _style(self):
        from pymorgan.gui.theme import is_dark_palette, style_figure

        style_figure(self.plot.figure, is_dark_palette())


def make_monitor(parent, names, *, fixed=None, settings=None):
    """The monitor asked for by the settings, or ``None``.

    ``fit_monitor_every = 0`` means "never", which is the default.
    """
    import pyrate_ta as pr

    s = settings if settings is not None else pr.get_settings()
    every = int(getattr(s, "fit_monitor_every", 0) or 0)
    if every < 1:
        return None
    try:
        monitor = ParameterMonitor(
            parent,
            names,
            fixed=fixed,
            every=every,
            scale=str(getattr(s, "fit_monitor_scale", "symlog")),
        )
    except Exception:
        logger.warning("Could not open the live parameter window.", exc_info=True)
        return None
    monitor.show()
    return monitor
