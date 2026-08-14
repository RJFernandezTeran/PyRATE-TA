"""Settings panel auto-generated from :meth:`pyrate.Settings.field_specs`.

Each field becomes the appropriate widget (combo / spin / check / line edit);
edits are written back to the active settings through
:func:`pyrate.update_settings` and a ``changed`` signal is emitted so the caller
can re-render.

Only PyRATE's *analysis-only* settings appear here — solver tolerances, default
component counts, IRF handling. Everything presentational (colourmaps, label
style, figure sizes) belongs to :class:`pymorgan.Settings` and is edited with
``pymorgan-settings``; PyRATE deliberately does not start a second aesthetics system.

"""

from __future__ import annotations

import os

os.environ.setdefault("QT_API", "pyqt6")

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QLineEdit,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QWidget,
)

import pyrate_ta as pr

from ..log import get_logger

logger = get_logger(__name__)

# The ``tab`` key in field_specs -> the page title it is shown on. Every
# section of settings.toml has a page: a setting the panel cannot reach is a
# setting the user does not really have.
_TAB_TITLES: dict[str, str] = {
    "fit": "Fit defaults",
    "solver": "Solver",
    "irf": "Instrument response",
    "lda": "Lifetime density",
    "plots": "Plots",
    "gui": "Interface",
    "paths": "Paths",
}


def _as_text(value) -> str:
    """Render a settings value for a line edit (``None`` as an empty field)."""
    if value is None:
        return ""
    if isinstance(value, (tuple, list)):
        return ", ".join(str(v) for v in value)
    return str(value)


class SettingsPanel(QTabWidget):
    """Editable view of the active :class:`pyrate.Settings`.

    The pages and their contents are generated from
    :meth:`pyrate.Settings.field_specs`, so a new setting appears here as soon
    as it has an entry there -- no widget code to add.
    """

    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._widgets: dict[str, QWidget] = {}
        self._forms: dict[str, QFormLayout] = {}

        settings = pr.get_settings()
        for name, spec in pr.Settings.field_specs().items():
            widget = self._build_widget(name, spec, getattr(settings, name, None))
            if widget is None:
                continue
            self._widgets[name] = widget
            form = self._form_for(str(spec.get("tab", "fit")))
            if spec.get("tooltip"):
                widget.setToolTip(str(spec["tooltip"]))
            form.addRow(str(spec.get("label", name)), widget)

    # ------------------------------------------------------------------ #
    #                            Construction                            #
    # ------------------------------------------------------------------ #
    def _form_for(self, tab: str) -> QFormLayout:
        """The form layout of ``tab``, creating the page on first use."""
        if tab not in self._forms:
            page = QWidget()
            form = QFormLayout(page)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            self.addTab(scroll, _TAB_TITLES.get(tab, tab.title()))
            self._forms[tab] = form
        return self._forms[tab]

    def _build_widget(self, name: str, spec: dict, value):
        """Map one ``field_specs`` entry to its editor widget."""
        kind = str(spec.get("kind", "text"))
        if kind == "choice":
            w = QComboBox()
            w.addItems([str(c) for c in spec.get("choices", [])])
            idx = w.findText(str(value))
            if idx >= 0:
                w.setCurrentIndex(idx)
            w.currentTextChanged.connect(lambda text, n=name: self._apply(n, text))
            return w
        if kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(value))
            w.toggled.connect(lambda on, n=name: self._apply(n, on))
            return w
        if kind == "int":
            w = QSpinBox()
            w.setRange(int(spec.get("min", 0)), int(spec.get("max", 10**6)))
            w.setValue(int(value or 0))
            w.valueChanged.connect(lambda v, n=name: self._apply(n, v))
            return w
        if kind in ("float", "text"):
            # Floats go through a line edit: several are optional (``None``
            # means "fit it"), which a spin box cannot express, and tolerances
            # are entered in scientific notation.
            w = QLineEdit(_as_text(value))
            w.editingFinished.connect(lambda n=name, e=w: self._apply(n, e.text()))
            return w
        logger.debug("no editor for setting %s of kind %r; skipped", name, kind)
        return None

    # ------------------------------------------------------------------ #
    #                              Editing                               #
    # ------------------------------------------------------------------ #
    def _apply(self, name: str, value):
        """Write one field back to the active settings.

        A value the settings layer rejects is logged and reverted in the widget
        rather than silently dropped.
        """
        try:
            pr.update_settings(**{name: value})
        except (ValueError, TypeError):
            logger.warning("rejected value %r for setting %s", value, name, exc_info=True)
            self.refresh()
            return
        self.changed.emit()

    def refresh(self):
        """Re-read every widget from the active settings."""
        settings = pr.get_settings()
        for name, w in self._widgets.items():
            value = getattr(settings, name, None)
            w.blockSignals(True)
            if isinstance(w, QComboBox):
                idx = w.findText(str(value))
                if idx >= 0:
                    w.setCurrentIndex(idx)
            elif isinstance(w, QCheckBox):
                w.setChecked(bool(value))
            elif isinstance(w, QSpinBox):
                w.setValue(int(value or 0))
            elif isinstance(w, QLineEdit):
                w.setText(_as_text(value))
            w.blockSignals(False)
