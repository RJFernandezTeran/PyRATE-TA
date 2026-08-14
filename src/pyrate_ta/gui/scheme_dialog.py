"""The K-matrix editor: write a scheme, see the graph it makes.

The dialog is a thin binding of ``scheme_dialog.ui``. All the understanding of
the notation lives in :mod:`pyrate_ta.models.scheme_text`, and the drawing in
:mod:`pyrate_ta.plot.scheme`; this module only moves text and pictures between
them.

*Update K graph* is deliberately both the redraw button and the check: a scheme
that cannot be parsed produces the error, with its line number, instead of a
graph, so the button answers "would this scheme work?" before any fit is
started. The derived matrix is shown beside the graph -- evaluated with every
rate set to 1, since the shape is what is being checked, not the values.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_API", "pyqt6")

from pathlib import Path

import numpy as np
from PyQt6 import uic
from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QTableWidgetItem

from ..log import get_logger
from ..models.scheme_text import SchemeSyntaxError, scheme_from_text

logger = get_logger(__name__)

_UI_FILE = Path(__file__).with_name("scheme_dialog.ui")

_DEFAULT_TEXT = """\
# One reaction per line. Rate names are yours; repeating one shares
# a single fitted parameter between those arrows.
A -> B  : k1
B -> C  : k2
C ->    : k3
init A = 1
"""

_OK_STYLE = "color: #196f3d;"
_ERROR_STYLE = "color: #a93226; font-weight: bold;"


class SchemeDialog(QDialog):
    """Edit a kinetic scheme as text, with the graph and matrix alongside."""

    def __init__(self, parent=None, text: str | None = None):
        super().__init__(parent)
        uic.loadUi(str(_UI_FILE), self)
        self.scheme = None

        self.SchemeText.setPlainText(text or _DEFAULT_TEXT)
        self._fill_templates()

        self.UpdateGraphButton.clicked.connect(self.update_graph)
        self.TemplateCombo.currentIndexChanged.connect(self._load_template)
        self.ButtonBox.accepted.connect(self._accept_if_valid)
        self.ButtonBox.rejected.connect(self.reject)

        self.update_graph()

    # ------------------------------------------------------------------ #
    #                             Templates                              #
    # ------------------------------------------------------------------ #
    def _fill_templates(self):
        """Offer the predefined schemes as a starting point, in the notation."""
        from ..models.schemes import TARGET_SCHEMES

        self.TemplateCombo.blockSignals(True)
        self.TemplateCombo.addItem("Start from a template...", userData=None)
        for key, scheme in TARGET_SCHEMES.items():
            self.TemplateCombo.addItem(scheme.label, userData=key)
        self.TemplateCombo.blockSignals(False)

    def _load_template(self, index: int):
        key = self.TemplateCombo.itemData(index)
        if not key:
            return
        from ..models.schemes import get_scheme

        scheme = get_scheme(key)
        self.SchemeText.setPlainText(_template_text(scheme))
        self.update_graph()

    # ------------------------------------------------------------------ #
    #                          Parse and draw                            #
    # ------------------------------------------------------------------ #
    def scheme_text(self) -> str:
        return self.SchemeText.toPlainText()

    def update_graph(self) -> bool:
        """Parse, report, and redraw. Returns whether the scheme is valid."""
        try:
            scheme = scheme_from_text(self.scheme_text())
        except (SchemeSyntaxError, ValueError) as exc:
            self.scheme = None
            self.StatusLabel.setStyleSheet(_ERROR_STYLE)
            self.StatusLabel.setText(str(exc))
            self._clear_views()
            self._set_ok_enabled(False)
            return False

        self.scheme = scheme
        self._draw_scheme(scheme)
        self._fill_matrix(scheme)
        self.StatusLabel.setStyleSheet(_OK_STYLE)
        self.StatusLabel.setText(
            f"OK: {scheme.n_species} species ({', '.join(scheme.species_labels())}), "
            f"{scheme.n_rates} rate constant(s) ({', '.join(scheme.parameter_names())})."
        )
        self._set_ok_enabled(True)
        return True

    def _draw_scheme(self, scheme):
        from ..plot.scheme import plot_scheme

        widget = self.SchemeAxes
        widget.figure.clear()
        ax = widget.figure.add_subplot(111)
        try:
            # Unit rates: the topology is what this view is about, and the
            # arrows are labelled with the symbols in any case.
            plot_scheme(
                scheme,
                taus=np.ones(scheme.n_rates),
                ax=ax,
                labels=scheme.species_labels(),
                title="",
            )
        except Exception as exc:  # a drawable scheme that will not draw is still news
            logger.debug("scheme diagram failed", exc_info=True)
            ax.text(0.5, 0.5, f"Could not draw: {exc}", ha="center", va="center")
        self._style(widget)
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                widget.figure.tight_layout()
        except Exception:
            pass
        widget.canvas.draw_idle()


    def _fill_matrix(self, scheme):
        """Show ``K`` with every rate set to 1, so the structure is visible."""
        K = scheme.rate_matrix(np.ones(scheme.n_rates))
        names = scheme.species_labels()
        table = self.MatrixTable
        table.setRowCount(len(names))
        table.setColumnCount(len(names))
        table.setHorizontalHeaderLabels(names)
        table.setVerticalHeaderLabels(names)
        for i in range(len(names)):
            for j in range(len(names)):
                table.setItem(i, j, QTableWidgetItem(f"{K[i, j]:g}"))
        table.resizeColumnsToContents()

    def _clear_views(self):
        self.SchemeAxes.figure.clear()
        self.SchemeAxes.canvas.draw_idle()
        self.MatrixTable.setRowCount(0)
        self.MatrixTable.setColumnCount(0)

    def _style(self, widget):
        from pymorgan.gui.theme import is_dark_palette, style_figure

        style_figure(widget.figure, is_dark_palette())

    def _set_ok_enabled(self, on: bool):
        button = self.ButtonBox.button(QDialogButtonBox.StandardButton.Ok)
        if button is not None:
            button.setEnabled(bool(on))
            button.setToolTip("" if on else "Fix the scheme first (see the message on the left)")

    def _accept_if_valid(self):
        """Never return an unparsed scheme, even if the text changed after a draw."""
        if self.update_graph():
            self.accept()


def _template_text(scheme) -> str:
    """Render a predefined scheme in the notation, as an editable starting point."""
    from ..plot.scheme import rate_symbols

    names = scheme.species_labels()
    rates = scheme.parameter_names()
    symbols = rate_symbols(scheme.build, scheme.n_rates, scheme.n_species)

    lines: list[str] = [f"# {scheme.label}"]
    written: set[tuple] = set()
    for (src, dst), indices in symbols.items():
        if (src, dst) in written:
            continue
        reverse = (dst, src) if dst is not None else None
        if reverse in symbols and reverse not in written:
            written.add(reverse)
            lines.append(
                f"{names[src]} <-> {names[dst]} : {rates[indices[0]]}, {rates[symbols[reverse][0]]}"
            )
        else:
            target = "" if dst is None else names[dst]
            rate = " + ".join(rates[i] for i in indices)
            # A summed decay is written as separate channels, which is what the
            # notation expresses and what the fit actually has.
            for name in (rates[i] for i in indices):
                lines.append(f"{names[src]} -> {target} : {name}")
            if len(indices) == 1:
                lines.pop()
                lines.append(f"{names[src]} -> {target} : {rate}")
        written.add((src, dst))
    lines.append(f"init {names[0]} = 1")
    return "\n".join(lines)


def edit_scheme(parent=None, text: str | None = None):
    """Open the editor; return the accepted scheme, or ``None`` if cancelled."""
    dialog = SchemeDialog(parent, text=text)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return dialog.scheme
    return None
