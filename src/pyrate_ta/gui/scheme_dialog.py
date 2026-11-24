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
from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QInputDialog,
    QMessageBox,
    QTableWidgetItem,
)

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
        self._current_user_key: str | None = None

        self.SchemeText.setPlainText(text or _DEFAULT_TEXT)
        self._fill_templates()

        self.UpdateGraphButton.clicked.connect(self.update_graph)
        self.TemplateCombo.currentIndexChanged.connect(self._load_template)
        self.ButtonBox.accepted.connect(self._accept_if_valid)
        self.ButtonBox.rejected.connect(self.reject)

        for btn_name, handler in (
            ("SaveSchemeButton", self._save_scheme),
            ("DeleteSchemeButton", self._delete_scheme),
            ("ImportSchemeButton", self._import_scheme),
            ("ExportSchemeButton", self._export_scheme),
            ("OpenSchemesDirButton", self._open_schemes_dir),
        ):
            btn = getattr(self, btn_name, None)
            if btn is not None:
                btn.clicked.connect(handler)

        self.update_graph()

    # ------------------------------------------------------------------ #
    #                             Templates                              #
    # ------------------------------------------------------------------ #
    def _fill_templates(self, selected_user_key: str | None = None):
        """Offer built-in templates and saved user schemes in the dropdown."""
        from ..models.schemes import TARGET_SCHEMES
        from ..models.user_schemes import list_user_schemes

        self.TemplateCombo.blockSignals(True)
        self.TemplateCombo.clear()
        self.TemplateCombo.addItem("Start from a template or saved model...", userData=None)
        for key, scheme in TARGET_SCHEMES.items():
            self.TemplateCombo.addItem(scheme.label, userData=("builtin", key))

        try:
            saved = list_user_schemes()
        except Exception:
            logger.debug("could not list user schemes", exc_info=True)
            saved = {}

        if saved:
            self.TemplateCombo.insertSeparator(self.TemplateCombo.count())
            for key, entry in saved.items():
                self.TemplateCombo.addItem(f"\u2605 {entry.name}", userData=("user", key))

        self._current_user_key = None
        if selected_user_key:
            for idx in range(self.TemplateCombo.count()):
                if self.TemplateCombo.itemData(idx) == ("user", selected_user_key):
                    self.TemplateCombo.setCurrentIndex(idx)
                    self._current_user_key = selected_user_key
                    break
        self.TemplateCombo.blockSignals(False)

        del_btn = getattr(self, "DeleteSchemeButton", None)
        if del_btn is not None:
            del_btn.setEnabled(self._current_user_key is not None)

    def _load_template(self, index: int):
        data = self.TemplateCombo.itemData(index)
        del_btn = getattr(self, "DeleteSchemeButton", None)
        if not data:
            self._current_user_key = None
            if del_btn is not None:
                del_btn.setEnabled(False)
            return

        if isinstance(data, tuple) and len(data) == 2:
            kind, key = data
        else:
            kind, key = "builtin", str(data)

        try:
            if kind == "user":
                from ..models.user_schemes import list_user_schemes

                entry = list_user_schemes().get(key)
                if entry is None:
                    self._fill_templates()
                    return
                self._current_user_key = key
                if del_btn is not None:
                    del_btn.setEnabled(True)
                self.SchemeText.setPlainText(entry.text)
            else:
                from ..models.schemes import get_scheme

                self._current_user_key = None
                if del_btn is not None:
                    del_btn.setEnabled(False)
                scheme = get_scheme(key)
                self.SchemeText.setPlainText(_template_text(scheme))
            self.update_graph()
        except Exception as exc:
            logger.exception("Failed to load scheme template %r: %s", key, exc)
            QMessageBox.critical(self, "Load Scheme Failed", f"Could not load scheme:\n{exc}")

    # ------------------------------------------------------------------ #
    #                         User Scheme Library                        #
    # ------------------------------------------------------------------ #
    def _save_scheme(self) -> None:
        """Validate the current scheme text and save it to the user library."""
        if not self.update_graph():
            QMessageBox.warning(
                self, "Invalid Scheme", "Please fix syntax errors before saving the scheme."
            )
            return

        from ..models.user_schemes import list_user_schemes, save_user_scheme

        default_name = ""
        try:
            existing = list_user_schemes()
        except Exception:
            existing = {}
        if self._current_user_key and self._current_user_key in existing:
            default_name = existing[self._current_user_key].name

        name, ok = QInputDialog.getText(
            self,
            "Save Kinetic Model",
            "Model name (saved as a plain-text .txt file):",
            text=default_name,
        )
        if not ok or not name.strip():
            return

        clean_name = name.strip()
        try:
            try:
                saved_path = save_user_scheme(clean_name, self.scheme_text(), overwrite=False)
            except FileExistsError:
                reply = QMessageBox.question(
                    self,
                    "Overwrite Saved Model",
                    f"A saved model named '{clean_name}' already exists.\nOverwrite it?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return
                saved_path = save_user_scheme(clean_name, self.scheme_text(), overwrite=True)
        except Exception as exc:
            logger.exception("Saving user scheme failed: %s", exc)
            QMessageBox.critical(self, "Save Scheme Failed", f"Could not save scheme:\n{exc}")
            return

        saved_text = saved_path.read_text(encoding="utf-8")
        self.SchemeText.setPlainText(saved_text)
        self._fill_templates(selected_user_key=saved_path.stem)
        self.update_graph()
        self.StatusLabel.setStyleSheet(_OK_STYLE)
        self.StatusLabel.setText(f"Saved model '{clean_name}' to {saved_path.name}.")

    def _delete_scheme(self) -> None:
        """Delete the currently selected saved user scheme after confirmation."""
        key = self._current_user_key
        if not key:
            data = self.TemplateCombo.currentData()
            if isinstance(data, tuple) and len(data) == 2 and data[0] == "user":
                key = data[1]
        if not key:
            return

        from ..models.user_schemes import delete_user_scheme, list_user_schemes

        entries = list_user_schemes()
        display = entries[key].name if key in entries else key
        reply = QMessageBox.question(
            self,
            "Delete Saved Model",
            f"Delete saved kinetic model '{display}' from disk?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            deleted = delete_user_scheme(key)
        except Exception as exc:
            logger.exception("Deleting user scheme %r failed: %s", key, exc)
            QMessageBox.critical(self, "Delete Failed", f"Could not delete scheme:\n{exc}")
            return

        self._fill_templates()
        if deleted:
            self.StatusLabel.setStyleSheet(_OK_STYLE)
            self.StatusLabel.setText(f"Deleted saved model '{display}'.")

    def _import_scheme(self) -> None:
        """Load a scheme ``.txt`` file into the editor and optionally save it to the library."""
        from ..models.user_schemes import save_user_scheme, user_schemes_dir

        try:
            start = str(user_schemes_dir())
        except Exception:
            start = str(Path.home())

        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Import Kinetic Scheme",
            start,
            "Scheme text files (*.txt);;All files (*)",
        )
        if not filepath:
            return

        try:
            text = Path(filepath).read_text(encoding="utf-8")
        except Exception as exc:
            logger.exception("Failed reading scheme file %s: %s", filepath, exc)
            QMessageBox.critical(self, "Import Failed", f"Could not read file:\n{exc}")
            return

        self._current_user_key = None
        del_btn = getattr(self, "DeleteSchemeButton", None)
        if del_btn is not None:
            del_btn.setEnabled(False)
        self.SchemeText.setPlainText(text)
        if not self.update_graph():
            return

        reply = QMessageBox.question(
            self,
            "Add to Model Library?",
            f"Imported '{Path(filepath).name}'. Save a copy in your PyRATE-TA model library?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            stem = Path(filepath).stem
            try:
                saved = save_user_scheme(stem, self.scheme_text(), overwrite=True)
                self._fill_templates(selected_user_key=saved.stem)
            except Exception as exc:
                logger.exception("Could not save imported scheme: %s", exc)
                QMessageBox.critical(self, "Save Scheme Failed", f"Could not save scheme:\n{exc}")

    def _export_scheme(self) -> None:
        """Export the current scheme text to a standalone ``.txt`` file."""
        from ..models.user_schemes import user_schemes_dir

        try:
            default_dir = user_schemes_dir()
        except Exception:
            default_dir = Path.home()
        default_name = f"{self._current_user_key or 'custom_scheme'}.txt"
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Kinetic Scheme",
            str(default_dir / default_name),
            "Scheme text files (*.txt);;All files (*)",
        )
        if not filepath:
            return

        try:
            out = Path(filepath)
            out.write_text(self.scheme_text().rstrip() + "\n", encoding="utf-8")
            QMessageBox.information(self, "Export Complete", f"Scheme exported to:\n{out}")
        except Exception as exc:
            logger.exception("Export scheme failed: %s", exc)
            QMessageBox.critical(self, "Export Failed", f"Could not export scheme:\n{exc}")

    def _open_schemes_dir(self) -> None:
        """Open the user schemes library directory in the OS file manager."""
        from ..models.user_schemes import user_schemes_dir

        try:
            folder = user_schemes_dir()
            folder.mkdir(parents=True, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
        except Exception as exc:
            logger.exception("Could not open schemes folder: %s", exc)
            QMessageBox.critical(
                self, "Open Folder Failed", f"Could not open schemes folder:\n{exc}"
            )

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
