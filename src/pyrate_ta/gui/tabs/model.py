"""Model selection, the lifetime table and the IRF / time-zero table.

Only the *presentation* of the kinetic model lives here: reading and writing the
tables declared in ``main_window.ui``. The model objects themselves belong to
:mod:`pyrate_ta.models` and the solvers to :mod:`pyrate_ta.fit`, both of which stay
importable without Qt.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_API", "pyqt6")

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QTableWidgetItem

import pyrate_ta as pr

from ...log import get_logger

logger = get_logger(__name__)

#: Column order of the lifetime and IRF tables. Lower bound before upper, as
#: they read on a number line; the flag last.
COL_TAU, COL_LB, COL_UB, COL_FIX = 0, 1, 2, 3

#: Rows of the IRF table: time zero, then the Gaussian width (delta).
ROW_T0, ROW_IRF = 0, 1

# Lifetime guesses seeded into a fresh table, one decade apart.
_DEFAULT_TAUS = (1.0, 10.0, 100.0, 1000.0, 10000.0)
_MAX_COMPONENTS = 12


def _numeric_item(value: float, decimals: int = 4) -> QTableWidgetItem:
    from ...helpers import format_lifetime

    item = QTableWidgetItem(format_lifetime(value, f"%.{int(decimals)}g"))
    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
    return item


def _check_item(checked: bool = False) -> QTableWidgetItem:
    item = QTableWidgetItem()
    item.setFlags(
        Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
    )
    item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
    return item


class ModelTabMixin:
    """Drive ``RateFitTable``, ``IRFtable`` and the model-selection buttons."""

    # ------------------------------------------------------------------ #
    #                            Construction                            #
    # ------------------------------------------------------------------ #
    def _init_model_controls(self):
        """Seed both tables from the PyRATE-TA settings defaults."""
        s = pr.get_settings()
        self._set_component_count(int(s.n_components))
        self._init_irf_table()
        self._select_model_type(str(s.model_type))
        self.GaussianIRFCheckBox.setChecked(str(s.irf_mode) == str(pr.IRFMode.GAUSSIAN))
        # The coherent artefact starts off every session: it is a property of
        # the dataset in front of you, not a standing preference.
        self._on_irf_toggled(self.GaussianIRFCheckBox.isChecked())
        self._on_model_changed()  # gates the scheme editor on the family

    def _init_irf_table(self):
        table = self.IRFtable
        table.setRowCount(2)
        for row, (val, lb, ub, fixed) in enumerate(
            ((0.0, -0.5, 0.5, True), (0.2, 0.05, 0.5, True))
        ):
            table.setItem(row, COL_TAU, _numeric_item(val))
            table.setItem(row, COL_LB, _numeric_item(lb))
            table.setItem(row, COL_UB, _numeric_item(ub))
            table.setItem(row, COL_FIX, _check_item(fixed))
        self._compact_columns(table)

    def _reset_tables(self):
        """Reset RateFitTable and IRFtable to their initial default values."""
        s = pr.get_settings()
        table = getattr(self, "RateFitTable", None)
        if table is not None:
            table.setRowCount(int(s.n_components))
            for row in range(table.rowCount()):
                tau = _DEFAULT_TAUS[row] if row < len(_DEFAULT_TAUS) else _DEFAULT_TAUS[-1] * 10
                table.setItem(row, COL_TAU, _numeric_item(tau, s.table_decimals))
                table.setItem(row, COL_LB, _numeric_item(s.table_default_lb, s.table_decimals))
                table.setItem(row, COL_UB, _numeric_item(s.table_default_ub, s.table_decimals))
                table.setItem(row, COL_FIX, _check_item(bool(s.new_rows_fixed)))
            table.setVerticalHeaderLabels([str(i + 1) for i in range(table.rowCount())])
            self._compact_columns(table)
            self._update_tau_units()
        self._init_irf_table()
        ca_box = getattr(self, "CoherentArtifactCheckBox", None)
        if ca_box is not None:
            ca_box.setChecked(False)



    # ------------------------------------------------------------------ #
    #                          Component table                           #
    # ------------------------------------------------------------------ #
    def _set_component_count(self, n: int):
        """Grow or shrink ``RateFitTable`` to ``n`` components."""
        n = max(1, min(int(n), _MAX_COMPONENTS))
        table = self.RateFitTable
        current = table.rowCount()
        table.setRowCount(n)
        s = pr.get_settings()
        for row in range(current, n):
            tau = _DEFAULT_TAUS[row] if row < len(_DEFAULT_TAUS) else _DEFAULT_TAUS[-1] * 10
            table.setItem(row, COL_TAU, _numeric_item(tau, s.table_decimals))
            table.setItem(row, COL_LB, _numeric_item(s.table_default_lb, s.table_decimals))
            table.setItem(row, COL_UB, _numeric_item(s.table_default_ub, s.table_decimals))
            table.setItem(row, COL_FIX, _check_item(bool(s.new_rows_fixed)))
        table.setVerticalHeaderLabels([str(i + 1) for i in range(n)])
        self._compact_columns(table)
        self._update_tau_units()

    @staticmethod
    def _compact_columns(table):
        """Keep the bound and flag columns narrow; the lifetime gets the room.

        The numbers in LB / UB / Fix? are short, and a table that fits without
        scrolling leaves more of the window for the plots.
        """
        from PyQt6.QtWidgets import QHeaderView

        header = table.horizontalHeader()
        header.setSectionResizeMode(COL_TAU, QHeaderView.ResizeMode.Stretch)
        for column, width in ((COL_LB, 58), (COL_UB, 58), (COL_FIX, 38)):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
            table.setColumnWidth(column, width)
        table.verticalHeader().setDefaultSectionSize(22)

    def _add_component(self):
        self._set_component_count(self.RateFitTable.rowCount() + 1)

    def _remove_component(self):
        self._set_component_count(self.RateFitTable.rowCount() - 1)

    def _update_tau_units(self):
        """Label the lifetime column with the dataset's own time unit."""
        unit = ""
        if getattr(self, "dataset", None) is not None:
            unit = str(getattr(self.dataset, "time_unit", "") or "")
        header = f"Tau ({unit})" if unit else "Tau"
        item = QTableWidgetItem(header)
        self.RateFitTable.setHorizontalHeaderItem(0, item)
        self.IRFtable.setHorizontalHeaderItem(
            0, QTableWidgetItem(f"Val. ({unit})" if unit else "Val.")
        )

    def _sync_model_tables(self):
        """Refresh anything in the tables that depends on the loaded dataset."""
        self._update_tau_units()

    # ------------------------------------------------------------------ #
    #                          Model selection                           #
    # ------------------------------------------------------------------ #
    def _select_model_type(self, model_type: str):
        """Check the toggle button matching a :class:`pyrate.ModelType`."""
        buttons = {
            str(pr.ModelType.PARALLEL): self.ParallelButton,
            str(pr.ModelType.SEQUENTIAL): self.SequentialButton,
            str(pr.ModelType.TARGET): self.TargetButton,
        }
        btn = buttons.get(str(model_type))
        if btn is None:
            logger.debug("unknown model type %r; leaving the selection unchanged", model_type)
            return
        btn.setChecked(True)

    def model_type(self) -> str:
        """The model family currently selected in the button group."""
        if self.ParallelButton.isChecked():
            return str(pr.ModelType.PARALLEL)
        if self.TargetButton.isChecked():
            return str(pr.ModelType.TARGET)
        return str(pr.ModelType.SEQUENTIAL)

    def _on_model_changed(self):
        """React to a change of model family.

        Only a target (or ODE-defined) model has a scheme to write, so the
        editor is enabled for those and disabled -- with the reason in its
        tooltip -- for parallel and sequential, where the connectivity is
        already fixed by the family.
        """
        needs_scheme = self.model_type() == str(pr.ModelType.TARGET) or (
            getattr(self, "ODEdefinedButton", None) is not None
            and self.ODEdefinedButton.isChecked()
        )
        button = getattr(self, "DefinemodelButton", None)
        if button is not None:
            button.setEnabled(bool(needs_scheme) and self.dataset is not None)
            button.setToolTip(
                "Write the kinetic scheme (K-matrix editor) and see its graph"
                if needs_scheme
                else "Only target and ODE-defined models have a scheme to define"
            )
        # The spectra a model produces have its own name (DAS / EAS / SAS).
        self._update_species_button()
        logger.debug("model type set to %s", self.model_type())

    def _on_irf_toggled(self, on: bool):
        """Show or hide the IRF width row; time zero always applies.

        Switching the Gaussian IRF off means "do not convolve", not "there is
        no time zero": the delay axis still has an origin, and it is still
        worth fitting. Only the width row (delta) goes.
        """
        self.IRFtable.setRowHidden(ROW_IRF, not bool(on))
        self.IRFtable.setEnabled(True)
        self._sync_artifact_checkbox()

    def _sync_artifact_checkbox(self):
        """The artefact basis *is* the IRF and its derivatives.

        Without a Gaussian IRF there is no shape to build it from, so the box
        is disabled and cleared rather than left ticked and quietly ignored.
        """
        box = getattr(self, "CoherentArtifactCheckBox", None)
        if box is None:
            return
        has_irf = bool(self.GaussianIRFCheckBox.isChecked())
        box.setEnabled(has_irf)
        if not has_irf and box.isChecked():
            box.blockSignals(True)
            box.setChecked(False)
            box.blockSignals(False)
        box.setToolTip(
            "Add the IRF and its 1st and 2nd derivatives to the fit, so cross-phase "
            "modulation and other time-zero artefacts are absorbed by their own "
            "amplitudes instead of distorting the shortest lifetime."
            if has_irf
            else "Needs a Gaussian IRF: the artefact basis is built from it."
        )

    def fit_coherent_artifact(self) -> bool:
        """Whether the next fit adds the coherent-artefact columns."""
        box = getattr(self, "CoherentArtifactCheckBox", None)
        return bool(box is not None and box.isChecked() and box.isEnabled())
