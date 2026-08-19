"""Reading the model out of the GUI, running a fit, and showing the result.

The window is a caller of :mod:`pyrate_ta.fit`, never a prerequisite: everything
here reads widgets, hands plain values to the engine, and puts the answer back
on screen. No fitting logic lives in this module.

Fit limits
----------
When *Restrict fit to display limits* is ticked, the delay and probe windows
currently shown in the plot-controls panel are what gets fitted. The probe
limits are converted from the display unit (nm / cm-1 / eV / THz) back to the
dataset's native unit before they reach the engine, so cutting the data in
whatever unit is on screen does the right thing. Both windows are recorded on
the result, because restricting them changes what the fit saw.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_API", "pyqt6")

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMessageBox,
    QProgressDialog,
    QTableWidgetItem,
)

import pyrate_ta as pr

from ...helpers import format_lifetime, parse_lifetime
from ...log import get_logger
from ..mw_common import _FIT_ONLY_WIDGETS, _TOOLTIPS

logger = get_logger(__name__)


class _FitCancelled(Exception):
    """Raised inside the residual callback when the user presses Cancel."""


class FitTabMixin:
    """Run a fit from the GUI and display it."""

    # ------------------------------------------------------------------ #
    #                            Fit limits                              #
    # ------------------------------------------------------------------ #
    def restrict_to_display(self) -> bool:
        chk = getattr(self, "RestrictFitCheckBox", None)
        return bool(chk is not None and chk.isChecked())

    def fit_limits(self):
        """``(delay_range, probe_range)`` for the fit, or ``(None, None)``.

        The probe window is converted to the dataset's native spectral unit;
        the delay window is already in the dataset's time unit. Limits that do
        not actually cut anything are returned as ``None``, so a result only
        records a restriction when there was one.
        """
        if not self.restrict_to_display() or self.plot_controls is None or self.dataset is None:
            return None, None

        delay_range = tuple(float(v) for v in self.plot_controls.ylim())
        probe_display = self.plot_controls.xlim()
        probe_native = tuple(float(v) for v in self.plot_controls.probe_to_native(probe_display))
        probe_range = (min(probe_native), max(probe_native))

        delays = np.asarray(self.dataset.delays, dtype=float)
        probe = np.asarray(self.dataset.probe, dtype=float)
        if delay_range[0] <= delays.min() and delay_range[1] >= delays.max():
            delay_range = None
        if probe_range[0] <= probe.min() and probe_range[1] >= probe.max():
            probe_range = None
        return delay_range, probe_range

    def describe_fit_limits(self) -> str:
        """One line naming what will actually be fitted, for the status bar."""
        delay_range, probe_range = self.fit_limits()
        if delay_range is None and probe_range is None:
            return "fitting the whole dataset"
        parts = []
        if delay_range is not None:
            parts.append(f"delays {delay_range[0]:.4g} to {delay_range[1]:.4g}")
        if probe_range is not None:
            parts.append(f"probe {probe_range[0]:.6g} to {probe_range[1]:.6g}")
        return "fitting " + ", ".join(parts)

    def _on_limits_changed(self):
        if self.dataset is not None:
            self.statusBar().showMessage(self.describe_fit_limits(), 4000)

    # ------------------------------------------------------------------ #
    #                        Reading the tables                          #
    # ------------------------------------------------------------------ #
    def _table_value(self, table, row: int, column: int, *, field: str):
        item = table.item(row, column)
        text = "" if item is None else item.text()
        return parse_lifetime(text, field=field)

    def read_lifetimes(self):
        """``(taus, fixed)`` from the component table.

        ``inf`` (in any of its spellings) is a non-decaying component and is
        always fixed. A malformed entry raises with the row named, rather than
        being silently replaced by a default.
        """
        table = self.RateFitTable
        taus, fixed = [], []
        for row in range(table.rowCount()):
            taus.append(self._table_value(table, row, 0, field=f"lifetime in row {row + 1}"))
            item = table.item(row, 3)
            fixed.append(bool(item is not None and item.checkState() == Qt.CheckState.Checked))
        return taus, fixed

    def read_irf(self):
        """``(t0, irf_fwhm, fit_t0, fit_irf)`` from the IRF table and its switch.

        Time zero is read either way: turning the Gaussian IRF off means the
        model is not convolved, not that the delay axis has no origin. Only the
        width is dropped.
        """
        table = self.IRFtable
        convolve = self.GaussianIRFCheckBox.isChecked()

        def value(row, default):
            item = table.item(row, 0)
            try:
                return float(item.text().replace(",", ".")) if item else default
            except ValueError:
                logger.debug("unreadable IRF table entry in row %d", row + 1, exc_info=True)
                return default

        def is_fixed(row):
            item = table.item(row, 3)
            return bool(item is not None and item.checkState() == Qt.CheckState.Checked)

        t0 = value(0, 0.0)
        if not convolve:
            return t0, None, not is_fixed(0), False
        return t0, value(1, 0.2), not is_fixed(0), not is_fixed(1)

    def fit_method(self) -> str:
        combo = getattr(self, "FitMethod", None)
        return str(combo.currentData() or "trf") if combo is not None else "trf"

    # ------------------------------------------------------------------ #
    #                              Fitting                               #
    # ------------------------------------------------------------------ #
    def run_fit(self, preview: bool = False):
        """Fit the loaded dataset with the model shown in the panel.

        ``preview=True`` evaluates the model at the current parameters without
        optimising, so the guess can be judged before committing to a fit.
        """
        if self.dataset is None:
            return
        try:
            taus, fixed = self.read_lifetimes()
        except ValueError as exc:
            QMessageBox.warning(self, "Check the lifetimes", str(exc))
            return

        t0, irf_fwhm, fit_t0, fit_irf = self.read_irf()
        delay_range, probe_range = self.fit_limits()
        model_type = self.model_type()
        use_weights = self.UseNoiseWeightsCheckBox.isChecked()

        kwargs = dict(
            taus=taus,
            fixed=list(fixed),
            t0=t0,
            irf_fwhm=irf_fwhm,
            coherent_artifact=self.fit_coherent_artifact(),
            fit_t0=fit_t0 and not preview,
            fit_irf=fit_irf and not preview,
            use_weights=use_weights,
            delay_range=delay_range,
            probe_range=probe_range,
            method=self.fit_method(),
        )

        self.statusBar().showMessage(
            ("Previewing: " if preview else "Fitting: ") + self.describe_fit_limits()
        )
        progress = None
        if not preview:
            self._open_monitor(taus, fit_t0, fit_irf, fixed)
            monitor = getattr(self, "_monitor", None)
            if monitor is not None:
                progress = monitor
            else:
                progress = QProgressDialog("Fitting...", "Cancel", 0, 0, self)
                progress.setWindowTitle("PyRATE-TA")
                progress.setWindowModality(Qt.WindowModality.WindowModal)
                progress.setMinimumDuration(400)
                progress.setAutoClose(True)
                progress.show()
            kwargs["callback"] = self._make_progress_callback(progress)

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            scheme = getattr(self, "custom_scheme", None)
            if model_type == str(pr.ModelType.TARGET) and scheme is None:
                QMessageBox.information(
                    self,
                    "No scheme defined",
                    "A target fit needs a kinetic scheme. Press 'Define model...' and "
                    "write one, or choose a template.",
                )
                return
            if model_type == str(pr.ModelType.TARGET):
                kwargs.pop("model_type", None)
                if preview:
                    for key in ("fixed", "fit_t0", "fit_irf", "method"):
                        kwargs.pop(key, None)
                    self.fit_result = pr.preview_global(self.dataset, scheme=scheme, **kwargs)
                else:
                    self.fit_result = pr.fit_target(self.dataset, scheme=scheme, **kwargs)
            elif preview:
                # Nothing is optimised: the model is evaluated at the parameters
                # as typed and the spectra follow from them.
                for key in ("fixed", "fit_t0", "fit_irf", "method"):
                    kwargs.pop(key, None)
                self.fit_result = pr.preview_global(self.dataset, model_type=model_type, **kwargs)
            else:
                self.fit_result = pr.fit_global(self.dataset, model_type=model_type, **kwargs)
        except _FitCancelled:
            # Must precede the generic handler: cancelling is not a failure.
            self.statusBar().showMessage("Fit cancelled", 4000)
            logger.info("Fit cancelled by the user.")
            return
        except pr.NoiseUnavailableError as exc:
            QMessageBox.warning(self, "No noise available", str(exc))
            return
        except Exception as exc:
            logger.debug("fit failed", exc_info=True)
            QMessageBox.critical(self, "Fit failed", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
            if progress is not None:
                progress.close()
            self._close_monitor()

        self._show_fit_result(preview=preview)
        if pr.get_settings().autosave_session:
            self.save_fit_session(ask=False)

    def _open_monitor(self, taus, fit_t0: bool, fit_irf: bool, fixed):
        """Open the live parameter window, if the settings ask for one.

        The names must match the optimiser's vector, which appends ``t0`` and
        the IRF width *only when they are free* -- the same order
        ``KineticModel.param_names()`` uses, rebuilt here because the model does
        not exist yet when the window opens.
        """
        from ..monitor import make_monitor

        names = [f"tau{i + 1}" for i in range(len(taus))]
        flags = list(fixed)
        if fit_t0:
            names.append("t0")
            flags.append(False)
        if fit_irf:
            names.append("IRF")
            flags.append(False)
        self._monitor = make_monitor(self, names, fixed=flags)

    def _close_monitor(self):
        """Close the live window, keeping the trajectory it recorded."""
        monitor = getattr(self, "_monitor", None)
        if monitor is None:
            return
        try:
            self._monitor_history = monitor.history()
            monitor.close()
        except Exception:
            logger.debug("could not close the parameter monitor", exc_info=True)
        self._monitor = None

    def _make_progress_callback(self, progress):
        """Adapt the optimiser's callback to the progress dialog.

        Cancelling raises inside the residual function, which is the only way
        to stop ``least_squares`` mid-run; the caller turns that into a plain
        "cancelled" message rather than an error, since the user asked for it.
        """

        def report(n_eval: int, params, cost: float) -> None:
            if progress.wasCanceled():
                raise _FitCancelled
            progress.setLabelText(
                f"Fitting: {n_eval} evaluation(s), cost {cost:.5g}\n"
                + ", ".join(format_lifetime(v) for v in params[: self.RateFitTable.rowCount()])
            )
            monitor = getattr(self, "_monitor", None)
            if monitor is not None:
                # The monitor decides whether this evaluation is due; it also
                # pumps the event loop, so no second processEvents is needed.
                monitor.update_parameters(n_eval, params, cost)
            else:
                QApplication.processEvents()

        return report

    # ------------------------------------------------------------------ #
    #                            Sessions                                #
    # ------------------------------------------------------------------ #
    def save_fit_session(self, ask: bool = True):
        """Write the current fit, with its reproducibility payload, to disk."""
        if self.fit_result is None:
            return
        default = pr.io.default_session_path(self.fit_result, self._last_dir)
        path = str(default)
        if ask:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save fit session", path, "PyRATE-TA fit session (*.prfit)"
            )
            if not path:
                return
        try:
            written = pr.save_fit(path, self.fit_result)
        except Exception as exc:
            logger.debug("saving the fit session failed", exc_info=True)
            QMessageBox.warning(self, "Could not save", str(exc))
            return
        self.statusBar().showMessage(f"Fit session saved to {written.name}", 6000)

    def load_fit_session(self):
        """Open a saved session and report it.

        The reopened fit is shown and summarised, but it is not made the live
        result: it has no dataset behind it, and pretending otherwise would
        invite a "refit" of something that cannot be refitted.
        """
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open fit session",
            str(self._last_dir or ""),
            "PyRATE-TA fit session (*.prfit);;All files (*)",
        )
        if not path:
            return
        try:
            loaded = pr.load_fit(path)
        except Exception as exc:
            logger.debug("loading the fit session failed", exc_info=True)
            QMessageBox.critical(self, "Could not open", str(exc))
            return
        logger.info("%s", loaded.summary())
        QMessageBox.information(self, "Fit session", loaded.summary())
        self.statusBar().showMessage(f"Opened {loaded!r}", 8000)

    def _show_fit_result(self, preview: bool = False):
        """Fill the read-outs, enable the views, and draw the outcome."""
        fit = self.fit_result
        if fit is None:
            return

        self.Chi2Value.setText(f"{fit.statistic.value:.5g}")
        self._update_statistic_label()
        for name in _FIT_ONLY_WIDGETS:
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(True)
                widget.setToolTip(_TOOLTIPS.get(name, ""))
        self._update_species_button()

        self._write_fitted_lifetimes(fit)
        # The panels now have a fit to overlay: data as points, model as lines.
        self.render_all()
        self.statusBar().showMessage(
            ("Preview" if preview else "Fit")
            + f": {', '.join(format_lifetime(v) for v in fit.taus)}"
            + f" | {fit.statistic}",
        )
        self._render_species_spectra()
        self._autoplot_after_fit(preview=preview)
        if fit.report is not None and not fit.converged:
            QMessageBox.warning(
                self,
                "Fit did not converge",
                f"{fit.report}\n\nThe values shown are the last trial parameters, not a result.",
            )
        if pr.get_settings().open_scheme_after_fit:
            self._popout_scheme()

    def _update_species_button(self):
        """Name the spectra button after the model's own spectra.

        A parallel fit gives DAS, a sequential one EAS and a target one SAS.
        The distinction is not cosmetic -- it is what the spectra may be
        interpreted as -- so the button says which one it will open, taking the
        model from the result when there is one and from the selection
        otherwise.
        """
        button = getattr(self, "PlotSpeciesSpectraButton", None)
        if button is None:
            return
        fit = getattr(self, "fit_result", None)
        if fit is not None:
            kind = fit.spectra_kind
        else:
            kind = {
                str(pr.ModelType.PARALLEL): "DAS",
                str(pr.ModelType.SEQUENTIAL): "EAS",
                str(pr.ModelType.TARGET): "SAS",
            }.get(self.model_type(), "spectra")
        button.setText(f"Plot {kind}")

    @staticmethod
    def _quoting_kwargs() -> dict:
        """How lifetimes are quoted in a legend, from the PyRATE-TA settings.

        ``plot_species_spectra`` builds its own labels, so the policy has to be
        handed to it per call -- PyMORGAN cannot read PyRATE-TA's settings.
        """
        s = pr.get_settings()
        return {
            "printErrors": bool(s.show_uncertainties),
            "pair_round": bool(s.round_uncertainties),
        }

    def _species_dataset(self):
        """The dataset to draw the species spectra against.

        Not necessarily the loaded one: a fit restricted to part of the
        spectral range has spectra on that window only, so the probe axis has
        to be the fit's, not the dataset's.
        """
        from ...plot.matrix import as_fit_dataset

        return as_fit_dataset(self.fit_result, self.dataset)

    def _popout_species_spectra(self):
        """Open the fitted species spectra in their own window."""
        fit = getattr(self, "fit_result", None)
        if fit is None or self.dataset is None:
            return
        import matplotlib.pyplot as plt
        import pymorgan as pm

        pm.apply_style()
        do_smooth = self._smooth_value() if hasattr(self, "_smooth_value") else 0
        norm = self._norm() if hasattr(self, "_norm") else False
        gsb_chk = getattr(self, "GSBRecoveryCheckBox", None)
        show_gsb = gsb_chk.isChecked() if gsb_chk is not None else False

        try:
            ax = self._species_dataset().plot_species_spectra(
                *fit.as_species_args(),
                doSmooth=do_smooth,
                normY=norm,
                species_labels=fit.species_labels() if hasattr(fit, "species_labels") else None,
                **self._quoting_kwargs(),
            )

            if show_gsb:
                from ...fit.groundstate import compute_absolute_spectra

                # Always use the fit's probe axis to avoid size mismatches
                # when the fit was restricted to a sub-range of the dataset.
                species_ds = self._species_dataset()
                fit_probe = np.asarray(species_ds.probe, dtype=float)
                Np_fit = len(fit_probe)

                if getattr(self.dataset, "gs_spectrum", None) is not None:
                    gs_full = np.asarray(self.dataset.gs_spectrum, dtype=float)
                    # Interpolate onto fit probe grid if sizes differ
                    if len(gs_full) != Np_fit:
                        gs_spectrum = np.interp(fit_probe, np.asarray(self.dataset.probe, dtype=float), gs_full)
                    else:
                        gs_spectrum = gs_full
                else:
                    Z_mat = np.asarray(self.dataset.Z)[:, :, 0] if np.ndim(self.dataset.Z) == 3 else np.asarray(self.dataset.Z)
                    t_arr = np.asarray(self.dataset.delays, dtype=float)
                    t_pos = t_arr > 0
                    early_D_full = Z_mat[t_pos, :][0, :] if np.sum(t_pos) > 0 else Z_mat[0, :]
                    # Restrict to fit probe range
                    full_probe = np.asarray(self.dataset.probe, dtype=float)
                    if len(early_D_full) != Np_fit:
                        early_D = np.interp(fit_probe, full_probe, early_D_full)
                    else:
                        early_D = early_D_full
                    gs_spectrum = np.maximum(0.0, -early_D)

                if np.any(gs_spectrum > 0):
                    _, a_scale, _ = compute_absolute_spectra(fit.S, gs_spectrum, gs_scale="auto")
                    gsb_trace = -1.0 * a_scale * gs_spectrum
                    if norm and np.max(np.abs(gsb_trace)) > 0:
                        gsb_trace = gsb_trace / np.max(np.abs(gsb_trace))
                    ax.fill_between(
                        fit_probe,
                        gsb_trace,
                        0,
                        alpha=0.25,
                        color="dimgray",
                        label="GSB (-1 × GS)",
                        zorder=0,
                    )
                    ax.plot(fit_probe, gsb_trace, color="dimgray", linewidth=1.2, zorder=1)
                    ax.legend(loc="best")
                    if hasattr(ax.figure.canvas, "draw_idle"):
                        ax.figure.canvas.draw_idle()

        except Exception as exc:
            logger.debug("species-spectra window failed", exc_info=True)
            QMessageBox.warning(self, f"Plot {fit.spectra_kind}", str(exc))
            return
        plt.show(block=False)

    def _popout_concentrations(self):
        """Open the concentration profiles C(t) of the fitted model."""
        fit = getattr(self, "fit_result", None)
        if fit is None:
            return
        import matplotlib.pyplot as plt

        try:
            pr.plot_concentrations(fit)
        except Exception as exc:
            logger.debug("concentration-profile window failed", exc_info=True)
            QMessageBox.warning(self, "Concentration profiles", str(exc))
            return
        plt.show(block=False)

    def _autoplot_after_fit(self, preview: bool = False):
        """Open the post-fit figures the settings ask for.

        Only after a real fit: a preview is a look at a guess, and spawning two
        windows on every click of it would be a nuisance. Each figure has its
        own switch in ``settings.toml``, so this can be turned off without
        losing the buttons.
        """
        if preview:
            return
        s = pr.get_settings()
        if s.plot_species_after_fit:
            self._popout_species_spectra()
        if s.plot_profiles_after_fit:
            self._popout_concentrations()
        chk = getattr(self, "ExtraPlotsAfterFit_CheckBox", None)
        trio_ticked = chk.isChecked() if chk is not None else getattr(s, "plot_trio_after_fit", True)
        if trio_ticked:
            self._popout_data_fit_residuals()

    def _popout_data_fit_residuals(self):
        """Open the 3-column side-by-side Data+Fit+Residuals contour plot in its own window."""
        fit = getattr(self, "fit_result", None)
        if fit is None or self.dataset is None:
            return
        import matplotlib.pyplot as plt
        import pymorgan as pm

        from ...plot.matrix import plot_data_fit_residuals

        pm.apply_style()
        pc = getattr(self, "plot_controls", None)
        kwargs = pc.contour_kwargs() if pc is not None else {}
        try:
            plot_data_fit_residuals(fit, self.dataset, **kwargs)
        except Exception as exc:
            logger.debug("Data+Fit+Residuals plot failed", exc_info=True)
            QMessageBox.warning(self, "Plot failed", str(exc))
            return
        plt.show(block=False)


    def _write_fitted_lifetimes(self, fit):
        """Put the fitted lifetimes, t0 and IRF width (delta) back in the initial guess tables."""
        decimals = int(pr.get_settings().table_decimals)

        # 1. Lifetime table (RateFitTable)
        table = self.RateFitTable
        for row, (tau, err) in enumerate(zip(fit.taus, fit.tau_err, strict=False)):
            if row >= table.rowCount():
                break
            item = QTableWidgetItem(format_lifetime(tau, f"%.{decimals}g"))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if np.isfinite(err):
                item.setToolTip(f"1-sigma uncertainty: {err:.3g}")
            table.setItem(row, 0, item)

        # 2. IRF table (IRFtable: Row 0 = t0, Row 1 = delta / irf_fwhm)
        irf_table = getattr(self, "IRFtable", None)
        if irf_table is not None and irf_table.rowCount() >= 2:
            t0 = getattr(fit, "t0", None)
            t0_err = getattr(fit, "t0_err", None)
            if t0 is not None and np.isfinite(t0):
                t0_item = QTableWidgetItem(f"{t0:.{decimals}g}")
                t0_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if t0_err is not None and np.isfinite(t0_err):
                    t0_item.setToolTip(f"1-sigma uncertainty: {t0_err:.3g}")
                irf_table.setItem(0, 0, t0_item)

            irf_fwhm = getattr(fit, "irf_fwhm", None)
            irf_err = getattr(fit, "irf_err", None)
            if irf_fwhm is not None and np.isfinite(irf_fwhm):
                irf_item = QTableWidgetItem(f"{irf_fwhm:.{decimals}g}")
                irf_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if irf_err is not None and np.isfinite(irf_err):
                    irf_item.setToolTip(f"1-sigma uncertainty: {irf_err:.3g}")
                irf_table.setItem(1, 0, irf_item)


    def _copy_lifetimes(self):
        """Copy the fitted lifetimes (with uncertainties) to the clipboard."""
        if self.fit_result is None:
            return
        rows = [
            f"{format_lifetime(tau)}\t{err:.4g}"
            if np.isfinite(err)
            else f"{format_lifetime(tau)}\t"
            for tau, err in zip(self.fit_result.taus, self.fit_result.tau_err, strict=True)
        ]
        QApplication.clipboard().setText("\n".join(rows))
        self.statusBar().showMessage("Lifetimes copied to the clipboard", 3000)

    def _reset_fit(self):
        """Discard the fit and return to the data view."""
        self.fit_result = None
        self.Chi2Value.clear()
        self.ShowDataButton.setChecked(True)
        for name in _FIT_ONLY_WIDGETS:
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(False)
                widget.setToolTip("No fit result yet")
        self._update_species_button()
        if hasattr(self, "_reset_tables"):
            self._reset_tables()
        self.render_all()
        self.statusBar().showMessage("Fit discarded and parameters reset", 3000)


    # ------------------------------------------------------------------ #
    #                            Displaying                              #
    # ------------------------------------------------------------------ #
    def current_view(self) -> str:
        if getattr(self, "ShowResidualsButton", None) is not None and (
            self.ShowResidualsButton.isChecked()
        ):
            return "residuals"
        if getattr(self, "ShowFitButton", None) is not None and self.ShowFitButton.isChecked():
            return "fit"
        return "data"

    def _on_view_changed(self):
        """Redraw the panels for the selected view (data / fit / residuals)."""
        if self.dataset is None:
            return
        self.render_all()

    def _render_species_spectra(self):
        """Replace the spectra panel with the fitted species spectra.

        Drawn by PyMORGAN's own ``plot_species_spectra``, which is why the
        result exposes ``as_species_args()``: the adapter is on this side, the
        rendering stays upstream.
        """
        fit = self.fit_result
        if fit is None or self.dataset is None:
            return
        ax = self.axis("spectra")
        ax.clear()
        try:
            self._species_dataset().plot_species_spectra(
                *fit.as_species_args(),
                ax=ax,
                fig=self.PlotArea.figure,
                settings=self.embedded_settings(),
                species_labels=fit.species_labels() if hasattr(fit, "species_labels") else None,
                **self._quoting_kwargs(),
            )

        except Exception:
            logger.debug("species-spectra preview failed", exc_info=True)
            return
        self._style_preview(self.PlotArea)
        self.PlotArea.canvas.draw_idle()

    def edit_scheme(self):
        """Open the K-matrix editor and adopt the scheme it returns."""
        from ..scheme_dialog import edit_scheme

        previous = getattr(self, "custom_scheme", None)
        scheme = edit_scheme(self, text=previous.source_text if previous else None)
        if scheme is None:
            return
        self.custom_scheme = scheme
        self.TargetButton.setChecked(True)
        # The rate table must match the scheme: one row per rate constant, named.
        self._set_component_count(scheme.n_rates)
        self.RateFitTable.setVerticalHeaderLabels(list(scheme.parameter_names()))
        self.statusBar().showMessage(f"Scheme: {scheme.label}", 8000)
        logger.info("Target scheme set: %s", scheme.label)

    def scheme_source(self):
        """``(object, taus)`` describing the model as it stands right now.

        Resolved rather than guessed, in this order:

        1. a scheme written in the editor, with the lifetimes from the table;
        2. otherwise the selected family, built for as many components as the
           table has rows.

        A target family with no scheme yet is refused with an instruction, not a
        traceback: there is nothing to compile until a scheme exists.
        """
        taus, _ = self.read_lifetimes()
        scheme = getattr(self, "custom_scheme", None)
        if scheme is not None:
            needed = int(scheme.n_rates)
            if len(taus) < needed:
                raise ValueError(
                    f"the scheme has {needed} rate constant(s) but the table has "
                    f"{len(taus)} row(s); add rows, or reopen 'Define model...'"
                )
            return scheme, taus[:needed]

        if self.model_type() == str(pr.ModelType.TARGET):
            raise ValueError(
                "a target model has no scheme yet: press 'Define model...' and write one "
                "(or pick a template) before drawing it"
            )
        return pr.make_model(self.model_type(), len(taus)), taus

    def _popout_scheme(self):
        """Open the kinetic-scheme diagram of the current model in its own window."""
        if self.dataset is None:
            return
        import matplotlib.pyplot as plt

        try:
            obj, taus = self.scheme_source()
            pr.plot_scheme(obj, taus=taus)
        except ValueError as exc:
            # A model that cannot be drawn yet is a state, not a failure.
            logger.debug("no drawable scheme", exc_info=True)
            QMessageBox.information(self, "Kinetic scheme", str(exc))
            return
        except Exception as exc:
            logger.debug("scheme diagram failed", exc_info=True)
            QMessageBox.warning(self, "Scheme diagram", str(exc))
            return
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plt.tight_layout()
        except Exception:
            pass
        plt.show(block=False)

