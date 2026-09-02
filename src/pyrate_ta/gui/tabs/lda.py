"""GUI mixin for Lifetime Density Analysis (LDA) tab."""

from __future__ import annotations

import os

os.environ.setdefault("QT_API", "pyqt6")

import numpy as np
from PyQt6.QtWidgets import QMessageBox

from pyrate_ta.fit.lda import solve_lda
from pyrate_ta.log import get_logger
from pyrate_ta.plot.lda import plot_l_curve, plot_lda_map

logger = get_logger(__name__)


class LDATabMixin:
    """GUI mixin for running and visualising LDA in the 2nd tab."""

    def _init_lda_tab_from_settings(self) -> None:
        """Pre-populate the LDA tab widgets from the active Settings.

        The ``[lda]`` block in ``settings.toml`` is the CLI/script mechanism
        for setting LDA defaults.  The GUI tab reads those defaults here once
        at start-up, so a user who edits the file sees the change reflected in
        the widgets the next time the window opens.  From that point the widgets
        are the canonical control surface: the settings dialog never shows the
        LDA fields.
        """
        from pyrate_ta.settings import Regularisation, get_settings

        s = get_settings()

        # Number of grid points
        spin = getattr(self, "LDANtSpinBox", None)
        if spin is not None:
            spin.setValue(int(s.lda_n_lifetimes))

        # Tau range
        tmin_spin = getattr(self, "LDATauMinDoubleSpinBox", None)
        if tmin_spin is not None and s.lda_tau_min is not None:
            tmin_spin.setValue(float(s.lda_tau_min))

        tmax_spin = getattr(self, "LDATauMaxDoubleSpinBox", None)
        if tmax_spin is not None and s.lda_tau_max is not None:
            tmax_spin.setValue(float(s.lda_tau_max))

        # Regularisation method
        alpha_combo = getattr(self, "LDAAlphaCombo", None)
        if alpha_combo is not None:
            reg = s.lda_regularisation
            # Map Regularisation enum -> combo text substring
            target = {
                Regularisation.LCURVE: "L-curve",
                Regularisation.GCV: "GCV",
                Regularisation.FIXED: "Fixed",
            }.get(reg, "L-curve")
            for i in range(alpha_combo.count()):
                if target.lower() in alpha_combo.itemText(i).lower():
                    alpha_combo.setCurrentIndex(i)
                    break

        # Fixed alpha value
        alpha_val_spin = getattr(self, "LDAAlphaValueDoubleSpinBox", None)
        if alpha_val_spin is not None:
            alpha_val_spin.setValue(float(s.lda_alpha))

        # Non-negative constraint
        nn_chk = getattr(self, "LDANonNegativeCheckBox", None)
        if nn_chk is not None:
            nn_chk.setChecked(bool(s.lda_non_negative))


    def run_lda_from_gui(self):
        """Triggered by the 'Run LDA' button."""
        if getattr(self, "dataset", None) is None:
            QMessageBox.information(
                self,
                "No dataset",
                "Load a dataset before running Lifetime Density Analysis.",
            )
            return


        # Read controls if available, otherwise fall back to defaults
        spin = getattr(self, "LDANtSpinBox", None)
        n_taus = int(spin.value()) if spin is not None else 100

        pen_combo = getattr(self, "LDAPenaltyCombo", None)
        penalty = "d2"
        if pen_combo is not None:
            txt = pen_combo.currentText().lower()
            if "1st" in txt:
                penalty = "d1"
            elif "ridge" in txt or "identity" in txt:
                penalty = "ridge"

        alpha_combo = getattr(self, "LDAAlphaCombo", None)
        alpha_val_spin = getattr(self, "LDAAlphaValueDoubleSpinBox", None)
        alpha_method = "lcurve"
        alpha = "auto"
        if alpha_combo is not None:
            txt = alpha_combo.currentText().lower()
            if "manual" in txt or "fixed" in txt:
                alpha_method = "manual"
                alpha = float(alpha_val_spin.value()) if alpha_val_spin is not None else 0.01
            elif "gcv" in txt:
                alpha_method = "gcv"
            elif "morozov" in txt:
                alpha_method = "morozov"

        tmin_spin = getattr(self, "LDATauMinDoubleSpinBox", None)
        tmax_spin = getattr(self, "LDATauMaxDoubleSpinBox", None)
        tmin = float(tmin_spin.value()) if tmin_spin is not None else 0.0
        tmax = float(tmax_spin.value()) if tmax_spin is not None else 0.0
        tau_range = (tmin, tmax) if (tmin > 0 and tmax > tmin) else None

        ca_chk = getattr(self, "LDACoherentArtifactCheckBox", None)
        if ca_chk is not None:
            use_ca = ca_chk.isChecked()
        else:
            disc_ca = getattr(self, "CoherentArtifactCheckBox", None)
            use_ca = disc_ca.isChecked() if disc_ca is not None else False

        svd_spin = getattr(self, "LDASVDFilterSpinBox", None)
        svd_comps = int(svd_spin.value()) if svd_spin is not None else 0

        nn_chk = getattr(self, "LDANonNegativeCheckBox", None)
        non_neg = nn_chk.isChecked() if nn_chk is not None else False

        boot_spin = getattr(self, "LDABootstrapSpinBox", None)
        n_boot = int(boot_spin.value()) if boot_spin is not None else 0

        peaks_chk = getattr(self, "LDAPeaksCheckBox", None)
        find_pks = peaks_chk.isChecked() if peaks_chk is not None else False

        # Restrict fit to display limits
        restrict_chk = getattr(self, "LDARestrictFitCheckBox", None)
        restrict = bool(restrict_chk is not None and restrict_chk.isChecked())
        if restrict and hasattr(self, "fit_limits"):
            delay_range, probe_range = self.fit_limits()
        else:
            delay_range, probe_range = None, None

        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QApplication, QProgressDialog

        progress = QProgressDialog("Running Lifetime Density Analysis...", "Cancel", 0, 100, self)
        progress.setWindowTitle("PyRATE-TA")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(300)
        progress.setAutoClose(True)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()

        class _LDACancelled(Exception):
            pass

        def progress_cb(current: int, total: int, msg: str) -> None:
            if progress.wasCanceled():
                raise _LDACancelled("LDA cancelled by user")
            progress.setMaximum(total)
            progress.setValue(current)
            progress.setLabelText(f"Lifetime Density Analysis:\n{msg}")
            QApplication.processEvents()

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.statusBar().showMessage("Running Lifetime Density Analysis...")
            res = solve_lda(
                self.dataset,
                n_taus=n_taus,
                tau_range=tau_range,
                penalty=penalty,
                alpha=alpha,
                alpha_method=alpha_method,
                coherent_artifact=use_ca,
                svd_components=svd_comps,
                non_negative=non_neg,
                n_bootstraps=n_boot,
                find_peaks=find_pks,
                delay_range=delay_range,
                probe_range=probe_range,
                callback=progress_cb,
            )
            self.lda_result = res
            self.statusBar().showMessage(res.summary())

            # Render inline plots if widgets exist
            self._render_lda_plots(res)

            # Pop out 2D Lifetime Map figure window automatically
            self.popout_lda_map()
        except _LDACancelled:
            self.statusBar().showMessage("LDA cancelled", 4000)
            logger.info("LDA cancelled by the user.")
            return
        except Exception as err:
            logger.exception("LDA solve failed: %s", err)
            QMessageBox.critical(self, "LDA Error", f"Lifetime Density Analysis failed:\n{err}")
            return
        finally:
            QApplication.restoreOverrideCursor()
            if progress is not None:
                progress.close()

    def _lda_metric(self) -> str:
        """Selected metric for 1D integrated dynamics: 'dynamical_content' or 'abs'."""
        dyn_chk = getattr(self, "LDADynamicalContentCheckBox", None)
        return "dynamical_content" if (dyn_chk is not None and dyn_chk.isChecked()) else "abs"

    def _lda_annotate_centroids(self) -> bool:
        """Whether 'Auto-detect lifetime peak centroids' is checked."""
        chk = getattr(self, "LDAPeaksCheckBox", None)
        return bool(chk is not None and chk.isChecked())

    def _asinh_params(self) -> tuple[bool, float]:
        """(asinh_enabled, asinh_pct) inherited from the plot controls panel."""
        asinh = False
        asinh_pct = 5.0
        if hasattr(self, "plot_controls") and self.plot_controls is not None:
            chk = getattr(self.plot_controls, "arcsinh_chk", None)
            if chk is not None:
                asinh = chk.isChecked()
            pct_spin = getattr(self.plot_controls, "arcsinh_pct", None)
            if pct_spin is not None:
                asinh_pct = float(pct_spin.value())
        else:
            try:
                s = pm.get_settings()
                asinh_pct = float(getattr(s, "asinh_pct", 5.0))
            except Exception:
                pass
        return asinh, asinh_pct

    def popout_lda_map(self):
        """Open the 2D Lifetime Density Map & Integrated Dynamics in its own figure window."""
        res = getattr(self, "lda_result", None)
        if res is None:
            QMessageBox.information(
                self, "No LDA Result", "Run Lifetime Density Analysis first."
            )
            return
        import matplotlib.pyplot as plt
        import pymorgan as pm

        pm.apply_style()
        discrete_taus = self.fit_result.taus if getattr(self, "fit_result", None) is not None else None
        metric = self._lda_metric()
        asinh, asinh_pct = self._asinh_params()
        annotate_centroids = self._lda_annotate_centroids()
        res_axes = plot_lda_map(
            res,
            discrete_taus=discrete_taus,
            metric=metric,
            annotate_centroids=annotate_centroids,
            asinh=asinh,
            asinh_pct=asinh_pct,
        )
        fig = res_axes[0].figure if isinstance(res_axes, tuple) else res_axes.figure
        fig.tight_layout()
        plt.show(block=False)

    def popout_l_curve(self):
        """Open the L-Curve plot in its own figure window."""
        res = getattr(self, "lda_result", None)
        if res is None:
            QMessageBox.information(
                self, "No LDA Result", "Run Lifetime Density Analysis first."
            )
            return
        import matplotlib.pyplot as plt
        import pymorgan as pm

        pm.apply_style()
        fig, ax = plt.subplots(figsize=(6, 5))
        plot_l_curve(res, ax=ax)
        fig.tight_layout()
        plt.show(block=False)

    def popout_lda_slice(self):
        """Plot 1D slice(s) of S(tau) at selected probe position(s).

        Inherits probe cuts from interactive picking or the selection dialog, normalisation
        from PP_NormaliseCheckBox, and probe unit formatting.
        """
        res = getattr(self, "lda_result", None)
        if res is None:
            QMessageBox.information(
                self, "No LDA Result", "Run Lifetime Density Analysis first."
            )
            return

        if hasattr(self, "_interactive") and self._interactive():
            self._pick_lda_slice()
            return

        cuts = self._ask_cuts("kinetics") if hasattr(self, "_ask_cuts") else None
        if not cuts:
            if hasattr(self, "_cut") and self._cut is not None and len(self._cut) > 0:
                cuts = [self._cut[0]]
            elif hasattr(self, "_auto_wavelengths"):
                cuts = self._auto_wavelengths()
            else:
                probe_arr = res.probe if res.probe is not None else np.arange(res.S_map.shape[0])
                cuts = [float(probe_arr[len(probe_arr) // 2])]

        self._plot_lda_slice(cuts)

    def _pick_lda_slice(self):
        """Interactive picking of probe positions on the contour for LDA slice."""
        from pymorgan.gui.picker import ContourPicker

        res = getattr(self, "lda_result", None)
        if res is None:
            return

        ax = (
            self.axis("main")
            if (hasattr(self, "axis") and getattr(self, "_axes", None))
            else getattr(getattr(self, "PlotArea", None), "ax", None)
        )
        if ax is None:
            return

        if hasattr(self, "statusBar") and self.statusBar():
            self.statusBar().showMessage(
                "Interactive LDA slice: left-click to add probe positions, right-click to remove, "
                "Enter to plot, Esc to cancel."
            )

        def done(values):
            self._picker = None
            if not values:
                if hasattr(self, "statusBar") and self.statusBar():
                    self.statusBar().showMessage("Interactive selection cancelled.")
                return
            if hasattr(self, "statusBar") and self.statusBar():
                self.statusBar().clearMessage()
            self._plot_lda_slice(values)

        canvas = getattr(getattr(self, "PlotArea", None), "canvas", None)
        if canvas is not None:
            self._picker = ContourPicker(canvas, ax, "x", done).start()

    def _plot_lda_slice(self, cuts: list[float]):
        """Render 1D LDA slice figure for given probe cuts."""
        res = getattr(self, "lda_result", None)
        if res is None or not cuts:
            return
        import matplotlib.pyplot as plt
        import pymorgan as pm

        probe_arr = res.probe if res.probe is not None else np.arange(res.S_map.shape[0])
        norm = self._norm() if hasattr(self, "_norm") else False

        style = "()"
        try:
            style = str(getattr(pm.get_settings(), "label_style", "()"))
        except Exception:
            pass

        units = getattr(res, "units", {}) or {}
        time_unit = units.get("unitsT_ltx") or units.get("time_unit") or ""
        probe_unit = units.get("unitsL_ltx") or units.get("probe_unit") or ""
        units_z_lbl = units.get("unitsZ_lbl") or units.get("signal_name") or "Amplitude"
        units_z_ltx = units.get("unitsZ_ltx") or units.get("signal_unit") or "mOD"

        def _fmt_lbl(lbl: str, u: str) -> str:
            if not u:
                return lbl
            match style:
                case "[]":
                    return f"{lbl} [{u}]"
                case "/":
                    return f"{lbl} / {u}"
                case _:
                    return f"{lbl} ({u})"

        pm.apply_style()
        fig, ax = plt.subplots(figsize=(7.5, 5))

        # Standard zero line following PyMORGAN
        ax.axhline(0, color="0.75", linewidth=0.75)

        for p_val in cuts:
            idx = int(np.argmin(np.abs(probe_arr - p_val)))
            slice_data = res.S_map[idx, :]
            if norm and np.max(np.abs(slice_data)) > 0:
                slice_data = slice_data / np.max(np.abs(slice_data))

            label_str = f"{probe_arr[idx]:.4g} {probe_unit}".strip()
            ax.plot(res.tau_grid, slice_data, "o-", label=label_str, markersize=3.5, linewidth=1.5)

        ylabel = "Normalised amplitude" if norm else _fmt_lbl(units_z_lbl, units_z_ltx)
        xlabel = _fmt_lbl(r"Lifetime $\tau$", time_unit)

        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(r"Lifetime Distribution $S(\tau)$")
        ax.legend(loc="best")
        fig.tight_layout()
        plt.show(block=False)

    def _render_lda_plots(self, res):
        """Draw the 2D lifetime map and L-curve on GUI canvases if available."""
        lda_map_widget = getattr(self, "LDAMapPlot", None)
        lcurve_widget = getattr(self, "LCCurvePlot", None)

        discrete_taus = self.fit_result.taus if getattr(self, "fit_result", None) is not None else None

        if lda_map_widget is not None and hasattr(lda_map_widget, "figure"):
            lda_map_widget.figure.clear()
            ax = lda_map_widget.figure.add_subplot(111)
            metric = self._lda_metric()
            asinh, asinh_pct = self._asinh_params()
            annotate_centroids = self._lda_annotate_centroids()
            plot_lda_map(
                res,
                ax=ax,
                discrete_taus=discrete_taus,
                metric=metric,
                annotate_centroids=annotate_centroids,
                asinh=asinh,
                asinh_pct=asinh_pct,
            )
            lda_map_widget.draw()

        if lcurve_widget is not None and hasattr(lcurve_widget, "figure"):
            lcurve_widget.figure.clear()
            ax = lcurve_widget.figure.add_subplot(111)
            plot_l_curve(res, ax=ax)
            lcurve_widget.draw()

    def save_lda_map_pdat(self):
        """Export the fitted 2D LDA map S(tau, lambda) as a PyMORGAN PDAT dataset file."""
        import os
        from pathlib import Path

        from PyQt6.QtWidgets import QFileDialog, QMessageBox

        res = getattr(self, "lda_result", None)
        if res is None:
            QMessageBox.information(
                self,
                "No LDA Result",
                "Run Lifetime Density Analysis first before saving the map.",
            )
            return

        default_dir = self._get_dialog_dir()
        default_name = "dataset_LDA_MAP.pdat"
        curr_path = getattr(self, "_current_path", None)
        if curr_path:
            default_name = Path(curr_path).stem + "_LDA_MAP.pdat"
            default_dir = str(Path(curr_path).parent)

        default_path = os.path.join(default_dir, default_name)

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save fitted LDA Map as PDAT",
            default_path,
            "PDAT (*.pdat);;All files (*)",
        )
        if not path:
            return
        self._last_dir = str(Path(path).parent)

        try:
            saved_path = res.to_pdat(path)
            QMessageBox.information(
                self,
                "Export complete",
                f"Fitted 2D LDA map successfully saved as PDAT:\n\n{saved_path}",
            )
            self.statusBar().showMessage(f"Exported LDA Map PDAT to {saved_path}")
        except Exception as exc:
            QMessageBox.warning(self, "Export failed", f"Failed to save LDA map PDAT: {exc}")

    def show_lda_citations(self):
        """Show dialog listing recommended literature citations for active/fitted LDA algorithms."""
        from PyQt6.QtWidgets import QMessageBox

        res = getattr(self, "lda_result", None)
        if res is not None:
            msg = res.format_citations()
        else:
            msg = (
                "Recommended Literature Citations for LDA Analysis:\n\n"
                "[1] Slavov, C., Hartmann, H., & Wachtveitl, J. (2015). Implementation and Evaluation of Data Analysis Strategies for Time-Resolved Optical Spectroscopy. Anal. Chem., 87(4), 2328-2336. DOI: 10.1021/ac504348h\n"
                "[2] Dorlhiac, G. F., Fare, C., & van Thor, J. J. (2017). PyLDM - An open source package for lifetime density analysis of time-resolved spectroscopic data. PLOS Comput. Biol., 13(5), e1005528. DOI: 10.1371/journal.pcbi.1005528\n"
                "[3] Steinbach, P. J., Ionescu, R., & Champion, P. M. (2002). Analysis of Kinetics Using a Hybrid Maximum-Entropy/Nonlinear-Least-Squares Method: Application to Protein Folding. Biophys. J., 82(4), 2244-2255. DOI: 10.1016/S0006-3495(02)75570-7\n"
                "[4] Hansen, P. C. (1992). Analysis of discrete ill-posed problems by means of the L-curve. SIAM Review, 34(4), 561-580. DOI: 10.1137/1034115\n"
                "[5] Golub, G. H., Heath, M., & Wahba, G. (1979). Generalized cross-validation as a method for choosing a good ridge parameter. Technometrics, 21(2), 215-223. DOI: 10.1080/00401706.1979.10489751"
            )
        QMessageBox.information(self, "Literature Citations for LDA", msg)
