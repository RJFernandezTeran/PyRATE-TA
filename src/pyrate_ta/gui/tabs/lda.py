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
            )
            self.lda_result = res
            self.statusBar().showMessage(res.summary())

            # Render inline plots if widgets exist
            self._render_lda_plots(res)

            # Pop out 2D Lifetime Map figure window automatically
            self.popout_lda_map()
        except Exception as err:
            logger.exception("LDA solve failed: %s", err)
            QMessageBox.critical(self, "LDA Error", f"Lifetime Density Analysis failed:\n{err}")

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
        res_axes = plot_lda_map(res, discrete_taus=discrete_taus)
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

        Inherits probe cuts from the 'Plot Kinetics' selection dialog, normalisation
        from PP_NormaliseCheckBox, and probe unit formatting.
        """
        res = getattr(self, "lda_result", None)
        if res is None:
            QMessageBox.information(
                self, "No LDA Result", "Run Lifetime Density Analysis first."
            )
            return

        import matplotlib.pyplot as plt
        import pymorgan as pm

        # Prompt for probe positions (inheriting Plot Kinetics dialog)
        cuts = self._ask_cuts("kinetics") if hasattr(self, "_ask_cuts") else None
        if not cuts:
            if hasattr(self, "_cut") and self._cut is not None and len(self._cut) > 0:
                cuts = [self._cut[0]]
            elif hasattr(self, "_auto_wavelengths"):
                cuts = self._auto_wavelengths()
            else:
                probe_arr = res.probe if res.probe is not None else np.arange(res.S_map.shape[0])
                cuts = [float(probe_arr[len(probe_arr) // 2])]

        probe_arr = res.probe if res.probe is not None else np.arange(res.S_map.shape[0])
        norm = self._norm() if hasattr(self, "_norm") else False

        pm.apply_style()
        fig, ax = plt.subplots(figsize=(7.5, 5))

        ylabel = "Amplitude (mOD)"
        for p_val in cuts:
            idx = int(np.argmin(np.abs(probe_arr - p_val)))
            slice_data = res.S_map[idx, :]
            if norm and np.max(np.abs(slice_data)) > 0:
                slice_data = slice_data / np.max(np.abs(slice_data))
                ylabel = "Normalised Amplitude"

            label_str = f"Probe = {probe_arr[idx]:.4g}"
            ax.plot(res.tau_grid, slice_data, "o-", label=label_str, markersize=3.5, linewidth=1.5)

        ax.set_xscale("log")
        ax.set_xlabel(r"Lifetime $\tau$")
        ax.set_ylabel(ylabel)
        ax.set_title("LDA Lifetime Distribution Slice S(tau)")
        ax.grid(True, linestyle=":", alpha=0.6)
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
            plot_lda_map(res, ax=ax, discrete_taus=discrete_taus)
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

        default_dir = getattr(self, "_rootdir_text", lambda: "")() or ""
        default_name = "dataset_LDA_MAP.pdat"
        curr_path = getattr(self, "_current_path", None)
        if curr_path:
            default_name = Path(curr_path).stem + "_LDA_MAP.pdat"

        default_path = os.path.join(default_dir, default_name)

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save fitted LDA Map as PDAT",
            default_path,
            "PDAT (*.pdat);;All files (*)",
        )
        if not path:
            return

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
                "[1] Megerle, U., Lechner, R., & Riedle, E. (2011). Lifetime density analysis of femtosecond transient absorption spectra. Phys. Chem. Chem. Phys., 13, 8869-8877.\n"
                "[2] Mullen, K. M., & van Stokkum, I. H. (2007). TIMP: an R package for modelling multi-way spectroscopic data. J. Stat. Softw., 18, 1-46.\n"
                "[3] van Stokkum, I. H. et al. (2004). Global and target analysis of time-resolved spectra. BBA-Bioenergetics, 1657, 82-104.\n"
                "[4] Hansen, P. C. (1992). Analysis of discrete ill-posed problems by means of the L-curve. SIAM Review, 34, 561-580."
            )
        QMessageBox.information(self, "Literature Citations for LDA", msg)
