"""Dataset loading, the embedded contour preview and the trace panels.

Part of the :class:`~pyrate_ta.gui.main_window.MainWindow` implementation, split
out as a mixin: the methods run as ``MainWindow`` methods and bind to widgets
declared in ``main_window.ui``, so ``self`` is always the main window.

Nothing here reads a file format or draws a standard spectroscopy figure --
loading goes through :func:`pymorgan.load_1D` and every plot through the
``Dataset1D.plot_*`` family.

"""

from __future__ import annotations

import os

os.environ.setdefault("QT_API", "pyqt6")

from pathlib import Path

import numpy as np
import pymorgan as pm
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from ...log import get_logger
from ...plot.style import overlay_styles, scale_kwargs
from ..mw_common import (
    _DATASET_ENABLED_WIDGETS,
    _DATASET_PANELS,
    _DATASET_WIDGETS,
    _FIT_ONLY_WIDGETS,
    _UNIMPLEMENTED,
    _safe_set_limits,
)

logger = get_logger(__name__)


# Delays (in the dataset time unit) proposed for the transient-spectra panel
# when the user has not picked any.
_DEFAULT_SPEC_DELAYS = [0.25, 0.5, 1, 2, 5, 10, 50, 100, 500, 1500, 5000]


class DataTabMixin:
    """Load a dataset and keep the three embedded axes in step with it."""

    # ------------------------------------------------------------------ #
    #                              Loading                               #
    # ------------------------------------------------------------------ #
    def open_file(self):
        """Ask for a dataset and load it (Load Data button / File -> Open)."""
        start = str(self._last_dir or Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load transient dataset",
            start,
            "Processed 1D transient (*.pdat);;All files (*)",
        )
        if not path:
            return
        self._last_dir = str(Path(path).parent)
        self.load_path(path)

    def load_path(self, path: str, data_type: str | None = None):
        """Load a 1-D dataset through PyMORGAN's loader registry.

        A fit needs a processed transient, so the format is ``PDAT`` -- with its
        sibling ``.pdatn`` picked up automatically when present. There is no
        format selector to get wrong; ``data_type`` remains an argument for the
        tests and for a caller with a reason.

        Also the entry point used by the tests, which is why it takes a plain
        path and never touches a file dialog.
        """
        dt = data_type or "PDAT"
        self.DataLoadedLamp.set_state("loading")
        try:
            self.dataset = pm.load_1D(path, data_type=dt)
        except Exception as exc:  # a failed load must be visible, not silent
            logger.debug("load_1D failed for %s (%s)", path, dt, exc_info=True)
            self.dataset = None
            self._current_path = None
            self.DataLoadedLamp.set_state("error")
            self._set_dataset_widgets_enabled(False)
            QMessageBox.critical(self, "Load failed", str(exc))
            return

        self._current_path = path
        self.DataLoadedLamp.set_state("loaded")
        logger.info("Loaded %s - %r", Path(path).name, self.dataset)
        self.statusBar().showMessage(f"Loaded {Path(path).name} - {self.dataset!r}")

        if self.plot_controls is not None:
            self.plot_controls.set_dataset(self.dataset)
        self._set_dataset_widgets_enabled(True)
        self._sync_model_tables()
        self.render_all()

    def clear_data(self):
        """Unload the dataset and blank every embedded axis."""
        self.dataset = None
        self._current_path = None
        self.DataLoadedLamp.set_state("off")
        self._set_noise_lamp(None)
        self._detach_crosshair()
        self._cut = None  # the next dataset has its own coordinates
        self._axes = {}
        self.PlotArea.figure.clear()
        self.PlotArea.canvas.draw_idle()
        self._set_dataset_widgets_enabled(False)
        self.statusBar().showMessage("No dataset loaded")

    # ------------------------------------------------------------------ #
    #                          Widget gating                             #
    # ------------------------------------------------------------------ #
    def _set_dataset_widgets_enabled(self, on: bool):
        """Enable and show/hide dataset-dependent widgets and panels."""
        for name in _DATASET_PANELS:
            panel = getattr(self, name, None)
            if panel is not None:
                panel.setVisible(bool(on))
        for name in _DATASET_WIDGETS + _DATASET_ENABLED_WIDGETS:
            w = getattr(self, name, None)
            if w is not None:
                w.setEnabled(bool(on))
        # A fit-dependent control needs a fit, not just a dataset.
        for name in _FIT_ONLY_WIDGETS:
            widget = getattr(self, name, None)
            if widget is not None and getattr(self, "fit_result", None) is None:
                widget.setEnabled(False)
                widget.setToolTip("No fit result yet")
        for name, reason in _UNIMPLEMENTED.items():
            w = getattr(self, name, None)
            if w is not None:
                w.setEnabled(False)
                w.setToolTip(reason)
        self._update_noise_controls()
        self._on_model_changed()  # the scheme editor depends on the family *and* the data


    # ------------------------------------------------------------------ #
    #                        Noise weighting                             #
    # ------------------------------------------------------------------ #
    def dataset_noise(self):
        """Per-point noise (standard deviation) of the dataset, or ``None``.

        Delegates to :meth:`pymorgan.Dataset1D.noise_array`, which takes it
        from ``Zstdv`` when a sibling ``.pdatn`` file was loaded and otherwise
        from the spread over single scans. PyRATE-TA does not compute noise itself.
        """
        if self.dataset is None:
            return None
        try:
            return self.dataset.noise_array()
        except Exception:
            logger.debug("noise array unavailable", exc_info=True)
            return None

    def _update_noise_controls(self):
        """Gate the noise-weighting box on the dataset actually carrying noise.

        Without a noise array a weighted fit is not possible, so the box is
        disabled and unchecked, with the reason in its tooltip -- never silently
        ignored while still looking active.
        """
        chk = getattr(self, "UseNoiseWeightsCheckBox", None)
        if chk is None:
            return
        noise = self.dataset_noise()
        has_noise = noise is not None
        self._set_noise_lamp(noise)
        chk.setEnabled(has_noise)
        if not has_noise:
            chk.blockSignals(True)
            chk.setChecked(False)
            chk.blockSignals(False)
            chk.setToolTip(
                "No per-point noise available: load a dataset with a sibling "
                ".pdatn file, or one that carries single scans."
            )
        else:
            import pyrate_ta as pr

            # Restore the user's standing preference now that it is applicable.
            chk.blockSignals(True)
            chk.setChecked(bool(pr.get_settings().use_noise_weights))
            chk.blockSignals(False)
            chk.setToolTip(
                "Weight the residuals by the per-point noise, so the fit "
                "statistic is a true reduced chi-squared."
            )
        self._update_statistic_label()

    def _set_noise_lamp(self, noise):
        """Light the noise lamp when the dataset carries a per-point sigma.

        Its own indicator rather than a line of text: whether a weighted fit is
        possible at all is decided by the presence of a ``.pdatn`` sibling (or
        single scans), and that is worth seeing at a glance next to the dataset
        lamp instead of discovering it from a greyed-out checkbox.
        """
        lamp = getattr(self, "NoiseLoadedLamp", None)
        if lamp is None:
            return
        if self.dataset is None:
            lamp.set_state("off")
        else:
            lamp.set_state("loaded" if noise is not None else "error")

    def _on_noise_weights_toggled(self, on: bool):
        """Persist the weighting choice and relabel the statistic read-out."""
        import pyrate_ta as pr

        pr.update_settings(use_noise_weights=bool(on))
        self._update_statistic_label()

    def _update_statistic_label(self):
        """Label the read-out with the statistic that will actually be reported.

        Weighted fits report a reduced chi-squared; unweighted ones report the
        sum of squared residuals. The two are not comparable, so the label
        always says which one is shown.
        """
        lbl = getattr(self, "Chi2Value_label", None)
        if lbl is None:
            return
        chk = getattr(self, "UseNoiseWeightsCheckBox", None)
        weighted = bool(chk is not None and chk.isChecked() and chk.isEnabled())
        lbl.setText("red. chi2:" if weighted else "SSR:")
        lbl.setToolTip(
            "Reduced chi-squared: weighted residuals, divided by the degrees of freedom."
            if weighted
            else "Sum of squared residuals (unweighted; not comparable across datasets)."
        )

    # ------------------------------------------------------------------ #
    #                            Rendering                               #
    # ------------------------------------------------------------------ #
    #: Where each panel sits in the 3x3 grid of the single figure: the contour
    #: takes the top-left 2x2 block, so it is twice the size of the other two;
    #: the kinetics panel is the 2x1 column beside it and the spectra panel the
    #: 1x2 row below. The bottom-right cell is left free -- the plots-and-cuts
    #: panel sits over it.
    _PANELS = {
        "main": (slice(0, 2), slice(0, 2)),
        "kinetics": (slice(0, 2), 2),
        "spectra": (2, slice(0, 2)),
    }

    def _build_axes(self):
        """Clear the shared figure and lay out the three panels.

        One figure with three axes, not three figures: a single canvas means a
        single navigation toolbar, and the panels stay aligned. Everything is
        rebuilt on each render because a contour cannot be updated in place --
        the level set itself changes -- and because PyMORGAN's plotters add
        their own colourbars, which would otherwise accumulate.
        """
        import pyrate_ta as pr

        s = pr.get_settings()
        widget = self.PlotArea
        widget.figure.clear()
        # Explicit margins rather than tight_layout: this figure shares its
        # canvas with the controls panel, and the automatic layout leaves it
        # noticeably smaller than the window allows. All six are in
        # settings.toml ([plots]).
        grid = widget.figure.add_gridspec(
            3,
            3,
            left=float(s.panel_left),
            right=float(s.panel_right),
            top=float(s.panel_top),
            bottom=float(s.panel_bottom),
            hspace=float(s.panel_hspace),
            wspace=float(s.panel_wspace),
        )
        self._axes = {
            name: widget.figure.add_subplot(grid[row, col])
            for name, (row, col) in self._PANELS.items()
        }
        widget.ax = self._axes["main"]
        return widget

    def axis(self, name: str):
        """One panel of the shared figure, building the layout if needed."""
        if not getattr(self, "_axes", None):
            self._build_axes()
        return self._axes[name]

    def _fresh_axis(self, name: str):
        """Kept for callers that want ``(widget, axis)``; clears that panel only."""
        ax = self.axis(name)
        ax.clear()
        return self.PlotArea, ax

    @staticmethod
    def embedded_settings():
        """PyMORGAN's settings with the legend forced *inside* the axis.

        ``legend_location`` is a real choice for a stand-alone figure -- beside
        the axis is the usual PyMORGAN look -- but these panels share one small
        figure with two neighbours, where an outside legend is drawn over the
        panel next to it or clipped off the canvas entirely. So the setting is
        honoured everywhere except here, and only for the placement: everything
        else in the settings is passed through untouched.
        """
        from dataclasses import replace

        import pymorgan as pm

        settings = pm.get_settings()
        if str(getattr(settings, "legend_location", "")).strip().lower().startswith("outside"):
            return replace(settings, legend_location="best")
        return settings

    def _style_preview(self, w):
        """Recolour the embedded figure for the current (dark/light) palette."""
        from pymorgan.gui.theme import is_dark_palette, style_figure

        style_figure(w.figure, is_dark_palette())

    def request_render(self):
        """Coalesce rapid control changes into a single re-render."""
        self._render_timer.start()

    def render_all(self):
        """Redraw every panel from the current dataset (and fit, if there is one)."""
        if self.dataset is None:
            return
        pm.apply_style()
        # Drop the old crosshair while its lines are still attached: clearing
        # the figure first would detach them and leave nothing to remove.
        self._detach_crosshair()
        widget = self._build_axes()
        self._render_contour(rebuild=False)
        self._render_kinetics(rebuild=False)
        self._render_spectra(rebuild=False)
        self._attach_crosshair()
        self._tidy_shared_axes()
        self._style_preview(widget)
        widget.canvas.draw_idle()

    def _attach_crosshair(self):
        """Put the draggable crosshair back on the freshly drawn contour.

        The figure is rebuilt on every render, so the old lines are gone with
        it; the *position* is kept on the window and restored here, which is
        what makes dragging feel continuous across redraws.
        """
        from ..crosshair import Crosshair

        ax = self._axes.get("main")
        if ax is None or self.dataset is None:
            return
        self._detach_crosshair()

        probe, delay = self.selected_cut()
        self.crosshair = Crosshair(
            self.PlotArea.canvas,
            ax,
            on_change=self._on_crosshair_moved,
            on_release=self._on_crosshair_released,
            x=probe,
            y=delay,
            x_values=self.dataset.probe,
            y_values=getattr(self.dataset, "delays", getattr(self.dataset, "delay", None)),

        )


    def _detach_crosshair(self):
        """Disconnect the current crosshair, if any, and forget it.

        Idempotent, and safe whether or not the figure it drew on still exists;
        the *position* lives on ``self._cut``, not on the lines, so nothing is
        lost by dropping them.
        """
        old = getattr(self, "crosshair", None)
        if old is not None:
            old.disconnect()
        self.crosshair = None

    def _on_crosshair_moved(self, x, y):
        """Store the position while dragging; the panels follow on release.

        Redrawing two PyMORGAN panels on every motion event would make the drag
        crawl, so the traces are updated when the button is let go.
        """
        self._cut = (float(x), float(y))
        probe, delay = self.selected_cut()
        self.statusBar().showMessage(f"Cut at probe {probe:.6g}, delay {delay:.4g}")

    def _on_crosshair_released(self, x, y):
        """Redraw the two trace panels for the new cut."""
        self._cut = (float(x), float(y))
        if self.dataset is None:
            return
        for name in ("kinetics", "spectra"):
            self._axes[name].clear()
        self._render_kinetics(rebuild=False)
        self._render_spectra(rebuild=False)
        if getattr(self, "fit_result", None) is not None and self.current_view() != "data":
            pass  # the contour view is unaffected by a cut change
        self._tidy_shared_axes()
        self._style_preview(self.PlotArea)
        self.PlotArea.canvas.draw_idle()

    def _tidy_shared_axes(self):
        """Drop the tick labels a neighbouring panel already shows.

        With the kinetics panel drawn rotated, both shared axes line up:

        * the contour and the spectra panel below it share the **probe** axis,
          so the contour's copy of it goes;
        * the contour and the kinetics panel beside it share the **delay**
          axis, so the kinetics copy goes.

        The scales are matched explicitly rather than by ``sharey``, because
        each panel is drawn by PyMORGAN with its own scaling and a shared axis
        would have to be established before either was drawn.
        """
        main = self._axes.get("main")
        kinetics = self._axes.get("kinetics")
        spectra = self._axes.get("spectra")

        if main is not None and spectra is not None:
            main.set_xlabel("")
            main.tick_params(axis="x", labelbottom=False)
            if not getattr(spectra, "_has_spacer", False):
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                try:
                    # Append an invisible spacer axis of identical geometry to the colorbar
                    # so the Matplotlib layout engine aligns spectra with main dynamically during draw.
                    spacer = make_axes_locatable(spectra).append_axes("right", size=0.15, pad=0.1)
                    spacer.set_axis_off()
                    spectra._has_spacer = True
                except Exception:
                    pass
                pos_m = main.get_position()
                pos_s = spectra.get_position()
                spectra.set_position([pos_m.x0, pos_s.y0, pos_m.width, pos_s.height])

        if main is not None and kinetics is not None:
            try:
                kinetics.set_yscale(main.get_yscale(), **scale_kwargs(main))
                kinetics.set_ylim(main.get_ylim())
            except Exception:
                logger.debug("could not match the kinetics delay axis", exc_info=True)
            kinetics.set_ylabel("")
            kinetics.tick_params(axis="y", labelleft=False)
            pos_m = main.get_position()
            pos_k = kinetics.get_position()
            kinetics.set_position([pos_k.x0, pos_m.y0, pos_k.width, pos_m.height])


    def _render_contour(self, rebuild: bool = True):
        """Draw the delay-vs-probe contour, or the fit / residual view."""
        if self.dataset is None:
            return
        if rebuild:
            self.render_all()
            return

        ax = self.axis("main")
        pc = self.plot_controls
        kwargs = pc.contour_kwargs() if pc is not None else {}
        kwargs.setdefault("cbarLbl", "top")
        view = self.current_view() if hasattr(self, "current_view") else "data"
        fit = getattr(self, "fit_result", None)

        try:
            if view == "data" or fit is None:
                self.dataset.plot_contour(ax=ax, **kwargs)
            else:
                # The fit surface and the residuals go through the same
                # PyMORGAN plotter as the data (see pyrate_ta.plot.matrix), so the
                # three views are directly comparable.
                from ...plot.matrix import plot_matrix

                M = fit.C @ fit.S.T if view == "fit" else fit.R
                plot_matrix(
                    fit.t,
                    fit.probe if fit.probe is not None else self.dataset.probe,
                    M,
                    ax=ax,
                    dataset=self.dataset,
                    title="Fit" if view == "fit" else "Residuals",
                    **kwargs,
                )
        except Exception as exc:
            logger.debug("contour render failed", exc_info=True)
            QMessageBox.warning(self, "Plot failed", str(exc))
            return
        if pc is not None and view == "data":
            _safe_set_limits(ax, pc.xlim(), pc.ylim())

    def _render_kinetics(self, rebuild: bool = True):
        """Kinetic traces: the data as points, the fit as a line over them."""
        if self.dataset is None:
            return
        if rebuild:
            self.render_all()
            return
        ax = self.axis("kinetics")
        cuts = self._auto_wavelengths()
        data_style, fit_style = overlay_styles("kinetics")
        try:
            # Rotated, so the delay axis runs vertically like the contour's
            # beside it (PyMORGAN's own plotter, via swap_axes).
            self.dataset.plot_kinetics(
                cuts,
                ax=ax,
                fig=self.PlotArea.figure,
                swap_axes=True,
                settings=self.embedded_settings(),
                **data_style,
            )
            self._overlay_fit(ax, "kinetics", cuts, fit_style)
        except Exception:
            logger.debug("plot_kinetics preview failed", exc_info=True)

    def _render_spectra(self, rebuild: bool = True):
        """Transient spectra: data as points, fit as a line over them."""
        if self.dataset is None:
            return
        if rebuild:
            self.render_all()
            return
        ax = self.axis("spectra")
        delays = self._auto_delays()
        data_style, fit_style = overlay_styles("spectra")
        try:
            self.dataset.plot_spectra(
                delays,
                ax=ax,
                fig=self.PlotArea.figure,
                settings=self.embedded_settings(),
                **data_style,
            )
            self._overlay_fit(ax, "spectra", delays, fit_style)
        except Exception:
            logger.debug("plot_spectra preview failed", exc_info=True)

    def _overlay_fit(self, ax, kind: str, cuts, style: dict):
        """Draw the fitted surface over the data, at the same cuts.

        The fit is wrapped as a dataset (:mod:`pyrate_ta.plot.matrix`) and drawn by
        the same PyMORGAN plotter, so the cut positions, colours and units match
        the data underneath exactly.
        """
        fit = getattr(self, "fit_result", None)
        if fit is None:
            return
        from ...plot.matrix import as_dataset

        probe = fit.probe if fit.probe is not None else self.dataset.probe
        wrapped = as_dataset(fit.t, probe, fit.C @ fit.S.T, self.dataset)
        before = list(ax.lines)
        try:
            if kind == "kinetics":
                wrapped.plot_kinetics(
                    cuts,
                    ax=ax,
                    fig=self.PlotArea.figure,
                    swap_axes=True,
                    settings=self.embedded_settings(),
                    **style,
                )
            else:
                wrapped.plot_spectra(
                    cuts,
                    ax=ax,
                    fig=self.PlotArea.figure,
                    settings=self.embedded_settings(),
                    **style,
                )
        except Exception:
            logger.debug("fit overlay failed", exc_info=True)
            return

        # Combine the data and fit handles under the same label (no duplication)
        new_lines = [line for line in ax.lines if line not in before]
        for line in new_lines:
            line.set_label("_nolegend_")

        data_handles, data_labels = ax.get_legend_handles_labels()
        if len(data_handles) == len(new_lines):
            handles = [(data_handles[i], new_lines[i]) for i in range(len(data_handles))]
            labels = data_labels
        else:
            handles = data_handles
            labels = data_labels

        if ax.get_legend() is not None and handles:
            # Rebuilt, so its placement has to be asked for again -- from
            # PyMORGAN, so the setting that governs every other legend governs
            # this one too.
            from pymorgan.oneD.plot import legend_placement
            leg_kwargs = legend_placement(self.embedded_settings())

            leg = ax.legend(
                handles,
                labels,
                frameon=False,
                fontsize="small",
                **leg_kwargs,
            )
            leg.set_draggable(True)
            if "bbox_to_anchor" in leg_kwargs:
                try:
                    leg.set_in_layout(False)
                except Exception:
                    pass

    def _apply_view_limits(self):
        """Re-apply the panel X/Y limits to the contour without re-plotting."""
        if self.dataset is None or self.plot_controls is None or not getattr(self, "_axes", None):
            return
        _safe_set_limits(self.axis("main"), self.plot_controls.xlim(), self.plot_controls.ylim())
        self.PlotArea.canvas.draw_idle()

    # ------------------------------------------------------------------ #
    #                        Cut selection helpers                       #
    # ------------------------------------------------------------------ #
    def peak_position(self) -> tuple[float, float]:
        """Probe and delay of the largest signal -- where the crosshair starts.

        A useful default: the strongest feature is what one looks at first, and
        it makes the two trace panels show something meaningful before anything
        has been dragged.
        """
        probe = np.asarray(self.dataset.probe, dtype=float)
        delays = np.asarray(self.dataset.delays, dtype=float)
        Z = np.abs(np.asarray(self.dataset.Z, dtype=float))
        if Z.ndim == 3:
            Z = Z[:, :, 0]
        Z = np.nan_to_num(Z)
        row, col = np.unravel_index(int(np.argmax(Z)), Z.shape)
        return float(probe[col]), float(delays[row])

    def selected_cut(self) -> tuple[float, float]:
        """The crosshair position, snapped to measured values."""
        from ..crosshair import nearest

        if getattr(self, "_cut", None) is None:
            self._cut = self.peak_position()
        probe, delay = self._cut
        return (
            nearest(self.dataset.probe, probe),
            nearest(self.dataset.delays, delay),
        )

    def _auto_wavelengths(self, n: int = 1) -> list:
        """The probe position the crosshair points at (the kinetics cut)."""
        return [self.selected_cut()[0]]

    def _auto_delays(self) -> list:
        """The delay the crosshair points at (the spectra cut)."""
        return [self.selected_cut()[1]]

    def _norm(self) -> bool:
        chk = getattr(self, "PP_NormaliseCheckBox", None)
        return bool(chk.isChecked()) if chk is not None else False

    def _smooth_value(self) -> int:
        pc = getattr(self, "plot_controls", None)
        return pc.smooth() if pc is not None else 0

    # ------------------------------------------------------------------ #
    #                        Pop-out plot buttons                        #
    # ------------------------------------------------------------------ #
    def _ask_cuts(self, kind: str):
        """Which cuts to plot, pre-filled with the crosshair's.

        The crosshair picks *one* cut, which is what the embedded panels show;
        a pop-out figure is where several are compared, so it asks -- through
        PyMORGAN's shared prompt, so the accepted syntax (numbers, ``1:0.1:5``
        ranges, ``all``) is the same as in PyMORGAN's own cut plots.

        Returns ``None`` when the prompt was cancelled or nothing usable was
        typed, in which case the caller draws nothing.
        """
        from pymorgan.gui.dialogs import ask_values

        if kind == "kinetics":
            values = ask_values(
                self,
                "Kinetic cuts",
                "Probe positions (comma-separated, a range like 1900:5:1950, or 'all'):",
                self._auto_wavelengths(),
                all_values=self.dataset.probe,
            )
        else:
            values = ask_values(
                self,
                "Transient spectra",
                "Delays (comma-separated, a range like 1:0.5:5, or 'all'):",
                self._auto_delays(),
                all_values=self.dataset.delays,
            )
        return values or None

    def _popout(self, kind: str):
        """Open one of the standard PyMORGAN figures in its own window."""
        if self.dataset is None:
            return
        import matplotlib.pyplot as plt

        pm.apply_style()
        pc = self.plot_controls
        try:
            if kind == "contour":
                self.dataset.plot_contour(**(pc.contour_kwargs() if pc else {}))
            elif kind == "surface":
                self.dataset.plot_surface()
            elif kind == "kinetics":
                cuts = self._ask_cuts(kind)
                if not cuts:
                    return
                fit = getattr(self, "fit_result", None)
                incl_res_cb = getattr(self, "PP_IncludeResiduals", None)
                incl_res = bool(incl_res_cb.isChecked()) if incl_res_cb is not None else False

                if incl_res and fit is None:
                    QMessageBox.information(
                        self,
                        "No fit result",
                        "No fit result available yet to calculate residuals; plotting kinetic traces only.",
                    )

                from ...plot.traces import plot_kinetics_with_residuals

                plot_kinetics_with_residuals(
                    self.dataset,
                    cuts,
                    fit=fit,
                    incl_residuals=incl_res,
                    normY=self._norm(),
                    doSmooth=self._smooth_value(),
                )
            elif kind == "spectra":
                cuts = self._ask_cuts(kind)
                if not cuts:
                    return
                fit = getattr(self, "fit_result", None)
                incl_res_cb = getattr(self, "PP_IncludeResiduals", None)
                incl_res = bool(incl_res_cb.isChecked()) if incl_res_cb is not None else False

                if incl_res and fit is None:
                    QMessageBox.information(
                        self,
                        "No fit result",
                        "No fit result available yet to calculate residuals; plotting transient spectra only.",
                    )

                from ...plot.traces import plot_spectra_with_residuals

                plot_spectra_with_residuals(
                    self.dataset,
                    cuts,
                    fit=fit,
                    incl_residuals=incl_res,
                    normY=self._norm(),
                    doSmooth=self._smooth_value(),
                )
            elif kind in ("trio", "data_fit_residuals"):
                fit = getattr(self, "fit_result", None)
                if fit is None:
                    QMessageBox.information(
                        self,
                        "No fit result",
                        "No fit result available yet; fit a model before plotting Data+Fit+Residuals.",
                    )
                    return
                from ...plot.matrix import plot_data_fit_residuals

                kwargs = pc.contour_kwargs() if pc else {}
                plot_data_fit_residuals(fit, self.dataset, **kwargs)
            else:
                raise NotImplementedError(f"unknown plot kind: {kind}")


        except Exception as exc:
            logger.debug("pop-out plot %s failed", kind, exc_info=True)
            QMessageBox.warning(self, "Plot failed", str(exc))
            return
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plt.tight_layout()
        except Exception:
            pass
        plt.show(block=False)


