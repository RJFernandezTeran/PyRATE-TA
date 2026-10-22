"""Main window: the Qt Designer layout wired to the PyRATE/PyMORGAN pipeline.

The shell is loaded from ``main_window.ui``, with the plot area promoted to
PyMORGAN's
``WidgetPlot`` and the whole plot-controls panel (``PC_box``) reused verbatim so
:class:`pymorgan.gui.plot_controls.PlotControlsPanel` drives it unmodified.

This module keeps construction, wiring, menus and theming only; the behaviour
lives in the mixins in :mod:`pyrate_ta.gui.tabs`. Wiring is defensive: each
connection is made only if the named widget exists, so the window still loads
while the layout is being edited in Designer.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_API", "pyqt6")

from pathlib import Path

import numpy as np
import pymorgan as pm
from PyQt6 import uic
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QMainWindow, QMessageBox

import pyrate_ta as pr

from ..log import get_logger
from .mw_common import (
    _BUTTON_FONT_SIZES,
    _BUTTON_PALETTES,
    _RENDER_DEBOUNCE_MS,
    _TOOLTIPS,
    FIT_METHODS,
    button_stylesheet,
)
from .tabs import DataTabMixin, FitTabMixin, LDATabMixin, ModelTabMixin

logger = get_logger(__name__)

_UI_FILE = Path(__file__).with_name("main_window.ui")
_ICON_DIR = Path(__file__).with_name("icons")


class MainWindow(DataTabMixin, ModelTabMixin, FitTabMixin, LDATabMixin, QMainWindow):
    """PyRATE-TA main window."""

    def __init__(self):
        super().__init__()
        uic.loadUi(str(_UI_FILE), self)
        self._default_size = self.size()
        settings = pr.get_settings()
        if settings.restore_window_size:
            self._resize_within_screen(settings.window_width, settings.window_height)
        else:
            self._resize_within_screen(self._default_size.width(), self._default_size.height())
        self.setWindowTitle(f"PyRATE-TA v{pr.__version__}")
        icon = _ICON_DIR / "pirate_ship.png"
        if icon.exists():
            self.setWindowIcon(QIcon(str(icon)))

        pm.apply_style()

        # --- state -------------------------------------------------------- #
        self.dataset = None
        self.fit_result = None
        self.lda_result = None
        self.custom_scheme = None
        self.crosshair = None
        self._cut = None  # (probe, delay) chosen with the crosshair
        self._axes = {}
        self._current_path = None
        self._last_dir = str(settings.default_datadir) if settings.default_datadir else None
        self.plot_controls = None
        self._ss_abs = None
        self._ss_abs_spectrum = None
        self._ss_em = None
        self._ss_em_spectrum = None

        self._init_fit_methods()
        self._init_plot_controls()
        self._init_model_controls()
        self._init_lda_tab_from_settings()
        self._wire()
        self._build_menus()
        self._apply_button_aesthetics()
        self._on_reference_spectra_changed()

        self.DataLoadedLamp.set_state("off")
        self._set_dataset_widgets_enabled(False)
        self.statusBar().showMessage("No dataset loaded")
        QTimer.singleShot(1500, self._start_bg_update_check)

    def _start_bg_update_check(self):
        """Perform a quiet, non-blocking background check for updates."""
        try:
            from pymorgan.gui.updater import check_for_updates_background

            check_for_updates_background(
                self,
                github_url="https://github.com/RJFernandezTeran/PyRATE-TA",
                current_version=pr.__version__,
                app_name="PyRATE-TA",
            )
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    #                            Construction                            #
    # ------------------------------------------------------------------ #
    def _init_fit_methods(self):
        combo = getattr(self, "FitMethod", None)
        if combo is None:
            return
        combo.clear()
        for label, method in FIT_METHODS.items():
            combo.addItem(label, userData=method)

    def _init_plot_controls(self):
        """Bind PyMORGAN's plot-controls panel to the ``PC_*`` widgets."""
        from pymorgan.gui.plot_controls import PlotControlsPanel

        # A contour re-plot cannot be updated artist-by-artist (the level set
        # itself changes), so rapid control changes -- a slider drag emits one
        # signal per step -- are coalesced into a single render.
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(_RENDER_DEBOUNCE_MS)
        self._render_timer.timeout.connect(self._render_contour)

        if getattr(self, "PC_box", None) is None:
            logger.warning("PC_box missing from main_window.ui; plot controls disabled")
            return
        try:
            self.plot_controls = PlotControlsPanel(self, parent=self)
        except RuntimeError:
            logger.warning("Plot controls could not be bound", exc_info=True)
            self.plot_controls = None
            return
        self.plot_controls.renderRequested.connect(self.request_render)
        self.plot_controls.limitsChanged.connect(self._apply_view_limits)
        # The fit window follows the display window when "restrict" is on, so a
        # limit change is worth echoing in the status bar.
        self.plot_controls.limitsChanged.connect(self._on_limits_changed)
        if getattr(self.plot_controls, "restore_btn", None) is not None:
            self.plot_controls.restore_btn.clicked.connect(self._apply_default_time_limits_to_plot_controls)

    # ------------------------------------------------------------------ #
    #                             Aesthetics                             #
    # ------------------------------------------------------------------ #
    def _apply_button_aesthetics(self):
        """Colour every button by its function, following the GUI palette.

        The mapping lives in :mod:`pyrate_ta.gui.mw_common`; buttons that are not
        mapped get the neutral style, so a newly added button is styled rather
        than left with the platform default.
        """
        from pymorgan.gui.theme import is_dark_palette
        from PyQt6.QtWidgets import QPushButton

        dark = is_dark_palette()
        for btn in self.findChildren(QPushButton):
            palette = _BUTTON_PALETTES.get(btn.objectName(), "neutral")
            size = _BUTTON_FONT_SIZES.get(btn.objectName(), "11px")
            btn.setStyleSheet(button_stylesheet(palette, dark, size, checkable=btn.isCheckable()))
        for name, tip in _TOOLTIPS.items():
            w = getattr(self, name, None)
            if w is not None and not w.toolTip():
                w.setToolTip(tip)

    def _resize_within_screen(self, width, height, margin: int = 40):
        """Resize to at most what the screen actually offers, and centre.

        A stored geometry is not a promise that the screen still has room for
        it: a window opened on a laptop after being saved on a large monitor
        would otherwise reach under the dock and past the menu bar, with its
        lower controls unreachable. The size is therefore clipped to the
        *available* geometry (which already excludes the menu bar and dock),
        less a small margin for the window frame.
        """
        from PyQt6.QtWidgets import QApplication

        screen = self.screen() or QApplication.primaryScreen()
        width, height = int(width), int(height)
        if screen is not None:
            available = screen.availableGeometry()
            width = min(width, available.width() - margin)
            height = min(height, available.height() - margin)
        self.resize(width, height)

        if screen is not None:
            frame = self.frameGeometry()
            frame.moveCenter(screen.availableGeometry().center())
            self.move(frame.topLeft())
        return self.size()

    def _restore_default_size(self):
        """Return the window to the size declared in the .ui, if it fits."""
        self.showNormal()  # resizing a maximised window has no visible effect
        self._resize_within_screen(self._default_size.width(), self._default_size.height())

    # ------------------------------------------------------------------ #
    #                              Wiring                                #
    # ------------------------------------------------------------------ #
    def _connect(self, name: str, signal: str, slot):
        """Connect ``self.<name>.<signal>`` to ``slot`` if the widget exists."""
        w = getattr(self, name, None)
        if w is None:
            logger.debug("widget %s not found in main_window.ui; not wired", name)
            return
        getattr(w, signal).connect(slot)

    def _wire(self):
        self._connect("LoadPDATButton", "clicked", self.open_file)
        self._connect("ClearDataButton", "clicked", self.clear_data)

        self._connect("AddComponentButton", "clicked", self._add_component)
        self._connect("RemoveComponentButton", "clicked", self._remove_component)
        self._connect("GaussianIRFCheckBox", "toggled", self._on_irf_toggled)
        self._connect("UseNoiseWeightsCheckBox", "toggled", self._on_noise_weights_toggled)
        for name in ("ParallelButton", "SequentialButton", "TargetButton"):
            self._connect(name, "toggled", lambda checked: checked and self._on_model_changed())

        self._connect("FITDATAButton", "clicked", lambda: self.run_fit(preview=False))
        self._connect("PreviewButton", "clicked", lambda: self.run_fit(preview=True))
        self._connect("ResetFitButton", "clicked", self._reset_fit)
        self._connect("CopyTaus_button", "clicked", self._copy_lifetimes)
        self._connect("DefinemodelButton", "clicked", self.edit_scheme)
        self._connect("ShowSchemeButton", "clicked", self._popout_scheme)
        self._connect("PlotSpeciesSpectraButton", "clicked", self._popout_species_spectra)
        self._connect("PlotConcProfileButton", "clicked", self._popout_concentrations)
        self._connect("PlotSteadyStateCheckBox", "toggled", self._on_plot_ss_toggled)
        self._connect(
            "GSBRecoveryCheckBox",
            "toggled",
            lambda *_: self._render_species_spectra() if getattr(self, "fit_result", None) is not None else None,
        )
        from PyQt6.QtWidgets import QButtonGroup
        self._view_button_group = getattr(self, "PlotWhatButtonGroup", None)
        if self._view_button_group is None:
            self._view_button_group = QButtonGroup(self)
            for name in ("ShowDataButton", "ShowFitButton", "ShowResidualsButton"):
                btn = getattr(self, name, None)
                if btn is not None:
                    self._view_button_group.addButton(btn)
        self._view_button_group.setExclusive(True)

        for name in ("ShowDataButton", "ShowFitButton", "ShowResidualsButton"):
            self._connect(name, "toggled", lambda checked: checked and self._on_view_changed())
        self._connect("RestrictFitCheckBox", "toggled", lambda *_: self._on_limits_changed())
        self._connect("LDARestrictFitCheckBox", "toggled", lambda *_: self._on_limits_changed())
        self._connect("LDAAlphaCombo", "currentTextChanged", self._on_lda_alpha_method_changed)
        self._connect("RunLDAButton", "clicked", self.run_lda_from_gui)
        self._connect("PlotLDAMapButton", "clicked", self.popout_lda_map)
        self._connect("PlotLCurveButton", "clicked", self.popout_l_curve)
        self._connect("PlotLDASliceButton", "clicked", self.popout_lda_slice)
        self._connect("SaveLDAMapPDATButton", "clicked", self.save_lda_map_pdat)
        self._connect("LDACitationsButton", "clicked", self.show_lda_citations)
        self._connect(
            "LDADynamicalContentCheckBox",
            "toggled",
            lambda *_: getattr(self, "lda_result", None) is not None and self._render_lda_plots(self.lda_result),
        )
        self._connect(
            "LDAPeaksCheckBox",
            "toggled",
            lambda *_: getattr(self, "lda_result", None) is not None and self._render_lda_plots(self.lda_result),
        )

        self._connect("PP_ContourplotButton", "clicked", lambda: self._popout("contour"))
        self._connect("PP_SurfaceplotButton", "clicked", lambda: self._popout("surface"))
        self._connect("PP_PlotKineticsButton", "clicked", lambda: self._popout("kinetics"))
        self._connect("PP_PlotTrSpectraButton", "clicked", lambda: self._popout("spectra"))
        self._connect("PP_LoadSSabs_btn", "clicked", self.load_abs_spectrum)
        self._connect("PP_LoadSSem_btn", "clicked", self.load_em_spectrum)
        self._connect("PP_ShowSSbutton", "clicked", self._show_steady_state)
        self._connect("PP_ClearSSbutton", "clicked", self.clear_ref_spectra)
        self._connect("PP_IncludeSS_CheckBox", "toggled", self._on_include_ss_toggled)

    def _build_menus(self):
        self._connect("actionOpen", "triggered", self.open_file)
        self._connect("actionClear", "triggered", self.clear_data)
        self._connect("actionSaveFit", "triggered", lambda: self.save_fit_session(ask=True))
        self._connect("actionLoadFit", "triggered", self.load_fit_session)
        self._connect("actionLoadAbsSpectrum", "triggered", self.load_abs_spectrum)
        self._connect("actionLoadEmSpectrum", "triggered", self.load_em_spectrum)
        self._connect("actionClearRefSpectra", "triggered", self.clear_ref_spectra)
        self._connect("actionPlotSteadyState", "triggered", self._show_steady_state)
        self._connect("actionQuit", "triggered", self.close)
        self._connect("actionRestoreSize", "triggered", self._restore_default_size)
        self._connect("actionSettings", "triggered", self.open_settings_dialog)
        self._connect("actionAbout", "triggered", self._about)

        menu_file = getattr(self, "menuFile", None)
        if menu_file is not None:
            from PyQt6.QtGui import QAction

            menu_file.addSeparator()
            act_pm = QAction("Open in PyMORGAN (Spectral Processing & 2D-IR)...", self)
            act_pm.setToolTip("Launch PyMORGAN to process, chirp-correct, or stitch transient spectra")
            act_pm.triggered.connect(self._launch_pymorgan)
            menu_file.addAction(act_pm)
            self.actionOpenPyMORGAN = act_pm

        menu_help = getattr(self, "menuHelp", None)
        if menu_help is not None and not hasattr(self, "actionCheckUpdates"):
            from PyQt6.QtGui import QAction

            act_upd = QAction("Check for Updates (Release / Bleeding Edge)...", self)
            act_upd.setToolTip(
                "Check GitHub for new releases or sync bleeding-edge updates from the main branch"
            )
            act_upd.triggered.connect(self._check_for_updates)
            about_act = getattr(self, "actionAbout", None)
            if about_act is not None:
                menu_help.insertAction(about_act, act_upd)
                menu_help.insertSeparator(about_act)
            else:
                menu_help.addAction(act_upd)
            self.actionCheckUpdates = act_upd

    def _launch_pymorgan(self):
        """Launch PyMORGAN, optionally passing the current dataset file."""
        from pyrate_ta.io.pymorgan_bridge import launch_pymorgan

        cur_file = getattr(self, "_current_path", None)
        try:
            launch_pymorgan(dataset_path=cur_file)
            self.statusBar().showMessage("Launched PyMORGAN.", 5000)
        except Exception as exc:
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.critical(self, "Launch Error", f"Could not launch PyMORGAN:\n{exc}")

    def _get_dialog_dir(self, start_dir: str | Path | None = None) -> str:
        """Resolve the directory for file dialogs: start_dir -> _last_dir -> default_datadir -> home.

        Resolution order:
        1. Explicitly passed ``start_dir`` if valid.
        2. ``self._last_dir`` (the last opened/saved directory or default_datadir at launch) if valid.
        3. PyRATE's ``default_datadir`` setting (from settings.toml) if set and valid.
        4. PyMORGAN's ``default_datadir`` setting as fallback.
        5. User's home directory.
        """
        if start_dir:
            try:
                p = Path(start_dir).expanduser()
                if p.is_dir():
                    return str(p)
                if p.parent.is_dir():
                    return str(p.parent)
            except Exception:
                pass

        if self._last_dir:
            try:
                p = Path(self._last_dir).expanduser()
                if p.is_dir():
                    return str(p)
                if p.parent.is_dir():
                    return str(p.parent)
            except Exception:
                pass

        datadir = getattr(pr.get_settings(), "default_datadir", None)
        if datadir:
            try:
                p = Path(datadir).expanduser()
                if p.is_dir():
                    return str(p)
                if p.parent.is_dir():
                    return str(p.parent)
                return str(p)
            except Exception:
                return str(datadir)

        try:
            pm_datadir = getattr(pm.get_settings(), "default_datadir", None)
            if pm_datadir:
                p = Path(pm_datadir).expanduser()
                if p.is_dir():
                    return str(p)
                if p.parent.is_dir():
                    return str(p.parent)
                return str(p)
        except Exception:
            pass

        return str(Path.home())

    def load_abs_spectrum(self):
        """Load a text-based steady-state absorption spectrum for GSB recovery.

        The spectrum is robustly parsed and interpolated onto the current
        dataset's probe axis, stored as ``dataset.gs_spectrum``.
        """
        from PyQt6.QtWidgets import QFileDialog, QMessageBox

        if self.dataset is None:
            QMessageBox.warning(
                self,
                "No dataset loaded",
                "Load a dataset first so the absorption spectrum can be mapped onto its probe axis.",
            )
            return

        start = self._get_dialog_dir()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Steady-State Absorption Spectrum",
            start,
            "Text / Data files (*.txt *.dat *.csv *.tsv *.asc);;All files (*)",
        )
        if not path:
            return
        self._last_dir = str(Path(path).parent)

        try:
            from ..io.reference import interpolate_reference_spectrum, load_reference_spectrum

            x_raw, y_raw = load_reference_spectrum(path)
            probe_ds = np.asarray(self.dataset.probe, dtype=float)
            gs_on_grid = interpolate_reference_spectrum(x_raw, y_raw, probe_ds)

            self.dataset.gs_spectrum = gs_on_grid
            self.dataset.gs_raw = (x_raw, y_raw)
            self.dataset.gs_filename = Path(path).name

            import pymorgan as pm

            spec = pm.steadyState.Spectrum(
                x_raw, y_raw, kind="absorption", label=Path(path).stem, source=str(path)
            )
            self._ss_abs_spectrum = spec
            self._ss_abs = spec.as_overlay_dict()
            self._set_ss_checkboxes(True)

            self.statusBar().showMessage(
                f"Absorption spectrum loaded: {Path(path).name} "
                f"({len(x_raw)} raw points, mapped to {len(probe_ds)} probe channels)",
                8000,
            )
            self._on_reference_spectra_changed()

        except Exception as exc:
            logger.debug("loading absorption spectrum failed", exc_info=True)
            led = getattr(self, "PP_led_SSabs", None)
            if led is not None:
                led.set_state("error")
            QMessageBox.warning(self, "Load absorption spectrum failed", str(exc))

    def load_em_spectrum(self):
        """Load a text-based steady-state emission spectrum for stimulated emission recovery.

        The spectrum is robustly parsed and interpolated onto the current
        dataset's probe axis, stored as ``dataset.em_spectrum``.
        """
        from PyQt6.QtWidgets import QFileDialog, QMessageBox

        if self.dataset is None:
            QMessageBox.warning(
                self,
                "No dataset loaded",
                "Load a dataset first so the emission spectrum can be mapped onto its probe axis.",
            )
            return

        start = self._get_dialog_dir()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Steady-State Emission Spectrum",
            start,
            "Text / Data files (*.txt *.dat *.csv *.tsv *.asc);;All files (*)",
        )
        if not path:
            return
        self._last_dir = str(Path(path).parent)

        try:
            from ..io.reference import interpolate_reference_spectrum, load_reference_spectrum

            x_raw, y_raw = load_reference_spectrum(path)
            probe_ds = np.asarray(self.dataset.probe, dtype=float)
            em_on_grid = interpolate_reference_spectrum(x_raw, y_raw, probe_ds)

            self.dataset.em_spectrum = em_on_grid
            self.dataset.em_raw = (x_raw, y_raw)
            self.dataset.em_filename = Path(path).name

            import pymorgan as pm

            spec = pm.steadyState.Spectrum(
                x_raw, y_raw, kind="emission", label=Path(path).stem, source=str(path)
            )
            self._ss_em_spectrum = spec
            self._ss_em = spec.as_overlay_dict()
            self._set_ss_checkboxes(True)

            self.statusBar().showMessage(
                f"Emission spectrum loaded: {Path(path).name} "
                f"({len(x_raw)} raw points, mapped to {len(probe_ds)} probe channels)",
                8000,
            )
            self._on_reference_spectra_changed()

        except Exception as exc:
            logger.debug("loading emission spectrum failed", exc_info=True)
            led = getattr(self, "PP_led_SSem", None)
            if led is not None:
                led.set_state("error")
            QMessageBox.warning(self, "Load emission spectrum failed", str(exc))

    def clear_ref_spectra(self):
        """Clear loaded steady-state absorption and emission reference spectra."""
        self._ss_abs = None
        self._ss_abs_spectrum = None
        self._ss_em = None
        self._ss_em_spectrum = None
        if self.dataset is not None:
            self.dataset.gs_spectrum = None
            self.dataset.gs_raw = None
            self.dataset.gs_filename = None
            self.dataset.em_spectrum = None
            self.dataset.em_raw = None
            self.dataset.em_filename = None
        self.statusBar().showMessage("Reference absorption and emission spectra cleared", 5000)
        self._set_ss_checkboxes(False)
        self._on_reference_spectra_changed()

    @property
    def ss_abs(self) -> dict | None:
        """Raw steady-state absorption overlay dict {'X': x, 'Y': y} or None."""
        if getattr(self, "_ss_abs", None) is not None:
            return self._dense_overlay_dict(self._ss_abs["X"], self._ss_abs["Y"])
        if self.dataset is not None and getattr(self.dataset, "gs_raw", None) is not None:
            x, y = self.dataset.gs_raw
            return self._dense_overlay_dict(x, y)
        if self.dataset is not None and getattr(self.dataset, "gs_spectrum", None) is not None:
            return self._dense_overlay_dict(self.dataset.probe, self.dataset.gs_spectrum)
        return None

    def _set_ss_checkboxes(self, checked: bool) -> None:
        """Synchronize the steady-state toggle checkboxes without re-triggering signals."""
        for chk_name in ("PlotSteadyStateCheckBox", "PP_IncludeSS_CheckBox"):
            chk = getattr(self, chk_name, None)
            if chk is not None:
                chk.blockSignals(True)
                chk.setChecked(bool(checked))
                chk.blockSignals(False)

    @staticmethod
    def _dense_overlay_dict(x, y, min_points: int = 200) -> dict:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if x_arr.size >= 2 and x_arr.size < min_points:
            x_dense = np.linspace(float(np.nanmin(x_arr)), float(np.nanmax(x_arr)), min_points)
            order = np.argsort(x_arr)
            y_dense = np.interp(x_dense, x_arr[order], y_arr[order])
            return {"X": x_dense, "Y": y_dense}
        return {"X": x_arr, "Y": y_arr}

    @property
    def ss_em(self) -> dict | None:
        """Raw steady-state emission overlay dict {'X': x, 'Y': y} or None."""
        if getattr(self, "_ss_em", None) is not None:
            return self._dense_overlay_dict(self._ss_em["X"], self._ss_em["Y"])
        if self.dataset is not None and getattr(self.dataset, "em_raw", None) is not None:
            x, y = self.dataset.em_raw
            return self._dense_overlay_dict(x, y)
        if self.dataset is not None and getattr(self.dataset, "em_spectrum", None) is not None:
            return self._dense_overlay_dict(self.dataset.probe, self.dataset.em_spectrum)
        return None

    @property
    def ss_abs_spectrum(self):
        """Steady-state absorption Spectrum object or None."""
        if getattr(self, "_ss_abs_spectrum", None) is not None:
            return self._ss_abs_spectrum
        if self.dataset is not None and getattr(self.dataset, "gs_raw", None) is not None:
            import pymorgan as pm

            x, y = self.dataset.gs_raw
            return pm.steadyState.Spectrum(
                x, y, kind="absorption", label=getattr(self.dataset, "gs_filename", None)
            )
        return None

    @property
    def ss_em_spectrum(self):
        """Steady-state emission Spectrum object or None."""
        if getattr(self, "_ss_em_spectrum", None) is not None:
            return self._ss_em_spectrum
        if self.dataset is not None and getattr(self.dataset, "em_raw", None) is not None:
            import pymorgan as pm

            x, y = self.dataset.em_raw
            return pm.steadyState.Spectrum(
                x, y, kind="emission", label=getattr(self.dataset, "em_filename", None)
            )
        return None

    @property
    def include_ss_spectra(self) -> bool:
        """Whether steady-state spectra should be included in spectral cuts and species plots."""
        chk_fit = getattr(self, "PlotSteadyStateCheckBox", None)
        if chk_fit is not None and chk_fit.isChecked():
            return True
        chk = getattr(self, "PP_IncludeSS_CheckBox", None)
        return bool(chk.isChecked()) if chk is not None else False

    @property
    def plot_ss_spectra(self) -> bool:
        """Whether steady-state spectra should be overlaid on species spectra (EAS/DAS/SAS)."""
        chk_fit = getattr(self, "PlotSteadyStateCheckBox", None)
        if chk_fit is not None:
            return bool(chk_fit.isChecked())
        chk = getattr(self, "PP_IncludeSS_CheckBox", None)
        return bool(chk.isChecked()) if chk is not None else False

    def _on_plot_ss_toggled(self, checked: bool = False):
        """Called when 'Plot steady-state spectra' checkbox is toggled."""
        self._set_ss_checkboxes(checked)
        if self.dataset is None:
            return
        tab_idx = (
            self.FittingTabWidget.currentIndex()
            if hasattr(self, "FittingTabWidget") and self.FittingTabWidget is not None
            else 0
        )
        fit = getattr(self, "fit_result", None)
        if tab_idx == 0 and fit is not None and hasattr(self, "_render_species_spectra"):
            self._render_species_spectra()
        elif hasattr(self, "render_all"):
            self.render_all()

    def _on_reference_spectra_changed(self):
        """Callback when absorption or emission reference spectra are loaded or cleared."""
        has_ds = getattr(self, "dataset", None) is not None
        has_abs = has_ds and self.ss_abs_spectrum is not None
        has_em = has_ds and self.ss_em_spectrum is not None
        has_ss = bool(has_abs or has_em)
        led_abs = getattr(self, "PP_led_SSabs", None)
        if led_abs is not None:
            led_abs.set_state("loaded" if has_abs else "off")
        led_em = getattr(self, "PP_led_SSem", None)
        if led_em is not None:
            led_em.set_state("loaded" if has_em else "off")
        act = getattr(self, "actionPlotSteadyState", None)
        if act is not None:
            act.setEnabled(has_ss)
        for btn_name in ("PP_ShowSSbutton", "PP_ClearSSbutton"):
            btn = getattr(self, btn_name, None)
            if btn is not None:
                btn.setEnabled(has_ss)

        gsb_chk = getattr(self, "GSBRecoveryCheckBox", None)
        if gsb_chk is not None and gsb_chk.isChecked():
            if hasattr(self, "_render_species_spectra"):
                self._render_species_spectra()
        elif self.dataset is not None:
            tab_idx = (
                self.FittingTabWidget.currentIndex()
                if hasattr(self, "FittingTabWidget") and self.FittingTabWidget is not None
                else 0
            )
            fit = getattr(self, "fit_result", None)
            if tab_idx == 0 and fit is not None and hasattr(self, "_render_species_spectra"):
                self._render_species_spectra()
            elif hasattr(self, "render_all"):
                self.render_all()

    def _on_include_ss_toggled(self, checked: bool = False):
        """Called when 'Include SS-spectra' checkbox is toggled."""
        self._set_ss_checkboxes(checked)
        if self.dataset is None:
            return
        tab_idx = (
            self.FittingTabWidget.currentIndex()
            if hasattr(self, "FittingTabWidget") and self.FittingTabWidget is not None
            else 0
        )
        fit = getattr(self, "fit_result", None)
        if tab_idx == 0 and fit is not None and hasattr(self, "_render_species_spectra"):
            self._render_species_spectra()
        elif hasattr(self, "render_all"):
            self.render_all()

    def _show_steady_state(self):
        """Plot the loaded steady-state absorption and/or emission spectra in a new figure."""
        has_abs = self.ss_abs_spectrum is not None
        has_em = self.ss_em_spectrum is not None

        if not has_abs and not has_em:
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.information(
                self,
                "No Steady-State Data",
                "No steady-state spectra have been loaded yet.\nPlease load an absorption or emission spectrum first.",
            )
            return

        import matplotlib.pyplot as plt
        import pymorgan as pm
        from pymorgan.settings import SteadyStatePlotMode
        from pymorgan.steadyState import plot_steady_state

        pm.apply_style()
        s = pm.get_settings()
        normalise = getattr(self, "_norm", lambda: False)()
        fig, ax = plt.subplots(figsize=s.spectra_figsize)

        plot_steady_state(
            abs_spectrum=self.ss_abs_spectrum,
            em_spectrum=self.ss_em_spectrum,
            ax=ax,
            settings=s,
            normalise=normalise,
        )

        title = "Steady-State Spectra"
        if has_abs and has_em:
            title = "Steady-State Absorption & Emission"
        elif has_abs:
            title = "Steady-State Absorption"
        elif has_em:
            title = "Steady-State Emission"
        if normalise or (has_abs and has_em and s.ss_plot_mode == SteadyStatePlotMode.NORMALISED):
            title += " (Normalised)"
        ax.set_title(title)
        fig.tight_layout()
        plt.show(block=False)

    def open_settings_dialog(self):
        """Edit the settings live, in one dialog with two halves.

        PyRATE-TA owns the analysis settings and PyMORGAN the presentation ones,
        so both panels are shown here rather than either package

        growing a copy of the other's. Edits apply immediately and the plots
        re-render; *Save permanently* writes each half back to its own
        ``settings.toml``, so closing without saving keeps the change for this
        session only.
        """
        from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QTabWidget, QVBoxLayout

        from .settings_panel import SettingsPanel

        dialog = QDialog(self)
        dialog.setWindowTitle("PyRATE-TA settings")
        layout = QVBoxLayout(dialog)

        tabs = QTabWidget(dialog)
        analysis = SettingsPanel(dialog)
        analysis.changed.connect(self._on_settings_changed)
        tabs.addTab(analysis, "Analysis (PyRATE-TA)")

        aesthetics = None
        try:
            from pymorgan.gui.settings_panel import SettingsPanel as MorganPanel

            aesthetics = MorganPanel(dialog)
            aesthetics.changed.connect(self._on_settings_changed)
            tabs.addTab(aesthetics, "Aesthetics (PyMORGAN)")
        except Exception:
            # Worth saying, not worth blocking: the analysis half still works.
            logger.debug("PyMORGAN's settings panel is unavailable", exc_info=True)
        layout.addWidget(tabs)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Close
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("Save permanently")
        buttons.accepted.connect(lambda: self._save_settings(dialog, bool(aesthetics)))
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        dialog.resize(max(620, dialog.sizeHint().width()), 640)
        dialog.exec()

    def _save_settings(self, parent, with_aesthetics: bool):
        """Write both halves to their own files, reporting each outcome."""
        from .settings_app import pymorgan_settings_path, settings_path

        written = []
        try:
            target = settings_path()
            pr.save_settings(target)
            written.append(f"PyRATE-TA -> {target}")
        except Exception as exc:
            QMessageBox.warning(parent, "Save failed", f"PyRATE-TA settings: {exc}")
        if with_aesthetics:
            try:
                target = pymorgan_settings_path() or pm.settings_path()
                pm.save_settings(target)
                written.append(f"PyMORGAN -> {target}")
            except Exception as exc:
                QMessageBox.warning(parent, "Save failed", f"PyMORGAN settings: {exc}")
        if written:
            self.statusBar().showMessage("Settings saved: " + "; ".join(written), 6000)

    def _on_settings_changed(self):
        """Re-render with the new settings and sync directory defaults."""
        settings = pr.get_settings()
        if settings.default_datadir:
            self._last_dir = str(settings.default_datadir)
        if self.dataset is not None:
            self.render_all()

    def _check_for_updates(self):
        """Interactive check for new GitHub releases or bleeding-edge main branch updates."""
        from pymorgan.gui.updater import check_for_updates_interactive

        check_for_updates_interactive(
            self,
            github_url="https://github.com/RJFernandezTeran/PyRATE-TA",
            current_version=pr.__version__,
            app_name="PyRATE-TA",
            status_callback=lambda msg: (
                self.statusBar().showMessage(msg, 4000) if msg else self.statusBar().clearMessage()
            ),
        )

    def _about(self):
        from .about_dialog import ModernAboutDialog

        icon = str(_ICON_DIR / "pirate_ship.png")
        manual_pdf = Path(__file__).resolve().parents[3] / "docs" / "main.pdf"
        dlg = ModernAboutDialog(
            parent=self,
            title="About PyRATE-TA",
            app_name="PyRATE-TA",
            version=pr.__version__,
            subtitle="Rate Analysis & Target-model Engine for Transient Absorption",
            description=(
                f"Interactive kinetic analysis, global and target fitting with rate-matrix models, "
                f"lifetime density analysis (LDA), and species-associated spectra (DAS / EAS / SAS).<br><br>"
                f"Loading, data processing and plotting powered by <b>PyMORGAN</b> v{pm.__version__}."
            ),
            author=pr.__author__,
            department="Department of Physical Chemistry",
            institution="University of Geneva, Switzerland",
            contact_email="Ricardo.FernandezTeran@unige.ch",
            website_url="https://www.unige.ch/sciences/chifi/fernandez-teran/",
            github_url="https://github.com/RJFernandezTeran/PyRATE-TA",
            license_name="GNU AGPLv3 License",
            banner_path=icon if os.path.exists(icon) else None,
            icon_path=icon if os.path.exists(icon) else None,
            manual_pdf_path=str(manual_pdf) if manual_pdf.exists() else "docs/main.pdf",
            ai_credit="Developed with AI assistance from <b>Google Antigravity</b>.",
        )
        dlg.exec()

