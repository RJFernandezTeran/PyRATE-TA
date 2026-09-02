"""Shared constants and small helpers for the main-window modules.

Deliberately thin: everything PyMORGAN already defines (the data-type display
names, the combo helpers, the log-safe limit setter) is imported from
:mod:`pymorgan.gui.mw_common` and re-exported here, so PyRATE-TA has one import
site without owning a second copy. Only the PyRATE-TA-specific pieces -- which
widget groups are gated on a loaded dataset, the render debounce, the button
palette -- live here.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_API", "pyqt6")

# Re-exported from PyMORGAN: never reimplemented here.
from pymorgan.gui.mw_common import (  # noqa: F401
    DATA_TYPE_DISPLAY_NAMES,
    _safe_set_limits,
    get_combo_datatype,
    populate_datatype_combo,
    set_combo_datatype,
)

# Coalescing window for embedded-contour re-renders (slider drags, spin boxes).
_RENDER_DEBOUNCE_MS = 60

# Panels on the main window hidden until data is loaded.
_DATASET_PANELS = [
    "ModelSelectionGroup",
    "ComponentsGroup",
    "IRFandtimezeroPanel",
    "FitControlGroup",
    "FitOptionsGroup",
    "PlotArea",
    "PP_PlotsandcutsPanel",
    "PC_box",
]

# Shown only while a dataset is loaded.
_DATASET_WIDGETS = [
    "PlotArea",
    "PC_box",
    "PP_PlotsandcutsPanel",
    "FitControlGroup",
]


# Enabled only while a dataset is loaded (they stay visible, greyed out).
_DATASET_ENABLED_WIDGETS = [
    "ClearDataButton",
    "ModelSelectionGroup",
    "ComponentsGroup",
    "RateFitTable",
    "AddComponentButton",
    "RemoveComponentButton",
    "GaussianIRFCheckBox",
    "CoherentArtifactCheckBox",
    "IRFandtimezeroPanel",
    "FitOptionsGroup",
    "NoiseLoadedLamp",
    "PreviewButton",
    "FITDATAButton",
    "RestrictFitCheckBox",
    "ExtraPlotsAfterFit_CheckBox",
    "ShowSchemeButton",
    "LDAGroup",
    "RunLDAButton",
    "LDACitationsButton",
    "LDADynamicalContentCheckBox",
    "LDARestrictFitCheckBox",
]


# Enabled only once there is a fit result: they show something a fit produced.
# One list, used by the gating, the post-fit enabling and the reset, so the
# three cannot drift apart.
_FIT_ONLY_WIDGETS = [
    "ShowFitButton",
    "ShowResidualsButton",
    "CopyTaus_button",
    "ResetFitButton",
    "PlotSpeciesSpectraButton",
    "PlotConcProfileButton",
]

# Buttons whose feature is not implemented yet: disabled, with the reason in the
# tooltip rather than silently doing nothing.

_UNIMPLEMENTED = {
    "ODEdefinedButton": "ODE-defined models - not implemented yet",
    "AddTracestoLivePlotsButton": "Live trace overlay - not implemented yet",
    "ExplorerModeButton": "Explorer mode - not implemented yet",
}

# Descriptive tooltips for the buttons that do work.
_TOOLTIPS = {
    "FITDATAButton": "Fit the model to the data (restricted to the display window if ticked)",
    "PreviewButton": "Evaluate the model at the parameters as typed, without optimising",
    "CoherentArtifactCheckBox": (
        "Fit the coherent artefact with the IRF and its first two derivatives"
    ),
    "ExtraPlotsAfterFit_CheckBox": (
        "Automatically open the 3-column Data+Fit+Residuals contour plot after a fit"
    ),
    "ResetFitButton": "Discard the fit and go back to the data view",

    "CopyTaus_button": "Copy the fitted lifetimes and uncertainties to the clipboard",
    "DefinemodelButton": "Write the kinetic scheme (K-matrix editor) and see its graph",
    "ShowSchemeButton": "Draw the kinetic scheme of the current model, with the rate symbols",
    "PlotSpeciesSpectraButton": "Open the fitted species spectra in their own window",
    "PlotConcProfileButton": "Open the concentration profiles C(t) of the fitted model",
    "LoadPDATButton": "Load a 1-D transient dataset (PyMORGAN loader registry)",
    "ClearDataButton": "Unload the dataset and clear every plot",
    "AddComponentButton": "Add a kinetic component to the model",
    "RemoveComponentButton": "Remove the last kinetic component",
    "ParallelButton": "Independent parallel decays -> DAS",
    "SequentialButton": "Sequential A->B->C scheme -> EAS",
    "TargetButton": "General K-matrix with branching -> SAS",
    "PP_ContourplotButton": "Open the delay-vs-probe contour in its own window",
    "PP_PlotKineticsButton": "Plot kinetic traces at selected probe positions",
    "PP_SurfaceplotButton": "Open a 3D surface plot of the dataset",
    "PP_PlotTrSpectraButton": "Plot transient spectra at selected delays",
    "PP_IncludeResiduals": "Include a linked residuals panel below kinetic trace plots",
    "PP_InteractivemodeSwitch": (
        "Pick kinetic / spectral cut positions interactively by clicking on the 2D contour map "
        "(left-click adds, right-click removes, Enter plots, Esc cancels)"
    ),
    "PP_SavetracesSwitch": "Save extracted cut traces to disk as text/data files",
    "PC_restore": "Restore the default axis limits and colour scale",
    "RunLDAButton": "Solve Lifetime Density Analysis across the grid",
    "PlotLDAMapButton": "Open 2D contour map of lifetime distribution S(tau, lambda)",
    "PlotLCurveButton": "Open L-Curve plot with optimal corner alpha selection",
    "PlotLDASliceButton": "Plot 1D slice of lifetime distribution S(tau) at selected probe wavelength",
    "SaveLDAMapPDATButton": "Save fitted 2D lifetime density map S(tau, lambda) as a PyMORGAN PDAT dataset file",
    "LDANtSpinBox": "Number of logarithmically spaced lifetime grid points (default 100)",
    "LDAPenaltyCombo": "Regularisation penalty method: 2nd derivative D2 (smoothness), 1st derivative D1, or Ridge (L=I)",
    "LDAAlphaCombo": "Selection method for regularisation parameter alpha (L-Curve Corner or GCV Minimum)",
    "LDATauMinDoubleSpinBox": "Minimum lifetime in grid (in dataset time units, 0 for auto: min(t)/2)",
    "LDATauMaxDoubleSpinBox": "Maximum lifetime in grid (in dataset time units, 0 for auto: 5*max(t))",
    "LDACoherentArtifactCheckBox": "Include IRF Gaussian and its first two derivatives at t0 to absorb cross-phase modulation / solvent artefact",
    "LDASVDFilterSpinBox": "Pre-filter raw dataset using top K SVD singular components to suppress probe noise (0 = Off)",
    "LDANonNegativeCheckBox": "Enforce non-negative amplitudes S >= 0 via regularised Non-Negative Least Squares (NNLS)",
    "LDABootstrapSpinBox": "Number of Monte Carlo bootstrap iterations for confidence intervals on integrated dynamics A(tau) (0 = Off)",
    "LDAPeaksCheckBox": "Automatically locate and annotate major lifetime peak centroids",
    "LDADynamicalContentCheckBox": (
        "Plot dynamical content D(tau) = sqrt(int S^2 dlambda) instead of absolute "
        "integral A(tau) = int |S| dlambda in the 2D map side panel"
    ),
    "LDARestrictFitCheckBox": "Fit only the delay/probe window currently displayed",
    "LDACitationsButton": "Show literature citations for the selected LDA algorithms",
    "GSBRecoveryCheckBox": "Overlay ground-state bleach recovery spectrum (-1 * GS) on species spectra",
}

# Non-linear least-squares algorithms offered in the FitMethod combo, in the
# order they appear -- the first is the default. The strings are
# ``scipy.optimize.least_squares`` method names.
#
# Trust Region Reflective first because it is the only one of the three that is
# both bounded and robust for this problem: lifetimes are bounded below by zero,
# and MINPACK's Levenberg-Marquardt (scipy's "lm") cannot honour bounds at all,
# so it is refused rather than silently ignoring them.
FIT_METHODS: dict[str, str] = {
    "Trust Region Reflective": "trf",
    "Dogbox": "dogbox",
    "Levenberg-Marquardt (unbounded only)": "lm",
}

# --------------------------------------------------------------------------- #
#                             Button aesthetics                               #
# --------------------------------------------------------------------------- #
# Greyed-out look for disabled buttons, matching PyMORGAN's.
_DISABLED_LIGHT_BG, _DISABLED_LIGHT_FG, _DISABLED_LIGHT_BORDER = "#ececec", "#9a9a9a", "#d4d4d4"
_DISABLED_DARK_BG, _DISABLED_DARK_FG, _DISABLED_DARK_BORDER = "#3a3a3a", "#7a7a7a", "#4a4a4a"

# Named palettes, as (light bg, light text, light border, light hover,
# dark bg, dark text, dark border, dark hover). The plot-button colours are the
# same ones PyMORGAN uses for the equivalent action, so a contour button looks
# like a contour button in both applications.
_PALETTES: dict[str, tuple[str, ...]] = {
    "contour": (
        "#d6eaf8",
        "#1b4f72",
        "#aed6f1",
        "#aed6f1",
        "#1e293b",
        "#38bdf8",
        "#0284c7",
        "#0369a1",
    ),
    "kinetics": (
        "#d5f5e3",
        "#196f3d",
        "#abebc6",
        "#abebc6",
        "#064e3b",
        "#6ee7b7",
        "#059669",
        "#047857",
    ),
    "surface": (
        "#fadbd8",
        "#78281f",
        "#f5b7b1",
        "#f5b7b1",
        "#450a0a",
        "#fca5a5",
        "#dc2626",
        "#b91c1c",
    ),
    "spectra": (
        "#ebf5fb",
        "#21618c",
        "#aed6f1",
        "#aed6f1",
        "#172554",
        "#93c5fd",
        "#3b82f6",
        "#1d4ed8",
    ),
    "load": (
        "#e6f4f1",
        "#115e59",
        "#99f6e4",
        "#ccfbf1",
        "#134e4a",
        "#5eead4",
        "#14b8a6",
        "#0f766e",
    ),
    "clear": (
        "#fadbd8",
        "#78281f",
        "#f5b7b1",
        "#f5b7b1",
        "#450a0a",
        "#fca5a5",
        "#dc2626",
        "#b91c1c",
    ),
    # One colour per model family, so the selected scheme is recognisable at a
    # glance: violet for parallel (DAS), teal for sequential (EAS), amber for
    # target (SAS).
    "model_parallel": (
        "#ede9fe",
        "#5b21b6",
        "#ddd6fe",
        "#ddd6fe",
        "#2e1065",
        "#c4b5fd",
        "#7c3aed",
        "#6d28d9",
    ),
    "model_sequential": (
        "#ccfbf1",
        "#0f766e",
        "#99f6e4",
        "#99f6e4",
        "#134e4a",
        "#5eead4",
        "#14b8a6",
        "#0f766e",
    ),
    "model_target": (
        "#fef3c7",
        "#92400e",
        "#fde68a",
        "#fde68a",
        "#451a03",
        "#fcd34d",
        "#d97706",
        "#b45309",
    ),
    "model_other": (
        "#f1f5f9",
        "#334155",
        "#cbd5e1",
        "#e2e8f0",
        "#334155",
        "#e2e8f0",
        "#64748b",
        "#475569",
    ),
    "add": ("#d1fae5", "#065f46", "#a7f3d0", "#a7f3d0", "#064e3b", "#34d399", "#10b981", "#059669"),
    "remove": (
        "#ffe4e6",
        "#9f1239",
        "#fecdd3",
        "#fecdd3",
        "#4c0519",
        "#fda4af",
        "#f43f5e",
        "#e11d48",
    ),
    "preview": (
        "#fdebd0",
        "#7e5109",
        "#f9e79f",
        "#f9e79f",
        "#451a03",
        "#fde047",
        "#ca8a04",
        "#a16207",
    ),
    "fit": ("#dbeafe", "#1e3a8a", "#93c5fd", "#bfdbfe", "#0b3b8f", "#bfdbfe", "#2563eb", "#1d4ed8"),
    "neutral": (
        "#f1f5f9",
        "#334155",
        "#cbd5e1",
        "#e2e8f0",
        "#334155",
        "#e2e8f0",
        "#64748b",
        "#475569",
    ),
}

# Which palette each button uses. Anything unmapped falls back to "neutral".
_BUTTON_PALETTES: dict[str, str] = {
    "LoadPDATButton": "load",
    "ClearDataButton": "clear",
    # One colour per model family, so the selected scheme is obvious.
    "ParallelButton": "model_parallel",
    "SequentialButton": "model_sequential",
    "TargetButton": "model_target",
    "ODEdefinedButton": "model_other",
    "DefinemodelButton": "model_target",
    "ShowSchemeButton": "model_other",
    # The two post-fit figures borrow the colours of the equivalent plot
    # buttons, so a spectra button looks like a spectra button throughout.
    "PlotSpeciesSpectraButton": "spectra",
    "PlotConcProfileButton": "kinetics",
    "AddComponentButton": "add",
    "RemoveComponentButton": "remove",
    "PreviewButton": "preview",
    "FITDATAButton": "fit",
    "ResetFitButton": "clear",
    "PP_ContourplotButton": "contour",
    "PP_PlotKineticsButton": "kinetics",
    "PP_SurfaceplotButton": "surface",
    "PP_PlotTrSpectraButton": "spectra",
    # Lifetime Density Analysis (LDA) tab buttons
    "RunLDAButton": "fit",
    "PlotLDAMapButton": "contour",
    "PlotLCurveButton": "preview",
    "PlotLDASliceButton": "kinetics",
    "SaveLDAMapPDATButton": "load",
    "LDACitationsButton": "neutral",
}

# The primary action buttons are set one step larger.
_BUTTON_FONT_SIZES: dict[str, str] = {
    "FITDATAButton": "13px",
    "RunLDAButton": "13px",
}


def button_stylesheet(palette: str, dark: bool, font_size: str = "11px") -> str:
    """Stylesheet for one button, in the light or dark variant of ``palette``.

    A disabled button keeps its per-button colours unless the sheet says
    otherwise, so every sheet ends with a neutral grey ``:disabled`` rule that
    also kills the hover effect.
    """
    lbg, ltx, lbd, lhv, dbg, dtx, dbd, dhv = _PALETTES.get(palette, _PALETTES["neutral"])
    if dark:
        bg, text, border, hover = dbg, dtx, dbd, dhv
        dis_fg, dis_bg, dis_bd = _DISABLED_DARK_FG, _DISABLED_DARK_BG, _DISABLED_DARK_BORDER
    else:
        bg, text, border, hover = lbg, ltx, lbd, lhv
        dis_fg, dis_bg, dis_bd = _DISABLED_LIGHT_FG, _DISABLED_LIGHT_BG, _DISABLED_LIGHT_BORDER
    return (
        f"QPushButton {{ font-weight: bold; font-size: {font_size}; color: {text};"
        f" background-color: {bg}; border: 1px solid {border}; border-radius: 4px;"
        f" padding: 3px 6px; }}"
        f"QPushButton:hover:enabled {{ background-color: {hover}; border-color: {text}; }}"
        f"QPushButton:pressed {{ background-color: {border}; }}"
        f"QPushButton:checked {{ background-color: {hover}; border: 2px solid {text}; }}"
        f"QPushButton:disabled {{ color: {dis_fg}; background-color: {dis_bg};"
        f" border: 1px solid {dis_bd}; }}"
    )
