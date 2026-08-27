"""Analysis settings for PyRATE.

**Aesthetics are not duplicated here.** Label style, colourmap, time-axis
convention and figure sizes belong to PyMORGAN's :class:`pymorgan.Settings`;
call :func:`pymorgan.apply_style` and read :func:`pymorgan.get_settings` for
those. This module holds only what is genuinely analysis-specific: solver
tolerances, iteration limits, defaults for new fits and how the instrument
response is treated.

Design follows PyMORGAN exactly, so the two feel the same and the GUI panels can
share code:

* :class:`Settings` is a dataclass and **the dataclass field is the single
  source of truth**. ``__post_init__`` (enum coercion), :meth:`Settings.to_dict`,
  :meth:`Settings.from_dict` and :func:`update_settings` are all derived from
  ``dataclasses.fields()`` plus the resolved annotations, so a new setting needs
  only the field -- no serialisation edits.
* :meth:`Settings.field_specs` drives the GUI settings panel and is *derived*
  from the fields, their annotations and :meth:`Settings.field_comments`, so a
  new setting is editable in the interface as soon as it exists. A small
  override table refines labels and ranges.
* Enums are :class:`enum.StrEnum` (3.11+): ``str(member)`` is the value, so they
  serialise and compare as plain strings while staying type-checked.
* ``settings.toml`` is read and written with ``tomlkit``, so user comments
  survive a save.

This module never imports Qt.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import StrEnum
from pathlib import Path
from typing import Any

from .log import get_logger

logger = get_logger(__name__)


class IRFMode(StrEnum):
    """How the instrument response is handled in a fit."""

    NONE = "None"  # no IRF; the model is evaluated directly
    GAUSSIAN = "Gaussian"  # analytic Gaussian-convolved exponentials
    CLIP = "Clip"  # skip early delays instead of modelling the IRF


class Regularisation(StrEnum):
    """How the LDA regularisation strength is chosen."""

    LCURVE = "L-curve"  # corner of the residual-vs-norm trade-off curve
    GCV = "GCV"  # generalised cross-validation
    FIXED = "Fixed"  # use lda_alpha as given, choose nothing


class ModelType(StrEnum):
    """Kinetic model family, which decides what the species spectra mean."""

    PARALLEL = "Parallel"  # independent decays -> DAS
    SEQUENTIAL = "Sequential"  # A -> B -> C -> ... -> EAS
    TARGET = "Target"  # general K-matrix with branching -> SAS


@dataclass
class Settings:
    """Analysis-only settings. Presentation lives in :class:`pymorgan.Settings`."""

    # --- Fit defaults ---------------------------------------------------- #
    # Number of kinetic components proposed for a new fit.
    n_components: int = 2
    # Default kinetic model family for a new fit.
    model_type: ModelType = ModelType.SEQUENTIAL
    # Add a constant offset (an infinite-lifetime component) by default.
    offset: bool = False

    # --- Solver ---------------------------------------------------------- #
    # Termination tolerance on the cost function.
    ftol: float = 1e-8
    # Termination tolerance on the parameter vector.
    xtol: float = 1e-8
    # Hard cap on optimiser iterations; exceeding it is a non-convergence.
    max_iterations: int = 2000
    # Solve the linear amplitudes by least squares at each step (variable
    # projection) instead of putting them in the optimiser's parameter vector.
    # Off is almost always the wrong choice; exposed for diagnostics only.
    variable_projection: bool = True

    # --- Instrument response --------------------------------------------- #
    irf_mode: IRFMode = IRFMode.GAUSSIAN
    # Fixed IRF FWHM in the dataset's time unit; None fits it as a parameter.
    irf_fwhm: float | None = None
    # Delays below this are dropped when irf_mode is CLIP (dataset time unit).
    # The coherent artefact is deliberately *not* a setting: it is a per-fit
    # choice (the checkbox beside the IRF table, or coherent_artifact=True on
    # any engine), not a standing preference.
    t_min: float | None = None

    # --- Weighting -------------------------------------------------------- #
    # Weight the residuals by the per-point noise when the dataset carries one
    # (a sibling .pdatn file, or the spread over single scans). With weights the
    # fit statistic is a true reduced chi-squared; without them it is the plain
    # sum of squared residuals, which is not comparable across datasets.
    use_noise_weights: bool = False
    # Noise values below this fraction of the median noise are clipped up to it,
    # so a single near-zero sigma cannot dominate the cost function.
    noise_floor_fraction: float = 1e-3

    # --- Reporting -------------------------------------------------------- #
    # Report 1-sigma uncertainties from the linearised covariance estimate.
    report_uncertainties: bool = True
    # Warn when two fitted lifetimes come within this factor of each other,
    # where the eigenvector matrix becomes ill-conditioned.
    degeneracy_warn_ratio: float = 1.2

    # --- Lifetime density analysis (LDA) ---------------------------------- #
    # Number of fixed lifetimes on the log-spaced grid the LDA solves over.
    lda_n_lifetimes: int = 100
    # Grid limits in the dataset time unit; None spans the measured window.
    lda_tau_min: float | None = None
    lda_tau_max: float | None = None
    # How the Tikhonov regularisation strength is chosen.
    lda_regularisation: Regularisation = Regularisation.LCURVE
    # Strength used when lda_regularisation is Fixed (and the starting point
    # for the automatic searches).
    lda_alpha: float = 1e-3
    # Constrain the amplitude distribution to be non-negative. Off by default:
    # a transient spectrum legitimately has both signs.
    lda_non_negative: bool = False

    # --- Overlay style (data vs fit) --------------------------------------- #
    # How the measured traces are drawn in the embedded panels: markers, so the
    # fitted line drawn over them stays readable.
    data_marker: str = "o"
    data_alpha: float = 0.5
    data_markersize: float = 3.0
    # ... and the fit itself, as a solid line over those points.
    fit_linestyle: str = "-"
    fit_linewidth: float = 1.0
    fit_alpha: float = 1.0
    # Height ratio between trace axis and linked residual axis (e.g. 4.0 for 4:1, 3.0 for 3:1)
    residuals_height_ratio: float = 4.0
    # Small vertical space between trace panel and residual panel
    residuals_hspace: float = 0.05
    n_contours: int = 40
    asinh_pct: float = 5.0


    # Margins of the embedded 3x3 figure, in figure fractions. Tuned so the
    # axis labels fit without wasting the window on white space; the panel is
    # not a stand-alone figure, so tight_layout's defaults are too generous.
    panel_left: float = 0.09
    panel_right: float = 0.975
    panel_top: float = 0.95
    panel_bottom: float = 0.11
    # Gaps between the three panels, as a fraction of the average axis size.
    panel_hspace: float = 0.45
    panel_wspace: float = 0.32

    # --- How results are quoted -------------------------------------------- #
    # Show the 1-sigma uncertainty beside each lifetime (plot legends and the
    # logged fit summary). Off quotes the value alone.
    show_uncertainties: bool = True
    # Round a value together with its uncertainty: the uncertainty to one
    # significant figure and the value to that same decimal place, so
    # 325.3 +/- 2.5 is quoted as 325 +/- 3. Keeps a fitted constant from being
    # presented with more precision than its own uncertainty supports.
    round_uncertainties: bool = True

    # --- Live fit monitor --------------------------------------------------- #
    # Show the parameters as they move, every N residual evaluations. 0 never
    # shows the window; 1 updates on every evaluation (slower, since each update
    # redraws and pumps the event loop); 5-10 is usually enough to watch a
    # lifetime run away without slowing the fit noticeably.
    fit_monitor_every: int = 0
    # Height scale of the monitor bars. Lifetimes span decades while t0 sits at
    # zero and may be negative, so symlog is the only one that shows them
    # together.
    fit_monitor_scale: str = "symlog"

    # --- GUI --------------------------------------------------------------- #
    # Qt style; a "dark" token anywhere selects the dark palette ("Fusion Dark").
    gui_theme: str = "Fusion"
    # Splash-screen duration in seconds; 0 disables it.
    splash_duration: float = 0.5
    # Default window geometry, and whether to restore it on the next start.
    # Taller than wide-ish on purpose: the controls stack vertically and the
    # plot grid is roughly square. The window is clipped to the available screen
    # area at start-up, so asking for more height than a laptop has is safe --
    # it simply gets what fits.
    window_width: int = 1010
    window_height: int = 860
    restore_window_size: bool = True
    # Defaults for a new row of the lifetime table.
    table_default_lb: float = 0.0
    table_default_ub: float = float("inf")
    table_decimals: int = 4
    new_rows_fixed: bool = False
    # What happens once a fit finishes. The two figures a fit is normally read
    # from -- the species spectra and the concentration profiles -- are opened
    # automatically; the same buttons reopen them at any time.
    plot_species_after_fit: bool = True
    plot_profiles_after_fit: bool = True
    plot_trio_after_fit: bool = True
    open_scheme_after_fit: bool = False
    autosave_session: bool = False

    @property
    def plot_data_fit_residuals_after_fit(self) -> bool:
        return self.plot_trio_after_fit

    @plot_data_fit_residuals_after_fit.setter
    def plot_data_fit_residuals_after_fit(self, value: bool):
        self.plot_trio_after_fit = bool(value)


    # --- Paths ------------------------------------------------------------ #
    default_datadir: Path | None = None

    def __post_init__(self) -> None:
        """Coerce every field to its declared type (enums from their value)."""
        for f in fields(self):
            setattr(self, f.name, _coerce_field(f.type, getattr(self, f.name)))

    # ------------------------------------------------------------------ #
    #                          Serialisation                             #
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """Plain mapping, with enums as their string values."""
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, StrEnum):
                value = str(value)
            elif isinstance(value, Path):
                value = str(value)
            out[f.name] = value
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Settings:
        """Build from a mapping, ignoring unknown keys."""
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            logger.debug("ignoring unknown settings: %s", ", ".join(sorted(unknown)))
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: str | Path) -> None:
        """Write the settings to ``path`` as TOML, preserving any comments."""
        save_settings(path, self)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Settings:
        """Read a :class:`Settings` from a TOML file."""
        return load_settings(path)

    # ------------------------------------------------------------------ #
    #                            GUI panel                               #
    # ------------------------------------------------------------------ #
    @staticmethod
    def field_sections() -> dict[str, str]:
        """Which ``settings.toml`` section each field is written to.

        Every field must appear here: a field with no section would be written
        at the top level and read back inconsistently, so
        :func:`save_settings` refuses rather than guessing.
        """
        return {
            # [fit]
            "n_components": "fit",
            "model_type": "fit",
            "offset": "fit",
            # [solver]
            "ftol": "solver",
            "xtol": "solver",
            "max_iterations": "solver",
            "variable_projection": "solver",
            "report_uncertainties": "solver",
            "degeneracy_warn_ratio": "solver",
            "use_noise_weights": "solver",
            "noise_floor_fraction": "solver",
            # [irf]
            "irf_mode": "irf",
            "irf_fwhm": "irf",
            "t_min": "irf",
            # [lda]
            "lda_n_lifetimes": "lda",
            "lda_tau_min": "lda",
            "lda_tau_max": "lda",
            "lda_regularisation": "lda",
            "lda_alpha": "lda",
            "lda_non_negative": "lda",
            # [plots]
            "data_marker": "plots",
            "data_alpha": "plots",
            "data_markersize": "plots",
            "fit_linestyle": "plots",
            "fit_linewidth": "plots",
            "fit_alpha": "plots",
            "residuals_height_ratio": "plots",
            "residuals_hspace": "plots",
            "n_contours": "plots",
            "asinh_pct": "plots",
            "panel_left": "plots",


            "panel_right": "plots",
            "panel_top": "plots",
            "panel_bottom": "plots",
            "panel_hspace": "plots",
            "panel_wspace": "plots",
            "fit_monitor_every": "gui",
            "fit_monitor_scale": "gui",
            "show_uncertainties": "plots",
            "round_uncertainties": "plots",
            # [gui]
            "gui_theme": "gui",
            "splash_duration": "gui",
            "window_width": "gui",
            "window_height": "gui",
            "restore_window_size": "gui",
            "table_default_lb": "gui",
            "table_default_ub": "gui",
            "table_decimals": "gui",
            "new_rows_fixed": "gui",
            "plot_species_after_fit": "gui",
            "plot_profiles_after_fit": "gui",
            "plot_trio_after_fit": "gui",
            "open_scheme_after_fit": "gui",

            "autosave_session": "gui",
            # [paths]
            "default_datadir": "paths",
        }

    @staticmethod
    def section_titles() -> dict[str, str]:
        """One-line description written above each section of the TOML file."""
        return {
            "fit": "Defaults proposed for a new fit.",
            "solver": "Optimiser limits, weighting and what is reported.",
            "irf": "Instrument response and the coherent-artefact window.",
            "lda": "Lifetime density analysis (the separate LDA window).",
            "plots": (
                "How data and fit are drawn in the embedded panels. Everything "
                "else about plot appearance belongs to PyMORGAN's settings."
            ),
            "gui": "Interface defaults. Plot aesthetics belong to PyMORGAN.",
            "paths": "Filesystem defaults.",
        }

    @staticmethod
    def field_comments() -> dict[str, str]:
        """Comment written above each key, so the file explains itself."""
        return {
            "n_components": "Components proposed for a new fit.",
            "model_type": "Parallel (DAS) / Sequential (EAS) / Target (SAS).",
            "offset": (
                "Deprecated: a non-decaying component is now created by entering a "
                "lifetime of inf, not by this flag."
            ),
            "ftol": "Termination tolerance on the cost function.",
            "xtol": "Termination tolerance on the parameter vector.",
            "max_iterations": "Cap on optimiser iterations; exceeding it is a non-convergence.",
            "variable_projection": (
                "Solve the amplitudes in closed form. Off is a diagnostic, not a normal mode."
            ),
            "report_uncertainties": "Report linearised 1-sigma uncertainties.",
            "degeneracy_warn_ratio": "Warn when two lifetimes come within this factor.",
            "use_noise_weights": (
                "Weight residuals by the per-point noise, making the statistic a reduced "
                "chi-squared instead of the SSR. Requires a .pdatn sibling or single scans."
            ),
            "noise_floor_fraction": "Clip sigma below this fraction of its median.",
            "irf_mode": "None / Gaussian / Clip.",
            "irf_fwhm": "Fixed IRF FWHM in the dataset time unit; empty fits it.",
            "t_min": "Drop delays below this when irf_mode = Clip.",
            "lda_n_lifetimes": "Points on the log-spaced lifetime grid.",
            "lda_tau_min": "Grid lower limit; empty spans the measured window.",
            "lda_tau_max": "Grid upper limit; empty spans the measured window.",
            "lda_regularisation": "How the Tikhonov strength is chosen: L-curve / GCV / Fixed.",
            "lda_alpha": "Strength used when lda_regularisation = Fixed.",
            "lda_non_negative": (
                "Constrain the distribution to be non-negative. Off by default: a transient "
                "spectrum legitimately has both signs."
            ),
            "data_marker": "Marker for the measured traces (any Matplotlib marker, e.g. o . s ^).",
            "data_alpha": "Opacity of the data markers, 0 to 1.",
            "data_markersize": "Marker size for the data points.",
            "fit_linestyle": "Line style of the fitted curve drawn over the data.",
            "fit_linewidth": "Line width of the fitted curve.",
            "fit_alpha": "Opacity of the fitted curve, 0 to 1.",
            "residuals_height_ratio": "Aspect ratio of kinetic trace plot height to residual plot height.",
            "residuals_hspace": "Vertical spacing between kinetic trace panel and residual panel.",
            "gui_theme": 'Qt style; a "dark" token selects the dark palette.',

            "splash_duration": "Splash-screen seconds; 0 disables it.",
            "window_width": "Default window width in pixels.",
            "window_height": "Default window height in pixels.",
            "restore_window_size": "Restore the last window size on the next start.",
            "table_default_lb": "Lower bound written into a new lifetime row.",
            "table_default_ub": "Upper bound written into a new lifetime row (inf = unbounded).",
            "table_decimals": "Decimal places shown in the lifetime table.",
            "new_rows_fixed": "New lifetime rows start with the Fix box ticked.",
            "plot_species_after_fit": "Open the DAS/EAS/SAS window automatically after a fit.",
            "plot_profiles_after_fit": (
                "Open the concentration-profile window automatically after a fit."
            ),
            "open_scheme_after_fit": "Open the kinetic-scheme diagram automatically after a fit.",
            "autosave_session": "Write the fit session next to the dataset after a fit.",
            "default_datadir": "Directory the file dialog opens in; empty means the home folder.",
        }

    @staticmethod
    def field_specs() -> dict[str, dict[str, Any]]:
        """Specs driving the GUI settings panel -- derived, then refined.

        Every field gets an entry automatically: the widget kind from its
        annotation, the page from :meth:`field_sections` and the tooltip from
        :meth:`field_comments`. A new setting therefore reaches the panel as
        soon as it exists, which is the point -- a setting the interface cannot
        reach is a setting the user does not really have.

        ``_SPEC_OVERRIDES`` refines the derived entry where the annotation
        cannot say enough (sensible ranges, a shorter label) and can drop a
        field from the panel with ``"skip": True``.
        """
        sections = Settings.field_sections()
        comments = Settings.field_comments()
        specs: dict[str, dict[str, Any]] = {}
        for f in fields(Settings):
            spec: dict[str, Any] = {
                "label": f.name.replace("_", " ").capitalize(),
                "tab": sections.get(f.name, "fit"),
            }
            spec.update(_kind_of(f.type))
            if comments.get(f.name):
                spec["tooltip"] = comments[f.name]
            spec.update(_SPEC_OVERRIDES.get(f.name, {}))
            if callable(spec.get("choices")):
                # Resolved late: the Qt styles this machine offers are only
                # known once the GUI has registered them.
                spec["choices"] = list(spec["choices"]())
            if not spec.pop("skip", False):
                specs[f.name] = spec
        return specs


#: Refinements on top of the derived specs (see :meth:`Settings.field_specs`).
_SPEC_OVERRIDES: dict[str, dict[str, Any]] = {
    "n_components": {
        "label": "Components",
        "kind": "int",
        "min": 1,
        "max": 10,
        "tab": "fit",
        "tooltip": "Number of kinetic components proposed for a new fit.",
    },
    "model_type": {
        "label": "Model",
        "kind": "choice",
        "choices": [str(m) for m in ModelType],
        "tab": "fit",
    },
    # ``offset`` is deliberately absent: an inf lifetime is the offset
    # now, so the field is deprecated and hidden (see below).
    "ftol": {"label": "Cost tolerance", "kind": "float", "tab": "solver"},
    "xtol": {"label": "Parameter tolerance", "kind": "float", "tab": "solver"},
    "max_iterations": {
        "label": "Max iterations",
        "kind": "int",
        "min": 1,
        "tab": "solver",
    },
    "irf_mode": {
        "label": "IRF",
        "kind": "choice",
        "choices": [str(m) for m in IRFMode],
        "tab": "fit",
    },
    "irf_fwhm": {
        "label": "IRF FWHM",
        "kind": "float",
        "tab": "fit",
        "tooltip": "Fixed IRF width; leave blank to fit it.",
    },
    "t_min": {
        "label": "Clip before",
        "kind": "float",
        "tab": "fit",
        "tooltip": "Drop delays below this value (IRF mode 'Clip').",
    },
    "report_uncertainties": {
        "label": "Report uncertainties",
        "kind": "bool",
        "tab": "solver",
    },
    "use_noise_weights": {
        "label": "Weight by noise",
        "kind": "bool",
        "tab": "solver",
        "tooltip": (
            "Weight residuals by the per-point noise, when the dataset "
            "carries one, so the statistic is a reduced chi-squared."
        ),
    },
    "noise_floor_fraction": {
        "label": "Noise floor (fraction of median)",
        "kind": "float",
        "tab": "solver",
        "tooltip": "Clip noise values below this fraction of the median noise.",
    },
    "offset": {"skip": True},  # deprecated: an inf lifetime is the offset now
    # LDA fields live in settings.toml as CLI/script defaults and are read by
    # the GUI LDA tab at start-up to pre-populate its own widgets.  They are
    # *not* shown in the settings dialog because those widgets are the canonical
    # control surface for LDA; exposing them in two places would invite
    # confusion about which one wins.
    "lda_n_lifetimes": {"skip": True},
    "lda_tau_min": {"skip": True},
    "lda_tau_max": {"skip": True},
    "lda_regularisation": {"skip": True},
    "lda_alpha": {"skip": True},
    "lda_non_negative": {"skip": True},
    # The themes are PyMORGAN's, and the list is only complete once the GUI has
    # asked Qt what this machine actually has -- hence a callable.
    "gui_theme": {"kind": "choice", "choices": lambda: _theme_choices()},
    "fit_monitor_every": {"label": "Live parameter update every N evals", "min": 0, "max": 1000},
    "fit_monitor_scale": {"kind": "choice", "choices": ["symlog", "log", "linear"]},
    "n_contours": {"label": "Default contour levels", "min": 2, "max": 500, "tab": "plots"},
    "asinh_pct": {"label": "arcsinh linear threshold (%)", "min": 0.1, "max": 100.0, "tab": "plots"},

    "table_decimals": {"min": 1, "max": 12},
    "window_width": {"min": 400, "max": 10000},
    "window_height": {"min": 300, "max": 10000},
    "default_datadir": {
        "label": "Default data directory",
        "kind": "directory",
        "tab": "paths",
        "tooltip": "Directory the file dialog opens in; empty means the home folder.",
    },
}


def _theme_choices() -> list[str]:
    """Qt styles offered for ``gui_theme``, from PyMORGAN (never a second list)."""
    try:
        from pymorgan.settings import gui_theme_choices

        return list(gui_theme_choices())
    except Exception:  # pragma: no cover - PyMORGAN is a hard dependency
        logger.debug("could not read the theme list from PyMORGAN", exc_info=True)
        return ["Fusion", "Fusion Dark"]


def _kind_of(annotation: Any) -> dict[str, Any]:
    """Widget kind for a field, read from its annotation.

    Annotations are strings here (``from __future__ import annotations``), so
    this matches on the text exactly as :func:`_coerce_field` does -- one
    convention, not two.
    """
    text = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", "")
    for enum_cls in (ModelType, IRFMode, Regularisation):
        if enum_cls.__name__ in text:
            return {"kind": "choice", "choices": [str(m) for m in enum_cls]}
    if "bool" in text:
        return {"kind": "bool"}
    if "int" in text and "float" not in text:
        return {"kind": "int", "min": 0, "max": 10**6}
    if "float" in text:
        return {"kind": "float"}
    if "Path" in text:
        return {"kind": "directory"}
    return {"kind": "text"}


def _coerce_field(annotation: Any, value: Any) -> Any:
    """Convert ``value`` to the type named by ``annotation``.

    Handles the cases the TOML layer produces: enum members from their string
    value, ``Path`` from a string, and a blank string as ``None`` for optional
    fields.
    """
    if value is None:
        return None
    text = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", "")
    if isinstance(value, str) and not value.strip() and "None" in text:
        return None
    if "ModelType" in text and not isinstance(value, ModelType):
        return ModelType(value)
    if "IRFMode" in text and not isinstance(value, IRFMode):
        return IRFMode(value)
    if "Regularisation" in text and not isinstance(value, Regularisation):
        return Regularisation(value)
    if "Path" in text and not isinstance(value, Path):
        return Path(value)
    if "bool" in text and not isinstance(value, bool):
        return bool(value)
    if "int" in text and "float" not in text and not isinstance(value, int):
        return int(value)
    if "float" in text and not isinstance(value, float):
        return float(value)
    return value


# --------------------------------------------------------------------------- #
#                              Active settings                                #
# --------------------------------------------------------------------------- #
_ACTIVE = Settings()


def get_settings() -> Settings:
    """The active PyRATE settings object."""
    return _ACTIVE


def set_settings(settings: Settings) -> Settings:
    """Replace the active settings wholesale."""
    global _ACTIVE
    _ACTIVE = settings
    return _ACTIVE


def update_settings(**kwargs: Any) -> Settings:
    """Update named fields on the active settings, rejecting unknown names."""
    known = {f.name for f in fields(Settings)}
    unknown = set(kwargs) - known
    if unknown:
        raise ValueError(f"unknown setting(s): {', '.join(sorted(unknown))}")
    for name, value in kwargs.items():
        annotation = next(f.type for f in fields(Settings) if f.name == name)
        setattr(_ACTIVE, name, _coerce_field(annotation, value))
    return _ACTIVE


def get_default_settings_text() -> str:
    """Return the raw text of the canonical default settings template."""
    pkg_default = Path(__file__).resolve().parent / "settings.default.toml"
    if pkg_default.is_file():
        return pkg_default.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Could not locate '{pkg_default}'.")


def get_default_settings_doc():
    """Parse and return the canonical default settings template as a tomlkit document."""
    import tomlkit

    return tomlkit.parse(get_default_settings_text())


def _flatten(document) -> dict[str, Any]:
    """Read a sectioned (or flat) TOML document into one mapping.

    Both layouts are accepted: sections are the current format, a flat file is
    what an older settings.toml looks like, and a key found at both levels is
    taken from its section (the newer form wins) with the duplicate logged.
    """
    flat: dict[str, Any] = {}
    for key, value in dict(document).items():
        if hasattr(value, "items"):
            for sub_key, sub_value in dict(value).items():
                if sub_key in flat:
                    logger.debug("setting %r appears twice; using the [%s] one", sub_key, key)
                flat[sub_key] = sub_value
        elif key not in flat:
            flat[key] = value
    # An empty string in the file means "not set"; the dataclass wants None.
    return {k: (None if v == "" else v) for k, v in flat.items()}


def settings_path(start: str | Path | None = None) -> Path:
    """Where ``settings.toml`` is looked for: working directory, then the root.

    The same lookup order PyMORGAN uses, so a project directory can override the
    defaults for both packages in the same way.
    """
    local = Path(start or Path.cwd()) / "settings.toml"
    if local.is_file():
        return local.resolve()
    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / "pyproject.toml").is_file() or (repo_root / ".git").is_dir():
        return (repo_root / "settings.toml").resolve()
    return local.resolve()


def ensure_settings_file(path: str | Path | None = None) -> Path:
    """Ensure that ``path`` (defaulting to ``settings_path()``) exists and is up to date.

    If the file does not exist, it is generated from ``settings.default.toml``.
    If the file exists, it is checked for any newly introduced settings from the
    default template. If missing keys are found, they are merged into the file
    along with their section comments while preserving all existing user modifications.
    """
    import tomlkit

    target = Path(path).resolve() if path is not None else settings_path()
    default_text = get_default_settings_text()

    if not target.exists():
        import warnings

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(default_text, encoding="utf-8")
        warnings.warn(
            f"Created default settings file at '{target}'. "
            "You can run `pyrate-ta-settings` to configure your preferences interactively, "
            "or edit this file directly.",
            UserWarning,
            stacklevel=2,
        )
        logger.info("Wrote default settings file to %s", target)
        return target

    try:
        user_doc = tomlkit.parse(target.read_text(encoding="utf-8"))
    except Exception:
        return target

    default_doc = tomlkit.parse(default_text)

    # Check whether any sections or keys from default_doc are missing in user_doc
    needs_merge = False
    for sec_name, sec_val in default_doc.items():
        if hasattr(sec_val, "items"):
            if sec_name not in user_doc:
                needs_merge = True
                break
            user_sec = user_doc[sec_name]
            if not hasattr(user_sec, "__getitem__"):
                needs_merge = True
                break
            for k in sec_val:
                if k not in user_sec:
                    needs_merge = True
                    break
        else:
            if sec_name not in user_doc:
                needs_merge = True
                break

    if needs_merge:
        # Merge by taking default_doc and overlaying all user-defined values and tables
        merged = tomlkit.parse(default_text)
        sections = Settings.field_sections()
        for key, val in user_doc.items():
            if hasattr(val, "items"):
                if key not in merged:
                    merged[key] = val
                else:
                    for sub_k, sub_v in val.items():
                        merged[key][sub_k] = sub_v
            else:
                # Handle unsectioned/flat key: place in appropriate section if known
                sec = sections.get(key)
                if sec and sec in merged and hasattr(merged[sec], "__setitem__"):
                    merged[sec][key] = val
                else:
                    merged[key] = val
        target.write_text(tomlkit.dumps(merged), encoding="utf-8")

    return target


def load_settings(path: str | Path | None = None) -> Settings:
    """Load settings from a TOML file and make them active.

    Ensures the settings file exists (creating from defaults if absent) and
    merges any newly introduced settings while keeping user modifications.
    """
    import tomlkit

    target = ensure_settings_file(path)
    text = target.read_text(encoding="utf-8")
    return set_settings(Settings.from_dict(_flatten(tomlkit.parse(text))))


def save_settings(path: str | Path, settings: Settings | None = None) -> None:
    """Write the settings to ``path`` as TOML, preserving comments."""
    import tomlkit

    active = settings or _ACTIVE
    target = Path(path)
    if target.exists():
        doc = tomlkit.parse(target.read_text(encoding="utf-8"))
    else:
        try:
            doc = tomlkit.parse(get_default_settings_text())
        except Exception:
            doc = tomlkit.document()

    sections = Settings.field_sections()
    values = active.to_dict()
    for name, value in values.items():
        sec = sections.get(name)
        val_str = "" if value is None else value
        if sec is None:
            doc[name] = val_str
            continue
        if sec not in doc:
            doc[sec] = tomlkit.table()
        doc[sec][name] = val_str

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(tomlkit.dumps(doc), encoding="utf-8")

