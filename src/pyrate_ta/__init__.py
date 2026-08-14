"""PyRATE-TA: kinetic analysis of time-resolved spectroscopy data.

Multi-exponential fitting, global analysis and target analysis with rate-matrix
(K-matrix) models, and the DAS/EAS/SAS spectra that come out of them.

    import pyrate_ta as pr
    import pymorgan as pm

    data = pm.load_1D("scan.pdat", data_type="PDAT")   # PyMORGAN loads
    data.background_correct(tmin=-20, tmax=-5)         # PyMORGAN processes
    fit = pr.fit_global(data, n_components=3)          # PyRATE-TA fits
    data.plot_species_spectra(*fit.as_species_args())  # PyMORGAN plots

**Scope.** PyRATE-TA does not read files, own the dataset objects, or draw the
standard spectroscopy figures -- that is PyMORGAN's job, and PyRATE-TA builds on
it. The dependency is strictly one-way: ``pyrate_ta`` imports ``pymorgan``, never
the reverse. PyMORGAN's ``plot_species_spectra`` deliberately takes plain arrays
rather than a PyRATE-TA object so the arrow never has to reverse; the adapter
(``as_species_args``) lives on this side.

**Lazy public API.** The solvers pull in scipy, and ``pymorgan`` itself chains
in matplotlib and the loader registries -- together roughly a second. Importing
this package therefore binds only the names below; the module that provides a
name is imported the first time that name is used (PEP 562). ``import pyrate_ta as
pr`` is thus nearly free, and the GUI can put its splash screen on screen before
paying for the heavy imports.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from .__about__ import __author__, __email__, __version__
from .log import configure_logging, get_logger

# Public name -> module that defines it. Kept in sync with ``__all__``; a new
# export needs one entry here.
_EXPORTS: dict[str, str] = {
    # settings
    "IRFMode": ".settings",
    "Regularisation": ".settings",
    "ensure_settings_file": ".settings",
    "settings_path": ".settings",
    "ModelType": ".settings",
    "Settings": ".settings",
    "get_settings": ".settings",
    # citations
    "cite": ".cite",
    "citation_text": ".cite",
    "citation_notice": ".cite",
    "print_citations": ".cite",
    "REFERENCES": ".cite",
    # fitting
    "fit_kinetics": ".fit",
    "fit_global": ".fit",
    "fit_target": ".fit",
    "preview_global": ".fit",
    "prepare": ".fit",
    "run_fit": ".fit",
    "FitOutcome": ".fit",
    "FitReport": ".fit",
    "FitStatistic": ".fit",
    "NoiseUnavailableError": ".fit",
    "build_weights": ".fit",
    "project": ".fit",
    "solve_amplitudes": ".fit",
    # plots
    "plot_scheme": ".plot",
    "scheme_text": ".plot",
    "plot_matrix": ".plot",
    "plot_data_fit_residuals": ".plot",
    "plot_concentrations": ".plot",

    "plot_kinetics_with_residuals": ".plot",
    "plot_parameter_bars": ".plot",
    "plot_parameter_history": ".plot",
    "overlay_styles": ".plot",

    # sessions
    "save_fit": ".io",
    "load_fit": ".io",
    "LoadedFit": ".io",
    # results
    "KineticFit": ".results",
    "GlobalFit": ".results",
    "TargetFit": ".results",
    # helpers
    "parse_lifetime": ".helpers",
    "format_lifetime": ".helpers",
    # models
    "KineticModel": ".models",
    "ParallelModel": ".models",
    "SequentialModel": ".models",
    "TargetModel": ".models",
    "TargetScheme": ".models",
    "TARGET_SCHEMES": ".models",
    "make_model": ".models",
    "get_scheme": ".models",
    "scheme_from_text": ".models",
    "scheme_to_text": ".models",
    "check_scheme_text": ".models",
    "SchemeSyntaxError": ".models",
    "load_settings": ".settings",
    "save_settings": ".settings",
    "set_settings": ".settings",
    "update_settings": ".settings",
}

# Submodules reachable as attributes (``pr.models``) without an import.
_SUBMODULES = (
    "models",
    "fit",
    "results",
    "plot",
    "io",
    "settings",
    "log",
    "cite",
    "helpers",
)

if TYPE_CHECKING:  # keeps type checkers and IDE completion fully informed
    from .cite import (
        REFERENCES,
        citation_notice,
        citation_text,
        cite,
        print_citations,
    )
    from .fit import (
        FitOutcome,
        FitReport,
        FitStatistic,
        NoiseUnavailableError,
        build_weights,
        fit_global,
        fit_kinetics,
        fit_target,
        prepare,
        preview_global,
        project,
        run_fit,
        solve_amplitudes,
    )
    from .helpers import format_lifetime, parse_lifetime
    from .io import LoadedFit, load_fit, save_fit
    from .models import (
        TARGET_SCHEMES,
        KineticModel,
        ParallelModel,
        SchemeSyntaxError,
        SequentialModel,
        TargetModel,
        TargetScheme,
        check_scheme_text,
        get_scheme,
        make_model,
        scheme_from_text,
        scheme_to_text,
    )
    from .plot import (
        overlay_styles,
        plot_concentrations,
        plot_kinetics_with_residuals,
        plot_matrix,
        plot_parameter_bars,
        plot_parameter_history,
        plot_scheme,
        scheme_text,
    )
    from .results import GlobalFit, KineticFit, TargetFit
    from .settings import (
        IRFMode,
        ModelType,
        Regularisation,
        Settings,
        ensure_settings_file,
        get_settings,
        load_settings,
        save_settings,
        set_settings,
        settings_path,
        update_settings,
    )


def __getattr__(name: str):
    """Import the module owning ``name`` on first access (PEP 562)."""
    module_name = _EXPORTS.get(name)
    if module_name is not None:
        value = getattr(importlib.import_module(module_name, __name__), name)
        globals()[name] = value  # cache, so this runs once per name
        return value
    if name in _SUBMODULES:
        value = importlib.import_module(f".{name}", __name__)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose the lazy names to ``dir()`` and tab completion."""
    return sorted(set(__all__) | set(_SUBMODULES))


__all__ = [
    "__version__",
    "__author__",
    "__email__",
    # settings
    "Settings",
    "IRFMode",
    "Regularisation",
    "ModelType",
    "ensure_settings_file",
    "settings_path",
    "get_settings",
    "set_settings",
    "update_settings",
    "load_settings",
    "save_settings",
    # citations
    "cite",
    "citation_text",
    "citation_notice",
    "print_citations",
    "REFERENCES",
    # fitting
    "fit_kinetics",
    "fit_global",
    "fit_target",
    "preview_global",
    "prepare",
    "run_fit",
    "FitOutcome",
    "FitReport",
    "FitStatistic",
    "NoiseUnavailableError",
    "build_weights",
    "project",
    "solve_amplitudes",
    # plots
    "plot_scheme",
    "scheme_text",
    "plot_matrix",
    "plot_data_fit_residuals",
    "plot_concentrations",
    "plot_kinetics_with_residuals",


    "plot_parameter_bars",
    "plot_parameter_history",
    "overlay_styles",
    # sessions
    "save_fit",
    "load_fit",
    "LoadedFit",
    # results
    "KineticFit",
    "GlobalFit",
    "TargetFit",
    # helpers
    "parse_lifetime",
    "format_lifetime",
    # models
    "KineticModel",
    "ParallelModel",
    "SequentialModel",
    "TargetModel",
    "TargetScheme",
    "TARGET_SCHEMES",
    "make_model",
    "get_scheme",
    "scheme_from_text",
    "scheme_to_text",
    "check_scheme_text",
    "SchemeSyntaxError",
]

configure_logging()
