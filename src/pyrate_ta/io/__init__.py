"""Import/export of fit sessions and tabular fit results.

**Not spectroscopy file formats.** Raw data readers (``PDAT``, ``P2DAT``, MESS
directories, ...) belong to PyMORGAN's loader registry; load through
``pymorgan.load_1D`` / ``load_2D``. A format PyRATE-TA needs and PyMORGAN lacks is
contributed upstream, or registered at runtime with
``pymorgan.register_loader`` -- never forked into this package.

What lives here is the analysis session (``.prfit`` -- a compressed ``.npz``
carrying both the arrays and a JSON reproducibility payload, so the numbers can
never be separated from the record of how they were produced) together with
clean single-header CSV exports of DAS/EAS/SAS spectra, concentration profiles
``C(t)``, and companion human-readable ``.txt`` fit summary reports.
"""

from __future__ import annotations

from .export import (
    ExportedFitPaths,
    ExportedSurfacePaths,
    compute_surface_matrix,
    export_concentrations,
    export_fit_results,
    export_fit_summary,
    export_spectra,
    export_surface,
    export_surfaces,
    format_fit_report,
    make_timestamp,
)
from .session import (
    FORMAT_VERSION,
    LoadedFit,
    default_session_path,
    load_fit,
    save_fit,
)

__all__ = [
    "save_fit",
    "load_fit",
    "LoadedFit",
    "default_session_path",
    "FORMAT_VERSION",
    "ExportedFitPaths",
    "ExportedSurfacePaths",
    "compute_surface_matrix",
    "export_spectra",
    "export_concentrations",
    "export_surface",
    "export_surfaces",
    "export_fit_summary",
    "export_fit_results",
    "format_fit_report",
    "make_timestamp",
]
