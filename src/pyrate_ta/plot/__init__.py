"""Analysis-only plots.

**Only what PyMORGAN cannot express.** Contour maps, transient spectra, kinetic
traces and species spectra all have PyMORGAN plotters already; call those.
Adding a second implementation here means two figures that drift apart in style
and in unit handling, which is exactly what the shared ``Settings`` layer exists
to prevent.

Delegate:

* species spectra (DAS/EAS/SAS) -> ``pymorgan.oneD.plot.plot_species_spectra``,
  via a result's ``as_species_args()``;
* traces and maps -> ``Dataset1D.plot_kinetics`` / ``plot_spectra`` /
  ``plot_contour``.

Aesthetics come from PyMORGAN too: call ``pymorgan.apply_style()`` and read
``pymorgan.get_settings()`` for label style, colourmap and figure sizes, so a
PyRATE-TA figure sits beside a PyMORGAN one without looking foreign.

Contents
--------
``scheme``
    Compartment (K-matrix) diagrams: one node per compartment, one arrow per
    rate, plus the ground state where population is lost to it.
``matrix``
    Contour views of a fit surface or a residual matrix, drawn by PyMORGAN's
    own plotter.
``style``
    Data-as-points / fit-as-line keyword styles, read from the settings.
``monitor``
    The live parameter view: one bar per parameter while a fit runs, and the
    trajectory afterwards.
``concentrations``
    The concentration profiles ``C(t)`` of a fitted model -- the populations
    themselves, which no PyMORGAN plotter can express because they are never
    measured.

Planned contents
----------------
``lifetime_density``
    Lifetime-density maps.
"""

from __future__ import annotations

from .concentrations import plot_concentrations, species_labels_for
from .matrix import as_dataset, as_fit_dataset, plot_data_fit_residuals, plot_matrix
from .monitor import plot_parameter_bars, plot_parameter_history
from .scheme import plot_scheme, scheme_text
from .style import overlay_styles, scale_kwargs
from .traces import plot_kinetics_with_residuals, plot_spectra_with_residuals

__all__ = [
    "plot_scheme",
    "scheme_text",
    "plot_matrix",
    "as_dataset",
    "as_fit_dataset",
    "plot_concentrations",
    "plot_kinetics_with_residuals",
    "plot_spectra_with_residuals",
    "plot_parameter_bars",
    "plot_parameter_history",
    "species_labels_for",
    "overlay_styles",
    "scale_kwargs",
]

