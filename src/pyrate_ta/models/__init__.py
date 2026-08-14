"""Kinetic model definitions.

A model owns three things:

1. its **parameter vector layout** (which lifetimes/rates, in what order) plus
   bounds and a fixed/free mask;
2. the mapping from parameters to a **concentration matrix** ``C(t)``, shape
   ``[Ndelays x Ncomponents]``;
3. the **model family** it belongs to (:class:`pyrate.ModelType`), which decides
   whether the resulting species spectra are DAS, EAS or SAS, and therefore what
   PyMORGAN's ``plot_species_spectra`` prints in the legend.

Everything here is pure NumPy and must import without Qt, without scipy at
module level, and without touching ``pymorgan``. The data matrix is
``D ~ C @ S.T``; solving for the spectra ``S`` is the fitters' job
(:mod:`pyrate_ta.fit`), not the model's.

Contents
--------
``base``
    :class:`KineticModel` and the three families: :class:`ParallelModel`
    (DAS), :class:`SequentialModel` (EAS) and :class:`TargetModel` (SAS), plus
    the :func:`make_model` factory.
``scheme_text``
    The text notation for a scheme (``A -> B : k1``) and its parser, used by
    the K-matrix editor.
``schemes``
    Rate-matrix builders and :data:`TARGET_SCHEMES`, the library of predefined
    branched schemes.
``propagator``
    ``C(t)`` from ``K`` and ``c0`` -- eigendecomposition where usable, matrix
    exponential otherwise -- and the near-degeneracy guard.
``irf``
    Gaussian instrument response and the analytic (EMG) convolution.
"""

from __future__ import annotations

from .base import (
    KineticModel,
    ParallelModel,
    SequentialModel,
    TargetModel,
    make_model,
)
from .irf import (
    ARTIFACT_LABELS,
    clip_delays,
    coherent_artifact_basis,
    convolved_exponential,
    fwhm_to_sigma,
    gaussian_irf,
    sigma_to_fwhm,
)
from .propagator import check_degeneracy, concentrations, eigen_decomposition
from .scheme_text import (
    SchemeSyntaxError,
    check_scheme_text,
    parse_scheme_text,
    scheme_from_text,
    scheme_to_text,
)
from .schemes import (
    TARGET_SCHEMES,
    TargetScheme,
    default_c0,
    get_scheme,
    parallel_K,
    rates_from_lifetimes,
    sequential_K,
)

__all__ = [
    # models
    "KineticModel",
    "ParallelModel",
    "SequentialModel",
    "TargetModel",
    "make_model",
    # schemes and rate matrices
    "TARGET_SCHEMES",
    "TargetScheme",
    "get_scheme",
    "scheme_from_text",
    "scheme_to_text",
    "parse_scheme_text",
    "check_scheme_text",
    "SchemeSyntaxError",
    "parallel_K",
    "sequential_K",
    "default_c0",
    "rates_from_lifetimes",
    # propagation
    "concentrations",
    "eigen_decomposition",
    "check_degeneracy",
    # IRF
    "gaussian_irf",
    "convolved_exponential",
    "fwhm_to_sigma",
    "sigma_to_fwhm",
    "clip_delays",
    "coherent_artifact_basis",
    "ARTIFACT_LABELS",
]
