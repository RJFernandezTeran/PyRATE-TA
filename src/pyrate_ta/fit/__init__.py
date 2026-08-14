"""Fitting engines.

Every routine here is **headless-first**: a fit must be reachable and testable
from a plain script with no ``QApplication``. The GUI is a caller, never a
prerequisite, and progress is reported through a callback rather than by
touching widgets.

Variable projection (CRITICAL)
------------------------------
For fixed non-linear parameters (the lifetimes), the linear amplitudes are the
least-squares solution ``S = pinv(C) @ D``. Only the lifetimes go to the
non-linear optimiser. This is far better conditioned than fitting ``S`` and
``tau`` jointly, and it is why ``Settings.variable_projection`` defaults to
``True`` -- turning it off is a diagnostic, not a normal mode.

Convergence
-----------
A fit that did not converge is never returned as if it had. It raises, or the
result object carries the failure explicitly. Parameters resting on a bound and
lifetimes within ``Settings.degeneracy_warn_ratio`` of each other are flagged
through ``logger.warning``.

Contents
--------
``varpro``
    The closed-form linear solve: amplitudes and residuals for a fixed set of
    non-linear parameters, weighted or not.
``cost``
    Weights from the per-point noise, the residual vector handed to the
    optimiser, and the fit statistic (reduced chi-squared or SSR).
``driver``
    Parameter packing, fixed/free masks, bounds, the call into
    ``scipy.optimize.least_squares`` and the convergence report.

``engines``
    ``fit_kinetics`` (one trace), ``fit_global`` (DAS/EAS) and ``fit_target``
    (SAS), each returning a result object from :mod:`pyrate_ta.results`.
``prepare``
    The ``Dataset1D`` -> ``(t, D, sigma)`` adapter: the only PyMORGAN-aware
    piece of the fitting stack.
"""

from __future__ import annotations

from .cost import (
    FitStatistic,
    NoiseUnavailableError,
    Weights,
    build_weights,
    fit_statistic,
    residual_vector,
)
from .driver import FitOutcome, FitReport, run_fit
from .engines import fit_global, fit_kinetics, fit_target, preview_global
from .prepare import ProblemData, prepare
from .varpro import project, solve_amplitudes

__all__ = [
    # varpro
    "solve_amplitudes",
    "project",
    # cost
    "Weights",
    "build_weights",
    "residual_vector",
    "FitStatistic",
    "fit_statistic",
    "NoiseUnavailableError",
    # engines
    "fit_kinetics",
    "fit_global",
    "fit_target",
    "preview_global",
    "prepare",
    "ProblemData",
    # driver
    "run_fit",
    "FitOutcome",
    "FitReport",
]
