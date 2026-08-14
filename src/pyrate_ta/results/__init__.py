"""Fit result objects.

Reproducibility is the contract: a result must carry enough to

reproduce itself. That means the model identity, the initial guesses, the
bounds, which parameters were held fixed, the delay/probe ranges actually used
(clipping early delays changes what the fit saw, so it belongs here), and the
solver's convergence report. **A lifetime without its uncertainty and its
fixed/free flag is not a result.**

Uncertainties are 1-sigma from the linearised covariance estimate. Report them
as such: they are conditional on the model being correct and are not confidence
intervals for the model choice. Fixed parameters get no uncertainty and are
flagged as fixed.

The PyMORGAN adapter
--------------------
Result objects expose ``as_species_args()``, returning the plain
``(Sfit, Taus, TauErr, isFixTau, modelType)`` tuple that
``pymorgan.oneD.plot.plot_species_spectra`` expects. Keeping the adapter on this
side is what lets PyMORGAN render PyRATE output without ever importing
``pyrate``.

Contents
--------
``KineticFit``
    Single-trace multi-exponential result.
``GlobalFit``
    Global-analysis result: shared lifetimes plus the species spectra matrix.
``TargetFit``
    Target-analysis result, adding the K-matrix, the scheme and the system
    eigenvalues (which are what a branched scheme actually determines).

Planned contents
----------------
``serialise``
    Save/load a fit session (parameters, masks, results).
"""

from __future__ import annotations

from .fits import GlobalFit, KineticFit, TargetFit, covariance_from_jacobian

__all__ = ["KineticFit", "GlobalFit", "TargetFit", "covariance_from_jacobian"]
