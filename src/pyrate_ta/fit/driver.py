"""The shared fit driver: packing, masks, bounds, solver, convergence report.

One driver serves every engine. It owns:

* the free/fixed split -- fixed parameters never reach the optimiser, and come
  back flagged rather than with a small uncertainty that could be mistaken for
  precision;
* the bounds, and the method that can honour them;
* the variable-projection cost function (:mod:`pyrate_ta.fit.varpro`) with the
  weights and statistic from :mod:`pyrate_ta.fit.cost`;
* the convergence report. **Exceeding the iteration cap is a non-convergence**,
  never a result: the outcome carries ``converged = False`` and a warning is
  logged, so nothing downstream can mistake it for a fit.

The driver is headless: no Qt, no widgets. Progress is reported through a plain
callback, which the GUI adapts to a progress dialog.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..cite import cite
from ..log import get_logger
from . import varpro
from .cost import FitStatistic, Weights, build_weights, fit_statistic, residual_vector

logger = get_logger(__name__)

# ``lm`` cannot honour bounds; the bounded methods are the other two.
_BOUNDED_METHODS = ("trf", "dogbox")


@dataclass
class FitReport:
    """What the optimiser did, in enough detail to judge the result."""

    converged: bool
    status: int
    message: str
    n_evaluations: int
    n_jacobian: int
    cost: float
    optimality: float
    method: str
    max_iterations: int
    at_bounds: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        head = "converged" if self.converged else "DID NOT CONVERGE"
        return (
            f"{head} ({self.method}, status {self.status}): {self.message} "
            f"[{self.n_evaluations} evaluations]"
        )


@dataclass
class FitOutcome:
    """Raw output of one driver run.

    Phase 4 wraps this in the public result objects (``GlobalFit`` and friends)
    together with the reproducibility payload; it is kept separate so the
    driver stays independent of the result classes.
    """

    params: np.ndarray
    param_names: list[str]
    fixed: np.ndarray
    S: np.ndarray
    R: np.ndarray
    statistic: FitStatistic
    report: FitReport
    weights: Weights
    jacobian: np.ndarray | None = None
    initial_params: np.ndarray | None = None
    bounds: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def converged(self) -> bool:
        return self.report.converged


def _resolve_fixed(model, fixed) -> np.ndarray:
    """Normalise the fixed/free mask to a boolean array of length ``n_params``."""
    n = model.n_params
    if fixed is None:
        return np.zeros(n, dtype=bool)
    fixed = np.asarray(fixed)
    if fixed.dtype.kind in "US":  # names
        names = model.param_names()
        unknown = set(np.atleast_1d(fixed).tolist()) - set(names)
        if unknown:
            raise ValueError(f"unknown parameter name(s) to fix: {sorted(unknown)}")
        return np.array([name in set(fixed.tolist()) for name in names], dtype=bool)
    fixed = fixed.astype(bool).ravel()
    if fixed.size == model.n_lifetimes < n:
        # A caller that only knows about the lifetimes -- the GUI reads its mask
        # from the lifetime table -- may pass just those. Time zero and the IRF
        # width are appended to the vector *only when they are being fitted*, so
        # the entries that follow are free by construction; padding them is the
        # correct completion, not a guess.
        pad = np.zeros(n - fixed.size, dtype=bool)
        logger.debug(
            "fixed mask covers the %d lifetime(s); %s taken as free",
            fixed.size,
            ", ".join(model.param_names()[fixed.size :]),
        )
        return np.concatenate([fixed, pad])
    if fixed.size != n:
        raise ValueError(
            f"fixed mask has {fixed.size} entries, expected {n} ({', '.join(model.param_names())})"
        )
    return fixed


def _bounds_hit(params, lo, hi, fixed, names, rtol: float = 1e-6) -> list[str]:
    """Names of free parameters resting on a finite bound."""
    hits = []
    for i, name in enumerate(names):
        if fixed[i]:
            continue
        for bound in (lo[i], hi[i]):
            if not np.isfinite(bound):
                continue
            scale = max(abs(bound), 1.0)
            if abs(params[i] - bound) <= rtol * scale:
                hits.append(name)
                break
    return hits


def run_fit(
    model,
    t,
    D,
    *,
    sigma=None,
    use_weights: bool = False,
    noise_source: str | None = None,
    p0=None,
    fixed=None,
    bounds=None,
    method: str | None = None,
    max_iterations: int | None = None,
    ftol: float | None = None,
    xtol: float | None = None,
    callback: Callable[[int, np.ndarray, float], None] | None = None,
    **model_kwargs: Any,
) -> FitOutcome:
    """Fit one detector's data with variable projection.

    Parameters
    ----------
    model : KineticModel
        Supplies ``C(t)``, the parameter layout, the bounds and the initial
        guess.
    t : array_like, shape ``(Nd,)``
        Delays, in the dataset's time unit.
    D : array_like, shape ``(Nd, Np)``
        Data for **one** detector, in mOD.
    sigma : array_like, optional
        Per-point noise, same shape as ``D``.
    use_weights : bool
        Weight the residuals by ``1/sigma``; raises when no ``sigma`` is
        available (see :mod:`pyrate_ta.fit.cost`).
    p0 : array_like, optional
        Initial parameter vector. Defaults to ``model.initial_guess(t)``.
    fixed : array_like of bool or str, optional
        Parameters held at their initial value, by mask or by name.
    bounds : tuple, optional
        ``(lower, upper)``; defaults to ``model.default_bounds(t)``.
    method : str, optional
        ``"trf"`` (default), ``"dogbox"`` or ``"lm"``. ``lm`` ignores bounds and
        is refused when finite bounds are present, rather than silently
        dropping them.
    max_iterations, ftol, xtol : optional
        Solver limits; default to the active :class:`pyrate.Settings`.
    callback : callable, optional
        ``callback(n_eval, params, cost)``, called on each residual evaluation.
        Progress reporting never touches widgets from here.

    Returns
    -------
    FitOutcome
    """
    from scipy.optimize import least_squares

    from ..settings import get_settings

    s = get_settings()
    method = method or "trf"
    max_iterations = int(s.max_iterations if max_iterations is None else max_iterations)
    ftol = float(s.ftol if ftol is None else ftol)
    xtol = float(s.xtol if xtol is None else xtol)

    t = np.atleast_1d(np.asarray(t, dtype=float)).ravel()
    D = np.atleast_2d(np.asarray(D, dtype=float))
    if D.shape[0] != t.size:
        raise ValueError(f"D has {D.shape[0]} delays but t has {t.size}")
    if D.ndim != 2:
        raise ValueError("D must be a single detector slice, shape [Ndelays x Npixels]")

    weights = build_weights(
        D, sigma, use_weights=use_weights, source=noise_source if use_weights else None
    )

    names = model.param_names()
    fixed_mask = _resolve_fixed(model, fixed)
    p_full = np.asarray(model.initial_guess(t) if p0 is None else p0, dtype=float).ravel()
    if p_full.size != len(names):
        raise ValueError(f"p0 has {p_full.size} entries, expected {len(names)}")
    p_initial = p_full.copy()

    lo, hi = (
        model.default_bounds(t)
        if bounds is None
        else (
            np.asarray(bounds[0], dtype=float),
            np.asarray(bounds[1], dtype=float),
        )
    )
    # ``inf`` is a legitimate lifetime (the non-decaying component) and is not
    # a bound violation, so it is exempt from this check.
    finite_p = np.isfinite(p_full)
    if np.any(p_full[finite_p] < lo[finite_p]) or np.any(p_full[finite_p] > hi[finite_p]):
        raise ValueError("the initial parameters lie outside the bounds")

    # An infinite lifetime is the non-decaying (offset) component: its rate is
    # zero and there is nothing for an optimiser to vary, so it is fixed
    # automatically rather than handed to the solver as an infinite start value.
    infinite = ~np.isfinite(p_full)
    if np.any(infinite & ~fixed_mask):
        held = [names[i] for i in np.flatnonzero(infinite & ~fixed_mask)]
        logger.info(
            "Non-decaying component(s) %s: an infinite lifetime is held fixed (rate = 0).",
            ", ".join(held),
        )
        fixed_mask = fixed_mask | infinite

    free = ~fixed_mask
    if not np.any(free):
        raise ValueError("every parameter is fixed; there is nothing to fit")

    bounded = bool(np.any(np.isfinite(lo[free])) or np.any(np.isfinite(hi[free])))
    if method not in _BOUNDED_METHODS and bounded:
        raise ValueError(
            f"method {method!r} cannot honour bounds; use one of {_BOUNDED_METHODS} "
            "or pass bounds=(-inf, inf) explicitly."
        )

    n_eval = 0

    def residuals(p_free):
        nonlocal n_eval
        p = p_full.copy()
        p[free] = p_free
        C = model.concentrations(t, p, check_degenerate=False)
        S, R = varpro.project(C, D, weights.w)
        vec = residual_vector(R, weights)
        n_eval += 1
        if callback is not None:
            callback(n_eval, p, 0.5 * float(np.dot(vec, vec)))
        return vec

    kwargs = {"ftol": ftol, "xtol": xtol, "max_nfev": max_iterations, "method": method}
    if method in _BOUNDED_METHODS:
        kwargs["bounds"] = (lo[free], hi[free])

    sol = least_squares(residuals, p_full[free], **kwargs)

    p_full[free] = sol.x
    C = model.concentrations(t, p_full)
    S, R = varpro.project(C, D, weights.w)
    stat = fit_statistic(
        R,
        weights,
        n_nonlinear=int(np.count_nonzero(free)),
        n_amplitudes=int(S.size),
    )

    converged = bool(sol.success) and sol.status > 0
    at_bounds = _bounds_hit(p_full, lo, hi, fixed_mask, names)
    report = FitReport(
        converged=converged,
        status=int(sol.status),
        message=str(sol.message),
        n_evaluations=int(sol.nfev),
        n_jacobian=int(getattr(sol, "njev", 0) or 0),
        cost=float(sol.cost),
        optimality=float(sol.optimality),
        method=method,
        max_iterations=max_iterations,
        at_bounds=at_bounds,
    )

    if not converged:
        logger.warning(
            "Fit did not converge (status %d): %s. The parameters below are the last "
            "trial values, not a result.",
            sol.status,
            sol.message,
        )
    else:
        logger.info("Fit %s; %s", report, stat)
        cite("vanstokkum2004")
    if at_bounds:
        logger.warning(
            "Parameter(s) resting on a bound: %s. A bounded optimum is a constraint, "
            "not an optimum.",
            ", ".join(at_bounds),
        )

    return FitOutcome(
        params=p_full,
        param_names=names,
        fixed=fixed_mask,
        S=S,
        R=R,
        statistic=stat,
        report=report,
        weights=weights,
        jacobian=np.asarray(sol.jac) if getattr(sol, "jac", None) is not None else None,
        initial_params=p_initial,
        bounds=(lo, hi),
    )
