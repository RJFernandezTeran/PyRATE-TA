"""Weights, the residual vector, and the fit statistic.

The weighting rules are the ones agreed in ``todo_fitting_routines.md`` §3:

* weights are ``1/sigma`` with ``sigma`` the per-point noise PyMORGAN supplies
  (:meth:`pymorgan.Dataset1D.noise_array`); PyRATE-TA never estimates noise;
* ``sigma`` is clipped from below at ``noise_floor_fraction`` of its median, so
  one near-zero value cannot dominate the cost function;
* points with non-finite or non-positive ``sigma``, or non-finite data, get
  zero weight and are excluded from the point count;
* requesting weights for a dataset that has no noise **raises**. Falling back
  to an unweighted fit while still calling the statistic a reduced chi-squared
  would misreport it, which is worse than a refusal.

The statistic follows from whether weights were used:

* weighted: :math:`\\chi^2_\\nu = \\sum (R/\\sigma)^2 / (N - p)`
* unweighted: :math:`\\mathrm{SSR} = \\sum R^2`

``SSR`` is not comparable between datasets and carries no "close to 1" reading,
so the two are never given the same name.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..log import get_logger

logger = get_logger(__name__)


class NoiseUnavailableError(ValueError):
    """Weighted fit requested for data that carries no noise estimate."""


@dataclass(frozen=True)
class Weights:
    """Per-point weights and the bookkeeping that goes with them.

    Attributes
    ----------
    w : numpy.ndarray or None
        ``1/sigma`` ``[Nd, Np]``, or ``None`` for an unweighted fit.
    mask : numpy.ndarray
        Boolean ``[Nd, Np]``, True where a point takes part in the fit.
    n_points : int
        Number of points actually fitted, i.e. ``mask.sum()``.
    n_masked : int
        Points excluded (bad sigma or bad data); part of the reproducibility
        payload, since they change what the fit saw.
    floor : float or None
        The absolute value ``sigma`` was clipped at.
    n_floored : int
        How many sigma values that clip touched. A large number means the noise
        estimate, not the floor, is the thing to look at.
    source : str or None
        Where the noise came from, e.g. ``"pdatn"`` / ``"single_scans"``.
    """

    w: np.ndarray | None
    mask: np.ndarray
    n_points: int
    n_masked: int = 0
    floor: float | None = None
    n_floored: int = 0
    source: str | None = None

    @property
    def weighted(self) -> bool:
        return self.w is not None


def build_weights(
    D,
    sigma=None,
    *,
    use_weights: bool = False,
    floor_fraction: float | None = None,
    source: str | None = None,
) -> Weights:
    """Build the weights (or the plain mask) for one detector's data.

    Parameters
    ----------
    D : array_like, shape ``(Nd, Np)``
        Data, in mOD. Non-finite entries are masked out either way.
    sigma : array_like, optional
        Per-point standard deviation, same shape as ``D``.
    use_weights : bool
        Whether the fit is to be weighted. ``True`` without ``sigma`` raises.
    floor_fraction : float, optional
        Clip ``sigma`` at this fraction of its median. Defaults to
        ``Settings.noise_floor_fraction``.
    source : str, optional
        Provenance of the noise, recorded on the result.

    Raises
    ------
    NoiseUnavailableError
        If ``use_weights`` is set but no usable ``sigma`` was given.
    """
    D = np.atleast_2d(np.asarray(D, dtype=float))
    finite_data = np.isfinite(D)

    if not use_weights:
        n_masked = int(np.count_nonzero(~finite_data))
        if n_masked:
            logger.info("%d non-finite data point(s) excluded from the fit.", n_masked)
        return Weights(
            w=None,
            mask=finite_data,
            n_points=int(np.count_nonzero(finite_data)),
            n_masked=n_masked,
        )

    if sigma is None:
        raise NoiseUnavailableError(
            "a weighted fit was requested but the dataset carries no noise estimate "
            "(no .pdatn sibling and no single scans). Load noise data, or fit "
            "unweighted -- in which case the statistic is the SSR, not a reduced "
            "chi-squared."
        )

    sigma = np.asarray(sigma, dtype=float)
    if sigma.shape != D.shape:
        raise ValueError(f"sigma shape {sigma.shape} does not match D shape {D.shape}")

    usable = np.isfinite(sigma) & (sigma > 0) & finite_data
    if not np.any(usable):
        raise NoiseUnavailableError(
            "every sigma value is non-finite or non-positive; there is nothing to weight with."
        )

    if floor_fraction is None:
        from ..settings import get_settings

        floor_fraction = get_settings().noise_floor_fraction
    floor = float(floor_fraction) * float(np.median(sigma[usable]))

    sigma_clipped = np.where(usable, sigma, np.inf)
    n_floored = int(np.count_nonzero(usable & (sigma < floor)))
    if floor > 0:
        sigma_clipped = np.maximum(sigma_clipped, floor)
    if n_floored:
        logger.info(
            "%d sigma value(s) below %.3g (%.3g x median) were clipped up to the floor.",
            n_floored,
            floor,
            floor_fraction,
        )

    w = np.where(usable, 1.0 / sigma_clipped, 0.0)
    n_masked = int(np.count_nonzero(~usable))
    if n_masked:
        logger.info("%d point(s) excluded from the weighted fit (bad sigma or data).", n_masked)
    return Weights(
        w=w,
        mask=usable,
        n_points=int(np.count_nonzero(usable)),
        n_masked=n_masked,
        floor=float(floor),
        n_floored=n_floored,
        source=source,
    )


def residual_vector(R, weights: Weights):
    """Flatten the residual matrix into the vector the optimiser wants.

    Weighted residuals are ``R/sigma``; masked points contribute zero, which is
    what a zero weight already gives, so no reshaping is needed and the vector
    length stays constant across iterations (``scipy.optimize.least_squares``
    requires that).
    """
    R = np.asarray(R, dtype=float)
    if weights.w is None:
        return np.where(weights.mask, np.nan_to_num(R, nan=0.0), 0.0).ravel()
    return (np.nan_to_num(R, nan=0.0) * weights.w).ravel()


@dataclass(frozen=True)
class FitStatistic:
    """The goodness-of-fit number, and everything needed to read it.

    Attributes
    ----------
    kind : str
        ``"chi2_red"`` or ``"ssr"``.
    value : float
        The statistic itself. For a weighted fit this is the reduced
        chi-squared with the linear amplitudes counted as parameters.
    value_nonlinear_dof : float or None
        The same chi-squared counting only the non-linear parameters, which is
        the convention many programs report. Kept alongside because the choice
        of denominator changes the number by a large factor and should never be
        implicit.
    n_points, n_nonlinear, n_amplitudes : int
        Points fitted and parameters spent (the amplitudes are ``Np * Nc``).
    dof : int
        ``n_points - n_nonlinear - n_amplitudes``.
    weighted : bool
    noise_source : str or None
    """

    kind: str
    value: float
    n_points: int
    n_nonlinear: int
    n_amplitudes: int
    dof: int
    weighted: bool
    value_nonlinear_dof: float | None = None
    noise_source: str | None = None
    extra: dict = field(default_factory=dict)

    def __str__(self) -> str:
        if self.kind == "chi2_red":
            return (
                f"reduced chi2 = {self.value:.4g} (N = {self.n_points}, "
                f"dof = {self.dof}; nonlinear-dof value {self.value_nonlinear_dof:.4g})"
            )
        return f"SSR = {self.value:.6g} (N = {self.n_points}, unweighted)"


def fit_statistic(
    R,
    weights: Weights,
    *,
    n_nonlinear: int,
    n_amplitudes: int,
) -> FitStatistic:
    """Compute the statistic for a converged (or trial) residual matrix.

    ``n_amplitudes`` is ``Np * Nc``: variable projection solves the amplitudes
    in closed form, but it still spends those degrees of freedom, so they are
    counted in the default reduced chi-squared. The value with only the
    non-linear parameters counted is reported alongside.
    """
    R = np.asarray(R, dtype=float)
    resid = residual_vector(R, weights)
    total = float(np.sum(resid**2))

    n_points = int(weights.n_points)
    dof = n_points - int(n_nonlinear) - int(n_amplitudes)
    dof_nl = n_points - int(n_nonlinear)

    if not weights.weighted:
        return FitStatistic(
            kind="ssr",
            value=total,
            n_points=n_points,
            n_nonlinear=int(n_nonlinear),
            n_amplitudes=int(n_amplitudes),
            dof=dof,
            weighted=False,
        )

    if dof <= 0:
        logger.info(
            "Non-positive degrees of freedom (N = %d, parameters = %d): the reduced "
            "chi-squared is not defined; reporting the unreduced sum instead.",
            n_points,
            int(n_nonlinear) + int(n_amplitudes),
        )
    value = total / dof if dof > 0 else total
    value_nl = total / dof_nl if dof_nl > 0 else total
    return FitStatistic(
        kind="chi2_red",
        value=float(value),
        n_points=n_points,
        n_nonlinear=int(n_nonlinear),
        n_amplitudes=int(n_amplitudes),
        dof=dof,
        weighted=True,
        value_nonlinear_dof=float(value_nl),
        noise_source=weights.source,
        extra={"chi2": total, "dof_nonlinear": dof_nl},
    )
