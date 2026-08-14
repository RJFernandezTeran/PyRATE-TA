"""Instrument-response handling for exponential kinetic models.

The response is convolved into the concentration matrix, never into the data,
and analytically rather than numerically: the convolution of a normalised
Gaussian with a decaying exponential is an exponentially modified Gaussian
(EMG), which is both faster and better conditioned than a numerical
convolution on a non-uniform delay axis.

The IRF is specified by its **FWHM** (``w``) and its centre (``t0``), and the
Gaussian is area-normalised,

.. math::

    g(t) = \\frac{2}{w}\\sqrt{\\frac{\\ln 2}{\\pi}}
           \\exp\\!\\left[-4\\ln 2\\left(\\frac{t-t_0}{w}\\right)^2\\right].

Times are in the dataset's own unit throughout; nothing here converts.
"""

from __future__ import annotations

import numpy as np

from ..log import get_logger

logger = get_logger(__name__)

# FWHM = 2 sqrt(2 ln 2) sigma. Kept as a module constant so the two conversion
# helpers cannot drift apart.
_FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def fwhm_to_sigma(fwhm: float) -> float:
    """Gaussian standard deviation from its full width at half maximum."""
    return float(fwhm) / _FWHM_PER_SIGMA


def sigma_to_fwhm(sigma: float) -> float:
    """Full width at half maximum from the Gaussian standard deviation."""
    return float(sigma) * _FWHM_PER_SIGMA


def gaussian_irf(t, t0: float = 0.0, fwhm: float = 1.0):
    """Area-normalised Gaussian instrument response evaluated at ``t``.

    Parameters
    ----------
    t : array_like
        Delays, in the dataset's time unit.
    t0 : float
        Centre of the response (time zero).
    fwhm : float
        Full width at half maximum, same unit as ``t``. Must be positive.

    Returns
    -------
    numpy.ndarray
        The response, normalised so that its integral over ``t`` is 1.
    """
    if not np.isfinite(fwhm) or fwhm <= 0:
        raise ValueError(f"IRF FWHM must be finite and positive, got {fwhm!r}")
    t = np.asarray(t, dtype=float)
    return (
        (2.0 / fwhm)
        * np.sqrt(np.log(2.0) / np.pi)
        * np.exp(-4.0 * np.log(2.0) * ((t - t0) / fwhm) ** 2)
    )


def convolved_exponential(t, rate, t0: float = 0.0, sigma: float = 0.0):
    """Gaussian-convolved exponential :math:`e^{\\lambda (t-t_0)}`.

    This is the EMG

    .. math::

        f(t) = \\tfrac{1}{2}
               \\exp\\!\\left[\\lambda (t-t_0) + \\tfrac{1}{2}\\lambda^2\\sigma^2\\right]
               \\left[1 + \\operatorname{erf}
               \\left(\\frac{t-t_0+\\lambda\\sigma^2}{\\sigma\\sqrt 2}\\right)\\right],

    with ``rate`` the eigenvalue :math:`\\lambda` (negative for a decay).

    Parameters
    ----------
    t : array_like, shape ``(Nd,)``
        Delays.
    rate : array_like, shape ``(Nc,)`` or scalar
        Exponential rate(s) :math:`\\lambda`. Complex values are accepted, so
        that eigenvalues of a non-symmetric rate matrix can be passed straight
        through.
    t0 : float
        Time zero.
    sigma : float
        Gaussian standard deviation. ``0`` returns the unconvolved exponential
        multiplied by the Heaviside step, i.e. the ``sigma -> 0`` limit.

    Returns
    -------
    numpy.ndarray, shape ``(Nd, Nc)``
        One column per rate.

    Notes
    -----
    The naive expression overflows for early delays, where the prefactor
    :math:`\\exp(\\lambda(t-t_0)+\\lambda^2\\sigma^2/2)` is huge and the
    bracket is vanishing. Using
    :math:`z^2 - \\left[\\lambda(t-t_0)+\\lambda^2\\sigma^2/2\\right]
    = (t-t_0)^2/2\\sigma^2` the product can be rewritten with the *scaled*
    complementary error function,

    .. math:: f(t) = \\tfrac{1}{2} e^{-(t-t_0)^2/2\\sigma^2}\\,\\operatorname{erfcx}(-z),

    which is evaluated wherever :math:`-z \\ge 0`; the direct form is used
    elsewhere, where it is the stable one. Both branches are computed and
    selected with :func:`numpy.where`, so the result is free of overflow
    warnings for the whole delay axis.
    """
    from scipy.special import erf, erfcx  # local: keeps ``import pyrate`` cheap

    t = np.asarray(t, dtype=float).reshape(-1, 1)
    lam = np.atleast_1d(np.asarray(rate)).reshape(1, -1)
    dt = t - float(t0)

    if sigma is None or sigma <= 0:
        # sigma -> 0 limit: a plain exponential, zero before time zero.
        # Mask dt < 0 in the exponent to avoid eager overflow evaluation in pre-zero region
        arg = np.where(dt >= 0, lam * dt, -700.0)
        with np.errstate(over="ignore", invalid="ignore"):
            res = np.exp(np.minimum(arg, 700.0))
        return np.where(dt >= 0, res, 0.0)

    s = float(sigma)
    z = (dt + lam * s**2) / (s * np.sqrt(2.0))

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        # Stable branch (early delays): 0.5 exp(-dt^2/2s^2) erfcx(-z)
        stable = 0.5 * np.exp(-(dt**2) / (2.0 * s**2)) * erfcx(-z)
        # Direct branch (late delays), where the prefactor is small.
        direct = 0.5 * np.exp(lam * dt + 0.5 * lam**2 * s**2) * (1.0 + erf(z))

    out = np.where(np.real(z) <= 0.0, stable, direct)
    if not np.all(np.isfinite(out)):
        n_bad = int(np.count_nonzero(~np.isfinite(out)))
        logger.debug("%d non-finite value(s) in the convolved exponential", n_bad)
    return out


#: Labels of the coherent-artefact columns, in the order they are built.
ARTIFACT_LABELS: tuple[str, ...] = ("IRF", "dIRF/dt", "d2IRF/dt2")


def coherent_artifact_basis(t, t0: float = 0.0, fwhm: float | None = None, orders: int = 2):
    """Gaussian and its first ``orders`` derivatives, as extra design columns.

    Around time zero a transient carries signal the kinetic model cannot
    describe: cross-phase modulation and two-photon absorption in the solvent
    and the cell windows. Their time profile is, to a good approximation, the
    instrument response itself and its first two derivatives, so adding those
    three columns to the design matrix lets the fit absorb the artefact into
    amplitudes of its own instead of distorting the shortest lifetime -- which
    is what happens when it is left in.

    The columns are the *shape* only: each is scaled to unit maximum absolute
    value, since its amplitude is solved for per probe pixel like any other
    spectrum. Scaling also keeps the design matrix conditioned -- the second
    derivative is otherwise larger than the Gaussian by ``1/sigma**2``.

    Parameters
    ----------
    t : array_like
        Delays, in the dataset's time unit.
    t0 : float
        Centre of the response.
    fwhm : float
        Width of the response. Required: without a width there is no artefact
        shape to build, and guessing one would invent a component.
    orders : int, default 2
        Highest derivative included; ``2`` gives the usual three columns.

    Returns
    -------
    numpy.ndarray
        ``[Nd, orders + 1]``, columns ordered ``g``, ``g'``, ``g''``.

    Notes
    -----
    The Gaussian-and-derivatives description follows Kovalenko *et al.*,
    Phys. Rev. A **59**, 2369 (1999).
    """
    if fwhm is None or not np.isfinite(fwhm) or float(fwhm) <= 0:
        raise ValueError(
            "the coherent-artefact basis needs a Gaussian IRF width; "
            "enable the Gaussian IRF or switch the artefact off"
        )
    orders = int(orders)
    if orders < 0:
        raise ValueError(f"orders must be >= 0, got {orders}")

    t = np.asarray(t, dtype=float).ravel()
    sigma = fwhm_to_sigma(float(fwhm))
    x = (t - float(t0)) / sigma
    g = np.exp(-0.5 * x**2)

    # Hermite recursion: the n-th derivative of a Gaussian is (-1)^n He_n(x) g/sigma^n.
    columns = [g]
    if orders >= 1:
        columns.append(-x * g / sigma)
    if orders >= 2:
        columns.append((x**2 - 1.0) * g / sigma**2)
    for n in range(3, orders + 1):
        # He_{n} = x He_{n-1} - (n-1) He_{n-2}; the sign alternates with n.
        raise NotImplementedError(f"derivative order {n} is not implemented")

    basis = np.column_stack(columns)
    peaks = np.nanmax(np.abs(basis), axis=0)
    peaks[peaks == 0] = 1.0
    return basis / peaks


def clip_delays(t, t_min: float | None):
    """Boolean mask of the delays kept when clipping at ``t_min``.

    Clipping is the alternative to modelling the IRF (``IRFMode.CLIP``): the
    early delays, where the coherent artefact lives, are dropped instead. It
    changes which data the fit saw, so the caller must record ``t_min`` in the
    result's reproducibility payload.
    """
    t = np.asarray(t, dtype=float)
    if t_min is None:
        return np.ones(t.shape, dtype=bool)
    return t >= float(t_min)
