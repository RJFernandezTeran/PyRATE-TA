"""Propagation of a first-order kinetic system, with or without an IRF.

One propagator serves every model family. Given a rate matrix ``K`` and initial
concentrations ``c0``, the populations obey

.. math:: \\frac{\\mathrm{d}\\mathbf{c}}{\\mathrm{d}t} = \\mathbf{K}\\mathbf{c},
          \\qquad \\mathbf{c}(t) = e^{\\mathbf{K}t}\\,\\mathbf{c}_0 .

The eigenvalue route (Berberan-Santos and Martinho) is used where ``K`` is
diagonalisable: with :math:`\\mathbf{K} = \\mathbf{V}\\Lambda\\mathbf{V}^{-1}`
and :math:`\\mathbf{a} = \\mathbf{V}^{-1}\\mathbf{c}_0`,

.. math:: \\mathbf{c}(t) = \\mathbf{V}\\left(\\mathbf{a} \\odot e^{\\Lambda t}\\right),

which is what makes the analytic IRF convolution possible at all: each
exponential is convolved separately (:mod:`pyrate_ta.models.irf`) and the same
linear combination is taken afterwards. Where ``K`` is defective or its
eigenvector matrix is ill-conditioned, the unconvolved case falls back to a
matrix-exponential propagator.

Reference: M. N. Berberan-Santos and J. M. G. Martinho, *J. Chem. Educ.* **67**
(1990) 375 -- the eigenvector solution of coupled first-order systems.
"""

from __future__ import annotations

import numpy as np

from ..cite import cite
from ..log import get_logger
from .irf import convolved_exponential, fwhm_to_sigma

logger = get_logger(__name__)

# Above this condition number of the eigenvector matrix, the eigen route is
# considered unreliable: amplitudes of opposite sign start to blow up while the
# residual barely moves.
_COND_LIMIT = 1e8

# Relative size of the diagonal nudge used to split degenerate eigenvalues; see
# :func:`_split_degeneracy`. Chosen far below any experimental noise level.
_SPLIT_EPS = 1e-6


def _split_degeneracy(K, eps: float = _SPLIT_EPS):
    """Nudge a degenerate rate matrix until its eigenvectors are usable.

    Two identical (or near-identical) rates make ``K`` defective: the
    eigenvector matrix is singular and the eigenvalue solution does not exist,
    even though the physical kinetics are perfectly well behaved. The fix is a
    tiny, antisymmetric perturbation of the diagonal, of relative size ``eps``,
    which separates the eigenvalues while changing the dynamics by a part in a
    million -- orders of magnitude below the noise on any real measurement.

    This matters during a fit, not only for a pathological model: an optimiser
    exploring parameter space routinely passes through equal lifetimes on its
    way to the optimum, and refusing to evaluate there would abort perfectly
    good fits. The perturbation is logged, so a *result* that depended on it is
    still visible to the user.

    Returns
    -------
    (K_split, lam, V, cond)
        The perturbed matrix and its decomposition, or the original values when
        the nudge did not help.
    """
    K = np.asarray(K, dtype=float)
    n = K.shape[0]
    if n < 2:
        return K, *eigen_decomposition(K)
    scale = float(np.max(np.abs(np.diag(K)))) or 1.0
    # Antisymmetric about the centre, so the mean rate is preserved.
    offsets = (np.arange(n) - (n - 1) / 2.0) * (eps * scale)
    K_split = K + np.diag(offsets)
    lam, V, cond = eigen_decomposition(K_split)
    return K_split, lam, V, cond


def eigen_decomposition(K):
    """Eigenvalues and eigenvectors of ``K``, with the conditioning reported.

    Returns
    -------
    (lam, V, cond) : tuple
        Eigenvalues ``lam`` ``[Nc]``, eigenvectors ``V`` ``[Nc, Nc]`` in
        columns, and the 2-norm condition number of ``V``. Complex results are
        returned as-is; the caller decides when to take the real part.
    """
    from scipy.linalg import eig

    K = np.asarray(K, dtype=float)
    lam, V = eig(K)
    try:
        cond = float(np.linalg.cond(V))
    except np.linalg.LinAlgError:  # pragma: no cover - only for singular V
        logger.debug("eigenvector matrix condition number unavailable", exc_info=True)
        cond = np.inf
    return lam, V, cond


def check_degeneracy(lifetimes, warn_ratio: float | None = None) -> list[tuple[int, int]]:
    """Report pairs of lifetimes closer together than ``warn_ratio``.

    Two lifetimes within a small factor of each other make the eigenvector
    matrix ill-conditioned; the symptom in a fit is a pair of amplitudes of
    opposite sign that grow without bound while the residual barely improves.
    The pairs are logged at ``info`` level -- the user should see them -- and
    returned so a fit result can carry them.
    """
    if warn_ratio is None:
        from ..settings import get_settings

        warn_ratio = get_settings().degeneracy_warn_ratio
    taus = np.atleast_1d(np.asarray(lifetimes, dtype=float))
    pairs: list[tuple[int, int]] = []
    for i in range(taus.size):
        for j in range(i + 1, taus.size):
            a, b = abs(taus[i]), abs(taus[j])
            if a <= 0 or b <= 0 or not np.isfinite(a) or not np.isfinite(b):
                continue  # a non-decaying component has no lifetime to compare
            ratio = max(a, b) / min(a, b)
            if ratio < warn_ratio:
                pairs.append((i, j))
                logger.info(
                    "Lifetimes %d (%.4g) and %d (%.4g) differ by a factor of only %.3f "
                    "(< %.3f): the model is close to degenerate and the amplitudes are "
                    "poorly determined. Consider one component fewer.",
                    i + 1,
                    taus[i],
                    j + 1,
                    taus[j],
                    ratio,
                    warn_ratio,
                )
    return pairs


def concentrations(
    K,
    c0,
    t,
    *,
    t0: float = 0.0,
    irf_fwhm: float | None = None,
    check_degenerate: bool = True,
):
    """Concentration matrix ``C(t)`` of shape ``[Nd, Nc]``.

    Parameters
    ----------
    K : array_like, shape ``(Nc, Nc)``
        Rate matrix, ``dc/dt = K c``. Off-diagonal ``K[i, j]`` is the rate from
        compartment ``j`` to compartment ``i``; the diagonal is minus the sum
        of every rate leaving that compartment.
    c0 : array_like, shape ``(Nc,)``
        Initial concentrations at ``t0``.
    t : array_like, shape ``(Nd,)``
        Delays, in the dataset's time unit.
    t0 : float
        Time zero.
    irf_fwhm : float or None
        Gaussian IRF width. ``None`` (or 0) gives the unconvolved solution,
        zero before ``t0``.
    check_degenerate : bool
        Run :func:`check_degeneracy` on the eigenvalue lifetimes and log any
        near-degenerate pair.

    Returns
    -------
    numpy.ndarray, shape ``(Nd, Nc)``

    Raises
    ------
    ValueError
        If the shapes are inconsistent, or if an IRF is requested for a matrix
        whose eigendecomposition is unusable (there is no analytic convolution
        in that case, and silently returning the unconvolved solution would be
        a wrong result rather than a slow one).
    """
    K = np.atleast_2d(np.asarray(K, dtype=float))
    c0 = np.atleast_1d(np.asarray(c0, dtype=float)).ravel()
    t = np.atleast_1d(np.asarray(t, dtype=float)).ravel()

    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be square, got shape {K.shape}")
    if c0.size != K.shape[0]:
        raise ValueError(f"c0 has {c0.size} entries but K is {K.shape[0]}x{K.shape[0]}")

    lam, V, cond = eigen_decomposition(K)
    usable = np.isfinite(cond) and cond < _COND_LIMIT

    if check_degenerate:
        rates = np.abs(np.real(lam))
        with np.errstate(divide="ignore"):
            taus = np.where(rates > 0, 1.0 / np.where(rates > 0, rates, 1.0), np.inf)
        check_degeneracy(taus[np.isfinite(taus)])

    convolve = irf_fwhm is not None and float(irf_fwhm) > 0

    if not usable:
        # Degenerate rates are a coordinate problem, not a physical one: split
        # them by a part in a million and try again. Only if that fails is the
        # matrix genuinely unusable.
        K_split, lam_s, V_s, cond_s = _split_degeneracy(K)
        if cond_s < _COND_LIMIT:
            logger.debug(
                "Degenerate rate matrix (cond = %.3g) split by %.0e; cond is now %.3g.",
                cond,
                _SPLIT_EPS,
                cond_s,
            )
            lam, V, cond, usable = lam_s, V_s, cond_s, True

    if not usable:
        if convolve:
            raise ValueError(
                "the eigenvector matrix of K is ill-conditioned "
                f"(cond = {cond:.3g}) and could not be split; the analytic IRF "
                "convolution is not applicable. Reduce the number of components, "
                "or fit without an IRF."
            )
        logger.info(
            "Eigenvector matrix ill-conditioned (cond = %.3g); propagating with the "
            "matrix exponential instead.",
            cond,
        )
        return _expm_propagate(K, c0, t, t0)

    # The eigenvector solution is the published method actually being used
    # here, so it names its source the first time it runs.

    cite("berberansantos1990")

    a = np.linalg.solve(V, c0.astype(complex))
    sigma = fwhm_to_sigma(float(irf_fwhm)) if convolve else 0.0
    # [Nd, Nc] of convolved (or plain, for sigma = 0) exponentials, one per mode.
    modes = convolved_exponential(t, lam, t0=t0, sigma=sigma)
    C = (modes * a.reshape(1, -1)) @ V.T

    imag = float(np.max(np.abs(np.imag(C)))) if np.iscomplexobj(C) else 0.0
    if imag > 1e-8 * max(1.0, float(np.max(np.abs(np.real(C))))):
        logger.debug("discarding an imaginary part of %.3g in C(t)", imag)
    return np.real(C)


def _expm_propagate(K, c0, t, t0: float):
    """Fallback propagation by matrix exponential, one delay at a time.

    Used only when the eigenvector matrix is unusable. Delays before ``t0`` are
    zero, matching the Heaviside convention of the unconvolved solution.
    """
    from scipy.linalg import expm

    C = np.zeros((t.size, c0.size), dtype=float)
    for i, ti in enumerate(t):
        dt = ti - t0
        if dt < 0:
            continue
        C[i, :] = expm(K * dt) @ c0
    return C
