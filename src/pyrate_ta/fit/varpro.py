"""Variable projection: the linear amplitudes, in closed form.

With the bilinear model ``D ~ C @ S.T``, the spectra ``S`` enter linearly. For
any fixed set of non-linear parameters the best-fit spectra are therefore
available without an optimiser,

.. math:: \\mathbf{S}^{\\mathsf{T}} = \\arg\\min_{\\mathbf{S}}
          \\lVert \\mathbf{W} \\odot (\\mathbf{D} - \\mathbf{C}\\mathbf{S}^{\\mathsf{T}}) \\rVert^2 ,

and only the lifetimes (plus ``t0`` and the IRF width) go to the non-linear
solver. For a 500-pixel three-component fit that is three free parameters
instead of about 1500, and it is far better conditioned than fitting both
together. Amplitudes must never be placed in the optimiser's parameter vector.

Three solve paths, chosen from the weights:

===================  ==========================================================
weights              path
===================  ==========================================================
``None``             one least-squares solve for all pixels at once
constant per delay   rows scaled once, then a single solve
per point            batched QR, one small system per pixel
===================  ==========================================================

The per-point case cannot share a factorisation: each pixel has its own weight
profile down the delay axis, so each has its own normal system. It is solved by
a *batched* QR rather than by forming ``C.T W C`` explicitly, which would square
the condition number of ``C``.

Reference: van Stokkum, Larsen and van Grondelle (2004); see
:mod:`pyrate_ta.cite`.
"""

from __future__ import annotations

import numpy as np

from ..log import get_logger

logger = get_logger(__name__)

# Weight columns within this relative spread are treated as delay-only, so the
# cheap single-solve path can be used.
_UNIFORM_TOL = 1e-12


def _weights_are_delay_only(w) -> bool:
    """True when every pixel shares the same weight profile in time."""
    if w.ndim != 2 or w.shape[1] < 2:
        return True
    first = w[:, :1]
    scale = float(np.max(np.abs(w))) or 1.0
    return bool(np.all(np.abs(w - first) <= _UNIFORM_TOL * scale))


def solve_amplitudes(C, D, weights=None):
    """Least-squares component spectra ``S`` of shape ``[Np, Nc]``.

    Parameters
    ----------
    C : array_like, shape ``(Nd, Nc)``
        Concentration matrix.
    D : array_like, shape ``(Nd, Np)``
        Data for one detector, in mOD.
    weights : array_like, optional
        ``1/sigma`` weights, either ``(Nd, Np)`` per point, ``(Nd,)`` per delay,
        or ``None`` for an unweighted solve. Zero weights drop a point from the
        fit without changing the matrix shapes.

    Returns
    -------
    numpy.ndarray, shape ``(Np, Nc)``

    Raises
    ------
    ValueError
        On inconsistent shapes, or if ``weights`` contains negative or
        non-finite values (a weight is ``1/sigma``; a negative one is a bug in
        the caller, not a datum to be silently repaired).
    """
    C = np.atleast_2d(np.asarray(C, dtype=float))
    D = np.atleast_2d(np.asarray(D, dtype=float))
    if C.shape[0] != D.shape[0]:
        raise ValueError(f"C has {C.shape[0]} delays but D has {D.shape[0]}")
    if not np.all(np.isfinite(C)):
        raise ValueError("C contains non-finite values; the model did not evaluate cleanly")

    if weights is None:
        return _solve_unweighted(C, D)

    w = np.asarray(weights, dtype=float)
    if w.ndim == 1:
        w = w[:, None]
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be finite and non-negative (they are 1/sigma)")
    if w.shape[0] != D.shape[0] or w.shape[1] not in (1, D.shape[1]):
        raise ValueError(f"weights shape {w.shape} does not match D shape {D.shape}")

    # A masked-out point (w = 0) must not poison the solve through a NaN in D.
    D = np.where(w > 0, np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

    if w.shape[1] == 1 or _weights_are_delay_only(w):
        return _solve_unweighted(w[:, :1] * C, w[:, :1] * D)
    return _solve_per_pixel(C, D, w)


def _solve_unweighted(C, D):
    """One least-squares solve for every pixel at once."""
    from scipy.linalg import lstsq

    S_T, _, rank, _ = lstsq(C, D, lapack_driver="gelsd")
    if rank < C.shape[1]:
        logger.info(
            "Concentration matrix is rank deficient (rank %d < %d components): the "
            "amplitudes are not uniquely determined. Consider fewer components.",
            rank,
            C.shape[1],
        )
    return np.asarray(S_T).T


def _solve_per_pixel(C, D, w):
    """Batched weighted solve: one small QR per pixel.

    ``Cw`` is ``[Np, Nd, Nc]`` -- the concentration matrix scaled by each
    pixel's own weights -- so ``numpy.linalg.qr`` factorises all of them in one
    call. Memory is ``Np * Nd * Nc`` floats, a few MB for a typical dataset.
    """
    Cw = w.T[:, :, None] * C[None, :, :]  # [Np, Nd, Nc]
    Dw = (w * D).T  # [Np, Nd]

    Q, R = np.linalg.qr(Cw)
    rhs = np.einsum("pnc,pn->pc", Q, Dw)
    # A pixel whose weights are all zero gives R = 0; solving it would raise.
    diag = np.abs(np.einsum("pcc->pc", R))
    singular = np.any(diag <= 0, axis=1)
    S = np.zeros((D.shape[1], C.shape[1]), dtype=float)
    good = ~singular
    if np.any(good):
        # ``solve`` with a stacked 2-D right-hand side is read as a matrix, not
        # as a stack of vectors, so the column axis is made explicit.
        S[good] = np.linalg.solve(R[good], rhs[good][..., None])[..., 0]
    if np.any(singular):
        logger.info(
            "%d pixel(s) carry no usable weight and were given zero amplitudes.",
            int(np.count_nonzero(singular)),
        )
    return S


def project(C, D, weights=None):
    """Amplitudes and residuals in one step.

    Returns
    -------
    (S, R) : tuple of numpy.ndarray
        ``S`` ``[Np, Nc]`` and the **unweighted** residual matrix
        ``R = D - C @ S.T`` ``[Nd, Np]``. Residuals are returned unweighted so
        the caller decides whether the weights belong in the cost function or
        only in the statistic; :mod:`pyrate_ta.fit.cost` applies them.
    """
    C = np.atleast_2d(np.asarray(C, dtype=float))
    D = np.atleast_2d(np.asarray(D, dtype=float))
    S = solve_amplitudes(C, D, weights)
    return S, D - C @ S.T
