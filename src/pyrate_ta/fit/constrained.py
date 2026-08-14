"""Constrained amplitude solver for sign-constrained and bounded species spectra.

When physical constraints are imposed on species spectra (e.g. non-negative
absorption for real excited states/products, or non-positive sign for bleach
contributions), variable projection solves a bounded linear least-squares problem
per pixel rather than an unconstrained linear solve:

.. math:: \\min_{\\mathbf{S}_p} \\lVert \\mathbf{C}_w \\mathbf{S}_p - \\mathbf{d}_{w, p} \\rVert^2
          \\quad \\text{subject to } \\mathbf{l} \\le \\mathbf{S}_p \\le \\mathbf{u}

Used by Phase 6 (Absolute species spectra and sign-constrained fitting).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear, nnls

from ..log import get_logger

logger = get_logger(__name__)


def solve_amplitudes_constrained(
    C: np.ndarray,
    D: np.ndarray,
    weights: np.ndarray | None = None,
    bounds: tuple[np.ndarray | float, np.ndarray | float] | None = None,
    sign_mask: np.ndarray | list[int] | None = None,
) -> np.ndarray:
    """Bounded least-squares component spectra ``S`` of shape ``[Np, Nc]``.

    Parameters
    ----------
    C : array_like, shape ``(Nd, Nc)``
        Concentration matrix.
    D : array_like, shape ``(Nd, Np)``
        Data for one detector, in mOD.
    weights : array_like, optional
        ``1/sigma`` weights, either ``(Nd, Np)``, ``(Nd,)``, or ``None``.
    bounds : tuple of (lb, ub), optional
        Lower and upper bounds for each component amplitude, either floats or
        arrays of shape ``(Nc,)``.
    sign_mask : array_like of int, optional
        Array of length ``Nc`` specifying sign constraints per component:
        ``+1`` for non-negative (>= 0), ``-1`` for non-positive (<= 0), ``0`` for unconstrained.

    Returns
    -------
    numpy.ndarray, shape ``(Np, Nc)``
    """
    C = np.atleast_2d(np.asarray(C, dtype=float))
    D = np.atleast_2d(np.asarray(D, dtype=float))
    Nd, Nc = C.shape
    Np = D.shape[1]

    if sign_mask is not None:
        sign_mask = np.asarray(sign_mask, dtype=int)
        if sign_mask.shape[0] != Nc:
            raise ValueError(f"sign_mask length ({len(sign_mask)}) must match Nc ({Nc})")

    # Determine lower and upper bounds per component
    lb = np.full(Nc, -np.inf)
    ub = np.full(Nc, np.inf)

    if bounds is not None:
        l_in, u_in = bounds
        if np.ndim(l_in) == 0:
            lb[:] = float(l_in)
        else:
            lb[:] = np.asarray(l_in, dtype=float)
        if np.ndim(u_in) == 0:
            ub[:] = float(u_in)
        else:
            ub[:] = np.asarray(u_in, dtype=float)

    if sign_mask is not None:
        for i in range(Nc):
            if sign_mask[i] > 0:  # Non-negative
                lb[i] = max(lb[i], 0.0)
            elif sign_mask[i] < 0:  # Non-positive
                ub[i] = min(ub[i], 0.0)

    # If all bounds are (-inf, inf), delegate to standard unconstrained solve
    if np.all(np.isneginf(lb)) and np.all(np.isposinf(ub)):
        from .varpro import solve_amplitudes
        return solve_amplitudes(C, D, weights)

    # Set up weights
    if weights is None:
        w = np.ones((Nd, Np), dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.ndim == 1:
            w = w[:, None]
        if w.shape[0] != Nd:
            raise ValueError(f"weights shape {w.shape} invalid for C shape {C.shape}")

    # Solve per pixel
    S = np.zeros((Np, Nc), dtype=float)

    # Check if simple non-negative least squares (NNLS) can be used (all lb=0, ub=inf)
    is_nnls = np.all(lb == 0.0) and np.all(np.isposinf(ub))

    for p in range(Np):
        wp = w[:, 0] if w.shape[1] == 1 else w[:, p]
        dp = D[:, p]
        mask = wp > 0
        if not np.any(mask):
            continue

        Cw = wp[mask, None] * C[mask, :]
        dw = wp[mask] * dp[mask]

        if is_nnls:
            sol, _ = nnls(Cw, dw)
            S[p, :] = sol
        else:
            res = lsq_linear(Cw, dw, bounds=(lb, ub))
            S[p, :] = res.x

    return S
