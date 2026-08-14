"""Ground-state bleach recovery and absolute species spectra computation.

Transient absorption and vibrational data are difference spectra:
``S_diff = S_excited - f * GS``. Adding back a scaled ground-state spectrum
``GS`` converts them to absolute absorption spectra:

.. math:: \\mathbf{S}_{\\text{abs}}[:, i] = \\mathbf{S}_{\\text{diff}}[:, i] + f_i \\cdot a \\cdot \\mathbf{GS}

where ``a`` is the ground-state scale factor and ``f_i`` is the fraction of ground
state bleached by species ``i``.
"""

from __future__ import annotations

import numpy as np

from ..log import get_logger

logger = get_logger(__name__)


def compute_max_gs_scale(
    S_diff: np.ndarray,
    gs_spectrum: np.ndarray,
    f: np.ndarray | float | None = None,
) -> float:
    """Calculate the maximum physical scale factor ``a_max`` for ground state.

    ``a_max`` is the largest value of ``a`` before any pixel in any species'
    absolute spectrum ``S_abs = S_diff + f * a * GS`` becomes negative.

    Parameters
    ----------
    S_diff : array_like, shape ``(Np, Nc)``
        Difference species spectra.
    gs_spectrum : array_like, shape ``(Np,)``
        Ground-state absorption spectrum on the same probe grid.
    f : array_like or float, optional
        Ground state bleach fraction per species (defaults to 1.0 for all).

    Returns
    -------
    float
        Maximum allowable scale factor ``a_max >= 0``.
    """
    S_diff = np.atleast_2d(np.asarray(S_diff, dtype=float))  # [Np, Nc]
    gs = np.asarray(gs_spectrum, dtype=float)  # [Np]
    Np, Nc = S_diff.shape

    if f is None:
        f_vec = np.ones(Nc, dtype=float)
    elif np.ndim(f) == 0:
        f_vec = np.full(Nc, float(f), dtype=float)
    else:
        f_vec = np.asarray(f, dtype=float)

    # Where f * GS > 0, S_diff + f * a * GS >= 0 implies a >= -S_diff / (f * GS)
    # We find the maximum lower bound across all probe points and species where S_diff < 0
    a_max = np.inf

    for i in range(Nc):
        if f_vec[i] <= 0:
            continue
        effective_gs = f_vec[i] * gs  # [Np]
        # Only check where ground state absorption is positive
        mask = effective_gs > 1e-12
        if not np.any(mask):
            continue
        # Ratio needed to bring S_diff to 0
        ratios = -S_diff[mask, i] / effective_gs[mask]
        # Any negative S_diff needs a positive a to become non-negative
        valid_ratios = ratios[ratios >= 0]
        if len(valid_ratios) > 0:
            a_needed_max = np.max(valid_ratios)
            # The upper bound before another region turns negative:
            # S_diff + a * GS >= 0 where S_diff < 0
            # If S_diff is negative, we MUST add at least a_needed_max
            a_max = min(a_max, a_needed_max)

    if np.isinf(a_max):
        return 1.0
    return float(max(0.0, a_max))


def compute_absolute_spectra(
    S_diff: np.ndarray,
    gs_spectrum: np.ndarray,
    f: np.ndarray | float | None = None,
    gs_scale: float | str | None = "auto",
) -> tuple[np.ndarray, float, float]:
    """Convert difference species spectra to absolute species absorption spectra.

    Parameters
    ----------
    S_diff : array_like, shape ``(Np, Nc)``
        Difference species spectra from global / target fit.
    gs_spectrum : array_like, shape ``(Np,)``
        Ground-state absorption spectrum.
    f : array_like or float, optional
        Bleach fraction per compartment (default 1.0).
    gs_scale : float or "auto", optional
        Scale factor ``a``. If "auto", determines the optimal scale factor ``a_max``.

    Returns
    -------
    S_abs : numpy.ndarray, shape ``(Np, Nc)``
        Absolute absorption species spectra.
    a : float
        Applied scale factor.
    a_max : float
        Maximum allowable scale factor.
    """
    S_diff = np.atleast_2d(np.asarray(S_diff, dtype=float))
    gs = np.asarray(gs_spectrum, dtype=float)
    Np, Nc = S_diff.shape

    if f is None:
        f_vec = np.ones(Nc, dtype=float)
    elif np.ndim(f) == 0:
        f_vec = np.full(Nc, float(f), dtype=float)
    else:
        f_vec = np.asarray(f, dtype=float)

    a_max = compute_max_gs_scale(S_diff, gs, f_vec)

    if gs_scale == "auto" or gs_scale is None:
        a = a_max
    else:
        a = float(gs_scale)

    S_abs = np.zeros_like(S_diff)
    for i in range(Nc):
        S_abs[:, i] = S_diff[:, i] + f_vec[i] * a * gs

    return S_abs, a, a_max
