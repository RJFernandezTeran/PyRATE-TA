"""Ground-state bleach (GSB) and stimulated emission (SE) recovery for pure positive component spectra.

Transient absorption data are difference spectra:
    S_diff(lambda) = S_ESA(lambda) - a * sigma_A(lambda) - b * F(lambda)

where:
    - S_ESA(lambda) >= 0 is the pure excited-state absorption (ESA) spectrum.
    - sigma_A(lambda) >= 0 is the steady-state ground-state absorption spectrum (producing GSB).
    - F(lambda) >= 0 is the steady-state emission spectrum (producing stimulated emission SE).
    - a >= 0 and b >= 0 are the bleach and emission amplitudes per component.

Adding back the scaled reference spectra converts difference spectra to absolute,
pure positive (non-negative) component spectra:
    S_abs[:, i] = S_diff[:, i] + a_i * sigma_A + b_i * F >= 0
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from ..log import get_logger

logger = get_logger(__name__)


class BleachRecoveryResult:
    """Container for ground-state bleach & stimulated emission recovery results.

    Supports attribute access (``.S_abs``, ``.gs_scale``, ``.em_scale``,
    ``.bleach_spectra``, ``.a_max``) as well as 3-item unpacking:
    ``S_abs, a, a_max = compute_absolute_spectra(...)`` or
    ``S_abs, gs_scale, em_scale = compute_absolute_spectra(...)``.
    """

    def __init__(
        self,
        S_abs: np.ndarray,
        gs_scale: np.ndarray | float,
        em_scale: np.ndarray | float,
        bleach_spectra: np.ndarray | None = None,
        a_max: float | None = None,
    ):
        self.S_abs = np.asarray(S_abs, dtype=float)
        self.gs_scale = gs_scale
        self.em_scale = em_scale
        self.bleach_spectra = bleach_spectra
        self.a_max = a_max if a_max is not None else (
            float(np.max(gs_scale)) if np.ndim(gs_scale) > 0 else float(gs_scale)
        )

    def __iter__(self):
        # 3-item unpacking yields (S_abs, gs_scale, third)
        third = self.em_scale if (isinstance(self.em_scale, (int, float, np.number)) and self.em_scale > 0) else self.a_max
        yield self.S_abs
        yield self.gs_scale
        yield third

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index):
        items = (self.S_abs, self.gs_scale, self.em_scale, self.bleach_spectra)
        return items[index]

    def __repr__(self) -> str:
        return (
            f"BleachRecoveryResult(S_abs.shape={self.S_abs.shape}, "
            f"gs_scale={self.gs_scale}, em_scale={self.em_scale})"
        )


def recover_component_bleach(
    s_diff: np.ndarray,
    basis_matrix: np.ndarray,
    *,
    clip_negative: bool = True,
    robust_margin: float = 0.90,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover the minimal non-negative basis coefficients to eliminate negative dips in a spectrum.

    Parameters
    ----------
    s_diff : numpy.ndarray of shape (Np,)
        One component difference spectrum (carrying negative GSB and/or SE dips).
    basis_matrix : numpy.ndarray of shape (Np, K)
        Reference basis spectra (e.g. [sigma_A, F]).
    clip_negative : bool, default True
        Clamp any remaining negative residuals (e.g. noise fluctuations) to zero.
    robust_margin : float, default 0.90
        Quantile used to scale basis profile over the active bleach region,
        avoiding sensitivity to single-pixel noise outliers.

    Returns
    -------
    (coeffs, s_recovered) : tuple
        Fitted coefficients of shape (K,) and the recovered non-negative spectrum of shape (Np,).
    """
    s = np.asarray(s_diff, dtype=float)
    B = np.asarray(basis_matrix, dtype=float)
    K = B.shape[1] if B.ndim == 2 else 1
    if B.ndim == 1:
        B = B[:, None]

    # Non-negative least squares against the inverted negative portion of s_diff
    y_neg = np.maximum(0.0, -s)
    if not np.any(y_neg > 0) or not np.any(B > 0):
        return np.zeros(K, dtype=float), np.maximum(0.0, s) if clip_negative else s.copy()

    coeffs, _ = nnls(B, y_neg)

    # Active bleach region check
    bleach_profile = B @ coeffs
    if np.any(bleach_profile > 0):
        active = bleach_profile > 0.05 * np.max(bleach_profile)
        res = s + bleach_profile
        # If negative dips still exist inside the active bleach window, scale up
        if np.any(active) and np.min(res[active]) < 0:
            ratios = -s[active] / np.maximum(1e-12, bleach_profile[active])
            valid_ratios = ratios[ratios > 1.0]
            if len(valid_ratios) > 0:
                alpha = float(np.percentile(valid_ratios, robust_margin * 100.0))
                coeffs = coeffs * max(1.0, alpha)

    s_recovered = s + B @ coeffs
    if clip_negative:
        s_recovered = np.maximum(0.0, s_recovered)

    return coeffs, s_recovered


def compute_max_gs_scale(
    S_diff: np.ndarray,
    gs_spectrum: np.ndarray,
    f: np.ndarray | float | None = None,
) -> float:
    """Calculate the maximum physical scale factor ``a_max`` for ground state.

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
    S_diff = np.atleast_2d(np.asarray(S_diff, dtype=float))
    gs = np.asarray(gs_spectrum, dtype=float)
    Np, Nc = S_diff.shape

    if f is None:
        f_vec = np.ones(Nc, dtype=float)
    elif np.ndim(f) == 0:
        f_vec = np.full(Nc, float(f), dtype=float)
    else:
        f_vec = np.asarray(f, dtype=float)

    a_max = 0.0
    for i in range(Nc):
        if f_vec[i] <= 0:
            continue
        effective_gs = f_vec[i] * gs
        mask = effective_gs > 1e-12
        if not np.any(mask):
            continue
        ratios = -S_diff[mask, i] / effective_gs[mask]
        valid_ratios = ratios[S_diff[mask, i] < 0]
        if len(valid_ratios) > 0:
            a_max = max(a_max, float(np.max(valid_ratios)))

    return float(a_max) if a_max > 0 else 1.0


def compute_absolute_spectra(
    S_diff: np.ndarray,
    gs_spectrum: np.ndarray | None = None,
    f: np.ndarray | float | None = None,
    gs_scale: float | str | None = "auto",
    *,
    em_spectrum: np.ndarray | None = None,
    em_scale: float | str | None = "auto",
) -> BleachRecoveryResult:
    """Convert difference species spectra to pure positive absolute spectra.

    Accounts for ground-state bleach (GSB) using ``gs_spectrum`` and stimulated
    emission (SE) using ``em_spectrum``.

    Parameters
    ----------
    S_diff : array_like, shape ``(Np, Nc)``
        Difference species spectra from global / target fit.
    gs_spectrum : array_like, shape ``(Np,)``, optional
        Ground-state steady-state absorption spectrum.
    f : array_like or float, optional
        Relative bleach fraction per compartment (default 1.0).
    gs_scale : float or "auto", default "auto"
        Absorption scale factor. If "auto", determines the optimal non-negative
        scale per species to eliminate bleach dips.
    em_spectrum : array_like, shape ``(Np,)``, optional
        Steady-state emission / fluorescence spectrum.
    em_scale : float or "auto", default "auto"
        Emission scale factor. If "auto", determines the optimal non-negative
        scale per species to eliminate stimulated emission dips.

    Returns
    -------
    BleachRecoveryResult
        Container with ``.S_abs``, ``.gs_scale``, ``.em_scale``, ``.bleach_spectra``,
        and ``.a_max``, supporting 3-tuple unpacking for backward compatibility.
    """
    S_diff = np.atleast_2d(np.asarray(S_diff, dtype=float))
    Np, Nc = S_diff.shape

    has_gs = gs_spectrum is not None and np.any(np.asarray(gs_spectrum) > 0)
    has_em = em_spectrum is not None and np.any(np.asarray(em_spectrum) > 0)

    if not has_gs and not has_em:
        return BleachRecoveryResult(
            S_abs=S_diff.copy(),
            gs_scale=0.0 if Nc == 1 else np.zeros(Nc),
            em_scale=0.0 if Nc == 1 else np.zeros(Nc),
            bleach_spectra=np.zeros_like(S_diff),
            a_max=0.0,
        )

    # Fractions per compartment
    if f is None:
        f_vec = np.ones(Nc, dtype=float)
    elif np.ndim(f) == 0:
        f_vec = np.full(Nc, float(f), dtype=float)
    else:
        f_vec = np.asarray(f, dtype=float)

    gs = np.asarray(gs_spectrum, dtype=float) if has_gs else np.zeros(Np)
    a_max = compute_max_gs_scale(S_diff, gs, f_vec) if has_gs else 0.0

    # Fast path: explicit manual scale with absorption only (matches legacy behavior)
    use_auto_gs = str(gs_scale).lower() in ("auto", "none")
    use_auto_em = str(em_scale).lower() in ("auto", "none")

    if has_gs and not has_em and not use_auto_gs:
        a = float(gs_scale)
        S_abs = np.zeros_like(S_diff)
        for i in range(Nc):
            S_abs[:, i] = S_diff[:, i] + f_vec[i] * a * gs
        bleach_spectra = -1.0 * (a * gs[:, None] * f_vec[None, :])
        return BleachRecoveryResult(
            S_abs=S_abs,
            gs_scale=a,
            em_scale=a_max,
            bleach_spectra=bleach_spectra,
            a_max=a_max,
        )

    # If only absorption is present and auto scale is requested
    if has_gs and not has_em and use_auto_gs:
        a = a_max
        S_abs = np.zeros_like(S_diff)
        for i in range(Nc):
            S_abs[:, i] = np.maximum(0.0, S_diff[:, i] + f_vec[i] * a * gs)
        bleach_spectra = -1.0 * (a * gs[:, None] * f_vec[None, :])
        return BleachRecoveryResult(
            S_abs=S_abs,
            gs_scale=a,
            em_scale=a_max,
            bleach_spectra=bleach_spectra,
            a_max=a_max,
        )

    # Multi-component basis recovery (GSB + SE, or SE only)
    basis_columns: list[np.ndarray] = []
    types: list[str] = []

    if has_gs:
        basis_columns.append(gs)
        types.append("gs")

    if has_em:
        em = np.asarray(em_spectrum, dtype=float)
        basis_columns.append(em)
        types.append("em")

    B = np.column_stack(basis_columns)  # (Np, K)

    S_abs = np.zeros_like(S_diff)
    bleach_spectra = np.zeros_like(S_diff)
    gs_scales = np.zeros(Nc, dtype=float)
    em_scales = np.zeros(Nc, dtype=float)

    for i in range(Nc):
        s_i = S_diff[:, i]
        eff_B = B.copy()
        if f_vec[i] != 1.0 and f_vec[i] > 0:
            eff_B = eff_B * f_vec[i]

        if use_auto_gs and use_auto_em:
            coeffs, s_rec = recover_component_bleach(s_i, eff_B, clip_negative=True)
        else:
            coeffs = np.zeros(len(types), dtype=float)
            for k, t in enumerate(types):
                if t == "gs":
                    coeffs[k] = float(gs_scale) if not use_auto_gs else 0.0
                elif t == "em":
                    coeffs[k] = float(em_scale) if not use_auto_em else 0.0
            if f_vec[i] != 1.0 and f_vec[i] > 0:
                coeffs = coeffs * f_vec[i]
            s_rec = np.maximum(0.0, s_i + eff_B @ coeffs)

        S_abs[:, i] = s_rec
        bleach_spectra[:, i] = -1.0 * (eff_B @ coeffs)

        for k, t in enumerate(types):
            if t == "gs":
                gs_scales[i] = coeffs[k]
            elif t == "em":
                em_scales[i] = coeffs[k]

    res_gs_scale = gs_scales[0] if Nc == 1 else gs_scales
    res_em_scale = em_scales[0] if Nc == 1 else em_scales

    return BleachRecoveryResult(
        S_abs=S_abs,
        gs_scale=res_gs_scale,
        em_scale=res_em_scale,
        bleach_spectra=bleach_spectra,
        a_max=a_max,
    )
