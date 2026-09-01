"""Lifetime Density Analysis (LDA) solver module.

Instead of fitting N discrete lifetimes, LDA solves for a continuous distribution
of amplitudes over a dense, logarithmically spaced grid of fixed lifetimes:

.. math:: \\min_{\\mathbf{S}^{\\mathsf{T}}} \\lVert \\mathbf{W} \\odot (\\mathbf{D} - \\mathbf{C}(\\boldsymbol{\\tau}_{grid}) \\mathbf{S}^{\\mathsf{T}}) \\rVert^2 + \\alpha^2 \\lVert \\mathbf{L} \\mathbf{S}^{\\mathsf{T}} \\rVert^2

Supports Tikhonov (ridge L=I), 1st-derivative (L=D1), and 2nd-derivative (L=D2)
smoothness penalties, with L-curve corner detection and GCV alpha selection.
"""

from __future__ import annotations

import numpy as np

from ..cite import cite
from ..log import get_logger
from ..models import ParallelModel
from ..results.lda_result import LDAResult

logger = get_logger(__name__)


def build_penalty_matrix(M: int, penalty: str = "d2") -> np.ndarray:
    """Build regularisation penalty matrix L of shape (K, M).

    Parameters
    ----------
    M : int
        Number of grid points in the lifetime axis.
    penalty : {"ridge", "identity", "d1", "d2"}
        Penalty type.

    Returns
    -------
    numpy.ndarray
    """
    p = penalty.lower()
    if p in ("ridge", "identity", "l0"):
        return np.eye(M, dtype=float)
    elif p in ("d1", "1st_derivative"):
        L = np.zeros((M - 1, M), dtype=float)
        for i in range(M - 1):
            L[i, i] = -1.0
            L[i, i + 1] = 1.0
        return L
    elif p in ("d2", "2nd_derivative", "smoothness"):
        L = np.zeros((M - 2, M), dtype=float)
        for i in range(M - 2):
            L[i, i] = 1.0
            L[i, i + 1] = -2.0
            L[i, i + 2] = 1.0
        return L
    else:
        raise ValueError(f"Unknown penalty matrix type: {penalty!r}")


def _find_l_curve_corner(log_res: np.ndarray, log_norm: np.ndarray, alphas: np.ndarray) -> int:
    """Find index of maximum curvature on the L-curve (log_res vs log_norm).

    Uses the 2D parametric curvature formula kappa(s) = (x'(s) y''(s) - y'(s) x''(s)) / (x'(s)^2 + y'(s)^2)^(3/2)
    parameterized by s = log10(alpha), where alpha values are log-spaced.
    """
    s = np.log10(np.asarray(alphas, dtype=float))
    x = np.asarray(log_res, dtype=float)
    y = np.asarray(log_norm, dtype=float)

    if len(s) < 5:
        return int(np.argmin(x**2 + y**2))

    # Compute 1st and 2nd numerical derivatives with respect to s = log10(alpha)
    dx = np.gradient(x, s)
    dy = np.gradient(y, s)
    ddx = np.gradient(dx, s)
    ddy = np.gradient(dy, s)

    # 2D parametric curvature: kappa = (x' * y'' - y' * x'') / (x'^2 + y'^2)^(3/2)
    denom = (dx**2 + dy**2) ** 1.5
    denom = np.where(denom <= 1e-12, 1e-12, denom)
    curvature = (dx * ddy - dy * ddx) / denom

    # The L-curve corner corresponds to the point of maximum geometric curvature
    corner_idx = int(np.argmax(curvature))
    return corner_idx


def solve_lda(
    data,
    t=None,
    *,
    n_taus: int = 100,
    tau_range: tuple[float, float] | None = None,
    irf_fwhm: float | None = None,
    t0: float = 0.0,
    penalty: str = "d2",
    alpha: float | str = "auto",
    alpha_method: str = "lcurve",
    alphas: np.ndarray | None = None,
    detector: int = 0,
    coherent_artifact: bool = False,
    svd_components: int = 0,
    non_negative: bool = False,
    n_bootstraps: int = 0,
    find_peaks: bool = False,
) -> LDAResult:
    """Run Lifetime Density Analysis (LDA) on time-resolved data.

    Parameters
    ----------
    data : Dataset1D or array_like (Nd, Np)
        Data matrix or PyMORGAN dataset.
    t : array_like, optional
        Delay axis if data is plain arrays.
    n_taus : int, default 100
        Number of log-spaced lifetimes in grid.
    tau_range : tuple of (tau_min, tau_max), optional
        Grid range. Defaults to [min_t/2, 5*max_t].
    irf_fwhm : float, optional
        Gaussian IRF width.
    t0 : float, default 0.0
        Time zero offset.
    penalty : {"d2", "d1", "ridge"}, default "d2"
        Regularisation operator.
    alpha : float or "auto"
        Regularisation parameter. If "auto", selected via `alpha_method`.
    alpha_method : {"lcurve", "gcv", "morozov"}, default "lcurve"
        Method for auto-selecting alpha.
    alphas : array_like, optional
        Array of candidate alpha values to scan.
    coherent_artifact : bool, default False
        Include IRF and derivative basis columns at t0 to absorb solvent artefact.
    svd_components : int, default 0
        Pre-filter raw data matrix using top K SVD components to suppress noise (0 = Off).
    non_negative : bool, default False
        Enforce non-negative amplitudes S >= 0 via regularised Non-Negative Least Squares (NNLS).
    n_bootstraps : int, default 0
        Number of Monte Carlo bootstrap iterations for confidence intervals on A(tau) (0 = Off).
    find_peaks : bool, default False
        Automatically locate major lifetime peak centroids from A(tau).

    Returns
    -------
    LDAResult
    """
    from .engines import _as_problem

    prob = _as_problem(data, t=t, detector=detector)
    t_arr = prob.t
    D = prob.D
    Nd, Np = D.shape
    probe = prob.probe

    # Optional SVD Noise Pre-Filtering
    if svd_components is not None and svd_components > 0:
        k_comp = min(int(svd_components), Nd, Np)
        U_svd, s_svd, Vt_svd = np.linalg.svd(D, full_matrices=False)
        D_fit = U_svd[:, :k_comp] @ np.diag(s_svd[:k_comp]) @ Vt_svd[:k_comp, :]
    else:
        D_fit = D

    if tau_range is None:
        t_pos = t_arr[t_arr > 0]
        t_min_pos = float(t_pos.min()) if len(t_pos) > 0 else 0.1
        t_max_pos = float(t_arr.max())
        tau_range = (max(0.01, t_min_pos * 0.5), t_max_pos * 5.0)

    tau_grid = np.geomspace(tau_range[0], tau_range[1], n_taus)

    # Build Parallel model concentration matrix for tau_grid
    model = ParallelModel(n_components=n_taus, fit_t0=False, fit_irf=False)
    C = model.concentrations(t_arr, taus=tau_grid, t0=t0, irf_fwhm=irf_fwhm, check_degenerate=False)  # (Nd, n_taus)

    if coherent_artifact:
        from ..models.irf import coherent_artifact_basis
        fwhm_val = irf_fwhm if irf_fwhm is not None and irf_fwhm > 0 else 0.1
        C_ca = coherent_artifact_basis(t_arr, t0=t0, fwhm=fwhm_val, orders=2)  # (Nd, 3)
        C_fit = np.hstack([C, C_ca])  # (Nd, n_taus + 3)
        n_extra = 3
    else:
        C_fit = C
        n_extra = 0

    L = build_penalty_matrix(n_taus, penalty=penalty)  # (K, n_taus)
    if n_extra > 0:
        L_fit = np.hstack([L, np.zeros((L.shape[0], n_extra))])  # (K, n_taus + 3)
    else:
        L_fit = L

    K_row = L_fit.shape[0]

    if alphas is None:
        alphas_scan = np.geomspace(1e-5, 1e2, 60)
    else:
        alphas_scan = np.asarray(alphas, dtype=float)

    log_residuals = []
    log_solution_norms = []
    gcv_scores = []
    solutions = []

    CtC = C_fit.T @ C_fit
    LtL = L_fit.T @ L_fit
    M_full = C_fit.shape[1]

    for a_val in alphas_scan:
        # Build augmented system: [ C_fit ; a_val * L_fit ] S_full^T = [ D_fit ; 0 ]
        C_aug = np.vstack([C_fit, a_val * L_fit])  # (Nd + K_row, n_taus + n_extra)
        D_aug = np.vstack([D_fit, np.zeros((K_row, Np))])  # (Nd + K_row, Np)

        # Solve linear least squares S_full_T shape (n_taus + n_extra, Np)
        S_full_T, _, _, _ = np.linalg.lstsq(C_aug, D_aug, rcond=None)
        S_map = S_full_T[:n_taus, :].T  # (Np, n_taus) -> lifetime density grid only!
        solutions.append(S_map)

        # Residual R = D_fit - C_fit @ S_full_T
        R = D_fit - C_fit @ S_full_T
        res_norm = float(np.linalg.norm(R))
        sol_norm = float(np.linalg.norm(L @ S_full_T[:n_taus, :]))

        log_residuals.append(np.log10(max(res_norm, 1e-12)))
        log_solution_norms.append(np.log10(max(sol_norm, 1e-12)))

        # Generalized Cross-Validation (GCV) formula
        A_mat = CtC + (a_val**2) * LtL
        try:
            G_mat = np.linalg.solve(A_mat, np.eye(M_full))
            tr_H = float(np.trace(G_mat @ CtC))
        except np.linalg.LinAlgError:
            G_mat = np.linalg.pinv(A_mat)
            tr_H = float(np.trace(G_mat @ CtC))
        denom = max(1.0 - tr_H / Nd, 1e-4) ** 2
        gcv_score = (res_norm**2 / (Nd * Np)) / denom
        gcv_scores.append(gcv_score)

    log_residuals = np.asarray(log_residuals)
    log_solution_norms = np.asarray(log_solution_norms)
    gcv_scores = np.asarray(gcv_scores)

    if isinstance(alpha, (int, float)):
        chosen_idx = int(np.argmin(np.abs(alphas_scan - float(alpha))))
        alpha_opt = float(alpha)
        method_used = "manual"
    elif alpha_method == "gcv":
        chosen_idx = int(np.argmin(gcv_scores))
        alpha_opt = float(alphas_scan[chosen_idx])
        method_used = "gcv"
    elif alpha_method == "morozov":
        neg_mask = t_arr < t0
        if np.sum(neg_mask) >= 3:
            sigma_noise = float(np.std(D[neg_mask, :]))
        else:
            sigma_noise = float(np.std(D - C @ solutions[0].T))
        target_res = sigma_noise * np.sqrt(Nd * Np)
        res_norms = 10**log_residuals
        chosen_idx = int(np.argmin(np.abs(res_norms - target_res)))
        alpha_opt = float(alphas_scan[chosen_idx])
        method_used = "morozov"
    else:  # lcurve
        chosen_idx = _find_l_curve_corner(log_residuals, log_solution_norms, alphas_scan)
        alpha_opt = float(alphas_scan[chosen_idx])
        method_used = "lcurve"

    # Re-solve optimal solution with Non-Negative constraint if requested
    C_aug_opt = np.vstack([C_fit, alpha_opt * L_fit])
    D_aug_opt = np.vstack([D_fit, np.zeros((K_row, Np))])

    if non_negative:
        if n_extra > 0:
            from scipy.optimize import lsq_linear

            lb = np.concatenate([np.zeros(n_taus), -np.full(n_extra, np.inf)])
            ub = np.full(n_taus + n_extra, np.inf)
            opt_S_full_T = np.zeros((C_fit.shape[1], Np))
            for j in range(Np):
                res_j = lsq_linear(C_aug_opt, D_aug_opt[:, j], bounds=(lb, ub))
                opt_S_full_T[:, j] = res_j.x
        else:
            from scipy.optimize import nnls

            opt_S_full_T = np.zeros((C_fit.shape[1], Np))
            for j in range(Np):
                sol_j, _ = nnls(C_aug_opt, D_aug_opt[:, j])
                opt_S_full_T[:, j] = sol_j
    else:
        opt_S_full_T, _, _, _ = np.linalg.lstsq(C_aug_opt, D_aug_opt, rcond=None)

    opt_S_map = opt_S_full_T[:n_taus, :].T
    opt_R = D - C_fit @ opt_S_full_T

    # Bootstrap Confidence Resampling
    bootstrap_std = None
    if n_bootstraps > 0 and n_bootstraps <= 500:
        A_boots = []
        for _ in range(int(n_bootstraps)):
            boot_idx = np.random.choice(Nd, size=Nd, replace=True)
            D_boot = D_fit + opt_R[boot_idx, :]
            D_aug_b = np.vstack([D_boot, np.zeros((K_row, Np))])
            if non_negative:
                if n_extra > 0:
                    from scipy.optimize import lsq_linear

                    lb = np.concatenate([np.zeros(n_taus), -np.full(n_extra, np.inf)])
                    ub = np.full(n_taus + n_extra, np.inf)
                    S_b_T = np.zeros((C_fit.shape[1], Np))
                    for j in range(Np):
                        r_j = lsq_linear(C_aug_opt, D_aug_b[:, j], bounds=(lb, ub))
                        S_b_T[:, j] = r_j.x
                else:
                    from scipy.optimize import nnls

                    S_b_T = np.zeros((C_fit.shape[1], Np))
                    for j in range(Np):
                        s_j, _ = nnls(C_aug_opt, D_aug_b[:, j])
                        S_b_T[:, j] = s_j
            else:
                S_b_T, _, _, _ = np.linalg.lstsq(C_aug_opt, D_aug_b, rcond=None)
            S_map_b = S_b_T[:n_taus, :].T
            A_b = np.sum(np.abs(S_map_b), axis=0)
            A_boots.append(A_b)
        bootstrap_std = np.std(np.array(A_boots), axis=0)

    # Peak Centroid Auto-Detection
    detected_peaks = []
    if find_peaks:
        from scipy.signal import find_peaks as sc_find_peaks

        A_opt = np.sum(np.abs(opt_S_map), axis=0)
        p_indices, _ = sc_find_peaks(A_opt, prominence=0.05 * (np.max(A_opt) if np.max(A_opt) > 0 else 1.0))
        for idx_p in p_indices:
            detected_peaks.append(
                {
                    "tau": float(tau_grid[idx_p]),
                    "amplitude": float(A_opt[idx_p]),
                    "index": int(idx_p),
                }
            )

    l_curve_pts = np.column_stack([log_residuals, log_solution_norms])
    units = dict(getattr(data, "units", {}) or {})

    cite("optimus2015")
    if method_used == "lcurve":
        cite("hansen1992")

    return LDAResult(
        tau_grid=tau_grid,
        S_map=opt_S_map,
        alpha_opt=alpha_opt,
        penalty=penalty,
        alpha_method=method_used,
        l_curve_points=l_curve_pts,
        alphas=alphas_scan,
        gcv_scores=gcv_scores,
        residuals=opt_R,
        t=t_arr,
        probe=probe,
        irf_fwhm=irf_fwhm,
        t0=t0,
        svd_components=int(svd_components) if svd_components else 0,
        non_negative=bool(non_negative),
        n_bootstraps=int(n_bootstraps) if n_bootstraps else 0,
        bootstrap_std=bootstrap_std,
        peaks=detected_peaks,
        units=units,
    )
