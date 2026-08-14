"""Plotting routines for Lifetime Density Analysis (LDA)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..results.lda_result import LDAResult


def _get_probe_xlabel(result: LDAResult) -> str:
    """Format probe axis label matching parent dataset units & PyMORGAN style."""
    units = getattr(result, "units", {}) or {}
    if not units and isinstance(getattr(result, "settings", None), dict):
        units = result.settings.get("units", {})
    if isinstance(units, dict) and units:
        lbl = units.get("unitsL_lbl") or units.get("probe_name") or "Probe"
        unit = units.get("unitsL_ltx") or units.get("probe_unit") or ""
        if lbl and unit:
            style = "()"
            try:
                import pymorgan as pm
                style = getattr(pm.get_settings(), "label_style", "()")
            except Exception:
                pass
            if style == "[]":
                return f"{lbl} [{unit}]"
            elif style == "/":
                return f"{lbl} / {unit}"
            else:
                return f"{lbl} ({unit})"
        elif lbl:
            return str(lbl)
    return "Probe"


def _get_time_ylabel(result: LDAResult) -> str:
    """Format time/lifetime Y axis label matching parent dataset units & PyMORGAN style."""
    units = getattr(result, "units", {}) or {}
    if not units and isinstance(getattr(result, "settings", None), dict):
        units = result.settings.get("units", {})
    if isinstance(units, dict) and units:
        unit = units.get("unitsT_ltx") or units.get("time_unit") or ""
        if unit:
            style = "()"
            try:
                import pymorgan as pm
                style = getattr(pm.get_settings(), "label_style", "()")
            except Exception:
                pass
            if style == "[]":
                return rf"Lifetime $\tau$ [{unit}]"
            elif style == "/":
                return rf"Lifetime $\tau$ / {unit}"
            else:
                return rf"Lifetime $\tau$ ({unit})"
    return r"Lifetime $\tau$"


def _get_z_cbar_label(result: LDAResult, two_lines: bool = True) -> str:
    """Format units-aware signal label matching dataset units and PyMORGAN style."""
    units = getattr(result, "units", {}) or {}
    if not units and isinstance(getattr(result, "settings", None), dict):
        units = result.settings.get("units", {})

    units_z_lbl = units.get("unitsZ_lbl") or units.get("signal_name") or ""
    units_z_ltx = units.get("unitsZ_ltx") or units.get("signal_unit") or units.get("z_unit") or ""

    try:
        import pymorgan.helpers as hlp
        style = "()"
        try:
            import pymorgan as pm
            style = getattr(pm.get_settings(), "label_style", "()")
        except Exception:
            pass
        if units_z_lbl or units_z_ltx:
            return hlp.fmtZlabel(style, units_z_lbl, units_z_ltx, twoLines=two_lines)
    except Exception:
        pass

    if units_z_lbl and units_z_ltx:
        return f"{units_z_lbl}\n({units_z_ltx})" if two_lines else f"{units_z_lbl} ({units_z_ltx})"
    return str(units_z_ltx or units_z_lbl or "mOD")


def plot_lda_map(
    result: LDAResult,
    ax: plt.Axes | None = None,
    cmap: str = "RdBu_r",
    discrete_taus: np.ndarray | list[float] | None = None,
    title: str = "Lifetime Density Map",
    show_integrated: bool = True,
):
    """Plot 2D heatmap/contour of lifetime amplitudes S(tau, probe) and integrated dynamics.

    Parameters
    ----------
    result : LDAResult
        The LDA result.
    ax : matplotlib.axes.Axes, optional
        If None, creates a 2-panel figure with synced Y-axis (Map + Integrated Dynamics).
    cmap : str, default "RdBu_r"
        Colourmap.
    discrete_taus : sequence of float, optional
        Discrete fit lifetimes to overlay as horizontal dashed lines.
    title : str
        Plot title.
    show_integrated : bool, default True
        Include the synced vertical panel of integrated absolute dynamics sum_lambda |S(tau, lambda)|.

    Returns
    -------
    matplotlib.axes.Axes or tuple of (Axes, Axes)
    """
    tau_grid = result.tau_grid
    probe = result.probe if result.probe is not None else np.arange(result.S_map.shape[0])
    S_map = result.S_map  # (Np, M)

    # Integrated absolute spectral content per lifetime: A(tau) = sum_lambda |S(tau, lambda)|
    integrated_dynamics = np.sum(np.abs(S_map), axis=0)  # shape (M,)

    if ax is None and show_integrated:
        fig, (ax_map, ax_int) = plt.subplots(
            1, 2, figsize=(9.0, 5.5), sharey=True, gridspec_kw={"width_ratios": [3.2, 1.2]}
        )
    elif ax is None:
        fig, ax_map = plt.subplots(figsize=(7, 5))
        ax_int = None
    else:
        ax_map = ax
        ax_int = None

    # 2D Meshgrid for pcolormesh
    P, T = np.meshgrid(probe, tau_grid)

    vmax = float(np.max(np.abs(S_map)))
    mesh = ax_map.pcolormesh(P, T, S_map.T, cmap=cmap, vmin=-vmax, vmax=vmax, shading="auto")
    ax_map.set_yscale("log")

    # Inherit parent contour plot probe X label
    probe_xlabel = _get_probe_xlabel(result)
    ax_map.set_xlabel(probe_xlabel)
    time_ylabel = _get_time_ylabel(result)
    ax_map.set_ylabel(time_ylabel)
    ax_map.set_title(title)

    # Units-aware colorbar label in vertical label mode (top title cbarLbl='top')
    cbar = plt.colorbar(mesh, ax=ax_map)
    cbar_label = _get_z_cbar_label(result, two_lines=True)
    cbar.ax.set_title(cbar_label, fontsize=12, pad=6, loc="left")

    # Integrated dynamical content vertical panel (synced Y-axis, smaller titles)
    if ax_int is not None:
        ax_int.plot(integrated_dynamics, tau_grid, "-", color="#1b4f72", linewidth=1.8)
        ax_int.fill_betweenx(tau_grid, 0, integrated_dynamics, color="#aed6f1", alpha=0.5)

        # Bootstrap confidence error bands
        if getattr(result, "bootstrap_std", None) is not None:
            b_err = np.asarray(result.bootstrap_std, dtype=float)
            if len(b_err) == len(tau_grid):
                ax_int.fill_betweenx(
                    tau_grid,
                    np.maximum(0.0, integrated_dynamics - b_err),
                    integrated_dynamics + b_err,
                    color="#e74c3c",
                    alpha=0.35,
                    label=r"$\pm 1\sigma$ Bootstrap",
                )

        # Peak centroids
        if getattr(result, "peaks", None):
            for pk in result.peaks:
                ax_int.plot(pk["amplitude"], pk["tau"], "r*", markersize=6)

        ax_int.set_yscale("log")
        ax_int.set_xlabel(r"Integrated $\int |S| d\lambda$", fontsize=8.5)
        ax_int.set_title("Integrated Dynamics", fontsize=9.0)
        ax_int.tick_params(axis="both", labelsize=8)
        ax_int.grid(True, linestyle=":", alpha=0.6)
        if getattr(result, "bootstrap_std", None) is not None:
            ax_int.legend(loc="best", fontsize=7.5)

    # Overlay discrete lifetimes across both panels
    if discrete_taus is not None:
        for tau in discrete_taus:
            if np.isfinite(tau) and tau > 0:
                ax_map.axhline(tau, color="black", linestyle="--", alpha=0.7, linewidth=1.2)
                if ax_int is not None:
                    ax_int.axhline(tau, color="black", linestyle="--", alpha=0.7, linewidth=1.2)

    if ax_int is not None:
        return ax_map, ax_int
    return ax_map


def plot_l_curve(
    result: LDAResult,
    ax: plt.Axes | None = None,
    title: str = "L-Curve Corner Detection",
):
    """Plot the L-curve (log residual vs log solution norm) and corner choice.

    Parameters
    ----------
    result : LDAResult
        The LDA result.
    ax : matplotlib.axes.Axes, optional
    title : str

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    if result.l_curve_points is None:
        ax.text(0.5, 0.5, "No L-curve points", ha="center", va="center")
        return ax

    pts = result.l_curve_points
    log_res, log_norm = pts[:, 0], pts[:, 1]

    ax.plot(log_res, log_norm, "o-", color="#2b5c8f", markersize=4, label="Alpha scan")

    # Highlight chosen alpha using Matplotlib TeX math syntax
    idx = int(np.argmin(np.abs(result.alphas - result.alpha_opt))) if result.alphas is not None else 0
    ax.plot(log_res[idx], log_norm[idx], "r*", markersize=12, label=r"Chosen $\alpha = " + f"{result.alpha_opt:.3g}$")

    ax.set_xlabel(r"$\log_{10}$ Residual Norm $\|\mathbf{D} - \mathbf{C} \mathbf{S}^T\|$")
    ax.set_ylabel(r"$\log_{10}$ Solution Norm $\|\mathbf{L} \mathbf{S}^T\|$")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best")

    return ax
