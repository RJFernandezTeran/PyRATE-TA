"""Plotting routines for Lifetime Density Analysis (LDA)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pymorgan as pm
from pymorgan import helpers as hlp

from ..results.lda_result import LDAResult


def _get_label_style() -> str:
    """Active PyMORGAN delimiter style: '()', '[]', or '/'."""
    try:
        return str(getattr(pm.get_settings(), "label_style", "()"))
    except Exception:
        return "()"


def _axis_label_text(lbl: str, unit: str) -> str:
    """Format label and unit string adhering to PyMORGAN delimiter style."""
    style = _get_label_style()
    if not unit:
        return lbl
    match style:
        case "[]":
            return f"{lbl} [{unit}]"
        case "/":
            return f"{lbl} / {unit}"
        case _:
            return f"{lbl} ({unit})"


def _get_probe_xlabel(result: LDAResult) -> str:
    """Format probe axis label matching parent dataset units & PyMORGAN style."""
    units = getattr(result, "units", {}) or {}
    if not units and isinstance(getattr(result, "settings", None), dict):
        units = result.settings.get("units", {})
    if isinstance(units, dict) and units:
        lbl = units.get("unitsL_lbl") or units.get("probe_name") or "Probe"
        unit = units.get("unitsL_ltx") or units.get("probe_unit") or ""
        return _axis_label_text(lbl, unit)
    return "Probe"


def _get_time_ylabel(result: LDAResult) -> str:
    """Format time/lifetime Y axis label matching parent dataset units & PyMORGAN style."""
    units = getattr(result, "units", {}) or {}
    if not units and isinstance(getattr(result, "settings", None), dict):
        units = result.settings.get("units", {})
    if isinstance(units, dict) and units:
        unit = units.get("unitsT_ltx") or units.get("time_unit") or ""
        return _axis_label_text(r"Lifetime $\tau$", unit)
    return r"Lifetime $\tau$"


def _get_z_cbar_label(result: LDAResult, two_lines: bool = True) -> str:
    """Format units-aware signal label matching dataset units and PyMORGAN style."""
    units = getattr(result, "units", {}) or {}
    if not units and isinstance(getattr(result, "settings", None), dict):
        units = result.settings.get("units", {})

    units_z_lbl = units.get("unitsZ_lbl") or units.get("signal_name") or ""
    units_z_ltx = units.get("unitsZ_ltx") or units.get("signal_unit") or units.get("z_unit") or ""

    style = _get_label_style()
    if units_z_lbl or units_z_ltx:
        try:
            return hlp.fmtZlabel(style, units_z_lbl, units_z_ltx, twoLines=two_lines)
        except Exception:
            pass

    if units_z_lbl and units_z_ltx:
        return f"{units_z_lbl}\n({units_z_ltx})" if two_lines else f"{units_z_lbl} ({units_z_ltx})"
    return str(units_z_ltx or units_z_lbl or "mOD")


def plot_lda_map(
    result: LDAResult,
    ax: plt.Axes | tuple[plt.Axes, plt.Axes] | None = None,
    ax_int: plt.Axes | None = None,
    cmap: str | None = None,
    discrete_taus: np.ndarray | list[float] | None = None,
    discrete_tau_color: str = "#27ae60",
    title: str = "Lifetime Density Map",
    show_integrated: bool = True,
    metric: str = "abs",
    annotate_centroids: bool = True,
    asinh: bool = False,
    Asinh: bool | None = None,
    asinh_pct: float | None = None,
):
    """Plot 2D heatmap of lifetime amplitudes S(tau, probe) and integrated dynamics.

    Parameters
    ----------
    result : LDAResult
        The LDA result.
    ax : matplotlib.axes.Axes or tuple of (Axes, Axes), optional
        If None, creates a 2-panel figure with synced Y-axis (Map + Integrated Dynamics).
        Can also be passed as a tuple ``(ax_map, ax_int)``.
    ax_int : matplotlib.axes.Axes, optional
        Dedicated axis for the integrated dynamics panel when ``ax`` is the map axis.
    cmap : str, optional
        Colourmap (defaults to PyMORGAN's active 2D colormap setting).
    discrete_taus : sequence of float, optional
        Discrete fit lifetimes to overlay as horizontal dashed lines.
    title : str
        Plot title.
    show_integrated : bool, default True
        Include the synced vertical panel of integrated absolute dynamics or dynamical content.
    metric : {"abs", "dynamical_content", "sqrt_sq"}
        - ``"abs"``: absolute integral :math:`A(\\tau) = \\int |S(\\lambda, \\tau)| d\\lambda`.
        - ``"dynamical_content"`` / ``"sqrt_sq"``: dynamical content :math:`D(\\tau) = \\sqrt{\\int S(\\lambda, \\tau)^2 d\\lambda}`.
    annotate_centroids : bool, default True
        Annotate detected lifetime peak centroids with their lifetime text.
    asinh, Asinh : bool, default False
        Apply arcsinh color scaling (compressing large amplitudes while keeping linear zero).
    asinh_pct : float, optional
        Linear threshold for arcsinh color scaling as percentage of vmax (defaults to settings.asinh_pct).

    Returns
    -------
    matplotlib.axes.Axes or tuple of (Axes, Axes)
    """
    pm.apply_style()

    if Asinh is not None:
        asinh = Asinh

    if cmap is None:
        try:
            cmap = str(getattr(pm.get_settings(), "cmap2D", "RdBu_r"))
        except Exception:
            cmap = "RdBu_r"

    tau_grid = result.tau_grid
    probe = result.probe if result.probe is not None else np.arange(result.S_map.shape[0])
    S_map = result.S_map  # (Np, M)

    is_dyn_content = str(metric).lower() in ("dynamical_content", "sqrt_sq", "rms", "d")
    if is_dyn_content:
        # Dynamical content D(tau) = sqrt(int S(lambda, tau)^2 dlambda)
        integrated_dynamics = np.sqrt(np.sum(S_map**2, axis=0))
        panel_title = r"Dynamical content, $D(\tau)$"
        panel_xlabel = r"Dynamical content, $D(\tau)$"
    else:
        # Absolute integral A(tau) = int |S(lambda, tau)| dlambda
        integrated_dynamics = np.sum(np.abs(S_map), axis=0)
        panel_title = "Integrated Dynamics"
        panel_xlabel = r"Integrated $\int |S| \, d\lambda$"

    if isinstance(ax, (tuple, list)):
        ax_map, ax_int = ax[0], ax[1]
    elif ax is not None and ax_int is not None:
        ax_map = ax
    elif ax is not None:
        ax_map = ax
        ax_int = None
    elif show_integrated:
        fig, (ax_map, ax_int) = plt.subplots(
            1, 2, figsize=(9.0, 5.5), sharey=True, gridspec_kw={"width_ratios": [3.2, 1.2]}
        )
    else:
        fig, ax_map = plt.subplots(figsize=(7, 5))
        ax_int = None

    # 2D Meshgrid for pcolormesh
    P, T = np.meshgrid(probe, tau_grid)

    vmax = float(np.nanmax(np.abs(S_map)))
    if vmax == 0 or np.isnan(vmax):
        vmax = 1.0

    import matplotlib.colors as mcolors

    if asinh:
        if asinh_pct is None:
            try:
                asinh_pct = float(getattr(pm.get_settings(), "asinh_pct", 5.0))
            except Exception:
                asinh_pct = 5.0
        s_val = max(1e-9, (float(asinh_pct) / 100.0) * vmax)
        norm = mcolors.AsinhNorm(linear_width=s_val, vmin=-vmax, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)

    mesh = ax_map.pcolormesh(P, T, S_map.T, cmap=cmap, norm=norm, shading="auto")
    ax_map.set_yscale("log")

    # Inherit parent contour plot probe X label and units
    probe_xlabel = _get_probe_xlabel(result)
    ax_map.set_xlabel(probe_xlabel)
    time_ylabel = _get_time_ylabel(result)
    ax_map.set_ylabel(time_ylabel)
    ax_map.set_title(title)

    # Units-aware colorbar label in vertical label mode (top title cbarLbl='top')
    cbar = plt.colorbar(mesh, ax=ax_map, pad=0.02)
    cbar_label = _get_z_cbar_label(result, two_lines=True)
    cbar.ax.set_title(cbar_label, fontsize=10, pad=6, loc="left")

    # Integrated dynamical content vertical panel (synced Y-axis, no grid, zero line)
    if ax_int is not None:
        # Standard zero line
        ax_int.axvline(0, color="0.75", linewidth=0.75)

        ax_int.plot(integrated_dynamics, tau_grid, "-", color="#1b4f72", linewidth=1.5)
        ax_int.fill_betweenx(tau_grid, 0, integrated_dynamics, color="#aed6f1", alpha=0.35)

        # Bootstrap confidence error bands
        if getattr(result, "bootstrap_std", None) is not None and not is_dyn_content:
            b_err = np.asarray(result.bootstrap_std, dtype=float)
            if len(b_err) == len(tau_grid):
                ax_int.fill_betweenx(
                    tau_grid,
                    np.maximum(0.0, integrated_dynamics - b_err),
                    integrated_dynamics + b_err,
                    color="#e74c3c",
                    alpha=0.3,
                    label=r"$\pm 1\sigma$ Bootstrap",
                )

        # Peak centroids: plot markers and annotate lifetime values only if enabled
        if annotate_centroids:
            peaks = getattr(result, "peaks", None) or []
            if not peaks:
                peaks = result.find_peaks(metric=metric)

            if peaks:
                time_unit = ""
                if isinstance(getattr(result, "units", None), dict):
                    time_unit = result.units.get("unitsT_ltx") or result.units.get("time_unit") or ""

                for pk in peaks:
                    pk_tau = float(pk["tau"])
                    idx_pk = int(pk.get("index", np.argmin(np.abs(tau_grid - pk_tau))))
                    pk_amp = float(integrated_dynamics[idx_pk]) if idx_pk < len(integrated_dynamics) else float(pk.get("amplitude", 0.0))

                    ax_int.plot(pk_amp, pk_tau, "r*", markersize=8, zorder=5)

                    try:
                        from pymorgan.helpers import ConvertTimeUnits
                        u_in = time_unit.strip() if time_unit else "ps"
                        if u_in in ("us", "µs", "μs", r"\mu s", r"$\mu s$"):
                            u_in = r"$\mu$s"
                        val_conv, unit_conv = ConvertTimeUnits(pk_tau, u_in)
                        lbl_text = f"{val_conv:.3g} {unit_conv}".strip()
                    except Exception:
                        lbl_text = f"{pk_tau:.3g} {time_unit}".strip() if time_unit else f"{pk_tau:.3g}"

                    ax_int.annotate(
                        lbl_text,
                        (pk_amp, pk_tau),
                        xytext=(6, 0),
                        textcoords="offset points",
                        va="center",
                        ha="left",
                        fontsize=8.5,
                        color="#900C3F",
                        fontweight="bold",
                    )

        ax_int.set_yscale("log")
        ax_int.set_xlabel(panel_xlabel, fontsize=8.5)
        ax_int.set_title(panel_title, fontsize=9.0)
        ax_int.tick_params(axis="both", labelsize=8)
        if getattr(result, "bootstrap_std", None) is not None and not is_dyn_content:
            ax_int.legend(loc="best", fontsize=7.5)

    # Overlay discrete lifetimes across both panels
    if discrete_taus is not None:
        for tau in discrete_taus:
            if np.isfinite(tau) and tau > 0:
                ax_map.axhline(tau, color=discrete_tau_color, linestyle="--", alpha=0.85, linewidth=1.2, zorder=4)
                if ax_int is not None:
                    ax_int.axhline(tau, color=discrete_tau_color, linestyle="--", alpha=0.85, linewidth=1.2, zorder=4)

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
    pm.apply_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    if result.l_curve_points is None:
        ax.text(0.5, 0.5, "No L-curve points", ha="center", va="center")
        return ax

    pts = result.l_curve_points
    log_res, log_norm = pts[:, 0], pts[:, 1]

    # Zero lines if spanning 0
    if np.min(log_res) <= 0 <= np.max(log_res):
        ax.axvline(0, color="0.75", linewidth=0.75)
    if np.min(log_norm) <= 0 <= np.max(log_norm):
        ax.axhline(0, color="0.75", linewidth=0.75)

    ax.plot(log_res, log_norm, "o-", color="#1b4f72", markersize=4, linewidth=1.2, label=r"$\alpha$ scan")

    # Highlight chosen alpha using Matplotlib TeX math syntax
    idx = int(np.argmin(np.abs(result.alphas - result.alpha_opt))) if result.alphas is not None else 0
    ax.plot(
        log_res[idx],
        log_norm[idx],
        "r*",
        markersize=10,
        label=rf"Optimal $\alpha = {result.alpha_opt:.3g}$",
    )

    ax.set_xlabel(r"$\log_{10} \|\mathbf{D} - \mathbf{C} \mathbf{S}^T\|_F$")
    ax.set_ylabel(r"$\log_{10} \|\mathbf{L} \mathbf{S}^T\|_F$")
    ax.set_title(title)
    ax.legend(loc="best")

    return ax
