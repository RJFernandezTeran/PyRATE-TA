"""Kinetic and spectral trace plots with optional fitted overlay and linked residuals panel."""

from __future__ import annotations

from dataclasses import replace

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from ..log import get_logger
from ..settings import get_settings
from .matrix import as_dataset, as_fit_dataset
from .style import overlay_styles

logger = get_logger(__name__)


def _artist_color(artist, default: str = "C0"):
    """Extract a representative colour from an artist (Line2D or collection)."""
    if hasattr(artist, "get_color"):
        c = artist.get_color()
        if c is not None:
            return c
    if hasattr(artist, "get_facecolor"):
        c = artist.get_facecolor()
        if c is not None and len(c) > 0:
            return c[0] if hasattr(c, "__len__") and not isinstance(c, (str, tuple)) else c
    return default


def setup_trace_legend(
    ax_trace: plt.Axes,
    *,
    fig: plt.Figure | None = None,
    fit_lines: list | None = None,
    legend_loc: str | None = None,
    settings=None,
) -> tuple[plt.matplotlib.legend.Legend | None, tuple[float, float, float, float] | None]:
    """Build a clean, readable legend on ``ax_trace`` that avoids clipping.

    By default, places the legend inside the trace axis (``loc='best'``) with a
    readable font size, clear combined data+fit handles (marker + line), and
    semi-transparent frame. If an outside location is explicitly requested,
    reserves margin space so that the legend is never clipped off the figure boundary.

    Parameters
    ----------
    ax_trace : matplotlib.axes.Axes
        The primary trace axis where the legend is placed.
    fig : matplotlib.figure.Figure, optional
        Figure owning the axes.
    fit_lines : list of matplotlib.lines.Line2D, optional
        Fitted curve lines corresponding to the data curves.
    legend_loc : str, optional
        Target location for the legend. Defaults to ``'best'`` (inside).
    settings : Settings, optional
        PyRATE-TA settings object.

    Returns
    -------
    (legend, layout_rect)
        The created Legend object (or None) and a tight_layout rect tuple
        (or None if inside layout is used).
    """
    import pymorgan as pm

    s = get_settings() if settings is None else settings
    pm_s = pm.get_settings()

    # Determine desired location
    if legend_loc is not None:
        loc_str = str(legend_loc).strip()
    else:
        cfg_loc = str(getattr(s, "legend_location", "") or getattr(pm_s, "legend_location", "")).strip()
        # For multi-panel traces with residuals, default outside locations to 'best'
        # unless explicitly passed to the plotting function
        if not cfg_loc or cfg_loc.lower().startswith("outside"):
            loc_str = "best"
        else:
            loc_str = cfg_loc

    # Retrieve existing data handles and labels
    data_handles, data_labels = ax_trace.get_legend_handles_labels()
    valid = [(h, l) for h, l in zip(data_handles, data_labels, strict=True) if l and not l.startswith("_")]
    if not valid:
        return None, None

    valid_handles = [v[0] for v in valid]
    valid_labels = [v[1] for v in valid]

    # Optional pruning if configured in PyMORGAN settings
    max_entries = int(getattr(pm_s, "max_legend_entries", 15))
    if getattr(pm_s, "prune_legend", False) and len(valid) > max_entries:
        idx = np.round(np.linspace(0, len(valid) - 1, max(1, max_entries))).astype(int)
        idx = np.unique(idx)
        valid_handles = [valid_handles[i] for i in idx]
        valid_labels = [valid_labels[i] for i in idx]
        if fit_lines is not None and len(fit_lines) == len(data_handles):
            fit_lines = [fit_lines[i] for i in idx]

    # Construct clean, highly readable handles
    marker = str(getattr(s, "data_marker", "o"))
    markersize = max(float(getattr(s, "data_markersize", 4.0)), 5.0)
    fit_lw = max(float(getattr(s, "fit_linewidth", 1.2)), 1.5)
    fit_ls = str(getattr(s, "fit_linestyle", "-"))

    handles = []
    has_fit = fit_lines is not None and len(fit_lines) == len(valid_handles)
    for h in valid_handles:
        c = _artist_color(h)
        ls = fit_ls if has_fit else (h.get_linestyle() if hasattr(h, "get_linestyle") else "none")
        m = marker if has_fit else (h.get_marker() if hasattr(h, "get_marker") else marker)
        if str(m).lower() == "none" and not has_fit:
            m = "none"
        handles.append(
            mlines.Line2D(
                [],
                [],
                color=c,
                marker=m if str(m) != "None" else marker,
                markersize=markersize,
                linestyle=ls if str(ls) != "None" else "-",
                linewidth=fit_lw,
            )
        )

    is_outside = loc_str.lower().startswith("outside")
    if is_outside:
        leg_kwargs = {"loc": "center left", "bbox_to_anchor": (1.02, 0.5)}
        layout_rect = (0.0, 0.0, 0.80, 1.0)
    else:
        leg_kwargs = {"loc": loc_str}
        layout_rect = None

    leg = ax_trace.legend(
        handles,
        valid_labels,
        frameon=True,
        framealpha=0.85,
        edgecolor="0.8",
        fontsize="medium",
        handlelength=2.2,
        labelspacing=0.35,
        borderpad=0.5,
        **leg_kwargs,
    )
    leg.set_draggable(True)
    return leg, layout_rect


def plot_kinetics_with_residuals(
    dataset,
    cuts,
    fit=None,
    *,
    incl_residuals: bool = True,
    ax=None,
    fig=None,
    normY: bool = False,
    doSmooth: int = 0,
    height_ratio: float | None = None,
    hspace: float | None = None,
    legend_loc: str | None = None,
    settings=None,
    **kwargs,
):
    """Plot kinetic traces at selected positions with fit overlay and linked residuals.

    Parameters
    ----------
    dataset : pymorgan.Dataset1D
        The dataset containing measured kinetic data.
    cuts : sequence of float
        Probe positions to cut at.
    fit : FitResult, optional
        Fitted model result object carrying ``.C``, ``.S``, ``.R``, ``.t``, ``.probe``.
        If provided, fitted curves are overlaid as solid lines.
    incl_residuals : bool, default True
        Whether to include a linked residuals panel below when ``fit`` is provided.
    ax : matplotlib.axes.Axes or sequence of Axes, optional
        Existing axis (or top trace axis / pair of subplots). If ``fit`` is provided and
        residuals are plotted, a 2x1 subplot figure with linked x-axis is created when ``ax`` is None.
    fig : matplotlib.figure.Figure, optional
        Figure owning the axes.
    normY : bool, default False
        Normalise each trace to its peak amplitude.
    doSmooth : int, default 0
        Smoothing width in points.
    height_ratio : float, optional
        Aspect ratio of trace axis height to residual axis height (e.g. 4.0 for 4:1, 3.0 for 3:1).
        Defaults to ``Settings.residuals_height_ratio``.
    hspace : float, optional
        Vertical spacing between trace panel and residual panel. Defaults to ``Settings.residuals_hspace``.
    legend_loc : str, optional
        Location for the legend. Defaults to ``'best'`` (inside the trace panel).
    settings : Settings, optional
        PyRATE-TA settings object.

    Returns
    -------
    (ax_trace, ax_res) or ax_trace
        The main trace axis (and residual axis if created).
    """
    import pymorgan as pm

    pm.apply_style()
    s = get_settings() if settings is None else settings
    pm_s = pm.get_settings()
    if height_ratio is None:
        height_ratio = float(getattr(s, "residuals_height_ratio", 4.0))
    if hspace is None:
        hspace = float(getattr(s, "residuals_hspace", 0.05))

    # Resolve target legend location: default outside to "best" for multi-panel trace plots
    if legend_loc is not None:
        target_loc = str(legend_loc).strip()
    else:
        cfg_loc = str(getattr(s, "legend_location", "") or getattr(pm_s, "legend_location", "")).strip()
        if not cfg_loc or cfg_loc.lower().startswith("outside"):
            target_loc = "best"
        else:
            target_loc = cfg_loc

    pm_plot_settings = replace(pm_s, legend_location=target_loc)
    data_style, fit_style = overlay_styles("kinetics", settings=s)

    show_residuals = incl_residuals and (fit is not None and getattr(fit, "R", None) is not None)

    if show_residuals:
        if ax is None:
            figsize = tuple(getattr(pm_s, "kinetics_figsize", (7.5, 5.0)))
            fig, (ax_trace, ax_res) = plt.subplots(
                2,
                1,
                sharex=True,
                figsize=figsize,
                gridspec_kw={"height_ratios": [height_ratio, 1.0], "hspace": hspace},
            )
        elif isinstance(ax, (list, tuple, np.ndarray)) and len(ax) >= 2:
            ax_trace, ax_res = ax[0], ax[1]
            if fig is None:
                fig = ax_trace.figure
        else:
            ax_trace = ax
            ax_res = None
            show_residuals = False
    else:
        if ax is None:
            fig, ax_trace = plt.subplots()
        else:
            ax_trace = ax
        ax_res = None

    # 1. Plot data as points (dots)
    dataset.plot_kinetics(
        cuts,
        ax=ax_trace,
        fig=fig,
        normY=normY,
        doSmooth=doSmooth,
        show_xlabel=not show_residuals,
        settings=pm_plot_settings,
        **data_style,
        **kwargs,
    )

    # 2. Overlay fit as solid lines
    fit_lines = None
    if fit is not None:
        fit_ds = as_fit_dataset(fit, dataset)
        before_lines = list(ax_trace.lines)
        fit_ds.plot_kinetics(
            cuts,
            ax=ax_trace,
            fig=fig,
            normY=normY,
            show_xlabel=not show_residuals,
            settings=pm_plot_settings,
            **fit_style,
        )
        new_lines = [line for line in ax_trace.lines if line not in before_lines]
        for line in new_lines:
            line.set_label("_nolegend_")
        fit_lines = new_lines

    # 3. Build unified, legible legend on ax_trace
    _, layout_rect = setup_trace_legend(
        ax_trace,
        fig=fig,
        fit_lines=fit_lines,
        legend_loc=target_loc,
        settings=s,
    )

    # 4. Plot residuals below on linked x axis
    if show_residuals and ax_res is not None:
        weights = getattr(fit, "weights", None)
        statistic = getattr(fit, "statistic", None)
        is_weighted = (
            (weights is not None and getattr(weights, "weighted", False))
            or (statistic is not None and getattr(statistic, "weighted", False))
        )

        res_matrix = np.asarray(fit.R, dtype=float).copy()
        res_label = "r/$\\sigma$" if is_weighted else "Res."

        if is_weighted:
            if weights is not None and getattr(weights, "w", None) is not None:
                w = np.asarray(weights.w, dtype=float)
                mask = getattr(weights, "mask", w > 0)
                res_matrix = np.where(mask, res_matrix * w, np.nan)
            elif getattr(dataset, "sigma", None) is not None:
                sigma = np.asarray(dataset.sigma, dtype=float)
                usable = np.isfinite(sigma) & (sigma > 0)
                res_matrix = np.where(usable, res_matrix / sigma, np.nan)

        probe = fit.probe if getattr(fit, "probe", None) is not None else dataset.probe
        res_ds = as_dataset(fit.t, probe, res_matrix, dataset)

        # Draw residual points
        res_ds.plot_kinetics(
            cuts,
            ax=ax_res,
            fig=fig,
            normY=normY,
            show_xlabel=True,
            settings=pm_plot_settings,
            **data_style,
        )
        if ax_res.get_legend() is not None:
            ax_res.get_legend().remove()

        # Zero reference line: identical to PyMORGAN plots (colour 0.75, lw 0.75)
        ax_res.axhline(0, color="0.75", linewidth=0.75, zorder=0)

        # Axis labels and cleanup
        ax_res.set_ylabel(res_label)
        ax_trace.set_xlabel("")
        plt.setp(ax_trace.get_xticklabels(), visible=False)

        # Align y-axis labels across subplots so they line up vertically
        if fig is not None:
            fig.align_ylabels([ax_trace, ax_res])
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if layout_rect is not None:
                        fig.tight_layout(rect=layout_rect)
                    else:
                        fig.tight_layout()
            except Exception:
                pass

        return ax_trace, ax_res

    if fig is not None:
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if layout_rect is not None:
                    fig.tight_layout(rect=layout_rect)
                else:
                    fig.tight_layout()
        except Exception:
            pass

    return ax_trace


def plot_spectra_with_residuals(
    dataset,
    cuts,
    fit=None,
    *,
    incl_residuals: bool = True,
    ax=None,
    fig=None,
    normY: bool = False,
    doSmooth: int = 0,
    height_ratio: float | None = None,
    hspace: float | None = None,
    legend_loc: str | None = None,
    settings=None,
    **kwargs,
):
    """Plot transient spectra at selected positions with fit overlay and linked residuals.

    Parameters
    ----------
    dataset : pymorgan.Dataset1D
        The dataset containing measured spectral data.
    cuts : sequence of float
        Delay times to cut at.
    fit : FitResult, optional
        Fitted model result object carrying ``.C``, ``.S``, ``.R``, ``.t``, ``.probe``.
        If provided, fitted curves are overlaid as solid lines.
    incl_residuals : bool, default True
        Whether to include a linked residuals panel below when ``fit`` is provided.
    ax : matplotlib.axes.Axes or sequence of Axes, optional
        Existing axis (or top trace axis / pair of subplots). If ``fit`` is provided and
        residuals are plotted, a 2x1 subplot figure with linked x-axis is created when ``ax`` is None.
    fig : matplotlib.figure.Figure, optional
        Figure owning the axes.
    normY : bool, default False
        Normalise each trace to its peak amplitude.
    doSmooth : int, default 0
        Smoothing width in points.
    height_ratio : float, optional
        Aspect ratio of trace axis height to residual axis height (e.g. 4.0 for 4:1, 3.0 for 3:1).
        Defaults to ``Settings.residuals_height_ratio``.
    hspace : float, optional
        Vertical spacing between trace panel and residual panel. Defaults to ``Settings.residuals_hspace``.
    legend_loc : str, optional
        Location for the legend. Defaults to ``'best'`` (inside the trace panel).
    settings : Settings, optional
        PyRATE-TA settings object.

    Returns
    -------
    (ax_trace, ax_res) or ax_trace
        The main trace axis (and residual axis if created).
    """
    import pymorgan as pm

    pm.apply_style()
    s = get_settings() if settings is None else settings
    pm_s = pm.get_settings()
    if height_ratio is None:
        height_ratio = float(getattr(s, "residuals_height_ratio", 4.0))
    if hspace is None:
        hspace = float(getattr(s, "residuals_hspace", 0.05))

    # Resolve target legend location: default outside to "best" for multi-panel trace plots
    if legend_loc is not None:
        target_loc = str(legend_loc).strip()
    else:
        cfg_loc = str(getattr(s, "legend_location", "") or getattr(pm_s, "legend_location", "")).strip()
        if not cfg_loc or cfg_loc.lower().startswith("outside"):
            target_loc = "best"
        else:
            target_loc = cfg_loc

    pm_plot_settings = replace(pm_s, legend_location=target_loc)
    data_style, fit_style = overlay_styles("spectra", settings=s)

    show_residuals = incl_residuals and (fit is not None and getattr(fit, "R", None) is not None)

    if show_residuals:
        if ax is None:
            figsize = tuple(getattr(pm_s, "spectra_figsize", (7.5, 4.5)))
            fig, (ax_trace, ax_res) = plt.subplots(
                2,
                1,
                sharex=True,
                figsize=figsize,
                gridspec_kw={"height_ratios": [height_ratio, 1.0], "hspace": hspace},
            )
        elif isinstance(ax, (list, tuple, np.ndarray)) and len(ax) >= 2:
            ax_trace, ax_res = ax[0], ax[1]
            if fig is None:
                fig = ax_trace.figure
        else:
            ax_trace = ax
            ax_res = None
            show_residuals = False
    else:
        if ax is None:
            fig, ax_trace = plt.subplots()
        else:
            ax_trace = ax
        ax_res = None

    # 1. Plot data as points (dots)
    dataset.plot_spectra(
        cuts,
        ax=ax_trace,
        fig=fig,
        normY=normY,
        doSmooth=doSmooth,
        show_xlabel=not show_residuals,
        settings=pm_plot_settings,
        **data_style,
        **kwargs,
    )

    # 2. Overlay fit as solid lines
    fit_lines = None
    if fit is not None:
        fit_ds = as_fit_dataset(fit, dataset)
        before_lines = list(ax_trace.lines)
        fit_ds.plot_spectra(
            cuts,
            ax=ax_trace,
            fig=fig,
            normY=normY,
            show_xlabel=not show_residuals,
            settings=pm_plot_settings,
            **fit_style,
        )
        new_lines = [line for line in ax_trace.lines if line not in before_lines]
        for line in new_lines:
            line.set_label("_nolegend_")
        fit_lines = new_lines

    # 3. Build unified, legible legend on ax_trace
    _, layout_rect = setup_trace_legend(
        ax_trace,
        fig=fig,
        fit_lines=fit_lines,
        legend_loc=target_loc,
        settings=s,
    )

    # 4. Plot residuals below on linked x axis
    if show_residuals and ax_res is not None:
        weights = getattr(fit, "weights", None)
        statistic = getattr(fit, "statistic", None)
        is_weighted = (
            (weights is not None and getattr(weights, "weighted", False))
            or (statistic is not None and getattr(statistic, "weighted", False))
        )

        res_matrix = np.asarray(fit.R, dtype=float).copy()
        res_label = "r/$\\sigma$" if is_weighted else "Res."

        if is_weighted:
            if weights is not None and getattr(weights, "w", None) is not None:
                w = np.asarray(weights.w, dtype=float)
                mask = getattr(weights, "mask", w > 0)
                res_matrix = np.where(mask, res_matrix * w, np.nan)
            elif getattr(dataset, "sigma", None) is not None:
                sigma = np.asarray(dataset.sigma, dtype=float)
                usable = np.isfinite(sigma) & (sigma > 0)
                res_matrix = np.where(usable, res_matrix / sigma, np.nan)

        probe = fit.probe if getattr(fit, "probe", None) is not None else dataset.probe
        res_ds = as_dataset(fit.t, probe, res_matrix, dataset)

        # Draw residual points
        res_ds.plot_spectra(
            cuts,
            ax=ax_res,
            fig=fig,
            normY=normY,
            show_xlabel=True,
            settings=pm_plot_settings,
            **data_style,
        )
        if ax_res.get_legend() is not None:
            ax_res.get_legend().remove()

        # Zero reference line
        ax_res.axhline(0, color="0.75", linewidth=0.75, zorder=0)

        # Axis labels and cleanup
        ax_res.set_ylabel(res_label)
        ax_trace.set_xlabel("")
        plt.setp(ax_trace.get_xticklabels(), visible=False)

        # Align y-axis labels across subplots so they line up vertically
        if fig is not None:
            fig.align_ylabels([ax_trace, ax_res])
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if layout_rect is not None:
                        fig.tight_layout(rect=layout_rect)
                    else:
                        fig.tight_layout()
            except Exception:
                pass

        return ax_trace, ax_res

    if fig is not None:
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if layout_rect is not None:
                    fig.tight_layout(rect=layout_rect)
                else:
                    fig.tight_layout()
        except Exception:
            pass

    return ax_trace
