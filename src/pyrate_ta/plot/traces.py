"""Kinetic trace plots with optional fitted overlay and linked residuals panel."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..log import get_logger
from ..settings import get_settings
from .matrix import as_dataset, as_fit_dataset
from .style import overlay_styles

logger = get_logger(__name__)


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
    settings : Settings, optional
        PyRATE settings object.

    Returns
    -------
    (ax_trace, ax_res) or ax_trace
        The main trace axis (and residual axis if created).
    """
    import pymorgan as pm

    pm.apply_style()
    s = get_settings() if settings is None else settings
    if height_ratio is None:
        height_ratio = float(getattr(s, "residuals_height_ratio", 4.0))
    if hspace is None:
        hspace = float(getattr(s, "residuals_hspace", 0.05))

    data_style, fit_style = overlay_styles("kinetics", settings=s)

    show_residuals = incl_residuals and (fit is not None and getattr(fit, "R", None) is not None)


    if show_residuals:
        if ax is None:
            figsize = tuple(getattr(pm.get_settings(), "kinetics_figsize", (7.5, 5.0)))
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
        **data_style,
        **kwargs,
    )

    # 2. Overlay fit as solid lines
    if fit is not None:
        fit_ds = as_fit_dataset(fit, dataset)
        before_lines = list(ax_trace.lines)
        fit_ds.plot_kinetics(
            cuts,
            ax=ax_trace,
            fig=fig,
            normY=normY,
            show_xlabel=not show_residuals,
            **fit_style,
        )
        # Suppress duplicate legend entries for fit lines and combine handles
        new_lines = [line for line in ax_trace.lines if line not in before_lines]
        for line in new_lines:
            line.set_label("_nolegend_")

        data_handles, data_labels = ax_trace.get_legend_handles_labels()
        if len(data_handles) == len(new_lines):
            handles = [(data_handles[i], new_lines[i]) for i in range(len(data_handles))]
            labels = data_labels
        else:
            handles = data_handles
            labels = data_labels

        if ax_trace.get_legend() is not None and handles:
            from pymorgan.oneD.plot import legend_placement
            leg_kwargs = legend_placement(None)
            leg = ax_trace.legend(
                handles,
                labels,
                frameon=False,
                fontsize="small",
                **leg_kwargs,
            )
            leg.set_draggable(True)
            if "bbox_to_anchor" in leg_kwargs:
                try:
                    leg.set_in_layout(False)
                except Exception:
                    pass

    # 3. Plot residuals below on linked x axis
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
                    fig.tight_layout()
            except Exception:
                pass

        return ax_trace, ax_res

    if fig is not None:
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
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
    settings : Settings, optional
        PyRATE settings object.

    Returns
    -------
    (ax_trace, ax_res) or ax_trace
        The main trace axis (and residual axis if created).
    """
    import pymorgan as pm

    pm.apply_style()
    s = get_settings() if settings is None else settings
    if height_ratio is None:
        height_ratio = float(getattr(s, "residuals_height_ratio", 4.0))
    if hspace is None:
        hspace = float(getattr(s, "residuals_hspace", 0.05))

    data_style, fit_style = overlay_styles("spectra", settings=s)

    show_residuals = incl_residuals and (fit is not None and getattr(fit, "R", None) is not None)

    if show_residuals:
        if ax is None:
            figsize = tuple(getattr(pm.get_settings(), "spectra_figsize", (7.5, 4.5)))
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
        **data_style,
        **kwargs,
    )

    # 2. Overlay fit as solid lines
    if fit is not None:
        fit_ds = as_fit_dataset(fit, dataset)
        before_lines = list(ax_trace.lines)
        fit_ds.plot_spectra(
            cuts,
            ax=ax_trace,
            fig=fig,
            normY=normY,
            show_xlabel=not show_residuals,
            **fit_style,
        )
        # Suppress duplicate legend entries for fit lines and combine handles
        new_lines = [line for line in ax_trace.lines if line not in before_lines]
        for line in new_lines:
            line.set_label("_nolegend_")

        data_handles, data_labels = ax_trace.get_legend_handles_labels()
        if len(data_handles) == len(new_lines):
            handles = [(data_handles[i], new_lines[i]) for i in range(len(data_handles))]
            labels = data_labels
        else:
            handles = data_handles
            labels = data_labels

        if ax_trace.get_legend() is not None and handles:
            from pymorgan.oneD.plot import legend_placement
            leg_kwargs = legend_placement(None)
            leg = ax_trace.legend(
                handles,
                labels,
                frameon=False,
                fontsize="small",
                **leg_kwargs,
            )
            leg.set_draggable(True)
            if "bbox_to_anchor" in leg_kwargs:
                try:
                    leg.set_in_layout(False)
                except Exception:
                    pass

    # 3. Plot residuals below on linked x axis
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
                    fig.tight_layout()
            except Exception:
                pass

        return ax_trace, ax_res

    if fig is not None:
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig.tight_layout()
        except Exception:
            pass

    return ax_trace



