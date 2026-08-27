"""Contour views of a fitted or residual matrix -- through PyMORGAN's plotter.

A fit surface and a residual matrix are not datasets, so PyMORGAN has no method
that takes them. What it *does* have is the whole contour engine (colourmaps,
white levels, symmetric scales, axis labels and units), and rewriting any of
that here would be redundant duplication.


So this module does not draw anything itself: it wraps the matrix in a
:class:`pymorgan.Dataset1D` carrying the original dataset's units, and hands it
to ``plot_contour``. The fit, the residuals and the data are then rendered by
the same code, which is what makes the three views comparable -- and it is why
a PyMORGAN colourmap identifier such as ``DkRd/Wh/DkBu`` works here without this
module knowing anything about colourmaps.
"""

from __future__ import annotations

import numpy as np

from ..log import get_logger

logger = get_logger(__name__)


def as_dataset(t, probe, M, dataset=None):
    """Wrap a ``[Nd, Np]`` matrix as a :class:`pymorgan.Dataset1D`.

    The units (and hence every axis label) are taken from the dataset the
    matrix came from, so a residual map is labelled exactly like the data it is
    the residual of.
    """
    import pymorgan as pm

    M = np.atleast_2d(np.asarray(M, dtype=float))
    t = np.asarray(t, dtype=float).ravel()
    probe = np.asarray(probe, dtype=float).ravel() if probe is not None else np.arange(M.shape[1])
    units = dict(getattr(dataset, "units", {}) or {})

    return pm.Dataset1D(
        M[:, :, None],
        t,
        probe,
        units,
        source=getattr(dataset, "source", None),
        data_type=getattr(dataset, "data_type", None),
    )


def as_fit_dataset(fit, dataset=None):
    """The fitted surface as a :class:`pymorgan.Dataset1D`, on the *fit's* axes.

    A fit restricted to part of the spectral range returns spectra on that
    window only, while the dataset still carries the full probe axis. Handing
    the full axis to a plotter that expects one point per row of ``S`` is a
    size mismatch at best and a mislabelled spectrum at worst, so anything that
    draws a fit gets this wrapper instead of the original dataset.

    The units, source and data type come from the dataset, so the axes still
    read like the data's.
    """
    probe = fit.probe
    if probe is None and dataset is not None:
        probe = dataset.probe
    return as_dataset(fit.t, probe, fit.C @ fit.S.T, dataset)


def plot_matrix(t, probe, M, ax=None, *, title=None, dataset=None, **contour_kwargs):
    """Contour a fit surface or residual matrix using PyMORGAN's plotter.

    Parameters
    ----------
    t, probe : array_like
        Delay and probe axes of ``M``.
    M : array_like ``(Nd, Np)``
        Fit surface or residual matrix, in mOD.
    dataset : pymorgan.Dataset1D, optional
        Source of the units, so the axes read like the data's.
    **contour_kwargs
        Passed straight to ``Dataset1D.plot_contour`` (``Zmin``/``Zmax``,
        ``Nlevels``, ``cmap_ID``, ``white_levels``, ``Yscale``, ...), so the
        plot-controls panel drives this view exactly as it drives the data.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    import pymorgan as pm

    # The style (and therefore the font stack) is PyMORGAN's, applied here so a
    # figure looks the same whether it was opened from the GUI -- which applies
    # it too -- or from a plain script. Every PyRATE-TA plotter does this.
    pm.apply_style()
    if ax is None:
        _, ax = plt.subplots()

    wrapped = as_dataset(t, probe, M, dataset)
    wrapped.plot_contour(ax=ax, **contour_kwargs)
    if title:
        ax.set_title(title, fontweight="bold")
    return ax


def plot_data_fit_residuals(
    fit,
    dataset,
    *,
    ax=None,
    fig=None,
    zscale_factor: float = 10.0,
    **contour_kwargs,
):
    """Plot 3-column side-by-side contour maps of (a) Data, (b) Fit, and (c) Residuals.

    The Data and Fit panels share the exact same Z contour scale (Zmin, Zmax derived from Data).
    The Residuals panel uses a Z contour scale zoomed in by ``zscale_factor`` (default 10x, i.e.
    Zmax_res = Zmax_data / 10.0).

    Parameters
    ----------
    fit : FitResult
        The fit result carrying ``.C``, ``.S``, ``.R``, ``.t``, ``.probe``.
    dataset : pymorgan.Dataset1D
        The measured dataset.
    ax : sequence of 3 Axes, optional
        3 subplot axes to render into.
    fig : matplotlib.figure.Figure, optional
        Parent figure.
    zscale_factor : float, default 10.0
        Scale factor for the residual panel Z contour range relative to Data and Fit.
    **contour_kwargs
        Passed to PyMORGAN's ``plot_contour``.

    Returns
    -------
    (ax_data, ax_fit, ax_res)
        Tuple of 3 matplotlib axes.
    """
    import matplotlib.pyplot as plt
    import pymorgan as pm

    pm.apply_style()

    if fit is None or dataset is None:
        raise ValueError("fit and dataset are required to plot Data+Fit+Residuals")

    # 1. Determine baseline Z limits for Data & Fit (shared scale)
    Z_data = np.asarray(dataset.Z, dtype=float)
    finite_data = Z_data[np.isfinite(Z_data)]
    zmax_data = float(np.nanmax(np.abs(finite_data))) if finite_data.size else 1.0
    if not np.isfinite(zmax_data) or zmax_data <= 0:
        zmax_data = 1.0

    zmax_user = contour_kwargs.get("Zmax")
    zmin_user = contour_kwargs.get("Zmin")

    if zmax_user is not None:
        zmax = float(zmax_user)
        zmin = float(zmin_user) if zmin_user is not None else -zmax
    else:
        zmax = zmax_data
        zmin = -zmax_data

    # Residual scale is zoomed in 10x (10x smaller range -> 10x higher contrast)
    zmax_res = zmax / float(zscale_factor) if zscale_factor > 0 else zmax
    zmin_res = zmin / float(zscale_factor) if zscale_factor > 0 else zmin

    # Create 1x3 subplot figure if axes not provided
    if ax is None:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True, sharey=True)
        ax_data, ax_fit, ax_res = axes[0], axes[1], axes[2]
    elif isinstance(ax, (list, tuple, np.ndarray)) and len(ax) >= 3:
        ax_data, ax_fit, ax_res = ax[0], ax[1], ax[2]
        if fig is None:
            fig = ax_data.figure
    else:
        raise ValueError("ax must be a sequence of 3 Axes")

    contour_kwargs = dict(contour_kwargs)
    contour_kwargs.setdefault("cbarLbl", "top")

    # 2. Draw Data panel
    data_kwargs = {**contour_kwargs, "Zmin": zmin, "Zmax": zmax}
    dataset.plot_contour(ax=ax_data, **data_kwargs)
    ax_data.set_title("(a) Data", fontweight="bold")

    # 3. Draw Fit panel (SAME Z contour scale as Data)
    probe_fit = fit.probe if getattr(fit, "probe", None) is not None else dataset.probe
    M_fit = fit.C @ fit.S.T
    fit_ds = as_dataset(fit.t, probe_fit, M_fit, dataset)
    fit_ds.plot_contour(ax=ax_fit, **data_kwargs)
    ax_fit.set_title("(b) Fit", fontweight="bold")
    ax_fit.set_ylabel("")
    ax_fit.tick_params(axis="y", labelleft=True)

    # 4. Draw Residuals panel (10x zoomed Z contour scale)
    res_kwargs = {**contour_kwargs, "Zmin": zmin_res, "Zmax": zmax_res}
    res_matrix = np.asarray(fit.R, dtype=float)
    res_ds = as_dataset(fit.t, probe_fit, res_matrix, dataset)
    res_ds.plot_contour(ax=ax_res, **res_kwargs)
    res_label = f"(c) Residuals ({int(zscale_factor)}×)" if zscale_factor != 1.0 else "(c) Residuals"
    ax_res.set_title(res_label, fontweight="bold")
    ax_res.set_ylabel("")
    ax_res.tick_params(axis="y", labelleft=True)

    if fig is not None:
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig.tight_layout()
        except Exception:
            pass

    return ax_data, ax_fit, ax_res

