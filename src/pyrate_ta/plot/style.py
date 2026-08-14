"""How data and fit are drawn on the same axes.

The measurement is drawn as points and the model as a thin line over them, so
both can be read at once: a line through a line hides the disagreement that a
fit is supposed to expose. Both styles come from ``settings.toml`` (``[plots]``)
and are handed to PyMORGAN's plotters as keyword arguments -- this module maps
settings to keywords and does no drawing of its own.
"""

from __future__ import annotations

from ..log import get_logger

logger = get_logger(__name__)


def overlay_styles(kind: str = "kinetics", settings=None) -> tuple[dict, dict]:
    """``(data_style, fit_style)`` keyword dictionaries for one plotter.

    The two PyMORGAN plotters take the style differently, which is why this
    takes ``kind``: :func:`~pymorgan.oneD.plot.plot_kinetics` has its own
    ``plotStyle`` format string, while :func:`~pymorgan.oneD.plot.plot_spectra`
    always draws a solid line and forwards the rest to Matplotlib, so the
    marker has to be given as ``marker`` / ``linestyle`` instead. Getting this
    wrong raises inside Matplotlib rather than being ignored, so the mapping
    lives in one place.

    Parameters
    ----------
    kind : {"kinetics", "spectra"}
        Which plotter the keywords are for.
    settings : pyrate.Settings, optional
        Defaults to the active settings.

    Returns
    -------
    (dict, dict)
        Keywords for the measured data (points) and for the fitted curve
        (line), ready to be forwarded to the plotter named by ``kind``.
    """
    if settings is None:
        from ..settings import get_settings

        settings = get_settings()

    marker = str(settings.data_marker)
    if kind == "kinetics":
        data = {
            "plotStyle": marker,
            "alpha": float(settings.data_alpha),
            "ms": float(settings.data_markersize),
        }
        fit = {
            "plotStyle": str(settings.fit_linestyle),
            "lw": float(settings.fit_linewidth),
            "alpha": float(settings.fit_alpha),
        }
        return data, fit

    if kind != "spectra":
        raise ValueError(f"unknown plot kind {kind!r}; expected 'kinetics' or 'spectra'")

    data = {
        "marker": marker,
        "linestyle": "none",
        "alpha": float(settings.data_alpha),
        "markersize": float(settings.data_markersize),
    }
    fit = {
        "linestyle": str(settings.fit_linestyle),
        "linewidth": float(settings.fit_linewidth),
        "alpha": float(settings.fit_alpha),
        "marker": "none",
    }
    return data, fit


def scale_kwargs(ax, axis: str = "y") -> dict:
    """Keywords needed to reproduce ``ax``'s scale on another axis.

    A symlog axis is not defined by its name alone: the linear threshold
    decides where the logarithmic region begins, and copying only the name
    would misalign two panels that are supposed to share the axis -- silently,
    which is the worst way for a plot to be wrong.
    """
    getter = ax.get_yscale if axis == "y" else ax.get_xscale
    if getter() != "symlog":
        return {}
    transform = (ax.yaxis if axis == "y" else ax.xaxis).get_transform()
    kwargs = {}
    for name in ("linthresh", "linscale"):
        value = getattr(transform, name, None)
        if value:
            kwargs[name] = value
    return kwargs
