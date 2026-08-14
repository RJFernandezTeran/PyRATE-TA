"""The live view of a fit: where each parameter stands, right now.

A bar per free parameter, its current value written above it and its name below.
Deliberately *not* a convergence plot of the cost: what one wants to see while a
global fit grinds is whether a lifetime is running away to the edge of the delay
window, whether two components are collapsing onto each other, or whether time
zero is drifting -- all of which are visible in the parameters themselves and
none of which show in the cost.

Qt-free on purpose: this draws into a Matplotlib axis and nothing else, so it
can be exercised headlessly and reused for a static "how did the fit get there"
figure. The window that hosts it lives in :mod:`pyrate_ta.gui.monitor`.
"""

from __future__ import annotations

import numpy as np

from ..log import get_logger

logger = get_logger(__name__)

#: Bars smaller than this fraction of the largest are drawn on the linear part
#: of the symlog axis; it keeps a fitted ``t0`` of ~0 from vanishing beside a
#: lifetime of 3000.
_LINTHRESH_FRACTION = 1e-3


def _linthresh(values) -> float:
    """Where the symlog axis should stop being logarithmic.

    Lifetimes in one fit routinely span four decades, and time zero sits near
    zero and may be negative, so neither a linear nor a log axis shows them
    together. Symlog does, provided the linear region is scaled to the data
    rather than left at Matplotlib's default of 1.
    """
    finite = np.abs(np.asarray(values, dtype=float))
    finite = finite[np.isfinite(finite) & (finite > 0)]
    if finite.size == 0:
        return 1.0
    return max(float(finite.max()) * _LINTHRESH_FRACTION, np.finfo(float).tiny)


def plot_parameter_bars(
    values,
    names=None,
    ax=None,
    *,
    fixed=None,
    scale: str = "symlog",
    title: str | None = None,
    value_format: str = "%.4g",
):
    """Draw one labelled bar per parameter.

    Parameters
    ----------
    values : array_like
        Current parameter values, in the model's own units.
    names : sequence of str, optional
        Parameter names, used as the tick labels. Defaults to ``p1``, ``p2``...
    fixed : sequence of bool, optional
        Which parameters are held fixed; those bars are drawn greyed, since a
        bar that never moves would otherwise look like a parameter that has
        converged.
    scale : {"symlog", "log", "linear"}
        Height scale. ``symlog`` is the default because lifetimes span decades
        while ``t0`` sits at zero and may be negative.
    title : str, optional
        Title line, e.g. the iteration count and cost.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    values = np.atleast_1d(np.asarray(values, dtype=float)).ravel()
    n = values.size
    if names is None:
        names = [f"p{i + 1}" for i in range(n)]
    names = [str(v) for v in names][:n]
    if fixed is None:
        fixed = np.zeros(n, dtype=bool)
    fixed = np.atleast_1d(np.asarray(fixed, dtype=bool)).ravel()
    if fixed.size < n:
        fixed = np.concatenate([fixed, np.zeros(n - fixed.size, dtype=bool)])

    if ax is None:
        _, ax = plt.subplots(figsize=(1.5 * max(n, 3), 3.2))
    ax.clear()

    # A non-decaying component has no finite bar to draw; it is marked instead.
    finite = np.isfinite(values)
    heights = np.where(finite, values, 0.0)
    colours = ["0.7" if f else "#2563eb" for f in fixed]
    ax.bar(range(n), heights, color=colours, width=0.62, zorder=2)

    if scale == "symlog":
        ax.set_yscale("symlog", linthresh=_linthresh(values[finite]), linscale=0.4)
    elif scale == "log":
        ax.set_yscale("log")
    ax.axhline(0.0, color="0.4", linewidth=0.8, zorder=1)

    span = float(np.nanmax(np.abs(heights))) if n else 1.0
    for i, value in enumerate(values):
        text = "inf" if not np.isfinite(value) else value_format % value
        offset = 0.06 * (span or 1.0)
        ax.text(
            i,
            heights[i] + (offset if heights[i] >= 0 else -offset),
            text,
            ha="center",
            va="bottom" if heights[i] >= 0 else "top",
            fontsize="small",
            fontweight="bold",
            zorder=3,
        )

    ax.set_xticks(range(n))
    ax.set_xticklabels(names, fontsize="small")
    ax.set_xlim(-0.6, n - 0.4)
    ax.tick_params(axis="x", length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title, fontsize="small", fontweight="bold")
    return ax


def plot_parameter_history(history, names=None, ax=None, *, scale: str = "symlog", title=None):
    """Every parameter against the evaluation number: the path the fit took.

    The companion to :func:`plot_parameter_bars` -- bars answer "where is it
    now?", this answers "how did it get here?", which is what tells a slow fit
    from a stuck one.
    """
    import matplotlib.pyplot as plt

    history = np.atleast_2d(np.asarray(history, dtype=float))
    n_steps, n_params = history.shape
    if names is None:
        names = [f"p{i + 1}" for i in range(n_params)]
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 3.6))

    cmap = plt.get_cmap("turbo")
    for i in range(n_params):
        ax.plot(
            np.arange(n_steps),
            history[:, i],
            marker="o",
            markersize=3,
            linewidth=1.2,
            color=cmap(0.1 + 0.8 * i / max(n_params - 1, 1)),
            label=str(names[i]),
        )
    if scale == "symlog":
        ax.set_yscale("symlog", linthresh=_linthresh(history[np.isfinite(history)]), linscale=0.4)
    elif scale == "log":
        ax.set_yscale("log")
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Parameter value")
    if title:
        ax.set_title(title, fontweight="bold")
    ax.legend(frameon=False, fontsize="small", loc="best")
    return ax
