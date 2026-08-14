"""Concentration profiles ``C(t)`` of a fitted kinetic model.

One of the few views PyMORGAN has no concept of: its plotters

draw *measured* quantities against delay, whereas ``C`` is the model's own
population of each compartment -- dimensionless, never loaded from a file, and
meaningful only next to the species spectra it multiplies.

What is *not* reimplemented here is the delay axis: the scale, limits and
tick-label convention come from ``pymorgan.oneD.plot.apply_delay_axis``, and the
colours from the same ``turbo`` ramp as ``plot_species_spectra``, so component
*i* is the same colour in both figures and the two can be read side by side.

For a parallel model the "profiles" are the bare exponentials of each
component; for a sequential or target model they are the populations of the
compartments, which is where the plot earns its keep -- it shows *when* each
species is present, and therefore which spectrum can be trusted where.
"""

from __future__ import annotations

import numpy as np

from ..log import get_logger

logger = get_logger(__name__)


def species_labels_for(fit, labels=None) -> list[str]:
    """Names for the columns of ``C``.

    Taken from the scheme when the fit has one (so a target fit reads ``ICT``,
    ``T1``, ... rather than ``A``, ``B``), and from the model family otherwise.
    """
    total = int(np.shape(fit.C)[1]) if getattr(fit, "C", None) is not None else len(fit.taus)
    n = max(total - int(getattr(fit, "n_artifact", 0) or 0), 0)
    if labels is not None:
        names = [str(v) for v in labels]
        names += [f"S{i + 1}" for i in range(len(names), n)]
        return _with_artifact(fit, names[:n], total)

    text = getattr(fit, "scheme_text", None)
    if text:
        try:
            # Parsed, not rebuilt: only the names are wanted here, and building
            # the whole scheme would re-announce it in the log on every redraw.
            from ..models.scheme_text import parse_scheme_text

            names = list(parse_scheme_text(text)[1])
            if len(names) >= n:
                return _with_artifact(fit, names[:n], total)
        except Exception:
            logger.debug("could not read species names from the scheme text", exc_info=True)

    key = getattr(fit, "scheme_key", None)
    if key:
        try:
            from ..models.schemes import get_scheme

            names = list(get_scheme(key).species_labels())
            if len(names) >= n:
                return _with_artifact(fit, names[:n], total)
        except Exception:
            logger.debug("unknown scheme key %r", key, exc_info=True)

    return _with_artifact(fit, [chr(ord("A") + i) for i in range(n)], total)


def _with_artifact(fit, names: list[str], total: int) -> list[str]:
    """Append the coherent-artefact column names, which are not species."""
    from ..models.irf import ARTIFACT_LABELS

    n_artifact = int(getattr(fit, "n_artifact", 0) or 0)
    if n_artifact:
        names = names + list(ARTIFACT_LABELS[:n_artifact])
    return names + [f"S{i + 1}" for i in range(len(names), total)]


def _trace_label(name: str, tau, err=None, *, unit=None, fixed=False, settings=None) -> str:
    """``A (320 fs)`` -- the compartment and the lifetime it decays with.

    The number is formatted by PyMORGAN's ``format_lifetime_label``, the same
    one the species spectra use, so the two figures agree on the unit, the
    rounding and the ``±`` -- and both follow ``round_labels`` / ``label_digits``.
    Whether the uncertainty is shown at all, and whether it is rounded together
    with the value, are PyRATE's ``[plots]`` settings.
    """
    from pymorgan.helpers import format_lifetime_label

    from ..settings import get_settings

    if tau is None:
        return name
    quoting = get_settings()  # PyRATE's; ``settings`` is PyMORGAN's aesthetics
    text = format_lifetime_label(
        tau,
        unit or "",
        err=err if quoting.show_uncertainties else None,
        fixed=fixed,
        pair_round=bool(quoting.round_uncertainties),
        settings=settings,
    )
    return f"{name} ({text})"


def plot_concentrations(
    fit,
    ax=None,
    fig=None,
    *,
    labels=None,
    normalise: bool = False,
    settings=None,
    time_axis_scale=None,
    title=None,
    **kwargs,
):
    """Plot the population of each species against delay.

    Parameters
    ----------
    fit : KineticFit
        A fit (or preview) result carrying ``C``. A result without ``C`` is
        refused rather than drawn empty -- there is nothing to show, and a blank
        axis would look like a fit with no population.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on; a new figure is created when omitted.
    labels : sequence of str, optional
        Species names. Defaults to the scheme's own names.
    normalise : bool, default False
        Scale each profile to its own maximum. Useful when one compartment
        never accumulates much; misleading if the relative populations are the
        point, which is why it is off by default.
    time_axis_scale : {"symlog", "log", "lin"}, optional
        Delay-axis scale; defaults to the active PyMORGAN setting, i.e. the
        same axis the contour and the kinetic traces use.
    **kwargs
        Passed to ``ax.plot``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    import pymorgan as pm
    from pymorgan import helpers as hlp
    from pymorgan.oneD.plot import apply_delay_axis

    C = getattr(fit, "C", None)
    if C is None:
        raise ValueError(
            "this result carries no concentration profiles (C is None), so there "
            "is nothing to plot; refit with an engine that returns them"
        )
    C = np.atleast_2d(np.asarray(C, dtype=float))
    t = np.asarray(fit.t, dtype=float).ravel()
    if C.shape[0] != t.size:
        raise ValueError(f"C has {C.shape[0]} rows but the delay axis has {t.size} points")

    pm.apply_style()
    s = settings if settings is not None else pm.get_settings()
    if ax is None:
        # Same size as a stand-alone kinetics figure: it is the same axis.
        _, ax = plt.subplots(figsize=tuple(getattr(s, "kinetics_figsize", (7.5, 4.5))))
    fig = fig if fig is not None else ax.figure

    names = species_labels_for(fit, labels)
    taus = list(np.asarray(fit.taus, dtype=float))
    errs = list(np.asarray(getattr(fit, "tau_err", []), dtype=float))
    fixed = list(np.asarray(getattr(fit, "is_fixed", []), dtype=bool))
    unit = (getattr(fit, "settings", None) or {}).get("time_unit") or ""

    n_artifact = int(getattr(fit, "n_artifact", 0) or 0)
    n_kinetic = getattr(fit, "n_kinetic", C.shape[1] - n_artifact)
    n_kinetic = max(int(n_kinetic), 0)

    # Shaded grey IRF area (alpha=0.1) if IRF parameters are available
    t0 = float(getattr(fit, "t0", 0.0) or 0.0)
    irf_fwhm = getattr(fit, "irf_fwhm", None)
    if irf_fwhm is not None and float(irf_fwhm) > 0:
        sigma_irf = float(irf_fwhm) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        irf_curve = np.exp(-0.5 * ((t - t0) / sigma_irf) ** 2)
        if normalise:
            irf_peak = 1.0
        else:
            irf_peak = float(np.nanmax(C[:, :n_kinetic])) if n_kinetic > 0 else 1.0
        irf_curve = irf_curve * (irf_peak if irf_peak > 0 else 1.0)
        ax.fill_between(t, 0, irf_curve, color="gray", alpha=0.1, zorder=0, label="IRF")

    # Same ramp as plot_species_spectra, so species i is the same colour there.
    cm = hlp.adjust_cmap(plt.cm.turbo(np.linspace(0, 1, n_kinetic + 1)), 0.9)

    for i in range(n_kinetic):
        y = C[:, i]
        if normalise:
            peak = np.nanmax(np.abs(y))
            y = y / peak if peak else y
        ax.plot(
            t,
            y,
            color=cm[i],
            linewidth=float(getattr(s, "kinetics_line_width", 1.5)),
            label=_trace_label(
                names[i] if i < len(names) else f"S{i + 1}",
                taus[i] if i < len(taus) else None,
                errs[i] if i < len(errs) else None,
                unit=unit,
                fixed=bool(fixed[i]) if i < len(fixed) else False,
                settings=s,
            ),
            **kwargs,
        )

    ax.axhline(0.0, color="0.75", linewidth=0.75, zorder=0)
    ax.axvline(0.0, color="0.75", linewidth=0.75, zorder=0)
    apply_delay_axis(ax, "x", time_axis_scale, t, settings=s)

    ax.set_xlabel(_delay_label(unit, getattr(s, "label_style", "()")))
    ax.set_ylabel("Norm. population" if normalise else "Relative population")
    if title is None:
        title = f"Concentration profiles - {getattr(fit, 'model_type', '')}".strip(" -")
    if title:
        ax.set_title(title, fontweight="bold")
    ax.legend(frameon=False, handlelength=0.9, loc="best")
    if fig is not None and ax.get_subplotspec() is not None:
        try:
            fig.tight_layout()
        except Exception:  # pragma: no cover - constrained layouts refuse
            logger.debug("tight_layout declined", exc_info=True)
    return ax



def _delay_label(unit, label_style) -> str:
    """``Delay (ps)`` in whichever bracket convention the settings ask for."""
    style = str(getattr(label_style, "value", label_style) or "()")
    if not unit:
        return "Delay"
    return {
        "()": f"Delay ({unit})",
        "[]": f"Delay [{unit}]",
        "/": f"Delay / {unit}",
    }.get(style, f"Delay ({unit})")
