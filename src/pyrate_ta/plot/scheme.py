"""Compartment (K-matrix) diagrams.

A kinetic scheme is a directed graph: one node per compartment, one arrow per
non-zero off-diagonal rate, plus an arrow to the ground state wherever a
compartment loses population to it (a column of ``K`` summing to a negative
number). PyMORGAN has no concept of this, so the plotter lives here.


Drawn with plain Matplotlib patches rather than a graph library: the layouts
worth having are simple (a chain, a stack, a ring), and the arrow geometry and
label placement need control that a generic spring layout does not give.

Everything aesthetic that PyMORGAN owns -- font sizes, the colour cycle -- is
taken from its style, so a scheme figure sits beside the spectra without
looking foreign.
"""

from __future__ import annotations

import re

import numpy as np

from ..log import get_logger

logger = get_logger(__name__)

# Rates below this fraction of the largest rate are treated as absent, so that
# a numerically-zero entry does not sprout a meaningless arrow.
_RATE_TOL = 1e-12

_GROUND = "0"


def rate_symbols(builder, n_rates: int, n_species: int, tol: float = _RATE_TOL) -> dict:
    """Which rate constants sit on which arrow.

    Feeding the ``k -> K`` builder a unit vector shows where ``k_i`` appears, so
    the diagram can be labelled with the symbols themselves rather than with
    numbers. This matters for setting up a fit: the arrow tells you which entry
    of the rate table you are about to bound.

    An edge may carry more than one constant -- decay to the ground state from a
    branching compartment is the sum of everything leaving it that goes nowhere
    else -- in which case the label is a sum.

    Returns
    -------
    dict
        ``(source, target) -> [rate indices]``, with ``target = None`` for the
        ground state. Indices are zero-based.
    """
    contributions: dict[tuple, list[int]] = {}
    for i in range(int(n_rates)):
        probe = np.zeros(int(n_rates))
        probe[i] = 1.0
        K = np.asarray(builder(probe), dtype=float)
        for j in range(n_species):
            for row in range(n_species):
                if row != j and abs(K[row, j]) > tol:
                    contributions.setdefault((j, row), []).append(i)
            leak = -float(np.sum(K[:, j]))
            if leak > tol:
                contributions.setdefault((j, None), []).append(i)
    return contributions


def _symbol_label(indices, names=None, symbol: str = "k") -> str:
    """``k_1``, or ``k_1+k_2`` when several constants share an arrow (mathtext).

    ``names`` are the scheme's own rate names when it has them, so a scheme
    written as ``S1 -> ICT : k_ct`` is labelled ``k_ct`` and not ``k_1``.
    """
    if not indices:
        return ""

    def one(i: int) -> str:
        if names is None or i >= len(names):
            return f"{symbol}_{{{i + 1}}}"
        name = str(names[i])
        base, _, sub = name.partition("_")
        if sub:
            return f"{base}_{{{sub}}}"
        # ``k1`` and friends are written with the digits as a subscript.
        trailing = re.match(r"^([A-Za-z]+)(\d+)$", name)
        return f"{trailing.group(1)}_{{{trailing.group(2)}}}" if trailing else name

    return "$" + "+".join(one(i) for i in sorted(indices)) + "$"


def _scheme_of(obj):
    """The scheme behind a fit result, whichever way it was defined.

    A result records its scheme twice: as ``scheme_text`` (the notation, always
    present for a scheme that was typed) and as ``scheme_key``. Text is tried
    first and parsed, so a *custom* scheme recovers its species names and rate
    symbols -- previously only the key was consulted, and a typed scheme stores
    the key ``"custom"``, which is in no registry, so drawing one raised
    ``KeyError`` instead of drawing.

    Returns ``None`` when neither is usable; the caller then falls back to
    ``A``, ``B``, ... and values on the arrows, which is a poorer diagram but
    still a diagram.
    """
    text = getattr(obj, "scheme_text", None)
    if text:
        try:
            from ..models.scheme_text import scheme_from_text

            return scheme_from_text(text)
        except Exception:
            logger.debug("could not parse the recorded scheme text", exc_info=True)

    key = getattr(obj, "scheme_key", None)
    if key:
        try:
            from ..models.schemes import get_scheme

            return get_scheme(key)
        except Exception:
            # "custom" and any renamed template land here: not an error, just
            # nothing to look up.
            logger.debug("no registered scheme named %r", key, exc_info=True)
    return None


def _rate_matrix_from(obj, taus=None):
    """Accept a model, a scheme, a result or a plain matrix.

    Returns ``(K, labels, model_type, builder, n_rates, rate_names)``;
    ``builder`` is the ``k -> K`` callable when one is known, so the arrows can
    be labelled with the rate symbols, and ``rate_names`` are the scheme's own
    names when it has them.
    """
    from ..models.base import KineticModel
    from ..models.schemes import TargetScheme

    if isinstance(obj, KineticModel):
        if taus is None:
            raise ValueError("a model needs lifetimes: plot_scheme(model, taus=[...])")
        scheme = getattr(obj, "scheme", None)
        return (
            obj.rate_matrix(taus),
            obj.species_labels(),
            str(obj.model_type),
            obj.rate_builder(),
            obj.n_lifetimes,
            scheme.parameter_names() if scheme is not None else None,
        )

    if isinstance(obj, TargetScheme):
        if taus is None:
            raise ValueError("a scheme needs lifetimes: plot_scheme(scheme, taus=[...])")
        from ..models.schemes import rates_from_lifetimes

        K = obj.rate_matrix(rates_from_lifetimes(taus))
        return (
            K,
            obj.species_labels(),
            "Target",
            obj.build,
            obj.n_rates,
            obj.parameter_names(),
        )

    K = getattr(obj, "K", None)  # a TargetFit
    if K is not None:
        n = np.shape(K)[0]
        scheme = _scheme_of(obj)
        return (
            np.asarray(K, dtype=float),
            scheme.species_labels() if scheme else [chr(ord("A") + i) for i in range(n)],
            "Target",
            scheme.build if scheme else None,
            scheme.n_rates if scheme else 0,
            scheme.parameter_names() if scheme else None,
        )

    if hasattr(obj, "taus") and hasattr(obj, "model_type"):  # a Kinetic/GlobalFit
        from ..models import make_model

        model = make_model(obj.model_type, int(obj.n_components))
        return (
            model.rate_matrix(obj.taus),
            model.species_labels(),
            str(obj.model_type),
            model.rate_builder(),
            model.n_lifetimes,
            None,
        )

    K = np.atleast_2d(np.asarray(obj, dtype=float))
    if K.shape[0] != K.shape[1]:
        raise ValueError(f"expected a square rate matrix, got shape {K.shape}")
    # A bare matrix carries no parameterisation, so no symbols can be derived.
    return K, [chr(ord("A") + i) for i in range(K.shape[0])], "Target", None, 0, None


def _edges(K, tol: float = _RATE_TOL):
    """Transfer and decay edges of ``K`` as ``(source, target, rate)`` triples.

    ``target`` is the index of the receiving compartment, or ``None`` for decay
    to the ground state. A compartment whose column sums to zero is closed and
    contributes no ground-state arrow.
    """
    K = np.asarray(K, dtype=float)
    n = K.shape[0]
    scale = float(np.max(np.abs(K))) or 1.0
    out = []
    for j in range(n):
        for i in range(n):
            if i != j and K[i, j] > tol * scale:
                out.append((j, i, float(K[i, j])))
        # What leaves compartment j but arrives nowhere went to the ground state.
        leak = -float(np.sum(K[:, j]))
        if leak > tol * scale:
            out.append((j, None, leak))
    return out


def _levels(edges, n: int) -> list[int]:
    """Distance of each compartment from the first one, following transfers.

    Breadth-first, so a branched scheme is drawn in layers -- the initially
    excited state first, its products after it -- rather than on a ring, where
    the arrows cross. A compartment unreachable from the first one starts its
    own layer 0.
    """
    adjacency: dict[int, list[int]] = {i: [] for i in range(n)}
    for src, dst, _ in edges:
        if dst is not None:
            adjacency[src].append(dst)

    level = [-1] * n
    for start in range(n):
        if level[start] != -1:
            continue
        level[start] = 0
        queue = [start]
        while queue:
            node = queue.pop(0)
            for nxt in adjacency[node]:
                if level[nxt] == -1:
                    level[nxt] = level[node] + 1
                    queue.append(nxt)
    return level


def _layout(n: int, model_type: str, has_ground: bool, edges=None):
    """Node positions for ``n`` compartments (plus the ground state).

    Every family is laid out **horizontally**: population flows left to right
    and the ground state sits below. The diagrams share a window with wide,
    short figures (the scheme dialog, the pop-out), where a tall column of
    compartments wastes most of the width and shrinks the nodes for nothing.
    """
    mt = str(model_type).lower()
    if mt.startswith("seq") and n > 1:
        pos = {i: (float(i), 0.0) for i in range(n)}
        ground = ((n - 1) / 2.0, -1.1)
    elif mt.startswith("par"):
        # Independent decays: a row of compartments over a shared ground state,
        # so the arrows fan in instead of stacking up a column.
        pos = {i: (float(i), 0.0) for i in range(n)}
        ground = ((n - 1) / 2.0, -1.2)
    else:
        level = _levels(edges or [], n)
        by_level: dict[int, list[int]] = {}
        for i, lv in enumerate(level):
            by_level.setdefault(lv, []).append(i)
        pos = {}
        for lv, members in by_level.items():
            spread = len(members) - 1
            for k, node in enumerate(members):
                # Layers advance along x; the members of a layer spread in y.
                pos[node] = (1.25 * lv, 1.3 * (spread / 2.0 - float(k)))
        ground = (1.25 * (max(by_level) + 1), 0.0)
    if has_ground:
        pos[_GROUND] = ground
    return pos


def _passes_through_a_node(pos, src, dst, node_size: float, tol: float = 1.2) -> bool:
    """Whether the straight line from ``src`` to ``dst`` runs over another node."""
    x0, y0 = pos[src]
    x1, y1 = pos[dst]
    dx, dy = x1 - x0, y1 - y0
    length = float(np.hypot(dx, dy))
    if length == 0:
        return False
    for key, (x, y) in pos.items():
        if key in (src, dst):
            continue
        # Projection of the node onto the segment, and its distance from it.
        s_along = ((x - x0) * dx + (y - y0) * dy) / length**2
        if not 0.0 < s_along < 1.0:
            continue
        distance = abs((x - x0) * dy - (y - y0) * dx) / length
        if distance < tol * node_size:
            return True
    return False


def plot_scheme(
    obj,
    taus=None,
    ax=None,
    *,
    labels=None,
    rate_labels: str = "symbol",
    show_ground_state: bool = True,
    node_size: float = 0.22,
    fontsize: float | None = None,
    title: str | None = None,
):
    """Draw the compartment diagram of a kinetic scheme.

    Parameters
    ----------
    obj : KineticModel, TargetScheme, fit result or array
        Anything that defines a rate matrix. A model or a bare scheme also
        needs ``taus``; a fit result already carries them.
    taus : sequence, optional
        Lifetimes, when ``obj`` does not carry them.
    ax : matplotlib.axes.Axes, optional
        Axis to draw into; a new figure is created when omitted.
    labels : sequence of str, optional
        Compartment names. Defaults to ``A``, ``B``, ``C``, ...
    rate_labels : {"symbol", "value", "none"}
        What to write on each arrow. ``"symbol"`` (the default) writes the rate
        constants themselves -- ``k_1``, ``k_2``, and a sum such as
        ``k_1+k_2`` where a compartment branches -- which is what tells you
        which row of the rate table an arrow belongs to when setting values and
        bounds. ``"value"`` writes the lifetime of that step (``1/k``), which
        for a branching compartment is *not* the lifetime with which it is
        observed to decay: that one is the inverse of the summed rate.
    show_ground_state : bool
        Draw the ground state as a separate grey node.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch

    try:
        import pymorgan as pm

        pm.apply_style()
    except Exception:  # PyMORGAN owns the aesthetics, but a scheme is drawable without it
        logger.debug("PyMORGAN style unavailable; using the Matplotlib defaults", exc_info=True)

    K, default_labels, model_type, builder, n_rates, rate_names = _rate_matrix_from(obj, taus)
    names = list(labels) if labels is not None else default_labels
    n = K.shape[0]
    if len(names) != n:
        raise ValueError(f"got {len(names)} labels for {n} compartments")

    edges = _edges(K)
    symbols: dict = {}
    if rate_labels == "symbol":
        if builder is None:
            logger.info(
                "No rate parameterisation available for this object; labelling the "
                "arrows with values instead of symbols."
            )
            rate_labels = "value"
        else:
            symbols = rate_symbols(builder, n_rates, K.shape[0])
    has_ground = show_ground_state and any(target is None for _, target, _ in edges)
    pos = _layout(n, model_type, has_ground, edges)

    if ax is None:
        # Wider than tall, to match the horizontal layout (see ``_layout``).
        _, ax = plt.subplots(figsize=(5.8, 3.4))
    if fontsize is None:
        fontsize = plt.rcParams.get("font.size", 10)

    cmap = plt.get_cmap("turbo")
    colours = [cmap(x) for x in np.linspace(0.1, 0.9, n)]

    # --- arrows first, so the nodes sit on top of the arrow ends ----------- #
    seen: set[tuple] = set()
    for src, dst, rate in edges:
        key_dst = _GROUND if dst is None else dst
        if key_dst not in pos:
            continue
        x0, y0 = pos[src]
        x1, y1 = pos[key_dst]
        # Curve both directions of an equilibrium by the same amount *in their
        # own frame*, which puts them on opposite sides of the line; negating
        # one would draw them on top of each other.
        reverse = any(s == key_dst and d == src for s, d, _ in edges)
        rad = 0.25 if reverse else 0.0
        if not reverse and _passes_through_a_node(pos, src, key_dst, node_size):
            # A straight arrow that runs over an intermediate compartment is
            # invisible; bow it out instead.
            rad = 0.45
        seen.add((src, key_dst))

        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                mutation_scale=14,
                shrinkA=node_size * 72,
                shrinkB=node_size * 72,
                linewidth=1.4,
                color="0.25",
                zorder=1,
            )
        )
        label = ""
        if rate_labels == "symbol":
            label = _symbol_label(symbols.get((src, dst), []), rate_names)
        elif rate_labels == "value" and rate > 0:
            label = f"{1.0 / rate:.3g}"
        if label:
            mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            # Offset the label off the arrow, on the side the curve bends to.
            dx, dy = x1 - x0, y1 - y0
            norm = np.hypot(dx, dy) or 1.0
            # A curved arrow's midpoint sits rad*|chord|/2 off the chord, so
            # the label follows the curve instead of floating beside it.
            off = 0.13 + abs(rad) * norm * 0.5
            # Matplotlib's arc3 bows towards +90 degrees from the direction of
            # travel for a positive rad, so the label goes to the same side.
            mx += dy / norm * off
            my += -dx / norm * off
            ax.text(
                mx,
                my,
                label,
                ha="center",
                va="center",
                fontsize=fontsize * 0.85,
                zorder=3,
            )

    # --- nodes -------------------------------------------------------------- #
    for i in range(n):
        x, y = pos[i]
        ax.add_patch(Circle((x, y), node_size, facecolor=colours[i], alpha=0.65, zorder=2))
        ax.text(
            x, y, names[i], ha="center", va="center", fontsize=fontsize, fontweight="bold", zorder=4
        )
    if has_ground:
        x, y = pos[_GROUND]
        ax.add_patch(Circle((x, y), node_size * 0.85, facecolor="0.8", alpha=0.9, zorder=2))
        ax.text(x, y, "GS", ha="center", va="center", fontsize=fontsize * 0.8, zorder=4)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    pad = node_size * 3.0
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    if title is None:
        title = f"{model_type} scheme"
    ax.set_title(title, fontweight="bold")
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax.figure.tight_layout()
    except Exception:
        pass
    return ax



def scheme_text(obj, taus=None, labels=None) -> str:
    """The scheme as a one-line string, e.g. ``A -> B; B <=> C; C ->``.

    Useful in a log line or a result summary, where a figure cannot go.
    """
    K, default_labels, *_ = _rate_matrix_from(obj, taus)
    names = list(labels) if labels is not None else default_labels
    parts = []
    seen = set()
    for src, dst, _ in _edges(K):
        if dst is None:
            parts.append(f"{names[src]} ->")
            continue
        if (dst, src) in seen:  # already written as the forward direction
            parts = [
                p.replace(f"{names[dst]} -> {names[src]}", f"{names[dst]} <=> {names[src]}")
                for p in parts
            ]
            continue
        seen.add((src, dst))
        parts.append(f"{names[src]} -> {names[dst]}")
    return "; ".join(parts)
