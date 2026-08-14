"""A text notation for kinetic schemes, and its parser.

Instead of typing a rate matrix -- where a sign error or a missing diagonal
term is invisible until the fit misbehaves -- a scheme is written as the
reactions themselves, one per line::

    A  ->  B   : k1        # A decays into B with rate k1
    B <->  C   : k2, k3     # equilibrium: forward k2, backward k3
    C  ->      : k4         # decay to the ground state (no product)
    B  ->      : k5         # a second channel out of B: B branches

    init A = 1              # optional; the default is all population in the
                            # first species mentioned

The matrix is then *derived*, so it is correct by construction: every reaction
contributes ``+k`` to the product's row and ``-k`` to the reactant's diagonal,
and the ground state is simply the absence of a product.

Why this notation
-----------------
* **Species are named, not numbered.** ``ICT``, ``T1``, ``GS_hot`` all work, and
  the names appear on the diagram and in the result, so a scheme reads as
  chemistry rather than as indices.
* **Rates are named, so they can be shared.** Writing ``k1`` on two arrows makes
  it *one* fitted parameter used twice -- the natural way to impose "these two
  channels have the same rate", which a matrix cannot express without a
  bespoke builder.
* **Any topology.** Chains, branches, equilibria, cycles, disconnected pools and
  non-decaying products are all just lines; nothing is special-cased.
* **It round-trips.** :func:`scheme_to_text` regenerates the notation from a
  parsed scheme, so a session can be saved, edited and reloaded as text.

Errors carry the line number and the offending text: a scheme that does not
parse is refused, never half-built.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from ..log import get_logger

logger = get_logger(__name__)

# ``A -> B : k1``  /  ``A <-> B : k1, k2``  /  ``A -> : k1``
_ARROW = re.compile(r"<->|<=>|-->|->|=>")
_INIT = re.compile(r"^\s*(?:init|c0)\s*[: ]\s*(.+)$", re.IGNORECASE)
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_*'+-]*$")
_GROUND_TOKENS = frozenset({"", "0", "gs", "ground", "g", "-", "none"})


class SchemeSyntaxError(ValueError):
    """A scheme could not be parsed; the message names the line."""


@dataclass(frozen=True)
class ParsedReaction:
    """One arrow: ``source -> target`` at rate ``rate``.

    ``target`` is ``None`` for decay to the ground state, which is not a
    compartment and therefore never a column of ``K``.
    """

    source: str
    target: str | None
    rate: str
    line: int


def _clean(text: str) -> list[tuple[int, str]]:
    """Numbered, comment-free, non-empty lines."""
    out = []
    for number, raw in enumerate(str(text).splitlines(), start=1):
        line = raw.split("#", 1)[0].split("%", 1)[0].strip()
        if line:
            out.append((number, line))
    return out


def _species_or_ground(token: str, line: int) -> str | None:
    name = token.strip()
    if name.lower() in _GROUND_TOKENS:
        return None
    if not _NAME.match(name):
        raise SchemeSyntaxError(f"line {line}: {name!r} is not a valid species name")
    return name


def _parse_rates(token: str, line: int, expected: int) -> list[str]:
    names = [r.strip() for r in token.split(",") if r.strip()]
    if len(names) != expected:
        word = "rate" if expected == 1 else "rates"
        raise SchemeSyntaxError(
            f"line {line}: expected {expected} {word} after ':', got {len(names)}"
        )
    for name in names:
        if not _NAME.match(name):
            raise SchemeSyntaxError(f"line {line}: {name!r} is not a valid rate name")
    return names


def parse_scheme_text(text: str):
    """Parse the notation into ``(reactions, species, rate_names, c0)``.

    Raises
    ------
    SchemeSyntaxError
        On any malformed line, with the line number. Nothing partial is
        returned: a scheme either parses or it does not.
    """
    reactions: list[ParsedReaction] = []
    species: list[str] = []
    rate_names: list[str] = []
    initial: dict[str, float] = {}

    def see(name: str | None) -> None:
        if name is not None and name not in species:
            species.append(name)

    for number, line in _clean(text):
        init_match = _INIT.match(line)
        if init_match:
            for chunk in re.split(r"[,;]", init_match.group(1)):
                if not chunk.strip():
                    continue
                if "=" not in chunk:
                    raise SchemeSyntaxError(f"line {number}: expected 'init NAME = value'")
                name, value = (part.strip() for part in chunk.split("=", 1))
                if not _NAME.match(name):
                    raise SchemeSyntaxError(f"line {number}: {name!r} is not a valid species name")
                try:
                    initial[name] = float(value.replace(",", "."))
                except ValueError:
                    raise SchemeSyntaxError(f"line {number}: {value!r} is not a number") from None
            continue

        if ":" not in line:
            raise SchemeSyntaxError(
                f"line {number}: no rate given. Write 'A -> B : k1' (rates follow a colon)."
            )
        reaction_part, rate_part = line.split(":", 1)
        arrow = _ARROW.search(reaction_part)
        if arrow is None:
            raise SchemeSyntaxError(
                f"line {number}: no arrow found. Use '->' for a step or '<->' for an equilibrium."
            )

        left = _species_or_ground(reaction_part[: arrow.start()], number)
        right = _species_or_ground(reaction_part[arrow.end() :], number)
        reversible = arrow.group() in ("<->", "<=>")

        if left is None:
            raise SchemeSyntaxError(
                f"line {number}: a reaction must start from a species, not the ground state."
            )
        if reversible and right is None:
            raise SchemeSyntaxError(f"line {number}: an equilibrium needs a species on both sides.")

        rates = _parse_rates(rate_part, number, 2 if reversible else 1)
        see(left)
        see(right)
        reactions.append(ParsedReaction(left, right, rates[0], number))
        if reversible:
            reactions.append(ParsedReaction(right, left, rates[1], number))
        for name in rates:
            if name not in rate_names:
                rate_names.append(name)

    if not reactions:
        raise SchemeSyntaxError(
            "the scheme is empty: write at least one reaction, e.g. 'A -> : k1'"
        )

    unknown = set(initial) - set(species)
    if unknown:
        raise SchemeSyntaxError(
            f"init refers to species that take part in no reaction: {sorted(unknown)}"
        )

    c0 = np.zeros(len(species), dtype=float)
    if initial:
        for name, value in initial.items():
            c0[species.index(name)] = value
    else:
        c0[0] = 1.0  # all population starts in the first species mentioned
    return reactions, species, rate_names, c0


def build_rate_matrix(reactions, species, rate_names, k) -> np.ndarray:
    """Assemble ``K`` from parsed reactions and a rate vector.

    Each reaction moves population out of its source (``-k`` on the diagonal)
    and, unless it decays to the ground state, into its target (``+k`` off the
    diagonal). Building it this way makes the matrix consistent by
    construction: the columns cannot fail to balance.
    """
    k = np.atleast_1d(np.asarray(k, dtype=float)).ravel()
    if k.size != len(rate_names):
        raise ValueError(f"expected {len(rate_names)} rate(s), got {k.size}")
    index = {name: i for i, name in enumerate(species)}
    rate_index = {name: i for i, name in enumerate(rate_names)}

    K = np.zeros((len(species), len(species)), dtype=float)
    for reaction in reactions:
        value = k[rate_index[reaction.rate]]
        source = index[reaction.source]
        K[source, source] -= value
        if reaction.target is not None:
            K[index[reaction.target], source] += value
    return K


def scheme_from_text(text: str, key: str = "custom"):
    """Build a :class:`~pyrate_ta.models.schemes.TargetScheme` from the notation.

    The returned scheme carries the species and rate names, so the diagram, the
    lifetime table and the result all speak the user's own vocabulary.
    """
    from .schemes import TargetScheme

    reactions, species, rate_names, c0 = parse_scheme_text(text)

    def build(k):
        return build_rate_matrix(reactions, species, rate_names, k)

    scheme = TargetScheme(
        key=key,
        label=scheme_to_text(reactions, compact=True),
        n_species=len(species),
        n_rates=len(rate_names),
        build=build,
        species=tuple(species),
        rate_names=tuple(rate_names),
        c0=tuple(float(v) for v in c0),
        source_text=str(text).strip(),
    )
    logger.info(
        "Parsed scheme: %d species (%s), %d rate constant(s) (%s).",
        len(species),
        ", ".join(species),
        len(rate_names),
        ", ".join(rate_names),
    )
    return scheme


def scheme_to_text(reactions, compact: bool = False) -> str:
    """Render parsed reactions back into the notation.

    Opposed pairs sharing the same two species are folded into a single
    ``<->`` line, so a scheme that was written as an equilibrium reads back as
    one.
    """
    lines: list[str] = []
    used: set[int] = set()
    for i, reaction in enumerate(reactions):
        if i in used:
            continue
        partner = next(
            (
                j
                for j, other in enumerate(reactions)
                if j > i
                and j not in used
                and other.source == reaction.target
                and other.target == reaction.source
            ),
            None,
        )
        if partner is not None:
            used.add(partner)
            lines.append(
                f"{reaction.source} <-> {reaction.target} : {reaction.rate}, "
                f"{reactions[partner].rate}"
            )
        else:
            target = reaction.target or ""
            lines.append(f"{reaction.source} -> {target} : {reaction.rate}".replace("  :", " :"))
        used.add(i)
    return "; ".join(lines) if compact else "\n".join(lines)


def check_scheme_text(text: str) -> tuple[bool, str]:
    """Validate without raising: ``(ok, message)`` for a GUI to display.

    The message is the reason on failure, and a short summary of what was
    understood on success -- species, rate constants, and which compartments
    decay to the ground state.
    """
    try:
        scheme = scheme_from_text(text)
    except (SchemeSyntaxError, ValueError) as exc:
        return False, str(exc)

    K = scheme.rate_matrix(np.ones(scheme.n_rates))
    leaking = [
        scheme.species[j] for j in range(scheme.n_species) if -float(np.sum(K[:, j])) > 1e-12
    ]
    closed = (
        "closed (population conserved)"
        if not leaking
        else ("decays to the ground state from " + ", ".join(leaking))
    )
    return True, (
        f"{scheme.n_species} species ({', '.join(scheme.species)}), "
        f"{scheme.n_rates} rate constant(s) ({', '.join(scheme.rate_names)}); {closed}."
    )
