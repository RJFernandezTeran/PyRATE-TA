"""Rate-matrix builders and the library of predefined target schemes.

Convention:

* ``dc/dt = K c``; off-diagonal ``K[i, j]`` is the rate **from** compartment
  ``j`` **to** compartment ``i``, and ``K[i, i]`` is minus the sum of every rate
  leaving compartment ``i``.
* Rates are ``k = 1 / tau``, with ``tau`` in the dataset's own time unit.
* A compartment whose column sums to a negative number decays to the ground
  state (which is not itself a compartment).

Parallel and sequential schemes are built directly; branched schemes come from
:data:`TARGET_SCHEMES`, keyed by a short identifier.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


def rates_from_lifetimes(taus):
    """``k = 1/tau``, with an **infinite lifetime meaning a zero rate**.

    A component entered as ``inf`` does not decay: it is the constant offset
    that accounts for signal outliving the measured delay window, and its rate
    is exactly zero. Every other lifetime must be finite and positive; zero and
    negative values are refused rather than clamped, since neither has a
    meaning here.
    """
    taus = np.atleast_1d(np.asarray(taus, dtype=float)).ravel()
    if taus.size == 0:
        raise ValueError("at least one lifetime is required")
    bad = np.isnan(taus) | (taus <= 0)
    if np.any(bad):
        raise ValueError(f"lifetimes must be positive (or inf), got {taus[bad].tolist()}")
    with np.errstate(divide="ignore"):
        return np.where(np.isinf(taus), 0.0, 1.0 / np.where(np.isinf(taus), 1.0, taus))


def parallel_K(rates):
    """``K`` for independent parallel decays: ``diag(-k)``."""
    k = np.atleast_1d(np.asarray(rates, dtype=float)).ravel()
    return np.diag(-k)


def sequential_K(rates):
    """``K`` for ``A -> B -> ... -> Z ->``, one branch, no back reactions.

    Compartment ``i`` decays with rate ``k[i]`` and feeds compartment ``i+1``;
    the last one decays to the ground state.
    """
    k = np.atleast_1d(np.asarray(rates, dtype=float)).ravel()
    K = np.diag(-k)
    if k.size > 1:
        K[1:, :-1] += np.diag(k[:-1])
    return K


def default_c0(n_components: int, model_type: str):
    """Initial concentrations for a family: all excited, or only the first.

    A parallel model starts every component at 1 (they are independent); a
    sequential or target model puts all the population in the first
    compartment.
    """
    n = int(n_components)
    if n < 1:
        raise ValueError(f"n_components must be >= 1, got {n}")
    if str(model_type).lower().startswith("parallel"):
        return np.ones(n, dtype=float)
    c0 = np.zeros(n, dtype=float)
    c0[0] = 1.0
    return c0


@dataclass(frozen=True)
class TargetScheme:
    """One predefined compartmental scheme.

    Attributes
    ----------
    key : str
        Short identifier, used in settings and in a saved fit session.
    label : str
        Reaction scheme in the notation used in the GUI and in publications.
    n_species : int
        Number of compartments, i.e. the size of ``K`` and the number of SAS.
    n_rates : int
        Number of independent rate constants (the fitted parameters).
    build : callable
        ``build(k) -> K`` with ``k`` the rate vector, length ``n_rates``.
    species : tuple of str
        Compartment names. Empty for the predefined schemes, which use
        ``A``, ``B``, ...; a scheme written in the text notation carries the
        user's own names, and they travel to the diagram and the result.
    rate_names : tuple of str
        Names of the rate constants, in parameter order. Empty means ``k1``,
        ``k2``, ...
    c0 : tuple of float
        Initial populations. Empty means all population in the first species.
    source_text : str
        The notation this scheme was parsed from, so a session round-trips.
    """

    key: str
    label: str
    n_species: int
    n_rates: int
    build: Callable[[np.ndarray], np.ndarray]
    species: tuple[str, ...] = ()
    rate_names: tuple[str, ...] = ()
    c0: tuple[float, ...] = ()
    source_text: str = ""

    def species_labels(self) -> list[str]:
        """Compartment names, defaulting to ``A``, ``B``, ..."""
        if self.species:
            return list(self.species)
        return [chr(ord("A") + i) for i in range(self.n_species)]

    def parameter_names(self) -> list[str]:
        """Rate-constant names, defaulting to ``k1``, ``k2``, ..."""
        if self.rate_names:
            return list(self.rate_names)
        return [f"k{i + 1}" for i in range(self.n_rates)]

    def rate_matrix(self, rates) -> np.ndarray:
        """Validate the rate vector and build ``K``."""
        k = np.atleast_1d(np.asarray(rates, dtype=float)).ravel()
        if k.size != self.n_rates:
            raise ValueError(f"scheme {self.key!r} needs {self.n_rates} rate(s), got {k.size}")
        K = np.asarray(self.build(k), dtype=float)
        if K.shape != (self.n_species, self.n_species):
            raise ValueError(
                f"scheme {self.key!r} built a {K.shape} matrix, "
                f"expected {(self.n_species, self.n_species)}"
            )
        return K


# Predefined branched schemes. The labels are the reaction schemes; "->"
# without a target means decay to the ground state.
TARGET_SCHEMES: dict[str, TargetScheme] = {
    s.key: s
    for s in (
        TargetScheme(
            "A_eq_B",
            "A <=> B",
            2,
            2,
            lambda k: np.array([[-k[0], k[1]], [k[0], -k[1]]]),
        ),
        TargetScheme(
            "A_eq_B_both_decay",
            "A <=> B; A ->; B ->",
            2,
            4,
            lambda k: np.array([[-(k[0] + k[1]), k[2]], [k[0], -(k[2] + k[3])]]),
        ),
        TargetScheme(
            "A_eq_B_A_decay",
            "A <=> B; A ->",
            2,
            3,
            lambda k: np.array([[-(k[0] + k[1]), k[2]], [k[0], -k[2]]]),
        ),
        TargetScheme(
            "A_eq_B_B_decay",
            "A <=> B; B ->",
            2,
            3,
            lambda k: np.array([[-k[0], k[1]], [k[0], -(k[1] + k[2])]]),
        ),
        TargetScheme(
            "A_eq_B_to_C",
            "A <=> B; B -> C; C ->",
            3,
            4,
            lambda k: np.array(
                [
                    [-k[0], k[1], 0.0],
                    [k[0], -(k[1] + k[2]), 0.0],
                    [0.0, k[2], -k[3]],
                ]
            ),
        ),
        TargetScheme(
            "A_eq_B_eq_C",
            "A <=> B; B <=> C; C ->",
            3,
            5,
            lambda k: np.array(
                [
                    [-k[0], k[1], 0.0],
                    [k[0], -(k[1] + k[2]), k[3]],
                    [0.0, k[2], -(k[3] + k[4])],
                ]
            ),
        ),
        TargetScheme(
            "A_to_B_eq_C",
            "A -> B; B <=> C; B ->; C ->",
            3,
            5,
            lambda k: np.array(
                [
                    [-k[0], 0.0, 0.0],
                    [k[0], -(k[1] + k[2]), k[3]],
                    [0.0, k[1], -(k[3] + k[4])],
                ]
            ),
        ),
        TargetScheme(
            "branch_to_common_D",
            "A -> B -> D; A -> C -> D; D ->",
            4,
            5,
            lambda k: np.array(
                [
                    [-(k[0] + k[1]), 0.0, 0.0, 0.0],
                    [k[1], -k[3], 0.0, 0.0],
                    [k[0], 0.0, -k[2], 0.0],
                    [0.0, k[3], k[2], -k[4]],
                ]
            ),
        ),
        TargetScheme(
            "two_chains_stable_ends",
            "A -> B -> C; A -> D -> E",
            5,
            4,
            lambda k: np.array(
                [
                    [-(k[0] + k[1]), 0.0, 0.0, 0.0, 0.0],
                    [k[0], -k[2], 0.0, 0.0, 0.0],
                    [0.0, k[2], 0.0, 0.0, 0.0],
                    [k[1], 0.0, 0.0, -k[3], 0.0],
                    [0.0, 0.0, 0.0, k[3], 0.0],
                ]
            ),
        ),
        TargetScheme(
            "two_chains_decaying_ends",
            "A -> B -> C ->; A -> D -> E ->",
            5,
            6,
            lambda k: np.array(
                [
                    [-(k[0] + k[1]), 0.0, 0.0, 0.0, 0.0],
                    [k[0], -k[2], 0.0, 0.0, 0.0],
                    [0.0, k[2], -k[3], 0.0, 0.0],
                    [k[1], 0.0, 0.0, -k[4], 0.0],
                    [0.0, 0.0, 0.0, k[4], -k[5]],
                ]
            ),
        ),
        TargetScheme(
            "branch_two_decays",
            "A -> B ->; A -> C ->",
            3,
            4,
            lambda k: np.array(
                [
                    [-(k[0] + k[1]), 0.0, 0.0],
                    [k[0], -k[2], 0.0],
                    [k[1], 0.0, -k[3]],
                ]
            ),
        ),
        TargetScheme(
            "chain_with_equilibrium_end",
            "A -> B -> C <=> D; C ->; D ->",
            4,
            6,
            lambda k: np.array(
                [
                    [-k[0], 0.0, 0.0, 0.0],
                    [k[0], -k[1], 0.0, 0.0],
                    [0.0, k[1], -(k[2] + k[3]), k[4]],
                    [0.0, 0.0, k[2], -(k[4] + k[5])],
                ]
            ),
        ),
    )
}


def get_scheme(key: str) -> TargetScheme:
    """Look up a target scheme by key, listing the alternatives on failure."""
    try:
        return TARGET_SCHEMES[key]
    except KeyError:
        raise KeyError(
            f"unknown target scheme {key!r}; available: {', '.join(sorted(TARGET_SCHEMES))}"
        ) from None
