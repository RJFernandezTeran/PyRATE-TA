"""Kinetic models: parameter layout, bounds, and the map to ``C(t)``.

A model owns the parameter vector and the rate matrix; it does **not** own the
spectra. The data matrix is ``D ~ C @ S.T`` and solving for ``S`` is the
fitters' job (:mod:`pyrate_ta.fit`), which is what makes variable projection
possible.

The non-linear parameter vector is, in this order:

    ``[tau_1, ..., tau_m]  (+ t0)  (+ irf_fwhm)``

with ``t0`` and ``irf_fwhm`` appended only when they are being fitted. The
lifetimes are in the dataset's own time unit and are never converted here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..cite import cite
from ..log import get_logger
from ..settings import ModelType
from .irf import ARTIFACT_LABELS, coherent_artifact_basis
from .propagator import concentrations
from .schemes import (
    TargetScheme,
    default_c0,
    get_scheme,
    parallel_K,
    rates_from_lifetimes,
    sequential_K,
)

logger = get_logger(__name__)


@dataclass
class KineticModel:
    """Base class: a rate matrix plus the parameter bookkeeping around it.

    Subclasses implement :meth:`rate_matrix`; everything else -- parameter
    names, bounds, initial guesses, and the concentration matrix -- follows
    from it.

    Attributes
    ----------
    n_components : int
        Number of compartments, i.e. the number of columns of ``C`` and hence
        of species spectra.
    fit_t0, fit_irf : bool
        Whether time zero and the IRF width are free parameters. They are
        appended to the parameter vector in that order when free.
    t0, irf_fwhm : float
        Values used when the corresponding flag is ``False``. ``irf_fwhm = 0``
        (or ``None``) means no IRF convolution at all.
    """

    n_components: int
    fit_t0: bool = False
    fit_irf: bool = False
    t0: float = 0.0
    irf_fwhm: float | None = None
    #: Add the IRF and its first two derivatives to the design matrix, to
    #: absorb cross-phase modulation and other time-zero artefacts.
    coherent_artifact: bool = False

    #: Family, which decides whether the spectra are DAS, EAS or SAS.
    model_type: ModelType = field(init=False, default=ModelType.PARALLEL)

    def __post_init__(self) -> None:
        if int(self.n_components) < 1:
            raise ValueError(f"n_components must be >= 1, got {self.n_components}")
        self.n_components = int(self.n_components)

    # ------------------------------------------------------------------ #
    #                          Parameter layout                          #
    # ------------------------------------------------------------------ #
    @property
    def n_lifetimes(self) -> int:
        """Number of independent lifetimes (not necessarily ``n_components``)."""
        return self.n_components

    def lifetime_names(self) -> list[str]:
        return [f"tau{i + 1}" for i in range(self.n_lifetimes)]

    def param_names(self) -> list[str]:
        """Names of the free parameters, in vector order."""
        names = self.lifetime_names()
        if self.fit_t0:
            names.append("t0")
        if self.fit_irf:
            names.append("irf_fwhm")
        return names

    @property
    def n_params(self) -> int:
        return len(self.param_names())

    def pack(self, taus, t0: float | None = None, irf_fwhm: float | None = None):
        """Assemble the parameter vector from named pieces."""
        vec = list(np.atleast_1d(np.asarray(taus, dtype=float)).ravel())
        if len(vec) != self.n_lifetimes:
            raise ValueError(f"expected {self.n_lifetimes} lifetime(s), got {len(vec)}")
        if self.fit_t0:
            vec.append(float(self.t0 if t0 is None else t0))
        if self.fit_irf:
            width = self.irf_fwhm if irf_fwhm is None else irf_fwhm
            vec.append(float(0.0 if width is None else width))
        return np.asarray(vec, dtype=float)

    def unpack(self, params) -> tuple[np.ndarray, float, float | None]:
        """Split a parameter vector into ``(taus, t0, irf_fwhm)``.

        Values that are not being fitted come from the model's own attributes,
        so a caller never has to remember which are free.
        """
        p = np.atleast_1d(np.asarray(params, dtype=float)).ravel()
        if p.size != self.n_params:
            raise ValueError(f"expected {self.n_params} parameter(s), got {p.size}")
        taus = p[: self.n_lifetimes]
        idx = self.n_lifetimes
        t0 = float(p[idx]) if self.fit_t0 else float(self.t0)
        if self.fit_t0:
            idx += 1
        irf = float(p[idx]) if self.fit_irf else self.irf_fwhm
        return taus, t0, irf

    # ------------------------------------------------------------------ #
    #                         Bounds and guesses                         #
    # ------------------------------------------------------------------ #
    def default_bounds(self, t=None):
        """``(lower, upper)`` arrays for the free parameters.

        Lifetimes are bounded below by zero and unbounded above. When a delay
        axis is given, ``t0`` is restricted to its range and the IRF width to
        the span of the data -- an IRF wider than the measurement is never a
        physical answer, and leaving it unbounded is how a fit runs away.
        """
        lo = [0.0] * self.n_lifetimes
        hi = [np.inf] * self.n_lifetimes
        if t is not None:
            t = np.asarray(t, dtype=float)
            span = float(np.nanmax(t) - np.nanmin(t))
        else:
            span = np.inf
        if self.fit_t0:
            lo.append(-np.inf if t is None else float(np.nanmin(t)))
            hi.append(np.inf if t is None else float(np.nanmax(t)))
        if self.fit_irf:
            lo.append(0.0)
            hi.append(span)
        return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)

    def initial_guess(self, t):
        """Lifetimes spread logarithmically over the measured delay range.

        The shortest guess is the smallest positive delay step and the longest
        is the last delay, which brackets everything the data can resolve.
        """
        t = np.atleast_1d(np.asarray(t, dtype=float)).ravel()
        finite = t[np.isfinite(t)]
        if finite.size < 2:
            raise ValueError("need at least two finite delays to guess lifetimes")
        steps = np.diff(np.sort(finite))
        steps = steps[steps > 0]
        lo = float(steps.min()) if steps.size else 1.0
        hi = float(np.nanmax(finite))
        if not np.isfinite(hi) or hi <= lo:
            hi = lo * 10.0**self.n_lifetimes
        taus = np.geomspace(max(lo, 1e-12), hi, self.n_lifetimes + 2)[1:-1]
        guess = self.pack(taus, self.t0, self.irf_fwhm)
        # A guess outside the bounds is rejected by the optimiser before it
        # starts, so clip it: with delays that begin after zero, the default
        # t0 = 0 would otherwise sit below its own lower bound.
        low, high = self.default_bounds(t)
        return np.clip(guess, low, high)

    # ------------------------------------------------------------------ #
    #                        Model evaluation                            #
    # ------------------------------------------------------------------ #
    @property
    def c0(self):
        """Initial concentrations at ``t0``."""
        return default_c0(self.n_components, str(self.model_type))

    def rate_matrix(self, taus):  # pragma: no cover - abstract
        """Rate matrix ``K`` for the given lifetimes."""
        raise NotImplementedError("subclasses must build their own rate matrix")

    def rate_builder(self):  # pragma: no cover - abstract
        """The raw ``k -> K`` callable, in terms of **rates** rather than lifetimes.

        Used to work out which rate constant sits on which arrow of the scheme
        diagram: feeding it a unit vector shows where ``k_i`` appears.
        """
        raise NotImplementedError("subclasses must expose their rate builder")

    def concentrations(self, t, params=None, **kwargs):
        """Concentration matrix ``C(t)``, shape ``[Nd, n_components]``.

        ``params`` is the packed parameter vector; alternatively pass
        ``taus=``, ``t0=`` and ``irf_fwhm=`` by keyword.
        """
        if params is not None:
            taus, t0, irf = self.unpack(params)
        else:
            taus = kwargs.pop("taus")
            t0 = float(kwargs.pop("t0", self.t0))
            irf = kwargs.pop("irf_fwhm", self.irf_fwhm)
        if self.model_type is not ModelType.PARALLEL:
            # EAS/SAS follow the global- and target-analysis formalism; the
            # reference is logged once, the first time such a model is used.
            cite("vanstokkum2004")
        K = self.rate_matrix(taus)
        C = concentrations(K, self.c0, t, t0=t0, irf_fwhm=irf, **kwargs)
        if self.coherent_artifact:
            C = np.hstack([C, self.artifact_basis(t, t0=t0, irf_fwhm=irf)])
        return C

    # ------------------------------------------------------------------ #
    #                        Coherent artefact                           #
    # ------------------------------------------------------------------ #
    @property
    def n_artifact(self) -> int:
        """Number of artefact columns appended to ``C`` (0 when switched off)."""
        return len(ARTIFACT_LABELS) if self.coherent_artifact else 0

    def artifact_basis(self, t, t0: float | None = None, irf_fwhm: float | None = None):
        """The artefact columns for this model's IRF (see :func:`coherent_artifact_basis`)."""
        cite("kovalenko1999")
        return coherent_artifact_basis(
            t,
            t0=float(self.t0 if t0 is None else t0),
            fwhm=self.irf_fwhm if irf_fwhm is None else irf_fwhm,
        )

    def column_labels(self) -> list[str]:
        """Labels of every column of ``C``: the species, then the artefact."""
        return self.species_labels() + (list(ARTIFACT_LABELS) if self.coherent_artifact else [])

    def species_labels(self) -> list[str]:
        """Compartment labels: ``A``, ``B``, ... then ``S1``, ``S2``, ..."""
        if self.n_components <= 26:
            return [chr(ord("A") + i) for i in range(self.n_components)]
        return [f"S{i + 1}" for i in range(self.n_components)]


@dataclass
class ParallelModel(KineticModel):
    """Independent exponential decays; the spectra are DAS."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.model_type = ModelType.PARALLEL

    def rate_matrix(self, taus):
        return parallel_K(rates_from_lifetimes(taus))

    def rate_builder(self):
        return parallel_K


@dataclass
class SequentialModel(KineticModel):
    """``A -> B -> C -> ...``, one branch; the spectra are EAS."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.model_type = ModelType.SEQUENTIAL

    def rate_matrix(self, taus):
        return sequential_K(rates_from_lifetimes(taus))

    def rate_builder(self):
        return sequential_K


@dataclass
class TargetModel(KineticModel):
    """General compartmental scheme with branching; the spectra are SAS.

    The scheme fixes both the number of compartments and the number of rate
    constants, which need not be equal -- an equilibrium ``A <=> B`` has two
    compartments and two rates, while ``A <=> B; A ->; B ->`` has two
    compartments and four.
    """

    scheme: TargetScheme | str = "A_eq_B"

    def __post_init__(self) -> None:
        if isinstance(self.scheme, str):
            self.scheme = get_scheme(self.scheme)
        # The scheme is authoritative: it decides how many compartments exist.
        if int(self.n_components) != self.scheme.n_species:
            logger.debug(
                "n_components=%s overridden by scheme %r (%d species)",
                self.n_components,
                self.scheme.key,
                self.scheme.n_species,
            )
            self.n_components = self.scheme.n_species
        super().__post_init__()
        self.model_type = ModelType.TARGET

    @property
    def n_lifetimes(self) -> int:
        return self.scheme.n_rates

    @property
    def c0(self):
        """Initial populations: the scheme's own, when it declares them."""
        if self.scheme.c0:
            return np.asarray(self.scheme.c0, dtype=float)
        return default_c0(self.n_components, str(self.model_type))

    def species_labels(self) -> list[str]:
        return self.scheme.species_labels()

    def lifetime_names(self) -> list[str]:
        """Named after the rate constants, so the table reads as the scheme does."""
        return [f"tau_{name}" for name in self.scheme.parameter_names()]

    def rate_matrix(self, taus):
        return self.scheme.rate_matrix(rates_from_lifetimes(taus))

    def rate_builder(self):
        return self.scheme.build


def make_model(
    model_type,
    n_components: int = 2,
    *,
    scheme: str | TargetScheme | None = None,
    **kwargs,
) -> KineticModel:
    """Build the model for a :class:`~pyrate_ta.ModelType` (or its string value).

    Parameters
    ----------
    model_type : ModelType or str
        ``"Parallel"``, ``"Sequential"`` or ``"Target"``.
    n_components : int
        Number of compartments. Ignored for a target model, where the scheme
        decides.
    scheme : str or TargetScheme, optional
        Required for a target model; see :data:`pyrate.models.TARGET_SCHEMES`.
    **kwargs
        Passed to the model (``fit_t0``, ``fit_irf``, ``t0``, ``irf_fwhm``).
    """
    mt = ModelType(str(model_type))
    if mt is ModelType.PARALLEL:
        return ParallelModel(n_components=n_components, **kwargs)
    if mt is ModelType.SEQUENTIAL:
        return SequentialModel(n_components=n_components, **kwargs)
    if mt is ModelType.TARGET:
        if scheme is None:
            raise ValueError("a target model needs a scheme; see pyrate.models.TARGET_SCHEMES")
        return TargetModel(n_components=n_components, scheme=scheme, **kwargs)
    raise NotImplementedError(f"no model implemented for {model_type!r}")
