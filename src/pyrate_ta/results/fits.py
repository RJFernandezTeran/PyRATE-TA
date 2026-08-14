"""Fit result objects and the linearised uncertainty estimate.

A result carries enough to reproduce itself: the model identity, the initial
guesses, the bounds, the fixed/free mask, the delay and probe ranges actually
fitted (clipping changes what the fit saw), the solver's report and the
statistic. **A lifetime without its uncertainty and its fixed/free flag is not
a result**, so ``tau_err`` and ``is_fixed`` travel with every lifetime.

Uncertainties are 1-sigma from the linearised covariance at the optimum. They
are conditional on the model being right: a three-component fit of
two-component data can return small uncertainties on three meaningless
lifetimes. Fixed parameters get ``nan``, never a small number that might read
as precision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..log import get_logger

logger = get_logger(__name__)


def covariance_from_jacobian(J, residuals, *, weighted: bool, dof: int):
    """Linearised parameter covariance at the optimum.

    Parameters
    ----------
    J : array_like ``(N, n_free)``
        Jacobian of the residual vector with respect to the free parameters, as
        ``scipy.optimize.least_squares`` returns it.
    residuals : array_like
        The residual vector at the optimum (already weighted, if weights were
        used -- that is what the optimiser saw).
    weighted : bool
        With proper ``1/sigma`` weights the residuals are dimensionless and the
        covariance is ``(J^T J)^-1``. Unweighted, it is scaled by the residual
        variance ``SSR/dof``, which assumes the model is correct and the noise
        uniform.
    dof : int
        Degrees of freedom used for that scaling.

    Returns
    -------
    numpy.ndarray or None
        ``(n_free, n_free)``, or ``None`` when the normal matrix is singular --
        in which case the parameters are not determined and no uncertainty
        should be invented for them.
    """
    J = np.atleast_2d(np.asarray(J, dtype=float))
    JTJ = J.T @ J
    try:
        cov = np.linalg.inv(JTJ)
    except np.linalg.LinAlgError:
        logger.warning(
            "The normal matrix is singular: the parameters are not independently "
            "determined and no uncertainties are reported."
        )
        return None
    if not weighted:
        ssr = float(np.dot(residuals, residuals))
        cov = cov * (ssr / dof if dof > 0 else np.nan)
    return cov


@dataclass
class KineticFit:
    """Result of a multi-exponential fit to one or more traces.

    Attributes
    ----------
    taus, tau_err : numpy.ndarray
        Lifetimes and their 1-sigma uncertainties, in the dataset's time unit.
        A fixed lifetime has ``nan`` uncertainty.
    is_fixed : numpy.ndarray of bool
        Per-lifetime fixed/free flag.
    S : numpy.ndarray ``(Np, Nc)``
        Component spectra (amplitudes for a single trace).
    R : numpy.ndarray ``(Nd, Np)``
        Residual matrix.
    model_type : str
        ``"Parallel"`` / ``"Sequential"`` / ``"Target"``; decides whether the
        spectra are DAS, EAS or SAS, and what PyMORGAN prints in the legend.
    """

    taus: np.ndarray
    tau_err: np.ndarray
    is_fixed: np.ndarray
    S: np.ndarray
    R: np.ndarray
    model_type: str
    t: np.ndarray
    probe: np.ndarray | None = None
    C: np.ndarray | None = None
    t0: float = 0.0
    t0_err: float | None = None
    irf_fwhm: float | None = None
    irf_err: float | None = None

    # --- Phase 6: Ground state bleach / absolute spectra -------------------- #
    S_abs: np.ndarray | None = None
    gs_scale: float | None = None
    gs_source: str | None = None
    a_max: float | None = None

    # --- reproducibility payload ------------------------------------------- #
    statistic: Any = None
    report: Any = None
    weights: Any = None
    param_names: list[str] = field(default_factory=list)
    initial_params: np.ndarray | None = None
    bounds: tuple[np.ndarray, np.ndarray] | None = None
    covariance: np.ndarray | None = None
    n_components: int = 0
    #: Number of coherent-artefact columns appended to ``C`` (and to ``S``).
    #: They are not kinetic components and carry no lifetime.
    n_artifact: int = 0
    delay_range: tuple[float, float] | None = None
    probe_range: tuple[float, float] | None = None
    t_min: float | None = None
    detector: int = 0
    source: str | None = None
    sorted_by_lifetime: bool = False
    permutation: np.ndarray | None = None
    settings: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    #                              Reading                               #
    # ------------------------------------------------------------------ #
    @property
    def converged(self) -> bool:
        return bool(getattr(self.report, "converged", False))

    @property
    def n_infinite(self) -> int:
        """Number of non-decaying (offset) components."""
        return int(np.count_nonzero(~np.isfinite(self.taus)))

    @property
    def spectra_kind(self) -> str:
        """What this model's amplitude spectra are called: DAS, EAS or SAS.

        On the base class because the name is a property of the model family,
        not of which engine produced the result -- a preview has to be able to
        label its own spectra too.
        """
        return {"Parallel": "DAS", "Sequential": "EAS", "Target": "SAS"}.get(
            str(self.model_type), "spectra"
        )

    @property
    def n_kinetic(self) -> int:
        """Columns of ``C``/``S`` that are kinetic components, artefact aside."""
        total = int(np.shape(self.S)[1]) if self.S is not None else len(self.taus)
        return max(total - int(self.n_artifact), 0)

    @property
    def artifact_spectra(self):
        """Amplitude spectra of the coherent-artefact columns, or ``None``.

        Worth looking at: a fitted artefact that does not look like a
        derivative of the IRF, or that carries amplitude far from time zero, is
        a sign the artefact is absorbing real kinetics.
        """
        if not self.n_artifact or self.S is None:
            return None
        return np.asarray(self.S)[:, self.n_kinetic :]

    def as_species_args(self):
        """The plain-array tuple ``plot_species_spectra`` expects.

        Returns ``(Sfit, Taus, TauErr, isFixTau, modelType)``. Keeping this
        adapter on the PyRATE side is what lets PyMORGAN render a PyRATE fit
        without ever importing ``pyrate``.

        Only the kinetic columns are handed over: an artefact column has no
        lifetime, and passing it would label it with someone else's.
        """
        return (
            np.asarray(self.S)[:, : self.n_kinetic],
            np.asarray(self.taus),
            np.asarray(self.tau_err),
            np.asarray(self.is_fixed, dtype=bool),
            str(self.model_type),
        )

    def species_labels(self) -> list[str]:
        """Names of the species/compartments for DAS/EAS/SAS plots and legends."""
        from ..plot.concentrations import species_labels_for as get_species_labels

        return get_species_labels(self)

    def summary(self) -> str:

        """One-line-per-lifetime summary, with the statistic and convergence.

        How the lifetimes are quoted follows the ``[plots]`` settings
        ``show_uncertainties`` and ``round_uncertainties``, so the summary agrees
        with the legends rather than reporting a different precision.
        """
        from pymorgan.helpers import format_uncertainty_pair

        from ..helpers import format_lifetime
        from ..settings import get_settings

        s = get_settings()
        lines = [
            f"{self.model_type} fit, {self.n_components} component(s)"
            + (f" from {self.source}" if self.source else ""),
            str(self.report) if self.report is not None else "no solver report",
        ]
        for i, (tau, err, fixed) in enumerate(
            zip(self.taus, self.tau_err, self.is_fixed, strict=True)
        ):
            flag = " (fixed)" if fixed else ""
            show = bool(s.show_uncertainties) and np.isfinite(err)
            if not np.isfinite(tau):
                lines.append(f"  tau{i + 1} = inf (non-decaying){flag}")
            elif show and s.round_uncertainties:
                value, error = format_uncertainty_pair(tau, err)
                lines.append(f"  tau{i + 1} = {value} +/- {error}{flag}")
            elif show:
                lines.append(f"  tau{i + 1} = {format_lifetime(tau)} +/- {err:.3g}{flag}")
            else:
                lines.append(f"  tau{i + 1} = {format_lifetime(tau)}{flag}")
        if self.t0_err is not None:
            lines.append(f"  t0 = {self.t0:.4g} +/- {self.t0_err:.3g}")
        if self.irf_fwhm is not None and self.irf_err is not None:
            lines.append(f"  IRF FWHM = {self.irf_fwhm:.4g} +/- {self.irf_err:.3g}")
        if self.statistic is not None:
            lines.append(f"  {self.statistic}")
        if self.sorted_by_lifetime:
            lines.append("  (components sorted by lifetime; spectra reordered to match)")
        return "\n".join(lines)

    def log_summary(self) -> None:
        """Send :meth:`summary` to the ``pyrate`` logger at info level."""
        logger.info("%s", self.summary())


@dataclass
class GlobalFit(KineticFit):
    """Global-analysis result: shared lifetimes, one spectrum per component.

    The spectra are DAS for a parallel model and EAS for a sequential one --
    which of the two is recorded in :attr:`model_type`, because the distinction
    is what the legend and the interpretation hang on (see
    :attr:`KineticFit.spectra_kind`).
    """


@dataclass
class TargetFit(GlobalFit):
    """Target-analysis result, adding the scheme and its rate matrix.

    The rates of a branched scheme are generally **not** identifiable from a
    single dataset even when the fit is perfect: the data fix the column space
    of ``C``, and several rate sets can span it. :attr:`eigenvalues` is what the
    data actually determine, and it is stored alongside the rates for that
    reason.
    """

    scheme_key: str | None = None
    scheme_label: str | None = None
    #: The scheme in its own notation, so a saved fit reopens as editable text.
    scheme_text: str | None = None
    K: np.ndarray | None = None
    eigenvalues: np.ndarray | None = None

    def summary(self) -> str:
        head = super().summary()
        if self.scheme_label:
            head += f"\n  scheme: {self.scheme_label}"
            head += "\n  (branched schemes are not generally identifiable from one dataset;"
            head += " the eigenvalues are what the data determine)"
        return head
