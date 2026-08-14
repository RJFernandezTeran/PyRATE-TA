"""The three fitting engines, and the result they build.

``fit_kinetics``
    One or a few traces, multi-exponential.
``fit_global``
    Every probe pixel at once with shared lifetimes; DAS or EAS.
``fit_target``
    The same over a compartmental K-matrix scheme; SAS.

All three take either a :class:`pymorgan.Dataset1D` or plain arrays, run
headlessly (no ``QApplication``), and return a result object from
:mod:`pyrate_ta.results` carrying the lifetimes *with* their uncertainties and
fixed/free flags.

Component ordering
------------------
Variable projection sees only the column space of ``C``, so permuting the
lifetimes fits equally well and the optimiser's ordering depends on the initial
guess. For **parallel and sequential** models the components are therefore
sorted by lifetime on output, the spectra permuted to match, and the fact
recorded on the result (``sorted_by_lifetime``). A **target** model is left
exactly as fitted: its compartments are defined by the scheme, and reordering
them would misrepresent it.
"""

from __future__ import annotations

import numpy as np

from ..cite import cite
from ..log import get_logger, log_run_header
from ..results.fits import GlobalFit, KineticFit, TargetFit, covariance_from_jacobian
from . import varpro
from .driver import run_fit
from .prepare import is_dataset, prepare

logger = get_logger(__name__)


def _uncertainties(model, outcome):
    """1-sigma errors for every parameter; ``nan`` for the fixed ones."""
    n = len(outcome.param_names)
    err = np.full(n, np.nan)
    free = ~outcome.fixed
    if outcome.jacobian is None or not np.any(free):
        return err, None

    from .cost import residual_vector

    resid = residual_vector(outcome.R, outcome.weights)
    cov = covariance_from_jacobian(
        outcome.jacobian,
        resid,
        weighted=outcome.weights.weighted,
        dof=max(outcome.statistic.dof, 1),
    )
    if cov is None:
        return err, None
    var = np.diag(cov)
    with np.errstate(invalid="ignore"):
        err[free] = np.sqrt(np.where(var >= 0, var, np.nan))
    return err, cov


def _sort_components(model, taus, tau_err, is_fixed, t, D, weights):
    """Order the components by lifetime and rebuild ``C`` and ``S`` to match.

    ``C`` and the spectra are recomputed rather than merely permuted, so the
    reported lifetimes, concentration profiles and spectra stay mutually
    consistent -- a permuted lifetime vector with an unpermuted ``C`` would be
    a silently wrong result.
    """
    order = np.argsort(np.where(np.isfinite(taus), taus, np.inf), kind="stable")
    if np.array_equal(order, np.arange(taus.size)):
        return taus, tau_err, is_fixed, None
    return taus[order], tau_err[order], is_fixed[order], order


def _finalise(
    model,
    outcome,
    problem,
    *,
    result_class=KineticFit,
    sort: bool,
    extra: dict | None = None,
):
    """Turn a :class:`~pyrate_ta.fit.driver.FitOutcome` into a result object."""
    err, cov = _uncertainties(model, outcome)
    taus, t0, irf = model.unpack(outcome.params)
    tau_err = err[: model.n_lifetimes]
    is_fixed = np.asarray(outcome.fixed[: model.n_lifetimes], dtype=bool)

    idx = model.n_lifetimes
    t0_err = float(err[idx]) if model.fit_t0 else None
    if model.fit_t0:
        idx += 1
    irf_err = float(err[idx]) if model.fit_irf else None

    permutation = None
    if sort:
        taus, tau_err, is_fixed, permutation = _sort_components(
            model, np.asarray(taus, dtype=float), tau_err, is_fixed, problem.t, problem.D, outcome
        )

    C = model.concentrations(problem.t, taus=taus, t0=t0, irf_fwhm=irf)

    sign_mask = (extra or {}).pop("sign_mask", None) if extra else None
    ground_state = (extra or {}).pop("ground_state", None) if extra else None

    if sign_mask is not None:
        from .constrained import solve_amplitudes_constrained
        S = solve_amplitudes_constrained(C, problem.D, outcome.weights.w, sign_mask=sign_mask)
        R = problem.D - C @ S.T
    else:
        S, R = varpro.project(C, problem.D, outcome.weights.w)

    if permutation is not None:
        logger.info("Components sorted by lifetime; spectra reordered to match.")

    res_extra = dict(extra or {})
    res_extra.pop("sign_mask", None)
    res_extra.pop("ground_state", None)

    if ground_state is not None:
        from .groundstate import compute_absolute_spectra
        gs_spectrum = np.asarray(ground_state, dtype=float)
        S_abs, gs_scale, a_max = compute_absolute_spectra(S, gs_spectrum)
        res_extra.update({
            "S_abs": S_abs,
            "gs_scale": gs_scale,
            "a_max": a_max,
            "gs_source": "supplied",
        })

    result = result_class(
        taus=np.asarray(taus, dtype=float),
        tau_err=np.asarray(tau_err, dtype=float),
        is_fixed=is_fixed,
        S=S,
        R=R,
        C=C,
        model_type=str(model.model_type),
        t=problem.t,
        probe=problem.probe,
        t0=float(t0),
        t0_err=t0_err,
        irf_fwhm=None if irf is None else float(irf),
        irf_err=irf_err,
        statistic=outcome.statistic,
        report=outcome.report,
        weights=outcome.weights,
        param_names=outcome.param_names,
        initial_params=outcome.initial_params,
        bounds=outcome.bounds,
        covariance=cov,
        n_components=int(model.n_components),
        n_artifact=int(model.n_artifact),
        delay_range=problem.delay_range,
        probe_range=problem.probe_range,
        t_min=problem.t_min,
        detector=problem.detector,
        source=problem.source,
        sorted_by_lifetime=permutation is not None,
        permutation=permutation,
        settings={"noise_source": problem.noise_source, "time_unit": problem.time_unit},
        **res_extra,
    )
    result.log_summary()
    return result


def _as_problem(
    data,
    t=None,
    *,
    detector=0,
    delay_range=None,
    probe_range=None,
    t_min=None,
    need_noise=False,
    sigma=None,
):
    """Accept a dataset or plain arrays and return a :class:`ProblemData`."""
    from .prepare import ProblemData

    if is_dataset(data):
        return prepare(
            data,
            detector=detector,
            delay_range=delay_range,
            probe_range=probe_range,
            t_min=t_min,
            need_noise=need_noise,
        )
    if t is None:
        raise ValueError("plain-array input needs the delay axis: fit_global(t, D, ...)")
    # ``data`` is D here; the argument order mirrors the dataset call.
    D = np.atleast_2d(np.asarray(data, dtype=float))
    t_arr = np.asarray(t, dtype=float).ravel()
    keep = np.ones(t_arr.size, dtype=bool)
    if t_min is not None:
        keep &= t_arr >= float(t_min)
    if delay_range is not None:
        keep &= (t_arr >= float(delay_range[0])) & (t_arr <= float(delay_range[1]))
    sig = None if sigma is None else np.atleast_2d(np.asarray(sigma, dtype=float))[keep]
    return ProblemData(
        t=t_arr[keep],
        D=D[keep],
        sigma=sig,
        delay_range=(float(t_arr[keep].min()), float(t_arr[keep].max())),
        t_min=None if t_min is None else float(t_min),
        detector=int(detector),
    )


def fit_global(
    data,
    t=None,
    *,
    n_components: int | None = None,
    model_type=None,
    taus=None,
    fit_t0: bool = False,
    fit_irf: bool = False,
    t0: float = 0.0,
    irf_fwhm: float | None = None,
    coherent_artifact: bool = False,
    use_weights: bool | None = None,
    detector: int = 0,
    delay_range=None,
    probe_range=None,
    t_min: float | None = None,
    sigma=None,
    **driver_kwargs,
) -> GlobalFit:
    """Global analysis of a dataset: shared lifetimes, one spectrum per component.

    ``data`` is either a :class:`pymorgan.Dataset1D` or the data matrix, in
    which case the delay axis ``t`` is the second argument.

    Parameters
    ----------
    n_components : int, optional
        Number of components; defaults to ``Settings.n_components``, or to the
        length of ``taus`` when those are given.
    model_type : ModelType or str, optional
        ``"Parallel"`` (DAS) or ``"Sequential"`` (EAS). Defaults to
        ``Settings.model_type``.
    taus : sequence, optional
        Initial lifetimes. An entry of ``inf`` (or "inf", "infinity", ...) adds
        a **non-decaying component**: the constant offset for signal that
        outlives the delay window. It is held fixed, since an optimiser cannot
        vary infinity.
    coherent_artifact : bool, default False
        Add the IRF and its first two derivatives to the design matrix, so
        cross-phase modulation and other time-zero artefacts are absorbed by
        their own amplitudes instead of distorting the shortest lifetime.
        Needs a Gaussian IRF width.
    use_weights : bool, optional
        Weight by the per-point noise. Defaults to ``Settings.use_noise_weights``.
        Requesting weights for data with no noise raises rather than quietly
        producing an SSR labelled as a chi-squared.

    Returns
    -------
    GlobalFit
    """
    from ..helpers import parse_lifetime
    from ..models import make_model
    from ..settings import get_settings

    s = get_settings()
    model_type = s.model_type if model_type is None else model_type
    if use_weights is None:
        use_weights = bool(s.use_noise_weights)

    if taus is not None:
        taus = [parse_lifetime(v, field=f"tau{i + 1}") for i, v in enumerate(taus)]
        n_components = len(taus)
    n_components = int(s.n_components if n_components is None else n_components)

    log_run_header(
        f"Global fit: {model_type}, {n_components} component(s)"
        + (" + coherent artefact" if coherent_artifact else ""),
        logger=logger,
    )
    problem = _as_problem(
        data,
        t,
        detector=detector,
        delay_range=delay_range,
        probe_range=probe_range,
        t_min=t_min,
        need_noise=bool(use_weights),
        sigma=sigma,
    )
    model = make_model(
        model_type,
        n_components,
        fit_t0=fit_t0,
        fit_irf=fit_irf,
        t0=t0,
        irf_fwhm=irf_fwhm,
        coherent_artifact=bool(coherent_artifact),
    )
    p0 = None if taus is None else model.pack(taus, t0, irf_fwhm)

    outcome = run_fit(
        model,
        problem.t,
        problem.D,
        sigma=problem.sigma,
        use_weights=bool(use_weights),
        noise_source=problem.noise_source,
        p0=p0,
        **driver_kwargs,
    )
    cite("vanstokkum2004")
    return _finalise(model, outcome, problem, result_class=GlobalFit, sort=True)


def preview_global(
    data,
    t=None,
    *,
    taus,
    model_type=None,
    scheme=None,
    t0: float = 0.0,
    irf_fwhm: float | None = None,
    coherent_artifact: bool = False,
    use_weights: bool | None = None,
    detector: int = 0,
    delay_range=None,
    probe_range=None,
    t_min: float | None = None,
    sigma=None,
) -> GlobalFit:
    """Evaluate a model at the given lifetimes **without** optimising.

    The spectra still follow from the data (they are the least-squares solution
    for the given ``C``), so a preview shows what those lifetimes actually imply
    and what the residual would be. Nothing is fitted, so there is no
    convergence report and no uncertainties -- and the result says so rather
    than presenting a guess as an answer.
    """
    from ..helpers import parse_lifetime
    from ..models import make_model
    from ..settings import get_settings
    from .cost import build_weights, fit_statistic

    s = get_settings()
    if use_weights is None:
        use_weights = bool(s.use_noise_weights)
    taus = [parse_lifetime(v, field=f"tau{i + 1}") for i, v in enumerate(taus)]
    log_run_header("Preview (no optimisation)", logger=logger)

    problem = _as_problem(
        data,
        t,
        detector=detector,
        delay_range=delay_range,
        probe_range=probe_range,
        t_min=t_min,
        need_noise=bool(use_weights),
        sigma=sigma,
    )
    artefact = dict(coherent_artifact=bool(coherent_artifact))
    model = (
        make_model("Target", 0, scheme=scheme, t0=t0, irf_fwhm=irf_fwhm, **artefact)
        if scheme is not None
        else make_model(
            s.model_type if model_type is None else model_type,
            len(taus),
            t0=t0,
            irf_fwhm=irf_fwhm,
            **artefact,
        )
    )
    weights = build_weights(
        problem.D, problem.sigma, use_weights=bool(use_weights), source=problem.noise_source
    )
    C = model.concentrations(problem.t, taus=taus, t0=t0, irf_fwhm=irf_fwhm)
    S, R = varpro.project(C, problem.D, weights.w)
    stat = fit_statistic(R, weights, n_nonlinear=0, n_amplitudes=int(S.size))

    result = GlobalFit(
        taus=np.asarray(taus, dtype=float),
        tau_err=np.full(len(taus), np.nan),
        is_fixed=np.ones(len(taus), dtype=bool),
        S=S,
        R=R,
        C=C,
        model_type=str(model.model_type),
        t=problem.t,
        probe=problem.probe,
        t0=float(t0),
        irf_fwhm=irf_fwhm,
        statistic=stat,
        report=None,
        weights=weights,
        param_names=model.param_names(),
        n_components=int(model.n_components),
        n_artifact=int(model.n_artifact),
        delay_range=problem.delay_range,
        probe_range=problem.probe_range,
        t_min=problem.t_min,
        detector=problem.detector,
        source=problem.source,
        settings={
            "noise_source": problem.noise_source,
            "time_unit": problem.time_unit,
            "preview": True,
        },
    )
    logger.info("Preview (no optimisation): %s", stat)
    return result


def fit_target(
    data,
    t=None,
    *,
    scheme,
    taus=None,
    fit_t0: bool = False,
    fit_irf: bool = False,
    t0: float = 0.0,
    irf_fwhm: float | None = None,
    coherent_artifact: bool = False,
    use_weights: bool | None = None,
    detector: int = 0,
    delay_range=None,
    probe_range=None,
    t_min: float | None = None,
    sigma=None,
    **driver_kwargs,
) -> TargetFit:
    """Target analysis over a compartmental scheme; the spectra are SAS.

    ``scheme`` is a key from :data:`pyrate.models.TARGET_SCHEMES` or a
    :class:`~pyrate_ta.models.TargetScheme`.

    The components are **not** reordered: a scheme defines which compartment is
    which. Note also that the rates of a branched scheme are generally not
    identifiable from a single dataset -- the fit can be perfect while the
    individual rates are not determined; the eigenvalues, which are, are stored
    on the result.
    """
    from ..helpers import parse_lifetime
    from ..models import get_scheme, make_model
    from ..models.propagator import eigen_decomposition
    from ..settings import get_settings

    s = get_settings()
    if use_weights is None:
        use_weights = bool(s.use_noise_weights)
    scheme_obj = get_scheme(scheme) if isinstance(scheme, str) else scheme
    log_run_header(
        f"Target fit: {getattr(scheme_obj, 'label', 'scheme')}"
        + (" + coherent artefact" if coherent_artifact else ""),
        logger=logger,
    )

    if taus is not None:
        taus = [parse_lifetime(v, field=f"tau{i + 1}") for i, v in enumerate(taus)]

    problem = _as_problem(
        data,
        t,
        detector=detector,
        delay_range=delay_range,
        probe_range=probe_range,
        t_min=t_min,
        need_noise=bool(use_weights),
        sigma=sigma,
    )
    model = make_model(
        "Target",
        scheme_obj.n_species,
        scheme=scheme_obj,
        fit_t0=fit_t0,
        fit_irf=fit_irf,
        t0=t0,
        irf_fwhm=irf_fwhm,
        coherent_artifact=bool(coherent_artifact),
    )
    p0 = None if taus is None else model.pack(taus, t0, irf_fwhm)

    outcome = run_fit(
        model,
        problem.t,
        problem.D,
        sigma=problem.sigma,
        use_weights=bool(use_weights),
        noise_source=problem.noise_source,
        p0=p0,
        **driver_kwargs,
    )
    cite("vanstokkum2004")

    fitted_taus, _, _ = model.unpack(outcome.params)
    K = model.rate_matrix(fitted_taus)
    lam = np.sort(np.real(eigen_decomposition(K)[0]))
    return _finalise(
        model,
        outcome,
        problem,
        result_class=TargetFit,
        sort=False,
        extra={
            "scheme_key": scheme_obj.key,
            "scheme_label": scheme_obj.label,
            "scheme_text": scheme_obj.source_text or None,
            "K": K,
            "eigenvalues": lam,
        },
    )


def fit_kinetics(
    t,
    y,
    *,
    n_components: int | None = None,
    model_type=None,
    taus=None,
    fit_t0: bool = False,
    fit_irf: bool = False,
    t0: float = 0.0,
    irf_fwhm: float | None = None,
    sigma=None,
    use_weights: bool = False,
    **driver_kwargs,
) -> KineticFit:
    """Multi-exponential fit of one trace (or a few), by the same machinery.

    A single trace is the ``Np = 1`` case of a global fit: the amplitudes are
    still solved in closed form, so only the lifetimes reach the optimiser.

    Parameters
    ----------
    t : array_like ``(Nd,)``
    y : array_like ``(Nd,)`` or ``(Nd, Ncuts)``
        One trace, or several sharing the same lifetimes.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)
        if sigma.ndim == 1:
            sigma = sigma[:, None]

    from ..helpers import parse_lifetime
    from ..models import make_model
    from ..settings import get_settings

    s = get_settings()
    model_type = s.model_type if model_type is None else model_type
    if taus is not None:
        taus = [parse_lifetime(v, field=f"tau{i + 1}") for i, v in enumerate(taus)]
        n_components = len(taus)
    n_components = int(s.n_components if n_components is None else n_components)

    problem = _as_problem(y, t, sigma=sigma)
    model = make_model(
        model_type, n_components, fit_t0=fit_t0, fit_irf=fit_irf, t0=t0, irf_fwhm=irf_fwhm
    )
    p0 = None if taus is None else model.pack(taus, t0, irf_fwhm)
    outcome = run_fit(
        model,
        problem.t,
        problem.D,
        sigma=problem.sigma,
        use_weights=bool(use_weights),
        p0=p0,
        **driver_kwargs,
    )
    return _finalise(model, outcome, problem, result_class=KineticFit, sort=True)
