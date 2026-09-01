"""Tests for :mod:`pyrate_ta.fit` -- variable projection, weighting, the driver.

The backbone is synthetic recovery: build data from known parameters, fit it,
and assert the parameters come back. Everything else tests a failure mode --
non-convergence flagged, weights refused when there is no noise, a bounded
optimum reported as such.
"""

from __future__ import annotations

import numpy as np
import pytest

import pyrate_ta as pr
from pyrate_ta.fit import build_weights, fit_statistic, project, run_fit, solve_amplitudes
from pyrate_ta.fit.cost import NoiseUnavailableError, residual_vector
from pyrate_ta.models import SequentialModel, make_model

# A delay axis with the usual pre-zero region and logarithmic tail.
T = np.concatenate([np.linspace(-2.0, 0.5, 20), np.geomspace(0.6, 3000.0, 140)])
PROBE = np.linspace(1900.0, 2010.0, 40)


def _spectra(centres=(1950.0, 1975.0), widths=(15.0, 25.0), amps=(1.0, -0.6)):
    """Gaussian component spectra, ``[Np, Nc]``."""
    return np.stack(
        [
            a * np.exp(-((PROBE - c) ** 2) / (2 * w**2))
            for c, w, a in zip(centres, widths, amps, strict=True)
        ],
        axis=1,
    )


def _dataset(taus=(8.0, 300.0), *, irf=0.3, t0=0.0, noise=0.0, seed=0):
    """Synthetic ``D = C @ S.T`` from a sequential model, optionally noisy."""
    model = SequentialModel(n_components=len(taus), t0=t0, irf_fwhm=irf)
    C = model.concentrations(T, taus=taus, t0=t0, irf_fwhm=irf)
    S = _spectra()
    D = C @ S.T
    if noise:
        D = D + noise * np.random.default_rng(seed).standard_normal(D.shape)
    return model, C, S, D


# --------------------------------------------------------------------------- #
#                           Variable projection                               #
# --------------------------------------------------------------------------- #
def test_amplitudes_are_exact_for_noiseless_data():
    _, C, S, D = _dataset()
    assert np.allclose(solve_amplitudes(C, D), S, atol=1e-8)


def test_project_returns_zero_residuals_for_an_exact_model():
    _, C, S, D = _dataset()
    S_fit, R = project(C, D)
    assert np.allclose(S_fit, S, atol=1e-8)
    assert np.max(np.abs(R)) < 1e-8


def test_uniform_weights_do_not_change_the_solution():
    """Scaling every point by the same sigma cannot move a least-squares fit."""
    _, C, S, D = _dataset(noise=0.02, seed=1)
    plain = solve_amplitudes(C, D)
    scaled = solve_amplitudes(C, D, np.full(D.shape, 1.0 / 0.7))
    assert np.allclose(plain, scaled, atol=1e-8)


def test_flat_per_pixel_weights_do_not_move_that_pixel():
    """Scaling one pixel's weights uniformly cannot change its own amplitudes.

    Each pixel is an independent least-squares solve, so a weight that is
    constant in time cancels out of it. Per-pixel noise matters for the
    *non-linear* parameters instead, where a noisy pixel contributes less to the
    cost -- see ``test_noisy_pixel_pulls_the_lifetimes_less_when_weighted``.
    """
    _, C, _, D = _dataset()
    sigma = np.full(D.shape, 0.01)
    sigma[:, 7] = 1e4
    assert np.allclose(solve_amplitudes(C, D, 1.0 / sigma), solve_amplitudes(C, D), atol=1e-8)


def test_weights_varying_in_time_do_change_the_amplitudes():
    """A weight profile down the delay axis is what actually reweights a pixel."""
    _, C, S, D = _dataset()
    corrupt = D.copy()
    late = T > 100.0
    corrupt[late, 4] += 5.0  # the late delays of one pixel are wrong

    w = np.ones_like(D)
    w[late, 4] = 0.0  # ... and we know it: drop them
    trusted = solve_amplitudes(C, corrupt, w)
    naive = solve_amplitudes(C, corrupt)

    assert np.allclose(trusted[4], S[4], atol=1e-6)
    assert not np.allclose(naive[4], S[4], atol=1e-3)


def test_zero_weight_pixels_get_zero_amplitude():
    _, C, _, D = _dataset()
    w = np.ones_like(D)
    w[:, 3] = 0.0
    S_fit = solve_amplitudes(C, D, w)
    assert np.allclose(S_fit[3], 0.0)


def test_solve_rejects_bad_shapes_and_weights():
    _, C, _, D = _dataset()
    with pytest.raises(ValueError):
        solve_amplitudes(C[:-1], D)
    with pytest.raises(ValueError):
        solve_amplitudes(C, D, -np.ones_like(D))
    with pytest.raises(ValueError):
        solve_amplitudes(C, D, np.ones((D.shape[0], 3)))


# --------------------------------------------------------------------------- #
#                          Weights and statistic                              #
# --------------------------------------------------------------------------- #
def test_weights_require_noise():
    _, _, _, D = _dataset()
    with pytest.raises(NoiseUnavailableError):
        build_weights(D, None, use_weights=True)


def test_unweighted_masks_non_finite_data():
    _, _, _, D = _dataset()
    D = D.copy()
    D[0, 0] = np.nan
    w = build_weights(D, None, use_weights=False)
    assert w.weighted is False
    assert w.n_masked == 1
    assert w.n_points == D.size - 1


def test_sigma_floor_limits_the_weight():
    """One near-zero sigma must not dominate the cost function."""
    _, _, _, D = _dataset()
    sigma = np.full(D.shape, 0.5)
    sigma[5, 5] = 1e-12
    w = build_weights(D, sigma, use_weights=True, floor_fraction=1e-3)
    assert w.n_floored == 1
    # Without the floor this weight would be 1e12 times the others.
    assert w.w[5, 5] / w.w[0, 0] == pytest.approx(1000.0, rel=1e-6)


def test_bad_sigma_points_are_masked_not_repaired():
    _, _, _, D = _dataset()
    sigma = np.full(D.shape, 0.5)
    sigma[1, 1] = 0.0
    sigma[2, 2] = np.nan
    w = build_weights(D, sigma, use_weights=True)
    assert w.n_masked == 2
    assert w.w[1, 1] == 0.0 and w.w[2, 2] == 0.0
    assert w.n_points == D.size - 2


def test_reduced_chi2_is_about_one_for_correctly_weighted_data():
    """The headline property: residuals at the noise level give chi2_red ~ 1."""
    sigma_true = 0.05
    _, C, S, D = _dataset(noise=sigma_true, seed=3)
    weights = build_weights(D, np.full(D.shape, sigma_true), use_weights=True, source="pdatn")
    _, R = project(C, D, weights.w)
    stat = fit_statistic(R, weights, n_nonlinear=2, n_amplitudes=S.size)

    assert stat.kind == "chi2_red"
    assert stat.noise_source == "pdatn"
    assert stat.value == pytest.approx(1.0, rel=0.1)
    # The non-linear-only denominator is larger, so that value is smaller.
    assert stat.value_nonlinear_dof < stat.value


def test_unweighted_statistic_is_the_plain_ssr():
    sigma_true = 0.05
    _, C, S, D = _dataset(noise=sigma_true, seed=4)
    weights = build_weights(D, None, use_weights=False)
    _, R = project(C, D)
    stat = fit_statistic(R, weights, n_nonlinear=2, n_amplitudes=S.size)

    assert stat.kind == "ssr"
    assert stat.weighted is False
    assert stat.value == pytest.approx(float(np.sum(R**2)), rel=1e-12)
    assert stat.value_nonlinear_dof is None


def test_residual_vector_length_is_constant():
    """``least_squares`` requires a fixed-length residual vector."""
    _, C, _, D = _dataset()
    sigma = np.full(D.shape, 0.1)
    sigma[0, 0] = np.nan  # masked, but the vector must not shrink
    weights = build_weights(D, sigma, use_weights=True)
    _, R = project(C, D, weights.w)
    assert residual_vector(R, weights).size == D.size


# --------------------------------------------------------------------------- #
#                          Synthetic recovery                                 #
# --------------------------------------------------------------------------- #
def test_recovers_known_lifetimes_without_noise():
    taus = (8.0, 300.0)
    model, _, S, D = _dataset(taus=taus)
    out = run_fit(model, T, D, p0=[3.0, 100.0])

    assert out.converged, out.report
    assert np.allclose(out.params, taus, rtol=1e-4)
    assert np.allclose(out.S, S, atol=1e-6)
    assert out.statistic.kind == "ssr"
    assert out.statistic.value < 1e-12


def test_recovers_known_lifetimes_from_noisy_data():
    taus = (8.0, 300.0)
    sigma_true = 0.02
    model, _, _, D = _dataset(taus=taus, noise=sigma_true, seed=5)
    out = run_fit(
        model,
        T,
        D,
        sigma=np.full(D.shape, sigma_true),
        use_weights=True,
        noise_source="pdatn",
        p0=[3.0, 100.0],
    )

    assert out.converged, out.report
    assert np.allclose(out.params, taus, rtol=0.05)
    assert out.statistic.kind == "chi2_red"
    assert out.statistic.value == pytest.approx(1.0, rel=0.15)


def test_noisy_pixel_pulls_the_lifetimes_less_when_weighted():
    """Per-pixel noise acts on the non-linear parameters, through the cost."""
    taus = (8.0, 300.0)
    model, _, _, D = _dataset(taus=taus)
    rng = np.random.default_rng(11)
    corrupt = D.copy()
    bad = 12
    corrupt[:, bad] += 5.0 * rng.standard_normal(D.shape[0])  # one ruined pixel

    sigma = np.full(D.shape, 0.01)
    sigma[:, bad] = 5.0  # ... and the noise array knows it

    weighted = run_fit(model, T, corrupt, sigma=sigma, use_weights=True, p0=[5.0, 200.0])
    unweighted = run_fit(model, T, corrupt, p0=[5.0, 200.0])

    err_w = np.max(np.abs(np.sort(weighted.params) - np.sort(taus)) / np.asarray(sorted(taus)))
    err_u = np.max(np.abs(np.sort(unweighted.params) - np.sort(taus)) / np.asarray(sorted(taus)))
    assert err_w < err_u
    assert err_w < 0.02


def test_lifetime_order_is_not_fixed_by_the_data():
    """Swapping two lifetimes spans the same subspace, so both fit equally well.

    Variable projection sees only the column space of ``C``; a permutation of
    the lifetimes leaves it unchanged (the spectra absorb the relabelling).
    Which order the optimiser lands on therefore depends on the starting point,
    and any physical reading of the EAS has to impose the order, not read it
    off the fit.
    """
    taus = (8.0, 300.0)
    model, _, _, D = _dataset(taus=taus)
    forward = run_fit(model, T, D, p0=[5.0, 200.0])
    swapped = run_fit(model, T, D, p0=[200.0, 5.0])

    assert forward.converged and swapped.converged
    assert np.allclose(np.sort(forward.params), np.sort(swapped.params), rtol=1e-3)
    assert np.allclose(np.sort(forward.params), np.sort(taus), rtol=1e-3)
    assert max(np.max(np.abs(forward.R)), np.max(np.abs(swapped.R))) < 1e-8


def test_recovers_time_zero_and_irf_width():
    taus, t0, irf = (10.0, 200.0), 0.15, 0.4
    model, _, _, D = _dataset(taus=taus, t0=t0, irf=irf)
    fit_model = SequentialModel(n_components=2, fit_t0=True, fit_irf=True, t0=0.0, irf_fwhm=0.3)
    out = run_fit(fit_model, T, D, p0=[5.0, 150.0, 0.0, 0.3])

    assert out.converged, out.report
    taus_fit, t0_fit, irf_fit = fit_model.unpack(out.params)
    assert np.allclose(taus_fit, taus, rtol=0.02)
    assert t0_fit == pytest.approx(t0, abs=0.02)
    assert irf_fit == pytest.approx(irf, rel=0.1)


def test_recovers_a_target_scheme():
    """A branched scheme with more rates than species."""
    model = make_model("Target", scheme="A_eq_B_to_C", irf_fwhm=0.3)
    truth = [5.0, 40.0, 20.0, 500.0]
    C = model.concentrations(T, taus=truth)
    S = np.stack([np.exp(-((PROBE - c) ** 2) / (2 * 20.0**2)) for c in (1940, 1960, 1990)], axis=1)
    D = C @ S.T

    out = run_fit(model, T, D, p0=[3.0, 30.0, 15.0, 400.0])
    assert out.converged, out.report
    # The data constrain the *column space* of C, not the individual rates: a
    # branched scheme with more rates than species is not identifiable from a
    # single dataset (van Stokkum 2004). What must come back are the system
    # eigenvalues and a vanishing residual, not the rate vector itself.
    from pyrate_ta.models.propagator import eigen_decomposition

    lam_true = np.sort(np.real(eigen_decomposition(model.rate_matrix(truth))[0]))
    lam_fit = np.sort(np.real(eigen_decomposition(model.rate_matrix(out.params))[0]))
    assert np.allclose(lam_fit, lam_true, rtol=1e-3)
    assert np.max(np.abs(out.R)) < 1e-6


# --------------------------------------------------------------------------- #
#                          Failure modes                                      #
# --------------------------------------------------------------------------- #
def test_non_convergence_is_flagged_not_returned_as_success(caplog):
    model, _, _, D = _dataset()
    with caplog.at_level("WARNING", logger="pyrate_ta"):
        out = run_fit(model, T, D, p0=[1.0, 1000.0], max_iterations=3)
    assert out.converged is False
    assert out.report.status <= 0
    assert "did not converge" in caplog.text.lower()


def test_fixed_parameters_stay_put():
    taus = (8.0, 300.0)
    model, _, _, D = _dataset(taus=taus)
    out = run_fit(model, T, D, p0=[8.0, 100.0], fixed=["tau1"])

    assert out.params[0] == 8.0  # untouched
    assert out.fixed.tolist() == [True, False]
    assert out.params[1] == pytest.approx(taus[1], rel=1e-3)


def test_all_parameters_fixed_is_refused():
    model, _, _, D = _dataset()
    with pytest.raises(ValueError, match="nothing to fit"):
        run_fit(model, T, D, fixed=[True, True])


def test_unknown_fixed_name_is_refused():
    model, _, _, D = _dataset()
    with pytest.raises(ValueError, match="unknown parameter"):
        run_fit(model, T, D, fixed=["tau9"])


def test_lm_with_bounds_is_refused_rather_than_dropping_them():
    """``lm`` ignores bounds; accepting it silently would fit the wrong problem."""
    model, _, _, D = _dataset()
    with pytest.raises(ValueError, match="cannot honour bounds"):
        run_fit(model, T, D, method="lm")


def test_parameter_on_a_bound_is_reported(caplog):
    model, _, _, D = _dataset(taus=(8.0, 300.0))
    lo = np.array([400.0, 400.0])  # both bounds sit above the true lifetimes
    hi = np.array([np.inf, np.inf])
    with caplog.at_level("WARNING", logger="pyrate_ta"):
        out = run_fit(model, T, D, p0=[500.0, 600.0], bounds=(lo, hi))
    assert out.report.at_bounds
    assert "resting on a bound" in caplog.text


def test_initial_guess_outside_bounds_is_refused():
    model, _, _, D = _dataset()
    with pytest.raises(ValueError, match="outside the bounds"):
        run_fit(
            model, T, D, p0=[8.0, 300.0], bounds=(np.array([10.0, 10.0]), np.array([20.0, 20.0]))
        )


def test_progress_callback_is_called_without_qt():
    calls = []
    model, _, _, D = _dataset()
    run_fit(model, T, D, p0=[5.0, 200.0], callback=lambda n, p, c: calls.append((n, c)))
    assert len(calls) > 1
    assert calls[-1][1] <= calls[0][1]  # the cost went down


def test_multi_detector_input_is_refused():
    """One detector at a time: a 3-D array is a caller mistake, not a slice."""
    model, _, _, D = _dataset()
    with pytest.raises(ValueError):
        run_fit(model, T, D[:, :, None])


def test_public_api_exposes_the_fitting_entry_points():
    for name in ("run_fit", "project", "solve_amplitudes", "build_weights", "FitOutcome"):
        assert hasattr(pr, name)


def test_default_bounds_are_strictly_positive():
    model = make_model("Sequential", 3, fit_t0=True, fit_irf=True)
    lo, hi = model.default_bounds(T)
    # Lifetimes and IRF must have strictly positive lower bounds to prevent 1/0 singularities
    assert np.all(lo[:3] > 0.0)
    assert lo[3] == float(np.min(T))  # t0 lower bound
    assert lo[4] > 0.0  # IRF lower bound


def test_summary_does_not_zero_out_lifetimes_with_large_uncertainty():
    """When a parameter has large standard error, summary must not round non-zero value to 0."""
    from pyrate_ta.results.fits import GlobalFit

    fit = GlobalFit(
        taus=np.array([0.1456, 0.652, 3.0, 0.05, 300.0]),
        tau_err=np.array([0.0005, 0.002, 2.0, 1e7, 200.0]),
        is_fixed=np.zeros(5, dtype=bool),
        S=np.zeros((10, 5)),
        R=np.zeros((20, 10)),
        model_type="Parallel",
        t=np.linspace(0, 10, 20),
        n_components=5,
    )
    summary = fit.summary()
    assert "tau4 = 0 +/-" not in summary
    assert "0.05" in summary or "0.050" in summary

