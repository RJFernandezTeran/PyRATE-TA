"""Tests for :mod:`pyrate_ta.models`.

The concentration matrices are checked against closed-form expressions rather
than against a stored reference, so a change in the propagator that happens to
be self-consistent still fails here.
"""

from __future__ import annotations

import numpy as np
import pytest

import pyrate_ta as pr
from pyrate_ta.models import (
    TARGET_SCHEMES,
    ParallelModel,
    SequentialModel,
    TargetModel,
    concentrations,
    convolved_exponential,
    fwhm_to_sigma,
    gaussian_irf,
    make_model,
    parallel_K,
    rates_from_lifetimes,
    sequential_K,
)

T = np.geomspace(0.01, 5000.0, 400)


# --------------------------------------------------------------------------- #
#                              Rate matrices                                  #
# --------------------------------------------------------------------------- #
def test_rates_from_lifetimes_rejects_nonpositive():
    assert np.allclose(rates_from_lifetimes([1.0, 4.0]), [1.0, 0.25])
    for bad in ([0.0], [-1.0], [np.nan]):
        with pytest.raises(ValueError):
            rates_from_lifetimes(bad)


def test_infinite_lifetime_is_a_zero_rate():
    """An ``inf`` lifetime is the non-decaying (offset) component, not an error."""
    assert np.allclose(rates_from_lifetimes([2.0, np.inf]), [0.5, 0.0])
    C = ParallelModel(n_components=2).concentrations(T, taus=[5.0, np.inf])
    assert np.allclose(C[:, 1], 1.0)  # it never decays


def test_parallel_K_is_diagonal():
    K = parallel_K([1.0, 0.1])
    assert np.allclose(K, [[-1.0, 0.0], [0.0, -0.1]])


def test_sequential_K_conserves_flux():
    """Every rate leaving a compartment reappears in the next one."""
    k = np.array([1.0, 0.5, 0.1])
    K = sequential_K(k)
    assert np.allclose(np.diag(K), -k)
    assert np.allclose(np.diag(K, -1), k[:-1])
    # Columns of a compartment that feeds another must sum to zero; only the
    # last compartment loses population to the ground state.
    sums = K.sum(axis=0)
    assert np.allclose(sums[:-1], 0.0)
    assert sums[-1] == pytest.approx(-k[-1])


@pytest.mark.parametrize("key", sorted(TARGET_SCHEMES))
def test_target_schemes_are_well_formed(key):
    """Each scheme builds a square matrix with non-positive column sums."""
    scheme = TARGET_SCHEMES[key]
    K = scheme.rate_matrix(np.linspace(0.1, 1.0, scheme.n_rates))
    assert K.shape == (scheme.n_species, scheme.n_species)
    assert np.all(np.diag(K) <= 0)
    # No compartment may gain more than it loses (population is never created).
    assert np.all(K.sum(axis=0) <= 1e-12)
    # Off-diagonal entries are transfer rates and cannot be negative.
    off = K - np.diag(np.diag(K))
    assert np.all(off >= 0)


def test_target_scheme_rejects_wrong_rate_count():
    scheme = TARGET_SCHEMES["A_eq_B"]
    with pytest.raises(ValueError):
        scheme.rate_matrix([1.0])


# --------------------------------------------------------------------------- #
#                          Concentration matrices                             #
# --------------------------------------------------------------------------- #
def test_parallel_concentrations_are_plain_exponentials():
    taus = [2.0, 50.0]
    model = ParallelModel(n_components=2)
    C = model.concentrations(T, taus=taus)
    expected = np.exp(-T[:, None] / np.asarray(taus)[None, :])
    assert C.shape == (T.size, 2)
    assert np.allclose(C, expected, atol=1e-10)


def test_sequential_two_component_matches_closed_form():
    """A -> B -> ground: B(t) has the standard consecutive-reaction form."""
    tau = np.array([3.0, 40.0])
    k = 1.0 / tau
    model = SequentialModel(n_components=2)
    C = model.concentrations(T, taus=tau)

    A = np.exp(-k[0] * T)
    B = k[0] / (k[1] - k[0]) * (np.exp(-k[0] * T) - np.exp(-k[1] * T))
    assert np.allclose(C[:, 0], A, atol=1e-10)
    assert np.allclose(C[:, 1], B, atol=1e-10)


def test_sequential_starts_in_first_compartment():
    model = SequentialModel(n_components=3)
    C = model.concentrations(np.array([0.0]), taus=[1.0, 10.0, 100.0])
    assert np.allclose(C[0], [1.0, 0.0, 0.0], atol=1e-12)


def test_closed_system_conserves_population():
    """An A <=> B equilibrium with no decay keeps the total constant."""
    K = TARGET_SCHEMES["A_eq_B"].rate_matrix([1.0 / 5.0, 1.0 / 20.0])
    C = concentrations(K, [1.0, 0.0], T)
    assert np.allclose(C.sum(axis=1), 1.0, atol=1e-8)


def test_equilibrium_reaches_the_rate_ratio():
    """At long times A/B tends to the ratio of the back and forward rates."""
    k_f, k_b = 1.0 / 2.0, 1.0 / 8.0
    K = TARGET_SCHEMES["A_eq_B"].rate_matrix([k_f, k_b])
    C = concentrations(K, [1.0, 0.0], np.array([1e4]))
    assert C[0, 0] / C[0, 1] == pytest.approx(k_b / k_f, rel=1e-6)


def test_eigen_and_expm_agree():
    """The fallback propagator reproduces the eigenvalue solution."""
    from pyrate_ta.models.propagator import _expm_propagate

    K = TARGET_SCHEMES["A_eq_B_to_C"].rate_matrix([0.5, 0.2, 0.1, 0.02])
    c0 = np.array([1.0, 0.0, 0.0])
    t = np.linspace(0.0, 60.0, 50)
    assert np.allclose(concentrations(K, c0, t), _expm_propagate(K, c0, t, 0.0), atol=1e-9)


def test_concentrations_are_zero_before_time_zero():
    model = SequentialModel(n_components=2)
    t = np.linspace(-10.0, 10.0, 41)
    C = model.concentrations(t, taus=[1.0, 5.0], t0=0.0)
    assert np.allclose(C[t < 0], 0.0)


def test_time_zero_shifts_the_response():
    model = ParallelModel(n_components=1)
    t = np.linspace(0.0, 20.0, 201)
    C0 = model.concentrations(t, taus=[4.0], t0=0.0)
    C1 = model.concentrations(t + 5.0, taus=[4.0], t0=5.0)
    assert np.allclose(C0, C1, atol=1e-12)


def test_shape_and_argument_validation():
    with pytest.raises(ValueError):
        concentrations(np.ones((2, 3)), [1.0, 1.0], T)
    with pytest.raises(ValueError):
        concentrations(np.eye(2), [1.0], T)


# --------------------------------------------------------------------------- #
#                          Instrument response                                #
# --------------------------------------------------------------------------- #
def test_gaussian_irf_is_area_normalised_and_has_the_right_fwhm():
    t = np.linspace(-20.0, 20.0, 20001)
    g = gaussian_irf(t, t0=0.0, fwhm=2.0)
    assert np.trapezoid(g, t) == pytest.approx(1.0, rel=1e-6)
    half = g.max() / 2.0
    above = t[g >= half]
    assert (above.max() - above.min()) == pytest.approx(2.0, rel=1e-3)
    with pytest.raises(ValueError):
        gaussian_irf(t, fwhm=0.0)


def test_convolved_exponential_matches_numerical_convolution():
    """The analytic EMG equals a fine numerical convolution."""
    fwhm, tau, t0, dt = 0.3, 2.0, 0.0, 1e-3
    sigma = fwhm_to_sigma(fwhm)
    grid = np.arange(-5.0, 40.0, dt)
    decay = np.where(grid >= 0, np.exp(-grid / tau), 0.0)
    # The kernel must live on its own symmetric, odd-length grid centred on
    # zero, otherwise ``mode="same"`` silently shifts the result.
    tk = np.arange(-10.0, 10.0 + dt, dt)
    kernel = gaussian_irf(tk, t0=0.0, fwhm=fwhm) * dt
    numeric = np.convolve(decay, kernel, mode="same")

    analytic = convolved_exponential(grid, -1.0 / tau, t0=t0, sigma=sigma).ravel()
    # Compare away from the padded edges of the numerical convolution.
    inside = (grid > -2.0) & (grid < 30.0)
    assert np.allclose(analytic[inside], numeric[inside], atol=2e-3)


def test_convolved_exponential_is_stable_at_early_delays():
    """No overflow where the naive prefactor explodes (long tau, wide IRF)."""
    t = np.linspace(-500.0, 10.0, 2001)
    out = convolved_exponential(t, -1.0 / 5000.0, t0=0.0, sigma=fwhm_to_sigma(50.0))
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0)
    assert out[0] == pytest.approx(0.0, abs=1e-12)


def test_narrow_irf_tends_to_the_unconvolved_limit():
    model = SequentialModel(n_components=2)
    t = np.linspace(0.5, 100.0, 200)
    sharp = model.concentrations(t, taus=[5.0, 50.0], irf_fwhm=None)
    narrow = model.concentrations(t, taus=[5.0, 50.0], irf_fwhm=1e-3)
    assert np.allclose(sharp, narrow, atol=1e-6)


def test_irf_smooths_the_rise():
    """With an IRF the response is non-zero before time zero."""
    model = ParallelModel(n_components=1)
    t = np.linspace(-2.0, 10.0, 121)
    C = model.concentrations(t, taus=[5.0], t0=0.0, irf_fwhm=1.0)
    assert C[t < -0.2].max() > 0
    assert C.max() <= 1.0 + 1e-9


# --------------------------------------------------------------------------- #
#                          Degeneracy guard                                   #
# --------------------------------------------------------------------------- #
def test_near_degenerate_lifetimes_warn_and_stay_finite(caplog):
    from pyrate_ta.models.propagator import check_degeneracy

    with caplog.at_level("INFO", logger="pyrate_ta.models.propagator"):
        pairs = check_degeneracy([10.0, 10.5], warn_ratio=1.2)
    assert pairs == [(0, 1)]
    assert "degenerate" in caplog.text.lower()

    model = SequentialModel(n_components=2)
    C = model.concentrations(T, taus=[10.0, 10.05])
    assert np.all(np.isfinite(C))
    assert np.abs(C).max() < 10.0  # no exploding amplitudes


def test_exactly_degenerate_rates_match_the_analytic_limit():
    """A defective K is split by a part in a million, not refused.

    For A -> B -> with equal rates the closed form is A = exp(-kt),
    B = k t exp(-kt); the eigenvector solution does not exist there, so the
    propagator nudges the rates apart. The result must still match the analytic
    limit to well below any experimental precision.
    """
    tau = 10.0
    k = 1.0 / tau
    t = np.linspace(0.0, 100.0, 200)
    C = SequentialModel(n_components=2).concentrations(t, taus=[tau, tau])

    assert np.allclose(C[:, 0], np.exp(-k * t), atol=1e-6)
    assert np.allclose(C[:, 1], k * t * np.exp(-k * t), atol=1e-5)


def test_degenerate_rates_still_allow_an_irf():
    """The split is what keeps a fit alive when it passes through degeneracy."""
    t = np.linspace(-1.0, 60.0, 200)
    C = SequentialModel(n_components=2).concentrations(t, taus=[10.0, 10.0], irf_fwhm=0.4)
    assert np.all(np.isfinite(C))
    assert C[:, 1].max() > 0.2  # the intermediate really is populated


def test_well_separated_lifetimes_do_not_warn():
    from pyrate_ta.models.propagator import check_degeneracy

    assert check_degeneracy([1.0, 100.0], warn_ratio=1.2) == []


# --------------------------------------------------------------------------- #
#                        Parameter bookkeeping                                #
# --------------------------------------------------------------------------- #
def test_parameter_vector_round_trip():
    model = SequentialModel(n_components=2, fit_t0=True, fit_irf=True)
    assert model.param_names() == ["tau1", "tau2", "t0", "irf_fwhm"]
    vec = model.pack([1.0, 10.0], t0=0.3, irf_fwhm=0.2)
    taus, t0, irf = model.unpack(vec)
    assert np.allclose(taus, [1.0, 10.0])
    assert (t0, irf) == (0.3, 0.2)
    with pytest.raises(ValueError):
        model.unpack([1.0, 2.0])


def test_fixed_parameters_are_not_in_the_vector():
    model = ParallelModel(n_components=3, t0=1.5, irf_fwhm=0.4)
    assert model.param_names() == ["tau1", "tau2", "tau3"]
    _, t0, irf = model.unpack([1.0, 2.0, 3.0])
    assert (t0, irf) == (1.5, 0.4)


def test_bounds_and_initial_guess_lie_inside_them():
    model = SequentialModel(n_components=3, fit_t0=True, fit_irf=True)
    lo, hi = model.default_bounds(T)
    guess = model.initial_guess(T)
    assert lo.size == hi.size == guess.size == model.n_params
    assert np.all(guess >= lo) and np.all(guess <= hi)
    assert np.all(np.diff(guess[:3]) > 0)  # lifetimes ordered, decade-spaced


def test_model_types_drive_the_spectra_label():
    assert ParallelModel(n_components=2).model_type == pr.ModelType.PARALLEL
    assert SequentialModel(n_components=2).model_type == pr.ModelType.SEQUENTIAL
    assert TargetModel(n_components=2, scheme="A_eq_B").model_type == pr.ModelType.TARGET


def test_target_model_takes_its_size_from_the_scheme():
    model = TargetModel(n_components=99, scheme="A_eq_B_to_C")
    assert model.n_components == 3  # the scheme wins
    assert model.n_lifetimes == 4  # ... and it has four rates, not three
    C = model.concentrations(T, taus=[2.0, 5.0, 20.0, 200.0])
    assert C.shape == (T.size, 3)


def test_make_model_factory():
    assert isinstance(make_model("Parallel", 2), ParallelModel)
    assert isinstance(make_model(pr.ModelType.SEQUENTIAL, 3), SequentialModel)
    assert isinstance(make_model("Target", scheme="A_eq_B"), TargetModel)
    with pytest.raises(ValueError):
        make_model("Target", 2)  # no scheme given


def test_species_labels():
    assert SequentialModel(n_components=3).species_labels() == ["A", "B", "C"]
