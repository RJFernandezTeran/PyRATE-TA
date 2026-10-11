"""Tests for Phase 6: Constrained solves and ground-state bleach recovery."""

from __future__ import annotations

import numpy as np

from pyrate_ta.fit.constrained import solve_amplitudes_constrained
from pyrate_ta.fit.groundstate import compute_absolute_spectra, compute_max_gs_scale


def test_solve_amplitudes_constrained_nnls():
    """Test non-negative least squares constraint (lb=0)."""
    np.random.seed(42)
    t = np.linspace(0, 10, 50)
    C = np.column_stack([np.exp(-t / 2.0), np.exp(-t / 5.0)])
    # True spectra with one positive and one negative amplitude
    S_true = np.array([[2.0, -1.5], [1.0, 0.5]])  # 2 pixels, 2 components
    D = C @ S_true.T

    # Constrain all amplitudes to be >= 0
    S_fit = solve_amplitudes_constrained(C, D, sign_mask=[1, 1])
    assert S_fit.shape == (2, 2)
    assert np.all(S_fit >= -1e-12)


def test_groundstate_max_scale():
    """Test max ground-state scale factor calculation."""
    # Synthetic difference spectrum with GSB feature (negative)
    S_diff = np.array([[-1.0, 0.5], [-0.5, 1.0]])  # 2 probe points, 2 components
    gs = np.array([1.0, 1.0])  # Steady state absorption

    a_max = compute_max_gs_scale(S_diff, gs)
    assert a_max > 0
    # At a_max, S_abs should be non-negative everywhere
    S_abs, a, _ = compute_absolute_spectra(S_diff, gs, gs_scale="auto")
    assert np.all(S_abs >= -1e-12)
    assert np.isclose(a, a_max)


def test_groundstate_custom_scale():
    """Test groundstate absolute spectra with explicit scale factor."""
    S_diff = np.array([[-0.8, 0.4], [0.2, 0.5]])
    gs = np.array([1.0, 0.5])
    scale = 0.5

    S_abs, a, a_max = compute_absolute_spectra(S_diff, gs, gs_scale=scale)
    assert a == 0.5
    assert np.allclose(S_abs[:, 0], S_diff[:, 0] + 0.5 * gs)
    assert np.allclose(S_abs[:, 1], S_diff[:, 1] + 0.5 * gs)


def test_groundstate_gsb_and_se_recovery():
    """Test recovering pure positive spectra from both GSB and SE."""
    # Probe grid: 400 to 700 nm
    probe = np.linspace(400, 700, 61)

    # Simulated ground-state absorption peak at 450 nm
    gs = np.exp(-0.5 * ((probe - 450) / 25) ** 2)
    # Simulated fluorescence emission peak at 550 nm
    em = np.exp(-0.5 * ((probe - 550) / 30) ** 2)

    # Pure positive excited-state absorption (ESA) with peak at 620 nm
    pure_esa = 0.8 * np.exp(-0.5 * ((probe - 620) / 35) ** 2)

    # Observed difference spectrum: ESA - a*GS - b*SE
    # Component 1 has both bleach and SE
    s_diff_1 = pure_esa - 0.6 * gs - 0.4 * em
    # Component 2 has primarily bleach
    s_diff_2 = 0.5 * pure_esa - 0.3 * gs

    S_diff = np.column_stack([s_diff_1, s_diff_2])

    res = compute_absolute_spectra(
        S_diff,
        gs_spectrum=gs,
        em_spectrum=em,
        gs_scale="auto",
        em_scale="auto",
    )

    # 1. Pure positive recovery
    assert np.all(res.S_abs >= -1e-12)
    # 2. Both scale factors recovered
    assert np.all(res.gs_scale > 0)
    assert np.all(res.em_scale >= 0)
    # 3. Bleach spectra are negative
    assert np.all(res.bleach_spectra <= 1e-12)
    # 4. Attribute and unpacking compatibility
    s_abs, a_gs, a_em = res
    assert np.allclose(s_abs, res.S_abs)


def test_groundstate_emission_only_recovery():
    """Test pure positive recovery when only emission spectrum is provided."""
    probe = np.linspace(500, 700, 41)
    em = np.exp(-0.5 * ((probe - 580) / 20) ** 2)
    pure_esa = 0.5 * np.exp(-0.5 * ((probe - 650) / 25) ** 2)

    # Only SE negative dip
    S_diff = (pure_esa - 0.4 * em)[:, None]

    res = compute_absolute_spectra(
        S_diff,
        gs_spectrum=None,
        em_spectrum=em,
        gs_scale="auto",
        em_scale="auto",
    )
    assert np.all(res.S_abs >= -1e-12)
    assert res.em_scale > 0
    assert np.all(res.bleach_spectra <= 1e-12)

