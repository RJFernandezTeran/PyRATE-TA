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
