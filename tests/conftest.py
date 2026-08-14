"""Shared fixtures.

Synthetic-recovery fixtures are the backbone of this suite:

build data from known parameters, fit it, assert the parameters come back. A
fitting project without those is untested regardless of line coverage.
"""

import matplotlib

matplotlib.use("Agg")  # no display in CI; must precede any pyplot import

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Seeded generator, so a failing test fails the same way twice."""
    return np.random.default_rng(20260728)


@pytest.fixture
def delays():
    """Delay axis with negative points and a log-spaced tail, as in a real scan."""
    return np.concatenate([np.linspace(-5, -0.5, 5), np.logspace(-1, 3.3, 60)])


@pytest.fixture
def probe():
    """Probe axis in cm-1."""
    return np.linspace(1950, 2150, 128)


@pytest.fixture
def two_component(delays, probe):
    """Synthetic sequential A -> B dataset with known lifetimes and spectra.

    Returns a dict with the ground truth (``taus``, ``spectra``) and the data
    matrix ``D`` ``[Ndelays x Npixels]``, so a fit can be checked against the
    parameters that generated it.
    """
    taus = np.array([5.0, 150.0])
    t = np.clip(delays, 0, None)

    # Sequential A -> B: A decays with tau0, B is fed by A and decays with tau1.
    c_a = np.exp(-t / taus[0])
    ratio = taus[1] / (taus[1] - taus[0])
    c_b = ratio * (np.exp(-t / taus[1]) - np.exp(-t / taus[0]))
    C = np.stack([c_a, c_b], axis=1)
    C[delays < 0] = 0.0

    def band(centre, width, amp):
        return amp * np.exp(-(((probe - centre) / width) ** 2))

    S = np.stack([band(2030, 12, 1.0), band(2065, 15, 0.6)], axis=1)

    return {
        "taus": taus,
        "delays": delays,
        "probe": probe,
        "C": C,
        "spectra": S,
        "D": C @ S.T,
    }


@pytest.fixture
def two_component_noisy(two_component, rng):
    """``two_component`` with 1 % Gaussian noise on the peak amplitude."""
    truth = dict(two_component)
    scale = 0.01 * np.nanmax(np.abs(truth["D"]))
    truth["D"] = truth["D"] + rng.normal(0.0, scale, truth["D"].shape)
    truth["noise_scale"] = scale
    return truth
