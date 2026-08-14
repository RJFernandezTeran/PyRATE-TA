"""Tests for the kinetic-scheme notation and its parser.

The matrix is derived from the reactions, so these tests check the derivation
itself: population balance, shared rates, and that a malformed scheme is
refused with the line named rather than half-built.
"""

from __future__ import annotations

import numpy as np
import pytest

import pyrate_ta as pr
from pyrate_ta.models.scheme_text import (
    SchemeSyntaxError,
    build_rate_matrix,
    parse_scheme_text,
    scheme_from_text,
    scheme_to_text,
)

CHAIN = """
A -> B : k1
B -> C : k2
C ->   : k3
"""


def test_a_chain_parses_into_species_and_rates():
    reactions, species, rates, c0 = parse_scheme_text(CHAIN)
    assert species == ["A", "B", "C"]
    assert rates == ["k1", "k2", "k3"]
    assert len(reactions) == 3
    assert np.allclose(c0, [1.0, 0.0, 0.0])  # all population starts in the first species


def test_the_derived_matrix_balances_population():
    """Every rate that leaves a compartment arrives somewhere, or at the ground state."""
    scheme = scheme_from_text(CHAIN)
    K = scheme.rate_matrix([0.5, 0.2, 0.1])
    # A and B feed the next compartment, so their columns sum to zero.
    assert np.allclose(K.sum(axis=0)[:2], 0.0)
    # C decays to the ground state, so its column sums to -k3.
    assert K.sum(axis=0)[2] == pytest.approx(-0.1)
    assert np.allclose(np.diag(K), [-0.5, -0.2, -0.1])


def test_it_reproduces_a_predefined_scheme():
    """The notation and the hand-written scheme must agree exactly."""
    text = "A <-> B : k1, k2\nB -> C : k3\nC ->   : k4"
    written = scheme_from_text(text)
    predefined = pr.get_scheme("A_eq_B_to_C")
    k = [0.3, 0.05, 0.1, 0.02]
    assert np.allclose(written.rate_matrix(k), predefined.rate_matrix(k))


def test_a_repeated_rate_name_is_one_shared_parameter():
    """Two arrows with the same rate name share a single fitted parameter."""
    scheme = scheme_from_text("A -> B : k_d\nA -> C : k_ct\nB -> : k_d\nC -> : k_d")
    assert scheme.n_species == 3
    assert scheme.n_rates == 2  # k_d and k_ct, not four
    K = scheme.rate_matrix([0.1, 0.4])  # k_d = 0.1, k_ct = 0.4
    assert K[0, 0] == pytest.approx(-0.5)  # A loses both channels
    assert K[1, 1] == pytest.approx(-0.1)  # B decays at the shared rate
    assert K[2, 2] == pytest.approx(-0.1)  # ... and so does C


def test_named_species_survive_into_the_model():
    scheme = scheme_from_text("S1 -> ICT : k_ct\nICT -> : k_r\ninit S1 = 1")
    model = pr.make_model("Target", 0, scheme=scheme)
    assert model.species_labels() == ["S1", "ICT"]
    assert model.lifetime_names() == ["tau_k_ct", "tau_k_r"]
    assert np.allclose(model.c0, [1.0, 0.0])


def test_initial_populations_can_be_split():
    scheme = scheme_from_text("A -> : k1\nB -> : k2\ninit A = 0.7, B = 0.3")
    assert np.allclose(scheme.c0, [0.7, 0.3])
    C = pr.make_model("Target", 0, scheme=scheme).concentrations([0.0], taus=[1.0, 10.0])
    assert np.allclose(C[0], [0.7, 0.3])


def test_a_closed_equilibrium_conserves_population():
    scheme = scheme_from_text("A <-> B : k_f, k_b")
    t = np.geomspace(0.01, 500.0, 100)
    C = pr.make_model("Target", 0, scheme=scheme).concentrations(t, taus=[5.0, 20.0])
    assert np.allclose(C.sum(axis=1), 1.0, atol=1e-8)


def test_comments_and_blank_lines_are_ignored():
    scheme = scheme_from_text("# a comment\n\nA -> B : k1  # trailing\n\nB -> : k2\n")
    assert scheme.n_species == 2 and scheme.n_rates == 2


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("A -> B", "no rate given"),
        ("A B : k1", "no arrow"),
        ("A <-> B : k1", "expected 2 rates"),
        ("A -> B : k1, k2", "expected 1 rate"),
        ("-> B : k1", "must start from a species"),
        ("A <-> : k1, k2", "species on both sides"),
        ("A -> B : 3k", "not a valid rate name"),
        ("", "empty"),
        ("init Q = 1\nA -> : k1", "take part in no reaction"),
    ],
)
def test_malformed_schemes_are_refused_with_a_reason(text, fragment):
    with pytest.raises(SchemeSyntaxError, match=fragment):
        scheme_from_text(text)


def test_the_error_names_the_line():
    with pytest.raises(SchemeSyntaxError, match="line 3"):
        scheme_from_text("A -> B : k1\nB -> C : k2\nC ~~ D : k3")


def test_check_scheme_text_reports_without_raising():
    ok, message = pr.check_scheme_text(CHAIN)
    assert ok and "3 species" in message and "ground state" in message

    ok, message = pr.check_scheme_text("A -> B")
    assert not ok and "no rate given" in message


def test_the_notation_round_trips():
    reactions, *_ = parse_scheme_text("A <-> B : k1, k2\nB -> : k3")
    text = scheme_to_text(reactions)
    assert "A <-> B : k1, k2" in text
    again = scheme_from_text(text)
    assert np.allclose(
        again.rate_matrix([1, 2, 3]),
        scheme_from_text("A <-> B : k1, k2\nB -> : k3").rate_matrix([1, 2, 3]),
    )


def test_build_rate_matrix_checks_the_rate_count():
    reactions, species, rates, _ = parse_scheme_text(CHAIN)
    with pytest.raises(ValueError, match="expected 3 rate"):
        build_rate_matrix(reactions, species, rates, [1.0])


def test_a_written_scheme_can_be_fitted():
    """End to end: notation in, lifetimes out."""
    t = np.concatenate([np.linspace(-1.0, 0.5, 10), np.geomspace(0.6, 2000.0, 120)])
    probe = np.linspace(1900.0, 2000.0, 30)
    scheme = scheme_from_text("A -> B : k1\nB -> : k2")
    model = pr.make_model("Target", 0, scheme=scheme, irf_fwhm=0.2)
    truth = [6.0, 250.0]
    C = model.concentrations(t, taus=truth)
    S = np.stack([np.exp(-((probe - c) ** 2) / (2 * 18.0**2)) for c in (1940, 1970)], axis=1)

    fit = pr.fit_target(C @ S.T, t, scheme=scheme, taus=[2.0, 100.0], irf_fwhm=0.2)
    assert fit.converged
    assert np.allclose(fit.taus, truth, rtol=1e-3)
    assert fit.scheme_label.startswith("A -> B")
