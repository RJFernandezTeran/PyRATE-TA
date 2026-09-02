"""Tests for the fitting engines, the result objects and the scheme diagram.

The dataset entry point is exercised with a stand-in object carrying the three
attributes ``prepare`` needs, so these tests neither import PyMORGAN nor need a
data file on disk.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import pyrate_ta as pr
from pyrate_ta.helpers import format_lifetime, is_infinite_lifetime, parse_lifetime

T = np.concatenate([np.linspace(-2.0, 0.5, 20), np.geomspace(0.6, 3000.0, 140)])
PROBE = np.linspace(1900.0, 2010.0, 40)
TAUS = (8.0, 300.0)


def _spectra(centres=(1950.0, 1975.0)):
    return np.stack([np.exp(-((PROBE - c) ** 2) / (2 * 20.0**2)) for c in centres], axis=1)


def _data(taus=TAUS, irf=0.3, noise=0.0, seed=0):
    model = pr.make_model("Sequential", len(taus), irf_fwhm=irf)
    C = model.concentrations(T, taus=taus)
    S = _spectra()
    D = C @ S.T
    if noise:
        D = D + noise * np.random.default_rng(seed).standard_normal(D.shape)
    return D, S


class _StubDataset:
    """The three attributes :func:`pyrate.fit.prepare` needs, and the noise hook."""

    def __init__(self, D, sigma=None, n_det=2):
        self.Z = np.repeat(D[:, :, None], n_det, axis=2)
        self.Z[:, :, 1] *= 0.5  # a second detector, deliberately different
        self.delays = T
        self.probe = PROBE
        self.units = {"unitsT_ltx": "ps"}
        self.source = "stub.pdat"
        self.data_type = "PDAT"
        self.Zstdv = None if sigma is None else np.repeat(sigma[:, :, None], n_det, axis=2)
        self.has_single_scans = False
        self._sigma = self.Zstdv

    def noise_array(self):
        return self._sigma


# --------------------------------------------------------------------------- #
#                          Lifetime parsing                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text", ["inf", "Inf", "INF", " infinity ", "Infinite", "∞", "oo", "constant", "offset"]
)
def test_infinite_spellings_are_all_accepted(text):
    assert is_infinite_lifetime(text)
    assert math.isinf(parse_lifetime(text))


def test_finite_lifetimes_parse_including_a_decimal_comma():
    assert parse_lifetime("12.5") == 12.5
    assert parse_lifetime("1,5") == 1.5  # a European decimal comma, not 15
    assert parse_lifetime(3) == 3.0


@pytest.mark.parametrize("bad", ["", "abc", "0", "-4", float("nan")])
def test_bad_lifetimes_are_refused(bad):
    with pytest.raises(ValueError):
        parse_lifetime(bad)


def test_format_lifetime_shows_infinity_as_inf():
    assert format_lifetime(math.inf) == "inf"
    assert format_lifetime(12.3456) == "12.35"


# --------------------------------------------------------------------------- #
#                             fit_global                                      #
# --------------------------------------------------------------------------- #
def test_fit_global_from_arrays_recovers_the_lifetimes():
    D, S = _data()
    fit = pr.fit_global(D, T, taus=[3.0, 100.0], irf_fwhm=0.3)

    assert fit.converged
    assert np.allclose(fit.taus, TAUS, rtol=1e-4)
    assert fit.model_type == "Sequential"
    assert fit.spectra_kind == "EAS"
    assert np.allclose(np.abs(fit.S), np.abs(S), atol=1e-5)


def test_fit_global_accepts_a_dataset_and_records_its_provenance():
    D, _ = _data()
    fit = pr.fit_global(_StubDataset(D), taus=[3.0, 100.0], irf_fwhm=0.3, detector=0)

    assert np.allclose(fit.taus, TAUS, rtol=1e-4)
    assert fit.source == "stub.pdat"
    assert fit.detector == 0
    assert fit.delay_range[0] >= 0.2
    assert fit.delay_range[1] == float(T.max())
    assert fit.probe_range == (float(PROBE.min()), float(PROBE.max()))


def test_detector_selection_changes_the_amplitudes_not_the_lifetimes():
    D, _ = _data()
    data = _StubDataset(D)
    first = pr.fit_global(data, taus=[3.0, 100.0], irf_fwhm=0.3, detector=0)
    second = pr.fit_global(data, taus=[3.0, 100.0], irf_fwhm=0.3, detector=1)

    assert np.allclose(first.taus, second.taus, rtol=1e-4)
    assert np.allclose(second.S, first.S * 0.5, atol=1e-6)  # detector 1 is halved
    with pytest.raises(ValueError):
        pr.fit_global(data, taus=[3.0, 100.0], detector=9)


def test_restricting_the_window_is_recorded():
    D, _ = _data()
    fit = pr.fit_global(
        _StubDataset(D),
        taus=[3.0, 100.0],
        irf_fwhm=0.3,
        t_min=1.0,
        probe_range=(1930.0, 1990.0),
    )
    assert fit.t_min == 1.0
    assert fit.delay_range[0] >= 1.0
    assert fit.probe_range[0] >= 1930.0
    assert fit.t.size < T.size


def test_components_are_sorted_for_a_sequential_model():
    D, _ = _data()
    fit = pr.fit_global(D, T, taus=[100.0, 3.0], irf_fwhm=0.3)

    assert fit.sorted_by_lifetime is True
    assert list(fit.permutation) == [1, 0]
    assert fit.taus[0] < fit.taus[1]
    # C, S and taus stay mutually consistent: C is rebuilt from the sorted
    # lifetimes and the spectra re-solved, so the residual is still zero.
    assert np.max(np.abs(fit.R)) < 1e-6
    assert np.allclose(
        fit.C, pr.make_model("Sequential", 2, irf_fwhm=0.3).concentrations(fit.t, taus=fit.taus)
    )


def test_target_fit_is_not_reordered_and_keeps_its_scheme():
    model = pr.make_model("Target", 3, scheme="A_eq_B_to_C", irf_fwhm=0.3)
    truth = [5.0, 40.0, 20.0, 500.0]
    C = model.concentrations(T, taus=truth)
    S = np.stack([np.exp(-((PROBE - c) ** 2) / (2 * 20.0**2)) for c in (1940, 1960, 1990)], axis=1)

    fit = pr.fit_target(
        C @ S.T, T, scheme="A_eq_B_to_C", taus=[3.0, 30.0, 15.0, 400.0], irf_fwhm=0.3
    )

    assert fit.sorted_by_lifetime is False
    assert fit.scheme_key == "A_eq_B_to_C"
    assert fit.spectra_kind == "SAS"
    assert fit.K.shape == (3, 3)
    # The eigenvalues are what a branched scheme actually determines.
    expected = np.sort(np.real(np.linalg.eigvals(model.rate_matrix(truth))))
    assert np.allclose(fit.eigenvalues, expected, rtol=1e-3)


def test_single_trace_fit():
    D, _ = _data()
    y = D[:, 20]
    fit = pr.fit_kinetics(T, y, taus=[3.0, 100.0], irf_fwhm=0.3)
    assert np.allclose(fit.taus, TAUS, rtol=1e-3)
    assert fit.S.shape == (1, 2)


# --------------------------------------------------------------------------- #
#                        Offset (infinite lifetime)                           #
# --------------------------------------------------------------------------- #
def test_infinite_lifetime_adds_a_non_decaying_component():
    D, _ = _data()
    offset = 0.4 * np.exp(-((PROBE - 1990.0) ** 2) / (2 * 15.0**2))
    D = D + np.where(T > 0, 1.0, 0.0)[:, None] * offset[None, :]

    fit = pr.fit_global(D, T, taus=[3.0, 100.0, "inf"], irf_fwhm=0.3)

    assert fit.n_infinite == 1
    assert math.isinf(fit.taus[-1])
    assert bool(fit.is_fixed[-1]) is True  # infinity cannot be optimised
    assert math.isnan(fit.tau_err[-1])  # ... and gets no uncertainty
    assert np.allclose(fit.taus[:2], TAUS, rtol=0.05)
    # The offset spectrum is recovered where it was put.
    assert PROBE[np.argmax(np.abs(fit.S[:, -1]))] == pytest.approx(1990.0, abs=6.0)


def test_infinite_lifetime_is_accepted_in_words():
    D, _ = _data()
    fit = pr.fit_global(D, T, taus=[3.0, 100.0, "Infinity"], irf_fwhm=0.3)
    assert math.isinf(fit.taus[-1])


# --------------------------------------------------------------------------- #
#                     Uncertainties and reporting                             #
# --------------------------------------------------------------------------- #
def test_uncertainties_are_reported_and_scale_with_the_noise():
    sigma_true = 0.02
    D, _ = _data(noise=sigma_true, seed=7)
    sigma = np.full(D.shape, sigma_true)
    fit = pr.fit_global(_StubDataset(D, sigma), taus=[3.0, 100.0], irf_fwhm=0.3, use_weights=True)

    assert fit.statistic.kind == "chi2_red"
    assert fit.settings["noise_source"] == "pdatn"
    assert np.all(np.isfinite(fit.tau_err))
    assert np.all(fit.tau_err > 0)
    # A 3x noisier dataset must give larger error bars.
    D3, _ = _data(noise=3 * sigma_true, seed=7)
    noisier = pr.fit_global(
        _StubDataset(D3, np.full(D.shape, 3 * sigma_true)),
        taus=[3.0, 100.0],
        irf_fwhm=0.3,
        use_weights=True,
    )
    assert np.all(noisier.tau_err > fit.tau_err)


def test_fixed_lifetime_gets_no_uncertainty():
    D, _ = _data()
    fit = pr.fit_global(D, T, taus=[8.0, 100.0], irf_fwhm=0.3, fixed=["tau1"])
    assert bool(fit.is_fixed[0]) is True
    assert math.isnan(fit.tau_err[0])
    assert np.isfinite(fit.tau_err[1])


def test_summary_names_the_statistic_and_the_fixed_flags():
    D, _ = _data()
    fit = pr.fit_global(D, T, taus=[3.0, "inf"], irf_fwhm=0.3)
    text = fit.summary()
    assert "Sequential fit" in text
    assert "inf (non-decaying)" in text
    assert "SSR" in text


def test_as_species_args_matches_the_pymorgan_signature():
    D, _ = _data()
    fit = pr.fit_global(D, T, taus=[3.0, 100.0], irf_fwhm=0.3)
    Sfit, taus, tau_err, is_fixed, model_type = fit.as_species_args()
    assert Sfit.shape == (PROBE.size, 2)
    assert taus.shape == tau_err.shape == is_fixed.shape == (2,)
    assert is_fixed.dtype == bool
    assert model_type == "Sequential"


def test_weighted_fit_without_noise_is_refused():
    D, _ = _data()
    with pytest.raises(ValueError):
        pr.fit_global(_StubDataset(D, sigma=None), taus=[3.0, 100.0], use_weights=True)


# --------------------------------------------------------------------------- #
#                           Scheme diagram                                    #
# --------------------------------------------------------------------------- #
def test_scheme_text_reads_as_a_reaction_scheme():
    assert (
        pr.scheme_text(pr.make_model("Sequential", 3), taus=[1, 10, 100]) == "A -> B; B -> C; C ->"
    )
    assert pr.scheme_text(pr.make_model("Parallel", 2), taus=[1, 10]) == "A ->; B ->"
    assert pr.scheme_text(pr.get_scheme("A_eq_B_to_C"), taus=[5, 40, 20, 500]) == (
        "A <=> B; B -> C; C ->"
    )


def test_plot_scheme_draws_one_node_per_compartment_plus_the_ground_state():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyArrowPatch

    _, ax = plt.subplots()
    pr.plot_scheme(pr.make_model("Sequential", 3), taus=[1.0, 10.0, 100.0], ax=ax)

    circles = [p for p in ax.patches if isinstance(p, Circle)]
    arrows = [p for p in ax.patches if isinstance(p, FancyArrowPatch)]
    assert len(circles) == 4  # A, B, C and the ground state
    assert len(arrows) == 3  # A->B, B->C, C->ground
    plt.close("all")


def test_plot_scheme_draws_both_directions_of_an_equilibrium():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    _, ax = plt.subplots()
    pr.plot_scheme(pr.get_scheme("A_eq_B"), taus=[5.0, 20.0], ax=ax)
    arrows = [p for p in ax.patches if isinstance(p, FancyArrowPatch)]
    assert len(arrows) == 2
    # A closed equilibrium loses no population, so there is no ground state.
    assert "GS" not in [t.get_text() for t in ax.texts]
    plt.close("all")


def test_plot_scheme_accepts_a_fit_result():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    D, _ = _data()
    fit = pr.fit_global(D, T, taus=[3.0, 100.0], irf_fwhm=0.3)
    ax = pr.plot_scheme(fit)
    assert ax.get_title()
    plt.close("all")


def test_plot_scheme_needs_lifetimes_for_a_bare_model():
    with pytest.raises(ValueError, match="needs lifetimes"):
        pr.plot_scheme(pr.make_model("Sequential", 2))


# --------------------------------------------------------------------------- #
#                               Settings                                      #
# --------------------------------------------------------------------------- #
def test_settings_round_trip_through_a_sectioned_file(tmp_path):
    """Every field survives a save/load, including inf and the empty-means-None ones."""
    from pyrate_ta.settings import Settings, load_settings, save_settings, set_settings

    original = pr.get_settings()
    try:
        target = tmp_path / "settings.toml"
        pr.set_settings(Settings())
        pr.update_settings(lda_n_lifetimes=42, gui_theme="Fusion Dark", table_default_ub=math.inf)
        save_settings(target)

        text = target.read_text(encoding="utf-8")
        assert "[fit]" in text and "[solver]" in text and "[lda]" in text and "[gui]" in text
        assert "# Points on the log-spaced lifetime grid." in text  # comments are written

        loaded = load_settings(target)
        assert loaded.lda_n_lifetimes == 42
        assert loaded.gui_theme == "Fusion Dark"
        assert math.isinf(loaded.table_default_ub)
        assert loaded.irf_fwhm is None  # an empty value reads back as None
    finally:
        set_settings(original)


def test_every_field_has_a_section():
    """A field with no section would be written where load cannot find it."""
    from dataclasses import fields

    from pyrate_ta.settings import Settings

    assert {f.name for f in fields(Settings)} == set(Settings.field_sections())


def test_a_flat_file_still_loads(tmp_path):
    """The older layout keeps working, so an existing settings.toml is not orphaned."""
    from pyrate_ta.settings import Settings, load_settings, set_settings

    original = pr.get_settings()
    try:
        flat = tmp_path / "flat.toml"
        flat.write_text("n_components = 5\nlda_alpha = 0.25\n", encoding="utf-8")
        loaded = load_settings(flat)
        assert loaded.n_components == 5
        assert loaded.lda_alpha == 0.25
    finally:
        set_settings(original)
        pr.set_settings(Settings())


def test_ensure_settings_file_creates_but_never_overwrites(tmp_path):
    import pytest

    from pyrate_ta.settings import ensure_settings_file, load_settings

    target = tmp_path / "settings.toml"
    with pytest.warns(UserWarning, match="pyrate-ta-settings"):
        ensure_settings_file(target)
    assert target.is_file()

    text = target.read_text(encoding="utf-8")
    assert "PyRATE-TA Settings Configuration" in text
    assert 'default_datadir = ""' in text
    assert "[fit]" in text
    assert "[solver]" in text

    # Hand edit with custom settings
    target.write_text("# edited by hand\n[fit]\nn_components = 7\n", encoding="utf-8")
    ensure_settings_file(target)

    content = target.read_text(encoding="utf-8")
    assert "n_components = 7" in content
    # Missing sections should be merged in
    assert "[solver]" in content
    assert "[lda]" in content

    loaded = load_settings(target)
    assert loaded.n_components == 7


# --------------------------------------------------------------------------- #
#                     Scheme diagram: rate symbols                            #
# --------------------------------------------------------------------------- #
def test_arrows_are_labelled_with_the_rate_symbols():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    pr.plot_scheme(pr.make_model("Sequential", 3), taus=[1.0, 10.0, 100.0], ax=ax)
    labels = {t.get_text() for t in ax.texts}
    assert {"$k_{1}$", "$k_{2}$", "$k_{3}$"} <= labels
    assert "1" not in labels  # the values are not written
    plt.close("all")


def test_symbols_follow_the_scheme_not_the_arrow_order():
    """In A <=> B, k_1 is the forward rate and k_2 the reverse, as the scheme defines."""
    from pyrate_ta.plot.scheme import rate_symbols

    scheme = pr.get_scheme("A_eq_B")
    symbols = rate_symbols(scheme.build, scheme.n_rates, scheme.n_species)
    assert symbols[(0, 1)] == [0]  # A -> B is k_1
    assert symbols[(1, 0)] == [1]  # B -> A is k_2


def test_a_branching_decay_is_labelled_as_a_sum():
    """Two rates leaving one compartment to the ground state add up on one arrow."""
    from pyrate_ta.plot.scheme import _symbol_label, rate_symbols

    scheme = pr.get_scheme("A_eq_B_both_decay")
    symbols = rate_symbols(scheme.build, scheme.n_rates, scheme.n_species)
    assert _symbol_label(symbols[(0, None)]) == "$k_{2}$"
    assert symbols[(1, None)] == [3]


def test_values_can_still_be_requested():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    pr.plot_scheme(pr.make_model("Parallel", 2), taus=[2.0, 20.0], ax=ax, rate_labels="value")
    labels = {t.get_text() for t in ax.texts}
    assert {"2", "20"} <= labels
    plt.close("all")


# --------------------------------------------------------------------------- #
#                                Preview                                      #
# --------------------------------------------------------------------------- #
def test_preview_evaluates_without_optimising():
    """A preview shows what the typed lifetimes imply; it does not fit them."""
    D, _ = _data()
    good = pr.preview_global(D, T, taus=list(TAUS), irf_fwhm=0.3)
    bad = pr.preview_global(D, T, taus=[3.0, 100.0], irf_fwhm=0.3)

    assert good.report is None  # nothing was optimised, so there is no report
    assert np.allclose(good.taus, TAUS)  # ... and the lifetimes are unchanged
    assert np.all(good.is_fixed)
    assert np.all(np.isnan(good.tau_err))  # no fit, no uncertainties
    assert good.statistic.value < bad.statistic.value  # the right guess fits better
    assert good.settings["preview"] is True


def test_preview_still_solves_the_spectra():
    D, S = _data()
    preview = pr.preview_global(D, T, taus=list(TAUS), irf_fwhm=0.3)
    assert np.allclose(np.abs(preview.S), np.abs(S), atol=1e-6)
    assert np.max(np.abs(preview.R)) < 1e-8


# --------------------------------------------------------------------------- #
#                           Coherent artefact                                 #
# --------------------------------------------------------------------------- #
def _artefact_data(irf=0.3, seed=3):
    """A sequential dataset with a coherent artefact sitting on time zero."""
    rng = np.random.default_rng(seed)
    t = np.concatenate([np.linspace(-2.0, 0.5, 40), np.geomspace(0.6, 3000.0, 140)])
    probe = np.linspace(1900.0, 2010.0, 64)
    C = pr.make_model("Sequential", 2, irf_fwhm=irf).concentrations(t, taus=[8.0, 300.0])
    S = np.stack(
        [
            a * np.exp(-((probe - c) ** 2) / (2 * w**2))
            for c, w, a in ((1950.0, 15.0, 1.0), (1975.0, 25.0, -0.6))
        ],
        axis=1,
    )
    art = pr.models.coherent_artifact_basis(t, t0=0.0, fwhm=irf)
    S_art = np.stack(
        [
            0.4 * np.ones_like(probe),
            0.8 * np.exp(-((probe - 1960.0) ** 2) / (2 * 40.0**2)),
            0.3 * np.ones_like(probe),
        ],
        axis=1,
    )
    D = C @ S.T + art @ S_art.T + 0.01 * rng.standard_normal((t.size, probe.size))
    return t, probe, D


def test_the_artefact_basis_is_the_irf_and_its_derivatives():
    from pyrate_ta.models import ARTIFACT_LABELS, coherent_artifact_basis

    t = np.linspace(-3.0, 3.0, 601)
    basis = coherent_artifact_basis(t, t0=0.0, fwhm=0.5)
    assert basis.shape == (t.size, len(ARTIFACT_LABELS))
    assert np.allclose(np.max(np.abs(basis), axis=0), 1.0)  # scaled, so conditioned

    # The Gaussian peaks at t0, its first derivative vanishes there and changes
    # sign, and the second derivative is at its minimum.
    assert abs(t[np.argmax(basis[:, 0])]) < 1e-9
    assert abs(basis[np.argmin(np.abs(t)), 1]) < 1e-9
    assert basis[0, 1] * basis[-1, 1] < 0
    assert abs(t[np.argmin(basis[:, 2])]) < 1e-9


def test_the_artefact_basis_needs_a_width():
    """No IRF, no artefact shape -- and guessing one would invent a component."""
    from pyrate_ta.models import coherent_artifact_basis

    with pytest.raises(ValueError, match="needs a Gaussian IRF"):
        coherent_artifact_basis(np.linspace(-1, 1, 10), fwhm=None)


def test_the_artefact_stops_it_distorting_the_shortest_lifetime():
    """The point of the option: without it, tau1 absorbs the artefact."""
    t, _, D = _artefact_data()
    plain = pr.fit_global(
        D, t, taus=[5.0, 200.0], model_type="Sequential", irf_fwhm=0.3, delay_range=(t.min(), t.max())
    )
    fitted = pr.fit_global(
        D,
        t,
        taus=[5.0, 200.0],
        model_type="Sequential",
        irf_fwhm=0.3,
        coherent_artifact=True,
        delay_range=(t.min(), t.max()),
    )

    assert fitted.statistic.value < 0.1 * plain.statistic.value
    assert abs(fitted.taus[0] - 8.0) < abs(plain.taus[0] - 8.0)
    assert np.allclose(fitted.taus, [8.0, 300.0], rtol=0.02)


def test_the_artefact_columns_are_not_species():
    """They have no lifetime, so they must not reach the species-spectra plot."""
    t, _, D = _artefact_data()
    fit = pr.fit_global(
        D,
        t,
        taus=[5.0, 200.0],
        model_type="Sequential",
        irf_fwhm=0.3,
        coherent_artifact=True,
        delay_range=(t.min(), t.max()),
    )
    assert fit.n_artifact == 3
    assert fit.C.shape[1] == fit.n_components + 3
    assert fit.n_kinetic == fit.n_components
    assert fit.as_species_args()[0].shape[1] == fit.n_components
    assert fit.artifact_spectra.shape[1] == 3

    from pyrate_ta.plot import species_labels_for

    assert species_labels_for(fit)[-3:] == ["IRF", "dIRF/dt", "d2IRF/dt2"]


# --------------------------------------------------------------------------- #
#                          Concentration profiles                             #
# --------------------------------------------------------------------------- #
def _fitted():
    D, _ = _data()
    return pr.preview_global(D, T, taus=list(TAUS), irf_fwhm=0.3)


def test_concentration_profiles_use_the_pymorgan_delay_axis():
    """The populations are PyRATE-TA's own view; the delay axis is not.

    Anything else would put a second time-axis convention beside the one the
    contour and the kinetic traces already use.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fit = _fitted()
    ax = pr.plot_concentrations(fit)
    import pymorgan as pm

    expected = str(getattr(pm.get_settings().time_axis_scale, "value", "symlog"))
    assert ax.get_xscale() == {"lin": "linear"}.get(expected, expected)
    # One labelled line per species (the zero line carries no label).
    labelled = [ln for ln in ax.lines if not ln.get_label().startswith("_")]
    assert len(labelled) == fit.C.shape[1]
    assert "population" in ax.get_ylabel().lower()
    plt.close("all")


def test_a_result_without_profiles_is_refused_not_drawn_empty():
    """A blank axis would read as a model with no population anywhere."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fit = _fitted()
    fit.C = None
    with pytest.raises(ValueError, match="no concentration profiles"):
        pr.plot_concentrations(fit)
    plt.close("all")


def test_species_names_come_from_the_scheme():
    """A target fit reads in its own species names, not A, B, C."""
    from pyrate_ta.plot import species_labels_for

    fit = _fitted()
    assert species_labels_for(fit) == ["A", "B"]  # no scheme: the plain default
    assert fit.species_labels() == ["A", "B"]
    fit.scheme_text = "S1 -> ICT : k1\nICT -> : k2\ninit S1 = 1"
    assert species_labels_for(fit) == ["S1", "ICT"]
    assert fit.species_labels() == ["S1", "ICT"]
    assert species_labels_for(fit, labels=["one", "two"]) == ["one", "two"]


def test_a_restricted_probe_window_still_plots_its_spectra():
    """The bug: S has the fitted window's rows, the dataset the full axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pyrate_ta.plot import as_dataset, as_fit_dataset

    D, _ = _data()
    stub = _StubDataset(D)
    window = (float(PROBE[8]), float(PROBE[-8]))
    fit = pr.fit_global(stub, taus=list(TAUS), model_type="Sequential", probe_range=window)
    assert fit.S.shape[0] < PROBE.size  # the fit really was restricted

    full = as_dataset(T, PROBE, D, stub)  # the full probe axis, as loaded
    with pytest.raises(ValueError, match="probe points"):
        full.plot_species_spectra(*fit.as_species_args())  # this is what used to crash

    ax = as_fit_dataset(fit, stub).plot_species_spectra(*fit.as_species_args())
    assert len(ax.lines) >= fit.n_kinetic
    plt.close("all")


def test_lifetime_labels_are_rounded_and_use_a_proper_plus_minus():
    """A legend is read, not parsed: 320 fs, not 320.95763496398745."""
    from pymorgan.helpers import format_lifetime_label

    assert format_lifetime_label(320.95763496398745, "fs") == "320 fs"
    assert format_lifetime_label(320.95763496398745, "fs", err=5.234) == r"320 $\pm$ 5.2 fs"
    # A fixed parameter has no uncertainty; printing one would invent a result.
    assert format_lifetime_label(8.0, "ps", err=0.1, fixed=True) == "8 ps"
    assert format_lifetime_label(float("inf"), "ps") == r"$\infty$"
    assert "+/-" not in format_lifetime_label(1.0, "ps", err=0.1)


def test_quoting_follows_the_settings():
    """Whether an uncertainty is shown, and how coarsely, is the user's choice."""
    D, _ = _data()
    fit = pr.fit_global(D, T, taus=list(TAUS), model_type="Sequential", delay_range=(T.min(), T.max()))
    original = pr.get_settings()
    try:
        pr.update_settings(show_uncertainties=True, round_uncertainties=True)
        rounded = [line for line in fit.summary().splitlines() if "tau1" in line][0]
        assert "+/-" in rounded

        pr.update_settings(show_uncertainties=False)
        hidden = [line for line in fit.summary().splitlines() if "tau1" in line][0]
        assert "+/-" not in hidden

        # ... and the legend follows the same two settings.
        pr.update_settings(show_uncertainties=True, round_uncertainties=False)
        loose = pr.plot_concentrations(fit).get_legend().get_texts()[0].get_text()
        pr.update_settings(round_uncertainties=True)
        tight = pr.plot_concentrations(fit).get_legend().get_texts()[0].get_text()
        assert r"\pm" in loose and r"\pm" in tight
        assert len(tight) <= len(loose)  # rounding cannot make it longer
    finally:
        pr.set_settings(original)
        import matplotlib.pyplot as plt

        plt.close("all")


def test_a_custom_scheme_can_be_drawn_from_its_own_fit():
    """The reported failure: a typed scheme records the key "custom".

    Looking that key up in the registry raised ``KeyError``, so 'Show kinetic
    graph' failed for exactly the schemes the editor produces. The text is
    parsed instead, which also recovers the species names and rate symbols.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pyrate_ta.plot.scheme import _scheme_of

    D, _ = _data()
    scheme = pr.scheme_from_text("S1 -> ICT : k1\nICT -> : k2\ninit S1 = 1")
    fit = pr.fit_target(D, T, scheme=scheme, taus=list(TAUS))
    assert fit.scheme_key == "custom"  # not in any registry
    assert _scheme_of(fit) is not None

    ax = pr.plot_scheme(fit)
    texts = {t.get_text() for t in ax.texts}
    assert {"S1", "ICT"} <= texts  # the scheme's own names, not A and B
    assert any("k_{1}" in t for t in texts)  # ... and its rate symbols
    plt.close("all")


def test_the_scheme_layouts_are_horizontal():
    """Population flows left to right, so a wide figure is used, not a column."""
    from pyrate_ta.plot.scheme import _layout

    for family in ("Sequential", "Parallel"):
        pos = _layout(3, family, has_ground=True)
        assert [pos[i][0] for i in range(3)] == [0.0, 1.0, 2.0]
        assert {pos[i][1] for i in range(3)} == {0.0}  # one row

    edges = [(0, 1, 1.0), (0, 2, 1.0), (1, None, 1.0), (2, None, 1.0)]
    branched = _layout(3, "Target", has_ground=True, edges=edges)
    assert branched[0][0] < branched[1][0]  # layers advance along x
    assert branched[1][1] != branched[2][1]  # the branch spreads in y


def test_the_spectra_are_named_after_the_model():
    """DAS, EAS or SAS -- the distinction is what they may be interpreted as."""
    D, _ = _data()
    assert pr.preview_global(D, T, taus=list(TAUS), model_type="Parallel").spectra_kind == "DAS"
    assert pr.preview_global(D, T, taus=list(TAUS), model_type="Sequential").spectra_kind == "EAS"


# --------------------------------------------------------------------------- #
#                    Fit / residual views via PyMORGAN                        #
# --------------------------------------------------------------------------- #
def test_plot_matrix_wraps_the_arrays_as_a_dataset():
    """The fit and residual views go through PyMORGAN's contour engine.

    Wrapping is what lets a PyMORGAN colourmap identifier such as
    ``DkRd/Wh/DkBu`` work here: this side never touches colourmaps.
    """
    from pyrate_ta.plot.matrix import as_dataset

    D, _ = _data()
    wrapped = as_dataset(T, PROBE, D, _StubDataset(D))
    assert wrapped.Z.shape == (T.size, PROBE.size, 1)
    assert np.allclose(wrapped.delays, T)
    assert wrapped.units["unitsT_ltx"] == "ps"  # the dataset's own units travel


def test_plot_matrix_accepts_the_pymorgan_colormap_name():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    D, _ = _data()
    _, ax = plt.subplots()
    # This is the setting PyMORGAN ships; passing it to contourf directly raises.
    pr.plot_matrix(T, PROBE, D, ax=ax, dataset=_StubDataset(D), cmap_ID="DkRd/Wh/DkBu")
    assert ax.collections or ax.get_children()
    plt.close("all")


def test_overlay_styles_come_from_the_settings():
    """Data as points, fit as a line -- both configurable in settings.toml."""
    from pyrate_ta.plot.style import overlay_styles

    original = pr.get_settings()
    try:
        pr.update_settings(data_marker="s", data_alpha=0.3, fit_linewidth=2.5)
        data, fit = overlay_styles("kinetics")
        assert data["plotStyle"] == "s"
        assert data["alpha"] == 0.3
        assert fit["lw"] == 2.5
        assert fit["plotStyle"] == "-"

        # plot_spectra takes no plotStyle: the marker goes in as Matplotlib
        # keywords instead, which is exactly the mistake this mapping avoids.
        data, fit = overlay_styles("spectra")
        assert data == {"marker": "s", "linestyle": "none", "alpha": 0.3, "markersize": 3.0}
        assert fit["linewidth"] == 2.5 and fit["marker"] == "none"
        assert "plotStyle" not in data and "plotStyle" not in fit
    finally:
        pr.set_settings(original)


def test_scale_kwargs_carries_the_symlog_threshold():
    """Copying a symlog scale by name alone would misalign the shared axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pyrate_ta.plot.style import scale_kwargs

    _, ax = plt.subplots()
    ax.set_yscale("symlog", linthresh=2.5, linscale=0.75)
    kwargs = scale_kwargs(ax)
    assert kwargs["linthresh"] == 2.5

    _, other = plt.subplots()
    other.set_yscale(ax.get_yscale(), **kwargs)
    assert other.yaxis.get_transform().linthresh == 2.5

    _, linear = plt.subplots()
    assert scale_kwargs(linear) == {}
    plt.close("all")


def test_rotated_kinetics_put_the_delays_on_the_vertical_axis():
    """PyMORGAN draws the rotated panel; PyRATE-TA only asks for it.

    The orientation is what allows the delay axis to be shared with the contour
    beside it, so the two panels line up.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pyrate_ta.plot.matrix import as_dataset

    D, _ = _data()
    dataset = as_dataset(T, PROBE, D)
    _, ax = plt.subplots()
    dataset.plot_kinetics([1950.0], ax=ax, swap_axes=True)

    x, y = ax.lines[0].get_data()
    assert np.allclose(y, T)  # delays vertical
    assert "Delay" in ax.get_ylabel()
    plt.close("all")


# --------------------------------------------------------------------------- #
#                              Crosshair                                      #
# --------------------------------------------------------------------------- #
def _crosshair(ax=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pyrate_ta.gui.crosshair import Crosshair

    if ax is None:
        _, ax = plt.subplots()
        ax.set_xlim(1900, 2000)
        ax.set_ylim(-1, 100)
    seen = []
    released = []
    hair = Crosshair(
        ax.figure.canvas,
        ax,
        on_change=lambda x, y: seen.append((x, y)),
        on_release=lambda x, y: released.append((x, y)),
        x=1950.0,
        y=10.0,
    )
    return hair, ax, seen, released


def test_crosshair_draws_two_guide_lines_and_reports_drags():
    from matplotlib.backend_bases import MouseButton, MouseEvent

    hair, ax, seen, released = _crosshair()
    assert hair.position == (1950.0, 10.0)

    # A press away from the lines moves the whole crosshair to that point.
    x_pix, y_pix = ax.transData.transform((1975.0, 50.0))
    press = MouseEvent("button_press_event", ax.figure.canvas, x_pix, y_pix, MouseButton.LEFT)
    hair._on_press(press)
    assert seen and abs(seen[-1][0] - 1975.0) < 1.0

    release = MouseEvent("button_release_event", ax.figure.canvas, x_pix, y_pix, MouseButton.LEFT)
    hair._on_release(release)
    assert released, "the panels are redrawn on release, so it must fire"

    hair.disconnect()


def test_crosshair_position_can_be_set_without_notifying():
    hair, _, seen, _ = _crosshair()
    hair.set_position(1910.0, 5.0)
    assert hair.position == (1910.0, 5.0)
    assert seen == []  # a programmatic move is not a user action
    hair.disconnect()


def test_a_crosshair_can_be_dropped_after_its_figure_was_cleared():
    """The sequence that crashed Preview: render -> figure.clear() -> disconnect.

    Every render rebuilds the figure, which detaches the guide lines; Matplotlib
    then has no remove method for them and ``Artist.remove`` raises
    ``NotImplementedError``. Dropping an already-detached crosshair is normal
    housekeeping, not an error, so it must not propagate.
    """
    hair, ax, _, _ = _crosshair()
    ax.figure.clear()
    hair.disconnect()  # must not raise
    hair.disconnect()  # and must stay idempotent


def test_a_cut_snaps_to_a_measured_value():
    """An interpolated coordinate is not a cut; the nearest measured one is."""
    from pyrate_ta.gui.crosshair import nearest

    assert nearest(PROBE, 1951.3) in PROBE
    assert nearest(PROBE, 1951.3) == pytest.approx(PROBE[np.argmin(np.abs(PROBE - 1951.3))])
    assert nearest([], 5.0) == 5.0


# --------------------------------------------------------------------------- #
#                    Fixed mask with free t0 / IRF                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "extra", [{"fit_t0": True}, {"fit_irf": True}, {"fit_t0": True, "fit_irf": True}]
)
def test_a_lifetime_only_mask_is_completed(extra):
    """The reported failure: "fixed mask has N entries, expected N+1".

    The GUI reads its mask from the lifetime table, while t0 and the IRF width
    are appended to the vector *only when they are free* -- so the entries that
    follow a lifetime-length mask are free by construction, and padding them is
    the correct completion rather than a guess.
    """
    D, _ = _data()
    fit = pr.fit_global(
        D,
        T,
        taus=list(TAUS),
        fixed=[False] * len(TAUS),
        model_type="Sequential",
        irf_fwhm=0.3,
        **extra,
    )
    assert fit.converged
    assert np.allclose(fit.taus, TAUS, rtol=0.05)


def test_a_wrong_length_mask_is_still_refused():
    """Padding must not turn a genuine mistake into a silent reinterpretation."""
    D, _ = _data()
    with pytest.raises(ValueError, match="expected"):
        pr.fit_global(D, T, taus=list(TAUS), fixed=[False] * 5, model_type="Sequential")


# --------------------------------------------------------------------------- #
#                          Live parameter monitor                             #
# --------------------------------------------------------------------------- #
def test_parameter_bars_show_every_parameter():
    """One labelled bar per parameter, with the value written on it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ax = pr.plot_parameter_bars(
        [8.03, 297.5, 0.001, 0.31],
        ["tau1", "tau2", "t0", "IRF"],
        fixed=[False, False, False, True],
        title="evaluation 12",
    )
    assert len(ax.patches) == 4
    assert [t.get_text() for t in ax.get_xticklabels()] == ["tau1", "tau2", "t0", "IRF"]
    # Symlog by default: a t0 of ~0 and a 300 ps lifetime on one axis.
    assert ax.get_yscale() == "symlog"
    # The fixed bar is greyed, so a parameter that never moves is not mistaken
    # for one that has converged.
    assert ax.patches[3].get_facecolor() != ax.patches[0].get_facecolor()
    plt.close("all")


def test_a_non_decaying_component_is_labelled_not_drawn():
    """``inf`` has no bar to draw; it is written instead of silently missing."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ax = pr.plot_parameter_bars([1.0, float("inf")], ["tau1", "tau2"])
    assert "inf" in [t.get_text() for t in ax.texts]
    assert np.isfinite([p.get_height() for p in ax.patches]).all()
    plt.close("all")


def test_the_trajectory_can_be_replayed():
    """The companion view: how the fit got where it did."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = np.array([[5.0, 200.0], [7.0, 250.0], [8.0, 297.0]])
    ax = pr.plot_parameter_history(history, ["tau1", "tau2"])
    assert len(ax.lines) == 2
    assert ax.get_xlabel() == "Evaluation"
    plt.close("all")


def test_the_monitor_is_off_unless_asked_for():
    """``fit_monitor_every = 0`` means never: no window, no cost."""
    original = pr.get_settings()
    try:
        pr.update_settings(fit_monitor_every=0)
        assert pr.get_settings().fit_monitor_every == 0
        pr.update_settings(fit_monitor_every=5)
        assert pr.get_settings().fit_monitor_every == 5
    finally:
        pr.set_settings(original)


def test_time_limits_setting_applied_to_fits():
    """Fits default to time_limits setting [0.2, max] and can be overridden."""
    D, _ = _data()
    original = pr.get_settings()
    try:
        # Default setting: (0.2, None)
        pr.update_settings(time_limits=(0.2, None))
        fit_default = pr.fit_global(_StubDataset(D), taus=[3.0, 100.0], irf_fwhm=0.3)
        assert fit_default.delay_range[0] >= 0.2
        assert fit_default.delay_range[1] == float(T.max())
        assert fit_default.t.min() >= 0.2

        # Plain arrays default
        fit_arr = pr.fit_global(D, T, taus=[3.0, 100.0], irf_fwhm=0.3)
        assert fit_arr.delay_range[0] >= 0.2
        assert fit_arr.delay_range[1] == float(T.max())
        assert fit_arr.t.min() >= 0.2

        # Explicit delay_range overrides setting
        fit_override = pr.fit_global(D, T, taus=[3.0, 100.0], irf_fwhm=0.3, delay_range=(1.0, 50.0))
        assert fit_override.delay_range[0] >= 1.0
        assert fit_override.delay_range[1] <= 50.0
        assert fit_override.t.min() >= 1.0
        assert fit_override.t.max() <= 50.0

        # Setting time_limits to (None, None) keeps all delays
        pr.update_settings(time_limits=(None, None))
        fit_all = pr.fit_global(_StubDataset(D), taus=[3.0, 100.0], irf_fwhm=0.3)
        assert fit_all.delay_range == (float(T.min()), float(T.max()))
    finally:
        pr.set_settings(original)
