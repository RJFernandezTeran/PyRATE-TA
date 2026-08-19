"""Tests for saving and reopening a fit session.

A saved fit must carry its provenance, not just its numbers: these check that
the reproducibility payload survives the round trip, and that a file which
cannot be read correctly is refused rather than half-interpreted.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import pyrate_ta as pr

T = np.concatenate([np.linspace(-1.0, 0.5, 10), np.geomspace(0.6, 2000.0, 120)])
PROBE = np.linspace(1900.0, 2000.0, 30)


def _fit(**kwargs):
    model = pr.make_model("Sequential", 2, irf_fwhm=0.2)
    C = model.concentrations(T, taus=[6.0, 250.0])
    S = np.stack([np.exp(-((PROBE - c) ** 2) / (2 * 18.0**2)) for c in (1940, 1970)], axis=1)
    return pr.fit_global(C @ S.T, T, taus=[2.0, 100.0], irf_fwhm=0.2, **kwargs)


def test_a_saved_fit_round_trips(tmp_path):
    fit = _fit()
    path = pr.save_fit(tmp_path / "demo", fit)
    assert path.suffix == ".prfit"

    back = pr.load_fit(path)
    assert np.allclose(back.taus, fit.taus)
    assert np.allclose(back.S, fit.S)
    assert np.allclose(back.t, fit.t)
    assert back.meta["model_type"] == "Sequential"
    assert back.converged is True


def test_the_reproducibility_payload_survives(tmp_path):
    """The record of what the fit saw is the point of saving it."""
    fit = _fit(t_min=1.0)
    back = pr.load_fit(pr.save_fit(tmp_path / "demo", fit))

    meta = back.meta
    assert meta["t_min"] == 1.0
    assert meta["delay_range"][0] >= 1.0
    assert meta["statistic"]["kind"] in ("ssr", "chi2_red")
    assert meta["report"]["method"] == "trf"
    assert meta["report"]["converged"] is True
    assert meta["param_names"] == ["tau1", "tau2"]
    assert meta["pyrate_version"] == pr.__version__


def test_a_target_fit_reopens_as_editable_text(tmp_path):
    """A scheme comes back in its own notation, not as an opaque matrix."""
    text = "A -> B : k1\nB -> : k2"
    scheme = pr.scheme_from_text(text)
    model = pr.make_model("Target", 0, scheme=scheme, irf_fwhm=0.2)
    C = model.concentrations(T, taus=[6.0, 250.0])
    S = np.stack([np.exp(-((PROBE - c) ** 2) / (2 * 18.0**2)) for c in (1940, 1970)], axis=1)
    fit = pr.fit_target(C @ S.T, T, scheme=scheme, taus=[2.0, 100.0], irf_fwhm=0.2)

    back = pr.load_fit(pr.save_fit(tmp_path / "target", fit))
    assert back.meta["scheme_text"] == text
    assert np.allclose(back.K, fit.K)
    assert np.allclose(back.eigenvalues, fit.eigenvalues)
    # ... and the notation really does rebuild the same scheme.
    assert np.allclose(
        pr.scheme_from_text(back.meta["scheme_text"]).rate_matrix([0.5, 0.1]),
        scheme.rate_matrix([0.5, 0.1]),
    )


def test_weighted_fits_record_the_noise_source(tmp_path):
    sigma_true = 0.02
    model = pr.make_model("Sequential", 2, irf_fwhm=0.2)
    C = model.concentrations(T, taus=[6.0, 250.0])
    S = np.stack([np.exp(-((PROBE - c) ** 2) / (2 * 18.0**2)) for c in (1940, 1970)], axis=1)
    D = C @ S.T + sigma_true * np.random.default_rng(0).standard_normal((T.size, PROBE.size))
    fit = pr.fit_global(
        D,
        T,
        taus=[2.0, 100.0],
        irf_fwhm=0.2,
        sigma=np.full(D.shape, sigma_true),
        use_weights=True,
    )
    back = pr.load_fit(pr.save_fit(tmp_path / "weighted", fit))
    assert back.meta["statistic"]["kind"] == "chi2_red"
    assert back.meta["statistic"]["weighted"] is True
    assert back.meta["weights"]["n_masked"] == 0


def test_summary_reads_like_a_live_result(tmp_path):
    back = pr.load_fit(pr.save_fit(tmp_path / "demo", _fit()))
    text = back.summary()
    assert "Sequential fit" in text
    assert "tau1" in text and "+/-" in text


def test_a_newer_format_is_refused(tmp_path):
    """Reading a file from a future PyRATE-TA would silently mis-interpret it."""
    path = pr.save_fit(tmp_path / "demo", _fit())
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files if k != "meta"}
        meta = json.loads(str(data["meta"]))
    meta["format_version"] = pr.io.FORMAT_VERSION + 1
    with path.open("wb") as handle:
        np.savez_compressed(handle, meta=json.dumps(meta), **arrays)

    with pytest.raises(ValueError, match="newer PyRATE-TA"):
        pr.load_fit(path)


def test_a_file_that_is_not_a_session_is_refused(tmp_path):
    path = tmp_path / "not_a_session.npz"
    np.savez(path, something=np.arange(3))
    with pytest.raises(ValueError, match="not a PyRATE-TA fit session"):
        pr.load_fit(path)


def test_default_path_sits_beside_the_dataset():
    from pyrate_ta.io.session import default_session_path

    class _Fit:
        source = "/data/2026-07-29/scan.pdat"

    assert default_session_path(_Fit()).name == "scan_fit.prfit"
