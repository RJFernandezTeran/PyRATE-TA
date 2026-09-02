"""Tests for Phase 7: Lifetime Density Analysis (LDA)."""

from __future__ import annotations

import numpy as np

from pyrate_ta.fit.lda import build_penalty_matrix, solve_lda
from pyrate_ta.results.lda_result import LDAResult


def test_build_penalty_matrix():
    M = 10
    L_ridge = build_penalty_matrix(M, "ridge")
    assert L_ridge.shape == (10, 10)
    assert np.allclose(L_ridge, np.eye(10))

    L_d1 = build_penalty_matrix(M, "d1")
    assert L_d1.shape == (9, 10)

    L_d2 = build_penalty_matrix(M, "d2")
    assert L_d2.shape == (8, 10)


def test_solve_lda_basic():
    # Synthetic two-exponential dataset
    t = np.linspace(0, 100, 60)
    _taus_true = [5.0, 30.0]
    C_true = np.column_stack([np.exp(-t / 5.0), np.exp(-t / 30.0)])
    S_true = np.array([[1.5, -0.8], [0.5, 1.2]])  # 2 pixels
    D = C_true @ S_true.T

    res = solve_lda(D, t=t, n_taus=30, penalty="d2", alpha_method="lcurve")
    assert isinstance(res, LDAResult)
    assert res.S_map.shape == (2, 30)
    assert res.alpha_opt > 0
    assert res.l_curve_points is not None
    assert len(res.l_curve_points) == 60


def test_solve_lda_manual_alpha():
    t = np.linspace(0, 50, 40)
    D = np.exp(-t[:, None] / 10.0) @ np.array([[1.0, 2.0]])

    res = solve_lda(D, t=t, n_taus=20, alpha=0.05, penalty="ridge")
    assert res.alpha_opt == 0.05
    assert res.alpha_method == "manual"


def test_solve_lda_restricted_limits():
    import pymorgan as pm
    t = np.linspace(0, 50, 40)
    probe = np.array([1000.0, 1500.0, 2000.0])
    D = (np.exp(-t[:, None] / 10.0) @ np.array([[1.0, 2.0, 3.0]]))[:, :, None]
    ds = pm.Dataset1D(D, t, probe, {})

    res = solve_lda(ds, n_taus=15, alpha=0.05, delay_range=(5.0, 30.0), probe_range=(1000.0, 1600.0))
    assert res.delay_range is not None
    assert res.probe_range == (1000.0, 1500.0)
    assert res.S_map.shape[0] == 2  # restricted to 1000 and 1500
    assert len(res.t) < 40  # restricted to 5.0 <= t <= 30.0
    assert res.t[0] >= 5.0 and res.t[-1] <= 30.0


def test_solve_lda_progress_callback():
    t = np.linspace(0, 50, 30)
    D = np.exp(-t[:, None] / 10.0) @ np.array([[1.0, 2.0]])

    calls = []

    def callback(step, total, msg):
        calls.append((step, total, msg))

    res = solve_lda(D, t=t, n_taus=10, alphas=[0.01, 0.1, 1.0], n_bootstraps=2, callback=callback)
    assert len(calls) == 5  # 3 alphas + 2 bootstraps
    assert calls[0][0] == 1
    assert calls[0][1] == 5
    assert "Alpha scan" in calls[0][2]
    assert "Bootstrap" in calls[-1][2]


def test_solve_lda_coherent_artifact():
    t = np.linspace(-2.0, 50.0, 50)
    D = np.exp(-np.maximum(t, 0)[:, None] / 10.0) @ np.array([[1.0, 2.0]])

    res = solve_lda(D, t=t, n_taus=20, coherent_artifact=True, irf_fwhm=0.3)
    assert isinstance(res, LDAResult)
    assert res.S_map.shape == (2, 20)


def test_lda_to_pdat(tmp_path):
    t = np.linspace(0, 50, 40)
    D = np.exp(-t[:, None] / 10.0) @ np.array([[1.0, 2.0]])
    res = solve_lda(D, t=t, n_taus=15, alpha=0.05)
    out_file = tmp_path / "test_lda_map.pdat"
    saved = res.to_pdat(out_file)
    assert saved.exists()

    import pymorgan as pm

    ds = pm.load_1D(saved, data_type="PDAT")
    assert ds.Z.shape[0] == 15
    assert ds.Z.shape[1] == 2


def test_solve_lda_literature_features():
    t = np.linspace(-2.0, 50.0, 45)
    D = np.exp(-np.maximum(t, 0)[:, None] / 10.0) @ np.array([[1.0, 2.0]])

    # 1. SVD & Morozov
    res_mor = solve_lda(D, t=t, n_taus=20, svd_components=2, alpha_method="morozov")
    assert res_mor.alpha_method == "morozov"
    assert res_mor.svd_components == 2

    # 2. Non-Negative (NNLS)
    res_nn = solve_lda(D, t=t, n_taus=20, non_negative=True)
    assert res_nn.non_negative is True
    assert np.all(res_nn.S_map >= -1e-12)

    # 3. Bootstrap & Peak Centroids & Citations
    res_full = solve_lda(D, t=t, n_taus=20, n_bootstraps=5, find_peaks=True, non_negative=True, svd_components=2)
    assert res_full.bootstrap_std is not None
    assert len(res_full.bootstrap_std) == 20
    cites = res_full.citations()
    assert len(cites) >= 4
    assert "Slavov" in res_full.format_citations()
    assert "10.1021/ac504348h" in res_full.format_citations()


def test_lda_plots_follow_pymorgan_rules():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pyrate_ta.plot.lda import plot_lda_map, plot_l_curve

    t = np.linspace(0, 50, 30)
    D = np.exp(-t[:, None] / 10.0) @ np.array([[1.0, 2.0]])
    res = solve_lda(D, t=t, n_taus=15, alpha=0.05)
    res.units = {
        "unitsL_lbl": "Wavelength",
        "unitsL_ltx": "nm",
        "unitsT_ltx": "ps",
        "unitsZ_lbl": r"$\Delta A$",
        "unitsZ_ltx": "mOD",
    }

    # 1. Test 2D Map + Integrated Dynamics (abs metric)
    ax_map, ax_int = plot_lda_map(res, discrete_taus=[10.0], metric="abs")
    assert not any(gl.get_visible() for gl in ax_map.xaxis.get_gridlines())
    assert not any(gl.get_visible() for gl in ax_int.xaxis.get_gridlines())
    assert "nm" in ax_map.get_xlabel()
    assert "ps" in ax_map.get_ylabel()
    assert "Integrated Dynamics" in ax_int.get_title()
    # Check zero line on integrated dynamics
    assert any(line.get_xdata()[0] == 0 for line in ax_int.lines)

    # Test Dynamical content metric D(tau) = sqrt(int S^2 dlambda)
    ax_map2, ax_int2 = plot_lda_map(res, discrete_taus=[10.0], metric="dynamical_content")
    assert "Dynamical content" in ax_int2.get_title()
    assert "Dynamical content" in ax_int2.get_xlabel()

    # Test Centroid Annotations (%.3g format and timescale unit conversion)
    res.peaks = [
        {"tau": 0.78, "amplitude": 1.0, "index": 2},
        {"tau": 12.3456, "amplitude": 1.5, "index": 7},
        {"tau": 1000.0, "amplitude": 0.8, "index": 12},
    ]
    ax_map3, ax_int3 = plot_lda_map(res, annotate_centroids=True)
    assert len(ax_int3.texts) >= 3
    assert "780 fs" in ax_int3.texts[0].get_text()
    assert "12.3 ps" in ax_int3.texts[1].get_text()
    assert "1 ns" in ax_int3.texts[2].get_text()

    # When annotate_centroids is False, no labels should be shown
    ax_map_none, ax_int_none = plot_lda_map(res, annotate_centroids=False)
    assert len(ax_int_none.texts) == 0

    # Test asinh color scaling option
    import matplotlib.colors as mcolors
    ax_map4, _ = plot_lda_map(res, asinh=True, asinh_pct=10.0)
    # The pcolormesh mesh norm should be AsinhNorm
    assert isinstance(ax_map4.collections[0].norm, mcolors.AsinhNorm)
    plt.close("all")

    # 2. Test L-Curve
    ax_lc = plot_l_curve(res)
    assert not any(gl.get_visible() for gl in ax_lc.xaxis.get_gridlines())
    assert ax_lc.get_legend() is not None
    plt.close("all")


def test_lda_result_metrics_and_peaks():
    t = np.linspace(0, 50, 30)
    D = np.exp(-t[:, None] / 10.0) @ np.array([[1.0, 2.0]])
    res = solve_lda(D, t=t, n_taus=15, alpha=0.05)

    a_abs = res.integrated_dynamics("abs")
    assert len(a_abs) == 15
    assert np.all(a_abs >= 0)

    d_dyn = res.integrated_dynamics("dynamical_content")
    assert len(d_dyn) == 15
    assert np.all(d_dyn >= 0)
    assert np.allclose(d_dyn, np.sqrt(np.sum(res.S_map**2, axis=0)))

    pks_abs = res.find_peaks("abs")
    pks_dyn = res.find_peaks("dynamical_content")
    assert isinstance(pks_abs, list)
    assert isinstance(pks_dyn, list)
