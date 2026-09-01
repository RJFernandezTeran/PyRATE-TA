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
