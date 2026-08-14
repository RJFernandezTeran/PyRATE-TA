import matplotlib.pyplot as plt
import numpy as np
import pymorgan as pm
import pytest

import pyrate_ta as pr
from pyrate_ta.plot.traces import plot_kinetics_with_residuals
from pyrate_ta.results.fits import GlobalFit


@pytest.fixture
def sample_dataset_and_fit():
    t = np.linspace(-1, 10, 30)
    probe = np.array([1000.0, 1500.0, 2000.0])
    T, P = np.meshgrid(t, probe, indexing="ij")
    Z_data = np.exp(-np.maximum(0, T) / 2.0) * np.sin(P / 100)
    ds = pm.Dataset1D(Z_data[:, :, None], t, probe, {"delay": "ps", "probe": "cm-1", "signal": "mOD"})

    # Mock GlobalFit result
    C = np.exp(-np.maximum(0, t)[:, None] / np.array([[2.0, 5.0]]))
    S = np.array([[1.0, 0.5, 0.2], [0.1, 0.8, 0.4]])
    R = 0.01 * np.random.randn(len(t), len(probe))
    fit = GlobalFit(
        taus=np.array([2.0, 5.0]),
        tau_err=np.array([0.1, 0.2]),
        is_fixed=np.array([False, False]),
        S=S.T,
        R=R,
        model_type=str(pr.ModelType.SEQUENTIAL),
        t=t,
        probe=probe,
        C=C,
    )
    return ds, fit



def test_plot_species_spectra_quoting_settings(sample_dataset_and_fit):
    ds, fit = sample_dataset_and_fit
    ax = ds.plot_species_spectra(
        *fit.as_species_args(),
        printErrors=True,
        pair_round=True,
    )
    legend = ax.get_legend()
    assert legend is not None
    labels = [t.get_text() for t in legend.get_texts()]
    assert len(labels) == 2
    assert "±" in labels[0] or "+/-" in labels[0] or "(" in labels[0]

    # With uncertainties disabled (printErrors=False), ± should be absent
    ax2 = ds.plot_species_spectra(
        *fit.as_species_args(),
        printErrors=False,
    )
    legend2 = ax2.get_legend()
    labels2 = [t.get_text() for t in legend2.get_texts()]
    assert "±" not in labels2[0]

    plt.close("all")



def test_plot_data_fit_residuals_scales(sample_dataset_and_fit):
    ds, fit = sample_dataset_and_fit
    ax_data, ax_fit, ax_res = pr.plot_data_fit_residuals(fit, ds, zscale_factor=10.0)
    assert ax_data is not None and ax_fit is not None and ax_res is not None

    # Verify titles
    assert "Data" in ax_data.get_title()
    assert "Fit" in ax_fit.get_title()
    assert "Residuals" in ax_res.get_title()

    plt.close("all")


def test_plot_kinetics_without_fit(sample_dataset_and_fit):


    ds, _ = sample_dataset_and_fit
    ax = plot_kinetics_with_residuals(ds, cuts=[1500.0])
    assert isinstance(ax, plt.Axes)
    plt.close("all")


def test_plot_kinetics_with_fit_and_residuals(sample_dataset_and_fit):
    ds, fit = sample_dataset_and_fit
    res = plot_kinetics_with_residuals(ds, cuts=[1500.0], fit=fit, incl_residuals=True)
    assert isinstance(res, tuple)
    ax_trace, ax_res = res

    # Verify linked X-axis
    assert ax_trace._sharex is ax_res or ax_res._sharex is ax_trace

    # Verify trace plot has both data lines/points and fit lines
    assert len(ax_trace.lines) > 0

    # Verify residual plot has zero line
    assert len(ax_res.lines) >= 2  # residual line + zero reference line

    # Verify residuals axis has no legend
    assert ax_res.get_legend() is None

    # Verify trace legend is set to not be in layout (for tight layout compatibility)
    legend = ax_trace.get_legend()
    assert legend is not None
    assert legend.get_in_layout() is False

    plt.close("all")


def test_plot_kinetics_aspect_ratios(sample_dataset_and_fit):
    ds, fit = sample_dataset_and_fit

    # Test 4:1 height ratio
    ax_tr4, ax_res4 = plot_kinetics_with_residuals(
        ds, cuts=[1500.0], fit=fit, incl_residuals=True, height_ratio=4.0
    )
    pos_tr4 = ax_tr4.get_position()
    pos_res4 = ax_res4.get_position()
    ratio4 = pos_tr4.height / pos_res4.height
    plt.close("all")

    # Test 3:1 height ratio
    ax_tr3, ax_res3 = plot_kinetics_with_residuals(
        ds, cuts=[1500.0], fit=fit, incl_residuals=True, height_ratio=3.0
    )
    pos_tr3 = ax_tr3.get_position()
    pos_res3 = ax_res3.get_position()
    ratio3 = pos_tr3.height / pos_res3.height
    plt.close("all")

    assert ratio4 > ratio3


def test_plot_kinetics_weighted_residuals(sample_dataset_and_fit):
    from pyrate_ta.fit.cost import Weights

    ds, fit = sample_dataset_and_fit

    # Attach mock weights with 1/sigma
    sigma = np.full_like(fit.R, 0.05)
    w = 1.0 / sigma
    fit.weights = Weights(w=w, mask=np.ones_like(w, dtype=bool), n_points=w.size)

    ax_trace, ax_res = plot_kinetics_with_residuals(ds, cuts=[1500.0], fit=fit, incl_residuals=True)

    # Verify Y label is r/\sigma
    assert ax_res.get_ylabel() == r"r/$\sigma$"

    # Verify zero reference line has PyMORGAN style (colour 0.75, lw 0.75)
    zero_lines = [l for l in ax_res.lines if l.get_ydata()[0] == 0 and l.get_ydata()[-1] == 0]
    assert len(zero_lines) >= 1
    zl = zero_lines[0]
    assert zl.get_color() == "0.75"
    assert zl.get_linewidth() == 0.75

    plt.close("all")


def test_plot_spectra_with_fit_and_residuals(sample_dataset_and_fit):
    from pyrate_ta.plot.traces import plot_spectra_with_residuals
    ds, fit = sample_dataset_and_fit
    res = plot_spectra_with_residuals(ds, cuts=[2.0], fit=fit, incl_residuals=True)
    assert isinstance(res, tuple)
    ax_trace, ax_res = res

    # Verify linked X-axis
    assert ax_trace._sharex is ax_res or ax_res._sharex is ax_trace

    # Verify trace plot has both data lines/points and fit lines
    assert len(ax_trace.lines) > 0

    # Verify residual plot has zero line
    assert len(ax_res.lines) >= 2  # residual line + zero reference line

    # Verify residuals axis has no legend
    assert ax_res.get_legend() is None

    # Verify trace legend is set to not be in layout (for tight layout compatibility)
    legend = ax_trace.get_legend()
    assert legend is not None
    assert legend.get_in_layout() is False

    plt.close("all")


