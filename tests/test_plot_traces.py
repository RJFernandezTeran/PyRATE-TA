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

    # Verify trace legend is created, draggable, readable, and placed outside right by default
    legend = ax_trace.get_legend()
    assert legend is not None
    assert len(legend.get_texts()) == 1
    assert "1500" in legend.get_texts()[0].get_text()
    assert legend.get_frame_on() is True
    assert legend.get_draggable() is True
    fig = ax_trace.figure
    fig.draw_without_rendering()
    bbox = legend.get_window_extent().transformed(fig.transFigure.inverted())
    assert bbox.x1 <= 1.0
    assert bbox.x0 >= ax_trace.get_position().x1

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

    # Verify trace legend is created, draggable, readable, and placed outside right by default
    legend = ax_trace.get_legend()
    assert legend is not None
    assert len(legend.get_texts()) == 1
    assert legend.get_frame_on() is True
    assert legend.get_draggable() is True
    fig = ax_trace.figure
    fig.draw_without_rendering()
    bbox = legend.get_window_extent().transformed(fig.transFigure.inverted())
    assert bbox.x1 <= 1.0
    assert bbox.x0 >= ax_trace.get_position().x1

    plt.close("all")


def test_plot_traces_legend_locations(sample_dataset_and_fit):
    from pyrate_ta.plot.traces import plot_spectra_with_residuals
    ds, fit = sample_dataset_and_fit

    # 1. Custom inside location
    ax_tr_in, _ = plot_kinetics_with_residuals(ds, cuts=[1500.0], fit=fit, legend_loc="upper left")
    leg_in = ax_tr_in.get_legend()
    assert leg_in is not None
    assert leg_in.get_draggable() is True
    fig_in = ax_tr_in.figure
    fig_in.draw_without_rendering()
    bbox_in = leg_in.get_window_extent().transformed(fig_in.transFigure.inverted())
    pos_in = ax_tr_in.get_position()
    assert bbox_in.x0 >= pos_in.x0 and bbox_in.x1 <= pos_in.x1

    # 2. Outside location
    ax_tr_out, _ = plot_spectra_with_residuals(ds, cuts=[2.0], fit=fit, legend_loc="outside right")
    leg_out = ax_tr_out.get_legend()
    assert leg_out is not None
    assert leg_out.get_draggable() is True
    fig_out = ax_tr_out.figure
    fig_out.draw_without_rendering()
    bbox_out = leg_out.get_window_extent().transformed(fig_out.transFigure.inverted())
    assert bbox_out.x1 <= 1.0
    assert bbox_out.x0 >= ax_tr_out.get_position().x1

    plt.close("all")

def test_plot_species_spectra_and_concentrations_target_model():
    """Target models should label species spectra and concentrations by species name only."""
    from pyrate_ta.plot.concentrations import plot_concentrations
    from pyrate_ta.results.fits import TargetFit

    t = np.linspace(0, 10, 20)
    probe = np.linspace(1000, 1500, 10)
    C = np.ones((len(t), 3))
    S = np.ones((len(probe), 3))
    R = np.zeros((len(t), len(probe)))

    fit = TargetFit(
        taus=np.array([1.0, 5.0, 20.0, 100.0]),  # 4 rate parameters
        tau_err=np.array([0.1, 0.5, 2.0, 10.0]),
        is_fixed=np.zeros(4, dtype=bool),
        S=S,
        R=R,
        C=C,
        model_type="Target",
        t=t,
        probe=probe,
        scheme_text="Q -> I : k1\nI -> D : k2\nD -> : k3\nD -> : k4",
    )

    # 1. Concentrations plot
    ax_conc = plot_concentrations(fit, labels=["Q", "I", "D"])
    legend_conc = ax_conc.get_legend()
    conc_labels = [t.get_text() for t in legend_conc.get_texts()]
    assert conc_labels == ["Q", "I", "D"]

    # 2. Species spectra plot
    ds = pm.Dataset1D(np.zeros((len(t), len(probe), 1)), t, probe, {})
    ax_spec = ds.plot_species_spectra(
        *fit.as_species_args(),
        species_labels=fit.species_labels(),
    )
    legend_spec = ax_spec.get_legend()
    spec_labels = [t.get_text() for t in legend_spec.get_texts()]
    assert spec_labels == ["Q", "I", "D"]

    plt.close("all")


def test_plot_spectra_with_residuals_steady_state_overlay(sample_dataset_and_fit):
    from pyrate_ta.plot.traces import plot_spectra_with_residuals

    ds, fit = sample_dataset_and_fit
    probe = ds.probe
    abs_dict = {"X": probe, "Y": np.exp(-((probe - 1500) / 200) ** 2)}
    em_dict = {"X": probe, "Y": np.exp(-((probe - 1700) / 150) ** 2)}

    # Without residuals
    ax = plot_spectra_with_residuals(ds, cuts=[1.0, 5.0], Abs=abs_dict, Em=em_dict)
    fig = ax.figure
    # Twin axis exists for steady state
    assert len(fig.axes) == 2
    twin_ax = fig.axes[1]
    assert twin_ax.get_legend() is not None
    twin_labels = [t.get_text() for t in twin_ax.get_legend().get_texts()]
    assert "Abs." in twin_labels
    assert "Em." in twin_labels

    # With residuals
    ax_trace, ax_res = plot_spectra_with_residuals(
        ds, cuts=[1.0, 5.0], fit=fit, incl_residuals=True, Abs=abs_dict, Em=em_dict
    )
    fig_res = ax_trace.figure
    assert len(fig_res.axes) == 3  # ax_trace, twin_ax, ax_res
    assert ax_res.get_legend() is None

    plt.close("all")


def test_plot_species_spectra_steady_state_overlay(sample_dataset_and_fit):
    ds, fit = sample_dataset_and_fit
    probe = ds.probe
    abs_dict = {"X": probe, "Y": np.exp(-((probe - 1500) / 200) ** 2)}
    em_dict = {"X": probe, "Y": np.exp(-((probe - 1700) / 150) ** 2)}

    ax = ds.plot_species_spectra(*fit.as_species_args(), Abs=abs_dict, Em=em_dict)
    fig = ax.figure
    assert len(fig.axes) == 2
    twin_ax = fig.axes[1]
    assert twin_ax.get_legend() is not None
    twin_labels = [t.get_text() for t in twin_ax.get_legend().get_texts()]
    assert "Abs." in twin_labels
    assert "Em." in twin_labels

    plt.close("all")




