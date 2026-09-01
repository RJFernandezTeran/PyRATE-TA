"""Static checks on ``main_window.ui``.

These run without a Qt runtime: the ``.ui`` file is parsed as XML and checked
for the widget names the Python layer binds to. The GUI wiring itself is tested
separately, in the Qt-dependent suite.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

UI_FILE = Path(__file__).resolve().parents[1] / "src" / "pyrate_ta" / "gui" / "main_window.ui"


def _require_qt():
    """Skip when there is no Qt *runtime*, not merely no PyQt6 package.

    ``importorskip`` only skips on ``ModuleNotFoundError``; a machine with
    PyQt6 installed but no display libraries raises a plain ``ImportError``
    (``libEGL.so.1``), which would fail the test instead of skipping it.
    """
    try:
        import PyQt6.QtWidgets  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"Qt runtime not available: {exc}")


@pytest.fixture(scope="module")
def ui_names() -> set[str]:
    root = ET.parse(UI_FILE).getroot()
    names = {w.get("name") for w in root.iter("widget")}
    names |= {a.get("name") for a in root.iter("action")}
    return names


def test_ui_file_exists_and_parses():
    assert UI_FILE.is_file()
    root = ET.parse(UI_FILE).getroot()
    assert root.tag == "ui"


@pytest.mark.parametrize(
    "name",
    [
        "LoadPDATButton",
        "ClearDataButton",
        "DataLoadedLamp",
        "NoiseLoadedLamp",
        "ModelSelectionGroup",
        "ParallelButton",
        "SequentialButton",
        "TargetButton",
        "RateFitTable",
        "IRFtable",
        "GaussianIRFCheckBox",
        "AddComponentButton",
        "RemoveComponentButton",
        "FitMethod",
        "PlotArea",
        "PC_box",
        "PP_PlotsandcutsPanel",
        "actionOpen",
        "actionAbout",
    ],
)
def test_required_widget_present(ui_names, name):
    assert name in ui_names


def test_plot_controls_widgets_present(ui_names):
    """Every ``PC_*`` widget PyMORGAN's PlotControlsPanel binds to must exist.

    The panel is reused verbatim rather than reimplemented, so the contract is
    the set of object names in its ``_WIDGETS`` mapping.
    """
    _require_qt()
    from pymorgan.gui import plot_controls as pc

    required = set(pc._WIDGETS.values())
    assert required <= ui_names


def test_there_is_one_canvas_not_three():
    """The three panels share a figure, so there is a single toolbar."""
    root = ET.parse(UI_FILE).getroot()
    canvases = [w.get("name") for w in root.iter("widget") if w.get("class") == "WidgetPlot"]
    assert canvases == ["PlotArea"]


def test_the_panel_sits_in_the_free_cell_of_the_plot_grid():
    """The canvas spans the 3x3 grid; the panel occupies the bottom-right cell."""
    root = ET.parse(UI_FILE).getroot()
    grid = next(la for la in root.iter("layout") if la.get("name") == "plotsGrid")
    cells = {
        item.find("widget").get("name"): (
            item.get("row"),
            item.get("column"),
            item.get("rowspan"),
            item.get("colspan"),
        )
        for item in grid.findall("item")
    }
    assert cells["PlotArea"] == ("0", "0", "3", "3")
    assert cells["PP_PlotsandcutsPanel"][:2] == ("2", "2")


def test_promoted_widgets_come_from_pymorgan():
    """The custom widgets are PyMORGAN's; PyRATE-TA must not fork them."""
    text = UI_FILE.read_text(encoding="utf-8")
    headers = set(re.findall(r"<header>([^<]+)</header>", text))
    assert headers
    assert all(h.startswith("pymorgan.gui.") for h in headers), headers


def test_every_button_palette_exists():
    """A renamed palette would silently make buttons fall back to grey.

    That happened once: the model-family palettes were renamed and the mapping
    was left pointing at the old key, so all three buttons looked identical.
    """
    import re

    source = (UI_FILE.parent / "mw_common.py").read_text(encoding="utf-8")
    palettes = set(re.findall(r'^    "([a-z_]+)": \(', source, flags=re.M))
    mapping = source[source.index("_BUTTON_PALETTES") : source.index("_BUTTON_FONT_SIZES")]
    used = set(re.findall(r'": "([a-z_]+)",', mapping))
    assert used <= palettes, f"palette(s) missing: {sorted(used - palettes)}"


def test_the_model_families_have_distinct_colours():
    import re

    source = (UI_FILE.parent / "mw_common.py").read_text(encoding="utf-8")
    mapping = source[source.index("_BUTTON_PALETTES") : source.index("_BUTTON_FONT_SIZES")]
    chosen = dict(re.findall(r'"([A-Za-z_]+)": "([a-z_]+)",', mapping))
    families = [chosen[n] for n in ("ParallelButton", "SequentialButton", "TargetButton")]
    assert len(set(families)) == 3


def test_the_irf_table_uses_a_delta_row_and_lb_before_ub():
    """Bounds read as they do on a number line, and the width row is delta."""
    root = ET.parse(UI_FILE).getroot()
    tables = {w.get("name"): w for w in root.iter("widget") if w.get("class") == "QTableWidget"}

    for name in ("RateFitTable", "IRFtable"):
        headers = [c.find("property/string").text for c in tables[name].findall("column")]
        assert headers[1:] == ["LB", "UB", "Fix?"], (name, headers)

    rows = [r.find("property/string").text for r in tables["IRFtable"].findall("row")]
    assert rows == ["t0", "Δ"]


def test_the_fit_controls_sit_in_the_left_column():
    """They were a strip above the plots; that height belongs to the figure."""
    root = ET.parse(UI_FILE).getroot()
    left = next(la for la in root.iter("layout") if la.get("name") == "leftColumn")
    names = {w.get("name") for w in left.iter("widget")}
    assert {"FitControlGroup", "FITDATAButton", "Chi2Value", "CopyTaus_button"} <= names

    right = next(la for la in root.iter("layout") if la.get("name") == "rightSide")
    assert "FitControlGroup" not in {w.get("name") for w in right.iter("widget")}


def test_there_is_no_format_selector(ui_names):
    """A fit needs a processed transient, so the format is not a choice.

    PDAT, with its .pdatn sibling picked up automatically -- a selector could
    only be set wrong. The two lamps say what was actually loaded instead.
    """
    assert "DataType_cbx" not in ui_names
    assert {"DataLoadedLamp", "NoiseLoadedLamp"} <= ui_names


def test_the_unimplemented_fit_options_button_is_gone(ui_names):
    """It opened nothing, and its settings live in the settings dialog."""
    assert "FitOptionsButton" not in ui_names


def test_the_artefact_checkbox_sits_with_the_irf(ui_names):
    """It is built *from* the IRF, so it belongs in that group, not elsewhere."""
    assert "CoherentArtifactCheckBox" in ui_names
    root = ET.parse(UI_FILE).getroot()
    group = next(w for w in root.iter("widget") if w.get("name") == "IRFandtimezeroPanel")
    names = {w.get("name") for w in group.iter("widget")}
    assert {"GaussianIRFCheckBox", "CoherentArtifactCheckBox", "IRFtable"} <= names


def test_the_settings_action_exists(ui_names):
    assert "actionSettings" in ui_names


def test_the_post_fit_plot_buttons_exist(ui_names):
    """The two figures a fit is read from, plus the scheme diagram."""
    assert {"PlotSpeciesSpectraButton", "PlotConcProfileButton", "ShowSchemeButton"} <= ui_names


def test_the_scheme_button_sits_next_to_define_model():
    """'Show kinetic graph' belongs with the model definition, not with the fit."""
    root = ET.parse(UI_FILE).getroot()
    group = next(w for w in root.iter("widget") if w.get("name") == "ModelSelectionGroup")
    names = {w.get("name") for w in group.iter("widget")}
    assert {"DefinemodelButton", "ShowSchemeButton"} <= names


def test_every_fit_only_widget_is_declared(ui_names):
    """The gating list and the layout must name the same widgets."""
    import re

    source = (UI_FILE.parent / "mw_common.py").read_text(encoding="utf-8")
    block = source[source.index("_FIT_ONLY_WIDGETS") : source.index("_UNIMPLEMENTED")]
    named = set(re.findall(r'"([A-Za-z_]+)"', block)) - {"_FIT_ONLY_WIDGETS"}
    assert named <= ui_names, sorted(named - ui_names)


def test_the_panel_margins_come_from_the_settings():
    """The embedded figure's margins are tuned, and tunable.

    Needs Qt only because the mixin lives beside the widgets; the method itself
    touches nothing but the figure, so a bare stub stands in for the window.
    """
    _require_qt()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import pyrate_ta as pr
    from pyrate_ta.gui.tabs.data import DataTabMixin

    class _Stub:
        _build_axes = DataTabMixin._build_axes
        _PANELS = DataTabMixin._PANELS

        def __init__(self, figure):
            self.PlotArea = type("W", (), {"figure": figure, "ax": None})()

    original = pr.get_settings()
    try:
        pr.update_settings(panel_left=0.2, panel_top=0.8)
        figure = plt.figure()
        stub = _Stub(figure)
        stub._build_axes()
        box = stub._axes["main"].get_subplotspec().get_gridspec()
        assert box.left == pytest.approx(0.2)
        assert box.top == pytest.approx(0.8)
    finally:
        pr.set_settings(original)
        plt.close("all")


def test_the_embedded_panels_keep_their_legend_inside():
    """PyMORGAN parks legends beside the axis; three panels sharing a figure cannot.

    The setting is honoured everywhere else -- only the placement is overridden,
    and only for the embedded panels.
    """
    _require_qt()
    import pymorgan as pm

    from pyrate_ta.gui.tabs.data import DataTabMixin

    original = pm.get_settings()
    try:
        pm.update_settings(legend_location="outside right")
        embedded = DataTabMixin.embedded_settings()
        assert embedded.legend_location == "best"
        # Everything else is passed through untouched.
        assert embedded.profile == original.profile
        assert embedded.label_digits == original.label_digits

        pm.update_settings(legend_location="lower left")
        assert DataTabMixin.embedded_settings().legend_location == "lower left"
    finally:
        pm.set_settings(original)


def test_the_window_fits_a_small_laptop_screen():
    """A 13-inch desktop is 1440x900 logical, less the menu bar and dock.

    Three things have to hold together: a default geometry that fits, minimums
    small enough that the window can actually be made that small, and the left
    column inside a scroll area so its controls are reachable when it cannot.
    """
    root = ET.parse(UI_FILE).getroot()
    rect = root.find("widget/property[@name='geometry']/rect")
    width = int(rect.find("width").text)
    height = int(rect.find("height").text)
    # The declared size is a *request*: it is clipped to the screen at start-up
    # (see ``_resize_within_screen``), so it only has to be sane, not tiny. The
    # window is taller than wide because the controls stack vertically.
    assert width <= 1400 and height <= 1000, (width, height)
    assert height > width * 0.7, "the window should not be short and wide"

    minimums = {
        w.get("name"): int(w.find("property[@name='minimumSize']/size/height").text)
        for w in root.iter("widget")
        if w.find("property[@name='minimumSize']/size") is not None
    }
    # The two that used to dominate the minimum height.
    assert minimums["PlotArea"] <= 360
    assert minimums["RateFitTable"] <= 110

    scrolls = [w for w in root.iter("widget") if w.get("class") == "QScrollArea"]
    assert [w.get("name") for w in scrolls] == ["leftScroll"]
    assert scrolls[0].find("property[@name='widgetResizable']/bool").text == "true"
    assert "leftPanel" in {w.get("name") for w in scrolls[0].iter("widget")}


def test_a_stored_geometry_is_clipped_to_the_screen():
    """A size saved on a large monitor must not reopen under the dock."""
    _require_qt()
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    from pyrate_ta.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        available = window.screen().availableGeometry()
        size = window._resize_within_screen(9000, 9000)
        assert size.width() <= available.width()
        assert size.height() <= available.height()
    finally:
        window.close()
        del app


def test_the_plot_area_stretches_more_than_the_controls():
    root = ET.parse(UI_FILE).getroot()
    canvas = next(w for w in root.iter("widget") if w.get("name") == "PlotArea")
    policy = canvas.find("property[@name='sizePolicy']/sizepolicy")
    assert int(policy.find("verstretch").text) >= 2


def test_write_fitted_parameters_to_tables():
    """Fitted t0 and irf_fwhm (delta) must be copied back to IRFtable."""
    _require_qt()
    import os

    import numpy as np

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication, QTableWidget, QWidget

    from pyrate_ta.gui.tabs.fitting import FitTabMixin
    from pyrate_ta.results.fits import GlobalFit

    _app = QApplication.instance() or QApplication([])
    win = type("Win", (QWidget, FitTabMixin), {})()
    win.RateFitTable = QTableWidget(2, 4)
    win.IRFtable = QTableWidget(2, 4)
    fit = GlobalFit(
        taus=np.array([1.5, 5.0]),
        tau_err=np.array([0.1, 0.2]),
        is_fixed=np.array([False, False]),
        S=np.ones((5, 2)),
        R=np.zeros((10, 5)),
        model_type="Sequential",
        t=np.linspace(0, 10, 10),
        probe=np.linspace(1000, 2000, 5),
        C=np.ones((10, 2)),
        t0=-0.05,
        t0_err=0.002,
        irf_fwhm=0.18,
        irf_err=0.01,
    )
    win._write_fitted_lifetimes(fit)
    assert win.IRFtable.item(0, 0).text() == "-0.05"
    assert win.IRFtable.item(1, 0).text() == "0.18"


def test_reset_fit_resets_tables_to_defaults():
    """Resetting a fit must reset lifetime and IRF tables to default values."""
    _require_qt()
    import os

    import numpy as np

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication, QLabel, QRadioButton, QTableWidget, QWidget

    from pyrate_ta.gui.tabs.fitting import FitTabMixin
    from pyrate_ta.gui.tabs.model import ModelTabMixin
    from pyrate_ta.results.fits import GlobalFit

    _app = QApplication.instance() or QApplication([])
    win = type("Win", (QWidget, ModelTabMixin, FitTabMixin), {})()
    win.RateFitTable = QTableWidget(2, 4)
    win.IRFtable = QTableWidget(2, 4)
    win.Chi2Value = QLabel()
    win.ShowDataButton = QRadioButton()
    win.render_all = lambda: None
    win.statusBar = lambda: type("SB", (), {"showMessage": lambda *a: None})()

    # Modify values in table
    win.IRFtable.setItem(0, 0, QTableWidget().item(0, 0))
    fit = GlobalFit(
        taus=np.array([12.3, 45.6]),
        tau_err=np.array([0.1, 0.2]),
        is_fixed=np.array([False, False]),
        S=np.ones((5, 2)),
        R=np.zeros((10, 5)),
        model_type="Sequential",
        t=np.linspace(0, 10, 10),
        probe=np.linspace(1000, 2000, 5),
        C=np.ones((10, 2)),
        t0=-0.05,
        irf_fwhm=0.18,
    )
    win._write_fitted_lifetimes(fit)
    assert win.IRFtable.item(0, 0).text() == "-0.05"

    # Now reset fit
    win._reset_fit()
    # Check that t0 returned to default 0 and tau returned to default 1
    assert win.IRFtable.item(0, 0).text() == "0"
    assert win.RateFitTable.item(0, 0).text() == "1"


def test_tidy_shared_axes_aligns_spectral_axis_bounds():
    """The spectral panel x-axis must align with the main contour panel x-axis despite the colorbar."""
    import matplotlib.pyplot as plt
    import numpy as np

    from pyrate_ta.gui.tabs.data import DataTabMixin

    fig = plt.figure()
    gs = fig.add_gridspec(3, 3)
    ax_main = fig.add_subplot(gs[0:2, 0:2])
    ax_kinetics = fig.add_subplot(gs[0:2, 2])
    ax_spectra = fig.add_subplot(gs[2, 0:2])

    im = ax_main.imshow(np.random.randn(10, 10))
    fig.colorbar(im, ax=ax_main)
    fig.canvas.draw()

    win = type("Win", (DataTabMixin,), {})()
    win._axes = {"main": ax_main, "kinetics": ax_kinetics, "spectra": ax_spectra}
    win._tidy_shared_axes()

    pos_m = ax_main.get_position()
    pos_s = ax_spectra.get_position()
    assert abs(pos_s.x0 - pos_m.x0) < 1e-5
    assert abs(pos_s.x1 - pos_m.x1) < 1e-5


def test_dataset_panels_hidden_until_data_loaded():
    """Everything on the main window except DataGroup must be hidden until data is loaded."""
    _require_qt()
    import numpy as np
    import pymorgan as pm
    from PyQt6.QtWidgets import QApplication

    from pyrate_ta.gui.main_window import MainWindow
    from pyrate_ta.gui.mw_common import _DATASET_PANELS

    _app = QApplication.instance() or QApplication([])


    win = MainWindow()
    win.show()

    # DataGroup ("Load data" panel) should be visible
    assert win.DataGroup.isVisible()

    # All _DATASET_PANELS must be hidden when no data is loaded
    for name in _DATASET_PANELS:
        panel = getattr(win, name, None)
        assert panel is not None
        assert not panel.isVisible(), f"Panel {name} should be hidden before data is loaded"

    # Simulate successful dataset load
    t = np.linspace(-1, 10, 20)
    probe = np.array([1000.0, 1500.0])
    ds = pm.Dataset1D(np.zeros((20, 2, 1)), t, probe, {})

    win.dataset = ds
    win._set_dataset_widgets_enabled(True)

    # All _DATASET_PANELS should now be visible
    for name in _DATASET_PANELS:
        panel = getattr(win, name, None)
        assert panel.isVisible(), f"Panel {name} should be visible after data is loaded"

    # Simulate clear_data
    win.clear_data()
    for name in _DATASET_PANELS:
        panel = getattr(win, name, None)
        assert not panel.isVisible(), f"Panel {name} should be hidden after clearing data"

    win.close()


def test_scheme_source_ignores_fit_result():
    """Ensure scheme_source ignores the fit_result in favor of the current model/lifetimes in the GUI."""
    _require_qt()
    from PyQt6.QtWidgets import QApplication

    from pyrate_ta.gui.main_window import MainWindow

    _app = QApplication.instance() or QApplication([])
    win = MainWindow()

    import numpy as np
    import pymorgan as pm
    t = np.linspace(-1, 10, 20)
    probe = np.array([1000.0, 1500.0])
    ds = pm.Dataset1D(np.zeros((20, 2, 1)), t, probe, {})
    win.dataset = ds

    from pyrate_ta.models.scheme_text import scheme_from_text
    win.custom_scheme = scheme_from_text("A -> B : k1\nB -> : k2\ninit A = 1")

    win.read_lifetimes = lambda: ([1.0, 2.0], ([0.0, 0.0], [np.inf, np.inf]), [False, False])

    class MockFit:
        pass
    win.fit_result = MockFit()

    obj, taus = win.scheme_source()
    assert obj is win.custom_scheme
    assert np.allclose(taus, [1.0, 2.0])

def test_about_dialog():
    """ModernAboutDialog initializes correctly with branding, manual, and GitHub buttons."""
    _require_qt()
    from PyQt6.QtWidgets import QApplication, QLabel, QPushButton

    from pyrate_ta.gui.about_dialog import ModernAboutDialog

    _app = QApplication.instance() or QApplication([])
    dlg = ModernAboutDialog(
        title="About PyRATE-TA",
        app_name="PyRATE-TA",
        version="1.0.0",
        subtitle="Rate Analysis & Target-model Engine",
        description="Sample description.",
        manual_pdf_path="docs/main.pdf",
        website_url="https://www.unige.ch/sciences/chifi/fernandez-teran/",
        github_url="https://github.com/RJFernandezTeran/PyRATE-TA",
    )
    assert dlg.windowTitle() == "About PyRATE-TA"
    assert dlg.isModal()
    assert dlg.width() == 580
    assert dlg.height() == 540

    btn_manual = dlg.findChild(QPushButton, "btnManual")
    assert btn_manual is not None
    assert "Open User Manual" in btn_manual.text()

    btn_github = dlg.findChild(QPushButton, "btnGithub")
    assert btn_github is not None
    assert "GitHub" in btn_github.text()

    lbl_details = dlg.findChild(QLabel, "lblDetails")
    assert lbl_details is not None
    assert "https://github.com/RJFernandezTeran/PyRATE-TA" in lbl_details.text()
    assert "https://www.unige.ch/sciences/chifi/fernandez-teran/" in lbl_details.text()
    dlg.close()

