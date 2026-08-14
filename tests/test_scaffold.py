"""Scaffold-level checks: the package imports, exports resolve, settings behave.

These are cheap invariants that break loudly when the lazy-API bookkeeping in
``pyrate/__init__.py`` falls out of sync -- the most common way to ship a broken
export.
"""

import numpy as np
import pytest

import pyrate_ta as pr


def test_version_scheme():
    """Version must parse as 0.<minor>.<yymmdd>.dev<N> (PEP 440)."""
    import re

    assert re.fullmatch(r"\d+\.\d+\.\d{6}\.dev\d+", pr.__version__), pr.__version__


def test_all_exports_resolve():
    """Every name in __all__ must be reachable through the PEP 562 __getattr__."""
    for name in pr.__all__:
        assert getattr(pr, name) is not None, name


def test_exports_and_all_agree():
    """_EXPORTS and __all__ must not drift apart."""
    missing = set(pr._EXPORTS) - set(pr.__all__)
    assert not missing, f"in _EXPORTS but not __all__: {sorted(missing)}"


def test_submodules_importable():
    """The declared submodules must all import."""
    for name in pr._SUBMODULES:
        assert getattr(pr, name) is not None, name


def test_no_reverse_dependency():
    """pymorgan must never import pyrate.

    Importing pyrate is expected to pull pymorgan in; the reverse would make the
    two packages circular and is a design error.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "import pymorgan, sys; print('pyrate' in sys.modules)"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip("pymorgan not installed in this environment")
    assert result.stdout.strip() == "False", "pymorgan imported pyrate"


def test_settings_roundtrip():
    """to_dict/from_dict must survive a round trip with enums intact."""
    s = pr.Settings(n_components=3, model_type=pr.ModelType.TARGET, n_contours=40)
    restored = pr.Settings.from_dict(s.to_dict())
    assert restored.n_components == 3
    assert restored.model_type is pr.ModelType.TARGET
    assert restored.n_contours == 40



def test_settings_rejects_unknown():
    """update_settings must not silently swallow a typo."""
    with pytest.raises(ValueError, match="unknown setting"):
        pr.update_settings(n_componets=3)  # typo on purpose


def test_field_specs_reference_real_fields():
    """Every GUI spec must name an actual dataclass field."""
    from dataclasses import fields

    names = {f.name for f in fields(pr.Settings)}
    for key in pr.Settings.field_specs():
        assert key in names, f"field_specs names unknown field {key!r}"


def test_every_setting_reaches_the_panel():
    """A setting the interface cannot edit is a setting the user does not have.

    The specs are derived from the dataclass, so this holds automatically; the
    test is here to catch a field being dropped from ``field_sections`` (which
    would leave it unreachable) or skipped by accident.
    """
    from dataclasses import fields

    names = {f.name for f in fields(pr.Settings)}
    specs = pr.Settings.field_specs()
    deliberately_hidden = {
        "offset",  # deprecated: an inf lifetime is the offset
        "lda_n_lifetimes",
        "lda_tau_min",
        "lda_tau_max",
        "lda_regularisation",
        "lda_alpha",
        "lda_non_negative",
    }
    assert names - set(specs) == deliberately_hidden
    assert all(spec.get("kind") for spec in specs.values())
    assert all(spec.get("tab") for spec in specs.values())


def test_the_theme_choices_come_from_pymorgan():
    """One list of Qt styles, owned by PyMORGAN."""

    from pymorgan.settings import gui_theme_choices

    spec = pr.Settings.field_specs()["gui_theme"]
    assert spec["kind"] == "choice"
    assert spec["choices"] == list(gui_theme_choices())


def test_two_component_fixture_is_consistent(two_component):
    """The synthetic fixture must satisfy D = C @ S.T exactly."""
    truth = two_component
    assert np.allclose(truth["D"], truth["C"] @ truth["spectra"].T)
    assert truth["D"].shape == (truth["delays"].size, truth["probe"].size)


def test_crosshair_keyboard_arrow_navigation():
    """Keyboard arrow keys must step crosshair horizontally/vertically."""
    import matplotlib.pyplot as plt

    from pyrate_ta.gui.crosshair import Crosshair

    fig, ax = plt.subplots()
    x_vals = np.linspace(400, 800, 41)
    y_vals = np.linspace(0, 10, 101)
    c = Crosshair(
        fig.canvas,
        ax,
        on_change=lambda x, y: None,
        on_release=lambda x, y: None,
        x=600,
        y=5,
        x_values=x_vals,
        y_values=y_vals,
    )
    # Right arrow -> probe + 10
    c._on_key_press(type("Event", (), {"inaxes": ax, "key": "right"})())
    assert abs(c.x - 610.0) < 1e-6

    # Up arrow -> delay + 0.1
    c._on_key_press(type("Event", (), {"inaxes": ax, "key": "up"})())
    assert abs(c.y - 5.1) < 1e-6

    # Left arrow -> probe - 10
    c._on_key_press(type("Event", (), {"inaxes": ax, "key": "left"})())
    assert abs(c.x - 600.0) < 1e-6

    # Down arrow -> delay - 0.1
    c._on_key_press(type("Event", (), {"inaxes": ax, "key": "down"})())
    assert abs(c.y - 5.0) < 1e-6

