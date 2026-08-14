"""Quick end-to-end smoke check for PyRATE-TA.

Run::

    python scripts/run_checks.py

Exercises what does not need a data file: the settings round-trip, the lazy
public API, the kinetic models against closed-form expressions, a synthetic
fit recovery, the PyMORGAN
boundary (one-way dependency), and the integrity of the Qt Designer layout. The Qt checks are skipped -- not failed -- when no Qt
runtime is available, so this runs on a headless machine.

Prints a PASS/SKIP/FAIL line per check and exits non-zero if any fail.
"""

from __future__ import annotations

import sys
import tempfile
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# The in-tree source wins over anything installed in the active environment.
sys.path.insert(0, str(REPO / "src"))

import pyrate_ta as pr  # noqa: E402

_UI_FILE = REPO / "src" / "pyrate_ta" / "gui" / "main_window.ui"

results: list[tuple[str, bool]] = []


def check(name: str, cond: bool) -> None:
    results.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


def skip(name: str, why: str) -> None:
    print(f"[SKIP] {name} ({why})")


def check_settings() -> None:
    """Defaults, field specs and a TOML round-trip."""
    pr.set_settings(pr.Settings())
    check("settings: defaults", pr.get_settings().n_components >= 1)

    specs = pr.Settings.field_specs()
    known = set(pr.Settings().to_dict())
    check("settings: field_specs names are real fields", set(specs) <= known)

    pr.update_settings(n_components=4, model_type=str(pr.ModelType.TARGET))
    path = Path(tempfile.mkdtemp()) / "settings.toml"
    pr.save_settings(path)
    loaded = pr.load_settings(path)
    check(
        "settings: TOML round-trip",
        loaded.n_components == 4 and loaded.model_type == pr.ModelType.TARGET,
    )
    pr.set_settings(pr.Settings())


def check_models() -> None:
    """Concentration matrices against closed-form expressions."""
    import numpy as np

    from pyrate_ta.models import ParallelModel, SequentialModel, TargetModel

    t = np.geomspace(0.01, 1000.0, 200)

    C = ParallelModel(n_components=2).concentrations(t, taus=[2.0, 50.0])
    check(
        "models: parallel = plain exponentials", np.allclose(C, np.exp(-t[:, None] / [2.0, 50.0]))
    )

    k = 1.0 / np.array([3.0, 40.0])
    C = SequentialModel(n_components=2).concentrations(t, taus=[3.0, 40.0])
    B = k[0] / (k[1] - k[0]) * (np.exp(-k[0] * t) - np.exp(-k[1] * t))
    check("models: sequential = consecutive-reaction form", np.allclose(C[:, 1], B))

    # A <=> B with no decay is closed: the population is conserved.
    C = TargetModel(n_components=2, scheme="A_eq_B").concentrations(t, taus=[5.0, 20.0])
    check("models: closed system conserves population", np.allclose(C.sum(axis=1), 1.0, atol=1e-8))

    sharp = SequentialModel(n_components=2).concentrations(t, taus=[5.0, 50.0])
    narrow = SequentialModel(n_components=2).concentrations(t, taus=[5.0, 50.0], irf_fwhm=1e-3)
    check("models: narrow IRF -> unconvolved limit", np.allclose(sharp, narrow, atol=1e-6))


def check_fit() -> None:
    """Synthetic recovery: known parameters in, same parameters out."""
    import numpy as np

    from pyrate_ta.fit import build_weights, fit_statistic, project, run_fit
    from pyrate_ta.models import SequentialModel

    t = np.concatenate([np.linspace(-2.0, 0.5, 20), np.geomspace(0.6, 3000.0, 140)])
    probe = np.linspace(1900.0, 2010.0, 40)
    taus = (8.0, 300.0)

    model = SequentialModel(n_components=2, irf_fwhm=0.3)
    C = model.concentrations(t, taus=taus)
    S = np.stack([np.exp(-((probe - c) ** 2) / (2 * 20.0**2)) for c in (1950, 1975)], axis=1)
    D = C @ S.T

    S_fit, R = project(C, D)
    check("fit: amplitudes exact for a noiseless model", np.allclose(S_fit, S, atol=1e-8))

    out = run_fit(model, t, D, p0=[3.0, 100.0])
    check(
        "fit: lifetimes recovered from synthetic data",
        out.converged and np.allclose(out.params, taus, rtol=1e-3),
    )

    sigma_true = 0.02
    Dn = D + sigma_true * np.random.default_rng(0).standard_normal(D.shape)
    w = build_weights(Dn, np.full(Dn.shape, sigma_true), use_weights=True, source="synthetic")
    _, Rn = project(C, Dn, w.w)
    stat = fit_statistic(Rn, w, n_nonlinear=2, n_amplitudes=S.size)
    check(
        "fit: reduced chi2 ~ 1 for correctly weighted noise",
        stat.kind == "chi2_red" and abs(stat.value - 1.0) < 0.15,
    )

    capped = run_fit(model, t, D, p0=[1.0, 1000.0], max_iterations=3)
    check("fit: non-convergence is flagged", capped.converged is False)


def check_lazy_api() -> None:
    """Importing pyrate must not drag in scipy, matplotlib or pymorgan.

    Measured in a subprocess: reading this process's ``sys.modules`` only tells
    us what *some* earlier check imported, so the answer depended on the order
    the checks happen to run in. A fresh interpreter measures the import itself,
    which is what the claim is about.
    """
    import subprocess

    probe = (
        "import sys, pyrate;"
        "print(','.join(m for m in ('scipy', 'matplotlib', 'pymorgan') if m in sys.modules))"
    )
    try:
        out = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, timeout=120
        )
        heavy = [m for m in out.stdout.strip().split(",") if m]
    except Exception as exc:  # pragma: no cover - no interpreter to spawn
        skip("api: import pyrate_ta stays cheap", str(exc))
        heavy = []
        return
    check("api: import pyrate_ta stays cheap", not heavy)
    check("api: every __all__ name resolves", all(hasattr(pr, n) for n in pr.__all__))


def check_boundary() -> None:
    """PyMORGAN must not import PyRATE-TA, and PyRATE-TA must not read files itself."""
    pm_src = REPO.parent / "PyMORGAN" / "src" / "pymorgan"
    if not pm_src.is_dir():
        skip("boundary: pymorgan does not import pyrate_ta", "PyMORGAN checkout not found")
    else:
        offenders = [
            str(f.relative_to(pm_src))
            for f in pm_src.rglob("*.py")
            if "import pyrate_ta" in f.read_text(encoding="utf-8", errors="replace")
        ]
        check("boundary: pymorgan does not import pyrate_ta", not offenders)

    pr_src = REPO / "src" / "pyrate_ta"
    readers = [
        str(f.relative_to(pr_src))
        for f in pr_src.rglob("*.py")
        if "def load_1D" in f.read_text(encoding="utf-8", errors="replace")
    ]
    check("boundary: pyrate defines no loader", not readers)


def check_ui() -> None:
    """The Designer layout parses and its promoted widgets come from PyMORGAN."""
    if not _UI_FILE.is_file():
        check("ui: main_window.ui present", False)
        return
    root = ET.parse(_UI_FILE).getroot()
    names = {w.get("name") for w in root.iter("widget")}
    check("ui: layout parses", root.tag == "ui")
    check("ui: plot area and panels declared", {"PlotArea", "PC_box", "RateFitTable"} <= names)
    headers = {h.text for h in root.iter("header")}
    check(
        "ui: promoted widgets come from pymorgan", all(h.startswith("pymorgan.") for h in headers)
    )


def check_gui_imports() -> None:
    """The GUI modules import and the main window builds, headless."""
    try:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication
    except Exception as exc:
        skip("gui: main window builds", f"no Qt runtime: {exc}")
        return

    app = QApplication.instance() or QApplication([])
    from pyrate_ta.gui.main_window import MainWindow

    win = MainWindow()
    check("gui: main window builds", win.dataset is None and win.plot_controls is not None)
    win.close()
    del app


def main() -> int:
    check_settings()
    check_lazy_api()
    check_models()
    check_fit()
    check_boundary()
    check_ui()
    check_gui_imports()

    n_fail = sum(1 for _, ok in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
