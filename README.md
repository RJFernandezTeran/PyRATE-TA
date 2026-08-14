# PyRATE-TA

<img src="src/pyrate_ta/gui/icons/pirate_ship.png" alt="" width="110" align="right">

**R**ate **A**nalysis & **T**arget-model **E**ngine for **T**ransient **A**bsorption in Python

[![PyPI - Version](https://img.shields.io/pypi/v/pyrate-ta.svg?logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/pyrate-ta/)
[![GitHub Release](https://img.shields.io/github/v/release/RJFernandezTeran/PyRATE-TA?logo=github&label=Release)](https://github.com/RJFernandezTeran/PyRATE-TA/releases)
[![Python >= 3.12](https://img.shields.io/badge/python-%3E%3D3.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![GUI: PyQt6](https://img.shields.io/badge/GUI-PyQt6-41CD52.svg?logo=qt&logoColor=white)](https://riverbankcomputing.com/software/pyqt/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tests: pytest](https://img.shields.io/badge/tests-pytest-0A9EDC.svg?logo=pytest&logoColor=white)](tests/)
[![Requires: PyMORGAN](https://img.shields.io/badge/requires-PyMORGAN-8A2BE2.svg)](https://github.com/RJFernandezTeran/PyMORGAN)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL_3.0-blue.svg)](LICENSE)
[![CodeFactor](https://www.codefactor.io/repository/github/RJFernandezTeran/PyRATE-TA/badge)](https://www.codefactor.io/repository/github/RJFernandezTeran/PyRATE-TA)
[![GitHub Stars](https://img.shields.io/github/stars/RJFernandezTeran/PyRATE-TA?style=flat&logo=github)](https://github.com/RJFernandezTeran/PyRATE-TA/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/RJFernandezTeran/PyRATE-TA?style=flat&logo=github)](https://github.com/RJFernandezTeran/PyRATE-TA/network/members)
[![GitHub Issues](https://img.shields.io/github/issues/RJFernandezTeran/PyRATE-TA?logo=github)](https://github.com/RJFernandezTeran/PyRATE-TA/issues)
[![Downloads](https://img.shields.io/pepy/dt/pyrate-ta?label=downloads)](https://pepy.tech/project/pyrate-ta)

Kinetic analysis of ultrafast time-resolved spectroscopy data: **multi-exponential**
fitting, **global** analysis and **target** analysis with rate-matrix (K-matrix)
models, **lifetime density analysis** (LDA) with regularisation and automatic
parameter selection, and the **DAS / EAS / SAS** spectra that come out of them.

---

## Relationship to PyMORGAN

PyRATE-TA is the analysis half of a two-project pair.
[PyMORGAN](https://github.com/RJFernandezTeran/PyMORGAN) owns **loading,
processing and plotting**: the file-format readers (`PDAT`, `P2DAT`, MESS
directories, …), the `Dataset1D` / `Dataset2D` objects, the presentation
`Settings` and every Matplotlib plotter.

**The dependency is one-way: PyRATE-TA imports PyMORGAN, never the reverse.**
PyRATE-TA does not reimplement readers, dataset objects or standard figures. The
seam is deliberately narrow — PyMORGAN's `plot_species_spectra` takes plain
arrays (`Sfit`, `Taus`, `TauErr`, `isFixTau`, `modelType`) rather than a PyRATE-TA
object, so the arrow never has to reverse; the adapter (`as_species_args`) lives
on this side.

## Features

- **Shared data layer** — datasets are loaded through PyMORGAN's pluggable
  loader registry, so every format it supports (`PDAT`, `MESS_TRIR`,
  `Helios_TA`, `UniGE_fsTA`, …) is available here without a second reader. New
  formats are contributed upstream, not forked.
- **Analysis-only settings** — solver tolerances, iteration limits, default
  component count and IRF handling, round-tripped to a commented
  `settings.toml`. The dataclass field is the single source of truth:
  serialisation, type coercion and the GUI widgets are all derived from it.
  Presentation settings are deliberately *not* duplicated — style, colourmap and
  label conventions stay in `pymorgan.Settings`.
- **Global analysis by variable projection** — for fixed lifetimes the linear
  amplitudes are the least-squares solution `S = pinv(C) @ D`, so only the
  non-linear parameters reach the optimiser. Far better conditioned than fitting
  amplitudes and lifetimes jointly.
- **Kinetic models** — parallel (independent decays → DAS), sequential
  (`A → B → C` → EAS) and general compartmental K-matrix models with branching
  (→ SAS). `C(t)` by eigendecomposition where `K` is diagonalisable, by
  matrix-exponential propagation otherwise, with the near-degenerate eigenvalue
  case guarded.
- **Analytic IRF convolution** — Gaussian-convolved exponentials (exponentially
  modified Gaussian) rather than numerical convolution, with clipping (`t_min`)
  as the alternative for the coherent artefact.
- **Coherent artefact suppression** — append the IRF and its first two
  derivatives to the design matrix (`coherent_artifact=True`) so cross-phase
  modulation and Raman signals are absorbed by their own amplitudes instead of
  distorting the shortest lifetime.
- **Lifetime density analysis (LDA)** — a dense regularised grid of fixed
  lifetimes with Tikhonov / first- or second-difference penalties. Automatic
  alpha selection by L-curve corner detection, GCV or the Morozov discrepancy
  principle. Optional non-negative constraint (NNLS), SVD pre-filtering, Monte
  Carlo bootstrap errors, and automatic peak centroid detection. Exports the 2D
  map to PDAT format.
- **Ground-state bleach / absolute spectra** — convert difference species
  spectra to absolute absorption spectra by adding back a scaled ground-state
  spectrum. The maximum physically allowable scale factor is computed
  automatically.
- **Constrained amplitudes** — a per-component sign mask (`sign_mask`) imposes
  non-negative (`+1`) or non-positive (`−1`) constraints on any species spectrum
  without touching the lifetimes.
- **Reproducible results** — every result carries the model identity, initial
  guesses, bounds, fixed/free masks, the delay and probe ranges actually used,
  and the solver's convergence report. A lifetime without its uncertainty and its
  fixed/free flag is not a result.
- **Fit session files** — `pr.save_fit` / `pr.load_fit` round-trip the full
  result (arrays + JSON metadata) to a `.prfit` file. A target fit reopens with
  its scheme in editable text so it can be modified without re-entering it.
- **Honest convergence** — a fit that did not converge raises or is flagged;
  it is never returned as though it had. Parameters resting on a bound and
  near-degenerate lifetimes are reported explicitly.
- **Preview mode** — `pr.preview_global` evaluates a model at given lifetimes
  *without* optimising, showing what those lifetimes imply and what the residual
  would be, clearly marked as a guess rather than a result.
- **Analysis-only plots** — residual matrices, concentration profiles `C(t)`,
  K-matrix diagrams, fit-monitor bars (live parameter tracking during a fit),
  kinetic traces with residuals panels, and spectra with residuals. Everything
  PyMORGAN can already draw is delegated upstream.
- **Headless-first fitting** — every fit is reachable and testable from a plain
  script with no `QApplication`; the GUI is a caller, never a prerequisite.
- **Bundled PyQt6 GUI** (`pyrate-ta-gui`) — data loading, model building,
  global / target / LDA fitting, fit control and results inspection, reusing
  PyMORGAN's theme so the two applications look identical side by side.
- **Fast start-up** — the public API is lazy (PEP 562), so `import pyrate_ta` does
  not pay for scipy or, through PyMORGAN, matplotlib and the loader registries.
- **Logging, not prints** — library messages go through the `pyrate` logger
  (`pyrate.configure_logging()`); a console handler is attached only when the
  host has not configured logging, so it does not fight PyMORGAN's.
- **uv-managed, ruff-linted, pytest-tested.**

## Installation

### From PyPI (Recommended)

Install the latest release directly from [PyPI](https://pypi.org/project/pyrate-ta/):

```bash
pip install pyrate-ta
# or with uv:
uv pip install pyrate-ta

# Register bundled fonts in matplotlib:
pymorgan-install-fonts
```

You can also run the GUI directly without installing using [`uvx`](https://docs.astral.sh/uv/concepts/tools/):
```bash
uvx --from pyrate-ta pyrate-ta-gui
```

### From Source (Development)

Clone the repository and install in editable mode with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/RJFernandezTeran/PyRATE-TA.git
cd PyRATE-TA

uv venv                       # create .venv (Python >= 3.12)

# Install the package (choose one):
uv pip install -e .           # core workflow (includes PyQt6 GUI)
# or:
uv pip install -e ".[dev]"    # + ruff and pytest

uv run pymorgan-install-fonts # register PyMORGAN's fonts in this environment
```

PyMORGAN is resolved from the sibling checkout (`../PyMORGAN`) through
`[tool.uv.sources]`, so upstream edits are picked up without a reinstall. **The
two folders must stay siblings**; replace that entry with a plain version pin
once PyMORGAN is published to an index.

Console scripts installed with the package:

| Command | Purpose |
|---|---|
| `pyrate-ta-gui` | Launch the fitting interface |
| `pyrate-ta-edit-gui` | Open `main_window.ui` in Qt Designer |
| `pyrate-ta-settings` | Standalone analysis-settings editor |

## Quick start

```python
import pyrate_ta as pr
import pymorgan as pm

pm.load_settings("settings.toml")   # aesthetics: profile, labels, cmap, ...
pm.apply_style()

# --- Load and process: PyMORGAN ---
data = pm.load_1D("scan.pdat", data_type="PDAT")
data.background_correct(tmin=-20, tmax=-5)

# --- Global fit: PyRATE-TA ---
fit = pr.fit_global(data, n_components=3, model_type="Sequential")
print(fit.taus, fit.tau_err)     # lifetimes and 1-sigma uncertainties

# --- Preview (no optimisation) ---
prev = pr.preview_global(data, taus=[1.0, 10.0, 100.0])

# --- Target analysis ---
fit_t = pr.fit_target(data, scheme="A_eq_B_to_C", taus=[2.0, 5.0, 20.0, 200.0])

# --- Lifetime density analysis ---
from pyrate_ta.fit.lda import solve_lda
lda = solve_lda(data, n_taus=100, alpha="auto", non_negative=True)

# --- Plot the species spectra: PyMORGAN ---
data.plot_species_spectra(*fit.as_species_args())

# --- Save the result ---
pr.save_fit("scan_fit", fit)          # -> scan_fit.prfit

pm.show_plots()
```

Analysis defaults come from `pyrate.Settings` (solver tolerances, component
count, IRF handling); everything about how the figures *look* comes from
`pymorgan.Settings`.

## Project layout

```
src/pyrate_ta/
  models/        kinetic models: parallel · sequential · target (K-matrix) · irf · schemes · scheme_text
  fit/           solvers & engines: driver · engines · varpro · cost · lda (lifetime density)
                                    groundstate (absolute spectra) · constrained · prepare
  results/       KineticFit · GlobalFit · TargetFit · LDAResult · serialisation
  plot/          analysis-only views: concentrations · scheme · matrix · monitor
                                      traces (kinetics + residuals) · lda · style
  io/            fit-session import/export (.prfit — raw formats belong upstream to PyMORGAN)
  gui/           PyQt6 application (pyrate-ta-gui)
    main_window.py / .ui   window assembly, wiring, menus (layout lives in the .ui)
    settings_app.py        standalone analysis-settings editor (pyrate-ta-settings)
    settings_panel.py      embedded settings editor widget
    scheme_dialog.py / .ui target analysis model scheme builder dialog
    tabs/                  per-tab mixins: data · model · fitting · lda
    crosshair.py · designer.py · monitor.py · mw_common.py
    icons/                 application icons (pirate_ship.png)
  settings.py            Settings (TOML auto-merge, solver, model and LDA defaults, GUI field specs)
  settings.default.toml  canonical default settings template shipped with the package
  cite.py                academic citation registry and logger (pyrate-ta-cite)
  helpers.py             shared numerical and plotting helpers
  log.py                 logger factory and console configuration
examples/        runnable example scripts (quick_fit.py, lda_fit.py)
tests/           pytest suite (headless matplotlib, offscreen Qt)
scripts/         bump_version.py · run_checks.py · run_changed_tests.py
docs/            LaTeX manual (main + boundary + models + fitting + gui + settings + extending + installation)
```

## Testing

```bash
uv pip install -e ".[dev]"
uv run pytest                             # full test suite
uv run python scripts/run_checks.py       # boundary, API and Qt-layout checks
```

Synthetic-recovery tests are the backbone of the suite: build data from known
parameters, fit it, assert the parameters come back within tolerance. A fitting
project without those is untested regardless of line coverage. The failure modes
are tested too — non-convergence must be flagged rather than silently returned,
and near-degenerate lifetimes must not produce exploding amplitudes unwarned.

## Linting

[Ruff](https://docs.astral.sh/ruff/) handles linting and formatting
(configured in `pyproject.toml`):

```bash
uvx ruff@latest check src tests scripts
uvx ruff@latest format src tests
```

The version string in `src/pyrate_ta/__about__.py` follows `0.x.yymmdd.devN`
(PEP 440) and is maintained by `python scripts/bump_version.py` (`--minor` for a
design bump, `--check` to verify it was bumped today).

## Citing

If you publish results obtained with PyRATE-TA, please cite:

1. The PyRATE-TA / PyMORGAN software paper — *manuscript in preparation*.
2. R. J. Fernández-Terán, E. Sucre-Rosales, L. Echevarria, F. E. Hernández,
   *A Sweet Introduction to the Mathematical Analysis of Time-Resolved Spectra
   and Complex Kinetic Mechanisms: The Chameleon Reaction Revisited*,
   J. Chem. Educ. **2022**, 99, 2327–2337.
   [10.1021/acs.jchemed.2c00104](https://doi.org/10.1021/acs.jchemed.2c00104)
3. I. H. M. van Stokkum, D. S. Larsen, R. van Grondelle, *Global and target
   analysis of time-resolved spectra*, Biochim. Biophys. Acta Bioenerg.
   **2004**, 1657, 82–104.
   [10.1016/j.bbabio.2004.04.011](https://doi.org/10.1016/j.bbabio.2004.04.011)
   — global/target analysis and variable projection.
4. M. N. Berberan-Santos, J. M. G. Martinho, *The integration of kinetic rate
   equations by matrix methods*, J. Chem. Educ. **1990**, 67, 375.
   [10.1021/ed067p375](https://doi.org/10.1021/ed067p375)
   — the eigenvector solution used by the propagator.
5. P. C. Hansen, *Analysis of discrete ill-posed problems by means of the
   L-curve*, SIAM Review **1992**, 34, 561–580.
   — L-curve corner detection for LDA alpha selection.

The list is printed by `pyrate-ta-cite` and logged when the GUI starts. Routines
that implement a published method also log their own reference when they run,
so the log of a fit records which formalism produced it.

## Acknowledgements

- Development assisted by **Google Antigravity**, with all code, algorithms, and implementations manually verified and tested.
- PyRATE-TA builds upon and modernizes the original MATLAB implementation from the now-deprecated [DataAnalysis](https://github.com/RJFernandezTeran/DataAnalysis) repository, written by Dr. Ricardo J. Fernández-Terán during his PhD and validated iteratively throughout the years.

## License

Released under the [GNU Affero General Public License v3.0 (AGPLv3)](LICENSE). © 2026 Dr. Ricardo J. Fernández-Terán
