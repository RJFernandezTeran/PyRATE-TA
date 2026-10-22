.. _installation:

Installation
============

PyRATE-TA requires **Python 3.11 or later** (Python 3.12+ recommended). It can be installed
directly from `PyPI <https://pypi.org/project/pyrate-ta/>`_, installed from source using
`uv <https://docs.astral.sh/uv/>`_, or deployed on **Windows**, **macOS**, and **Linux** lab
workstations using the cross-platform **PyMORGAN Suite** standalone installers.

Quick Install via PyPI (Recommended)
------------------------------------

To install the latest official release into your active Python environment:

.. code-block:: bash

   pip install pyrate-ta
   # or with uv:
   uv pip install pyrate-ta

   # Register PyMORGAN's bundled fonts into matplotlib:
   pymorgan-install-fonts

You can also launch the graphical interface directly without prior installation using ``uvx``:

.. code-block:: bash

   uvx --from pyrate-ta pyrate-ta-gui

Cross-Platform Standalone Installers (PyMORGAN Suite)
-----------------------------------------------------

PyRATE-TA is bundled alongside PyMORGAN in the cross-platform **PyMORGAN Suite** standalone
distributions available on `GitHub Releases <https://github.com/RJFernandezTeran/PyMORGAN/releases>`_:

- **Windows (**\ ``x64``\ **):** ``PyMORGAN_Suite_Setup_v*.exe`` (Inno Setup wizard with Start
  Menu / Desktop shortcuts and ``.pdat`` / ``.p2dat`` file associations) and
  ``PyMORGAN_Suite_v*_Windows_x64_Portable.zip``.
- **macOS (**\ ``Apple Silicon / Intel``\ **):** ``PyMORGAN_Suite_v*_macOS.dmg`` and code-signed
  ``PyMORGAN Suite.app`` bundle containing ``PyMORGAN_Suite``, ``PyMORGAN``, and ``PyRATE-TA``.
- **Linux (**\ ``x86_64``\ **):** ``PyMORGAN_Suite_v*_Linux_x86_64.tar.gz`` with standalone
  binaries (``bin/PyMORGAN_Suite``, ``bin/PyMORGAN``, ``bin/PyRATE-TA``), freedesktop
  ``.desktop`` launchers, and automated ``install.sh`` / ``uninstall.sh`` scripts.
- **1-Click Local Python Environment Setup (**\ ``pymorgan-setup-env``\ **):** Users can click
  **Tools → Setup in Local Python Environment...** in the Suite Launcher (or **Setup in
  Python...** in the **About** dialog) to automatically configure any local Conda, Homebrew,
  ``uv``, or system Python environment so ``import pyrate_ta`` and ``import pymorgan`` work
  directly in Jupyter notebooks and scripts.

Installation from Source (Development)
--------------------------------------

From the repository root:

.. code-block:: bash

   uv venv                          # create .venv (Python >= 3.11)
   uv pip install -e ".[gui,dev]"   # editable install + dependencies
   uv run pymorgan-install-fonts    # PyMORGAN's fonts, in *this* environment

The font step is essential in practice: PyRATE-TA uses PyMORGAN's font stack and ships no second
installer, and Matplotlib registers fonts per virtual environment. Skipping it causes figures to
fall back to DejaVu Sans. The GUI checks at start-up and logs the command if the preferred fonts
are missing.

Co-Development with PyMORGAN
----------------------------

PyRATE-TA depends on PyMORGAN. While the two are developed side by side, PyMORGAN can be picked
up from a sibling checkout as an editable path dependency so upstream edits are visible
immediately without reinstalling:

.. code-block:: toml

   # pyproject.toml
   [tool.uv.sources]
   pymorgan = { path = "../PyMORGAN", editable = true }

Core Dependencies
-----------------

.. list-table:: Core Runtime Dependencies
   :header-rows: 1
   :widths: 28 18 54

   * - Package
     - Minimum
     - Purpose
   * - ``pymorgan``
     - ``>=0.6``
     - Loading, dataset containers, plotting, and shared aesthetics
   * - ``numpy``
     - ``>=1.26.0``
     - Matrix and array operations
   * - ``scipy``
     - ``>=1.12.0``
     - Non-linear least squares, NNLS, and matrix exponentials
   * - ``tomlkit``
     - ``>=0.12.0``
     - Comment-preserving ``settings.toml`` read/write
   * - ``PyQt6`` / ``PyQt6-sip``
     - ``>=6.6.0``
     - Graphical user interface framework
   * - ``defusedxml``
     - ``>=0.7.1``
     - Hardened XML parsing for Qt Designer ``.ui`` files

Console Scripts
---------------

.. list-table:: Installed Console Entry Points
   :header-rows: 1
   :widths: 32 68

   * - Command
     - Purpose
   * - ``pyrate-ta`` / ``pyrate-ta-gui``
     - Launch the PyQt6 kinetic fitting interface
   * - ``pyrate-ta-edit-gui``
     - Open ``src/pyrate_ta/gui/main_window.ui`` in Qt Designer
   * - ``pyrate-ta-settings``
     - Standalone editor for analysis settings
   * - ``pyrate-ta-cite``
     - Print literature citations for the implemented kinetic & LDA algorithms

Verifying the Installation
--------------------------

.. code-block:: bash

   uv run python -c "import pyrate_ta; print(pyrate_ta.__version__)"
   uv run pytest
   uv run python scripts/run_checks.py

``scripts/run_checks.py`` exercises the settings round-trip, the lazy public API, the PyMORGAN
boundary, and the integrity of the Qt Designer layout.

Versioning
----------

Every edit to the codebase bumps the version in ``src/pyrate_ta/__about__.py`` using
``python scripts/bump_version.py``. The scheme is ``1.YYMMDD.N`` (PEP 440): ``YYMMDD`` is the
build date and ``N`` is the 1-indexed daily counter (starting at ``.1`` on a new date and
incrementing sequentially as ``.2``, ``.3``, etc.). ``--check`` exits non-zero when the recorded
version was not produced today.
