.. _installation:

Installation
============

PyRATE-TA requires **Python 3.11 or later** (Python 3.12+ recommended). It can be installed
directly from `PyPI <https://pypi.org/project/pyrate-ta/>`_ or from source using
`uv <https://docs.astral.sh/uv/>`_.

Quick Install via PyPI (Recommended)
------------------------------------

.. code-block:: bash

   pip install pyrate-ta
   # or with uv:
   uv pip install pyrate-ta

   # Register PyMORGAN's bundled fonts into matplotlib:
   pymorgan-install-fonts

You can also launch the graphical interface directly without prior installation using ``uvx``:

.. code-block:: bash

   uvx --from pyrate-ta pyrate-ta-gui

Installation from Source (Development)
--------------------------------------

From the repository root:

.. code-block:: bash

   uv venv                          # create .venv (Python >= 3.11)
   uv pip install -e ".[gui,dev]"   # editable install + dependencies
   uv run pymorgan-install-fonts    # register PyMORGAN's font stack in this environment

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
   * - ``pyrate-ta-gui``
     - Launch the PyQt6 kinetic fitting interface
   * - ``pyrate-ta-edit-gui``
     - Open ``src/pyrate_ta/gui/main_window.ui`` in Qt Designer
   * - ``pyrate-ta-settings``
     - Standalone editor for analysis settings
   * - ``pyrate-ta-cite``
     - Print literature citations for the implemented kinetic & LDA algorithms

Versioning
----------

Every edit to the codebase bumps the version in ``src/pyrate_ta/__about__.py`` using
``python scripts/bump_version.py``. The scheme is ``1.YYMMDD.N`` (PEP 440): ``YYMMDD`` is the
build date and ``N`` is the 1-indexed daily counter (starting at ``.1`` on a new date and
incrementing sequentially as ``.2``, ``.3``, etc.).
