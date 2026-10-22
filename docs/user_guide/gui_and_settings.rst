.. _user_guide_gui_and_settings:

Graphical Interface & Settings
==============================

Launching the GUI
-----------------

.. code-block:: bash

   uv run pyrate-ta-gui

Main Window Layout
------------------

- **Left Column:** Stacked control groups for *Data* (dataset and noise lamps), *Model Selection*
  (*Parallel* / *Sequential* / *Target* / *ODE-defined*), *Components* (lifetime table with
  bounds and fixed flags), *IRF and time-zero*, *Fit options*, and *Fit control* (including
  *Plot DAS/EAS/SAS*, *Plot Conc. Profile*, and Steady-State Reference GSB/SE de-bleaching
  controls).
- **Plot Area:** A :math:`3 \times 3` synchronised grid combining the 2D contour map, rotated
  kinetic traces, and transient spectra on a single canvas.
- **K-Matrix Editor:** *Define model...* opens the interactive text-scheme editor with live syntax
  validation, horizontal compartment graph preview, and derived :math:`\mathbf{K}` matrix
  inspection.
- **LDA Tab:** Organised into three side-by-side control groups (*Regularisation & Solver*,
  *Lifetime Grid & Options*, and *Actions & Diagnostics*) displaying the integrated amplitude
  spectrum :math:`A(\tau)`, L-curve/GCV diagnostics, 2D lifetime-density heatmap
  :math:`S(\tau, \lambda)`, and detected peak table.

Configuration (``settings.toml``)
---------------------------------

Analysis settings are managed via :class:`~pyrate_ta.settings.Settings` and persisted to
``settings.toml`` while preserving comments:

.. code-block:: python

   import pyrate_ta as pr

   pr.load_settings("settings.toml")
   pr.update_settings(n_components=3, irf_mode="Clip", t_min=0.3)
   pr.save_settings("settings.toml")
