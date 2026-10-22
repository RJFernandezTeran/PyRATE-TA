.. _user_guide_gui_and_settings:

Graphical Interface & Settings
==============================

Launching the GUI
-----------------

Launch with ``uv run pyrate-ta-gui`` (or ``python -m pyrate_ta.gui``). The window reuses
PyMORGAN's ``WidgetPlot``, ``PlotControlsPanel``, and theme so the two applications look
identical side by side.

Window Layout
-------------

- **Left Column (Scrollable Control Panel):**
  - **Data:** Load, clear, and two status lamps (one for the dataset, one for per-point noise
    from a sibling ``.pdatn`` file or single-scan spread).
  - **Model Selection:** *Parallel* / *Sequential* / *Target* / *ODE-defined*.
  - **Components:** Lifetime table with initial guesses, lower/upper bounds, fixed checkboxes,
    and add/remove row controls.
  - **IRF and Time-Zero:** IRF mode (``None``, ``Gaussian``, ``Clip``), FWHM, :math:`t_0`, and
    *Fit coherent artefact* checkbox.
  - **Fit Options:** Algorithm (defaults to Trust Region Reflective to honour positivity bounds),
    restrict-to-display-limits, and noise weighting.
  - **Fit Control:** View selector (*Data* / *Fit* / *Residuals*), *Preview*, *FIT DATA*,
    *Reset*, statistic read-out (*SSR* or *red. chi2*), *Copy taus*, *Plot DAS/EAS/SAS*,
    *Plot Conc. Profile*, and Steady-State Reference GSB/SE de-bleaching controls.
- **Plot Area (**\ :math:`3 \times 3` **Synchronised Grid):**
  - Top-left :math:`2 \times 2` block: 2D contour map.
  - Top-right :math:`2 \times 1` column: Rotated kinetic traces (sharing the vertical delay axis
    with the contour map).
  - Bottom-left :math:`1 \times 2` row: Transient spectra (sharing the horizontal probe axis with
    the contour map).
  - Bottom-right cell: *Additional plots and cuts* buttons for multi-cut stand-alone figures.
- **Bottom Panel:** Shared PyMORGAN ``PlotControlsPanel`` (axis limits, colour scale,
  :math:`\operatorname{arcsinh}` toggle, contour levels, time-axis convention, and probe units).

The K-Matrix Editor & Scheme Graph
----------------------------------

- **Define model...** opens the interactive reaction-scheme editor: reaction text notation on the
  left, live horizontal compartment diagram and derived :math:`\mathbf{K}` matrix on the right.
  Clicking **Update K graph** validates the syntax (reporting line numbers on error) and enables
  **OK** once valid.
- **Show kinetic graph** opens a stand-alone compartment diagram of the active model before or
  after fitting.

The LDA Tab
-----------

The **LDA** tab provides interactive Lifetime Density Analysis organised into three side-by-side
control groups:

1. **Regularisation & Solver:** Penalty type (``ridge`` / ``d1`` / ``d2``), alpha selection
   method (``lcurve`` / ``gcv`` / ``morozov`` / ``fixed``), and explicit :math:`\alpha` value.
2. **Lifetime Grid & Options:** Number of log-spaced lifetimes, :math:`[\tau_{\min}, \tau_{\max}]`
   bounds, SVD rank pre-filtering (:math:`K`), non-negative constraint, coherent-artefact
   suppression, and bootstrap iterations.
3. **Actions & Diagnostics:** **Run LDA** button, diagnostic view toggles (integrated amplitude
   spectrum :math:`A(\tau)`, L-curve/GCV curve, 2D heatmap :math:`S(\tau, \lambda)`, and detected
   peak centroid table), and **Export map...** to save as a ``.pdat`` file.

Analysis & LDA Settings (``settings.toml``)
-------------------------------------------

Presentation belongs to :class:`pymorgan.settings.Settings`; PyRATE-TA's own
:class:`~pyrate_ta.settings.Settings` stores analysis-only fields in ``settings.toml``:

.. list-table:: Analysis Settings (``[fit]``, ``[solver]``, ``[paths]``)
   :header-rows: 1
   :widths: 28 18 14 40

   * - Field
     - Default
     - Group
     - Meaning
   * - ``n_components``
     - ``2``
     - ``fit``
     - Components proposed for a new fit
   * - ``model_type``
     - ``Sequential``
     - ``fit``
     - Default model family
   * - ``time_limits``
     - ``[0.2, None]``
     - ``fit``
     - Default fitting window ``[min, max]`` delays
   * - ``irf_mode``
     - ``Gaussian``
     - ``fit``
     - ``None`` / ``Gaussian`` / ``Clip``
   * - ``irf_fwhm``
     - ``None``
     - ``fit``
     - Fixed IRF width; ``None`` fits it
   * - ``t_min``
     - ``None``
     - ``fit``
     - Clip delays below this (mode ``Clip``)
   * - ``ftol``
     - ``1e-8``
     - ``solver``
     - Termination tolerance on the cost
   * - ``xtol``
     - ``1e-8``
     - ``solver``
     - Termination tolerance on the parameters
   * - ``max_iterations``
     - ``2000``
     - ``solver``
     - Cap on optimiser iterations
   * - ``variable_projection``
     - ``True``
     - ``solver``
     - Solve linear amplitudes in closed form
   * - ``report_uncertainties``
     - ``True``
     - ``solver``
     - Report linearised :math:`1\sigma` errors
   * - ``use_noise_weights``
     - ``False``
     - ``solver``
     - Weight residuals by per-point noise :math:`1/\sigma`
   * - ``noise_floor_fraction``
     - ``1e-3``
     - ``solver``
     - Clip :math:`\sigma` below this fraction of the median
   * - ``degeneracy_warn_ratio``
     - ``1.2``
     - ``solver``
     - Warn when two lifetimes come within this ratio
   * - ``default_datadir``
     - ``None``
     - ``paths``
     - Default directory offered by the file dialog

.. list-table:: LDA Default Settings (``[lda]`` Section of ``settings.toml``)
   :header-rows: 1
   :widths: 28 18 54

   * - Field
     - Default
     - Meaning
   * - ``lda_n_lifetimes``
     - ``100``
     - Grid points in the lifetime axis
   * - ``lda_tau_min``
     - ``None``
     - Grid lower bound (``None`` = auto from delay axis)
   * - ``lda_tau_max``
     - ``None``
     - Grid upper bound (``None`` = auto from delay axis)
   * - ``lda_regularisation``
     - ``L-curve``
     - Alpha selection: ``L-curve`` / ``GCV`` / ``Fixed``
   * - ``lda_alpha``
     - ``1e-3``
     - Regularisation strength when ``Fixed`` is selected
   * - ``lda_non_negative``
     - ``False``
     - Constrain lifetime density distribution to :math:`\ge 0`

.. code-block:: python

   import pyrate_ta as pr

   pr.load_settings("settings.toml")
   pr.update_settings(n_components=3, irf_mode="Clip", t_min=0.3)
   pr.save_settings("settings.toml")

In the GUI, **View → Settings...** opens a dual-tab editor containing both **Analysis
(PyRATE-TA)** and **Aesthetics (PyMORGAN)** so both packages' settings can be adjusted and saved
live from one dialog.
