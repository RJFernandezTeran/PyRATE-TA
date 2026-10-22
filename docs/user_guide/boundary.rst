.. _user_guide_boundary:

The PyMORGAN Boundary
=====================

Ownership of Concerns
---------------------

Most architectural questions of the form *"where does this code belong?"* are answered by the
table below:

.. list-table:: Ownership of Each Concern Across PyMORGAN and PyRATE-TA
   :header-rows: 1
   :widths: 36 20 44

   * - Concern
     - Owner
     - Entry Point
   * - File formats & loader registry
     - **PyMORGAN**
     - ``pm.load_1D``, ``pm.register_loader``
   * - Dataset objects
     - **PyMORGAN**
     - ``pm.Dataset1D``, ``pm.Dataset2D``
   * - Background, chirp, solvent, SVD, stitching
     - **PyMORGAN**
     - ``Dataset1D.background_correct``, ``apply_chirp_correction``, ...
   * - Contour / spectra / kinetics plots
     - **PyMORGAN**
     - ``Dataset1D.plot_*``
   * - Species-spectra rendering
     - **PyMORGAN**
     - ``Dataset1D.plot_species_spectra``
   * - Aesthetics, colourmaps, unit labels
     - **PyMORGAN**
     - ``pm.Settings``, ``pm.apply_style()``
   * - Kinetic models & rate matrices (:math:`\mathbf{K}`)
     - **PyRATE-TA**
     - :mod:`pyrate_ta.models`
   * - Fitting engines (Global, Target, LDA)
     - **PyRATE-TA**
     - :mod:`pyrate_ta.fit`
   * - Fit results, uncertainties & sessions
     - **PyRATE-TA**
     - :mod:`pyrate_ta.results`, :mod:`pyrate_ta.io`
   * - Analysis-only plots (:math:`\mathbf{c}(t)`, :math:`\mathbf{K}`-graph, L-curve)
     - **PyRATE-TA**
     - :mod:`pyrate_ta.plot`

Three Architectural Rules
-------------------------

1. **The dependency is strictly one-way:** ``import pymorgan`` inside PyRATE-TA is expected;
   ``import pyrate_ta`` inside PyMORGAN is a design error. PyMORGAN's ``plot_species_spectra``
   deliberately takes plain arrays (``Sfit``, ``Taus``, ``TauErr``, ``isFixTau``, ``modelType``)
   rather than a PyRATE-TA result object so the dependency arrow never reverses; the adapter
   (``fit.as_species_args()``) lives on the PyRATE-TA side. ``scripts/run_checks.py`` asserts
   this boundary.
2. **No duplicate file readers:** Loading goes through ``pm.load_1D(path, data_type=...)``, which
   resolves the format through PyMORGAN's registry and automatically picks up every format
   PyMORGAN supports.
3. **One aesthetics system:** Call ``pm.apply_style()`` and read ``pm.get_settings()`` for label
   style, colourmap, time-axis convention, and figure sizes. PyRATE-TA's own
   :class:`~pyrate_ta.settings.Settings` holds analysis-only fields.

A Minimal Session
-----------------

.. code-block:: python

   import pymorgan as pm
   import pyrate_ta as pr

   data = pm.load_1D("scan.pdat", data_type="PDAT")  # PyMORGAN loads
   data.background_correct(tmin=-20, tmax=-5)        # PyMORGAN processes
   fit  = pr.fit_global(data, n_components=3)        # PyRATE-TA fits
   data.plot_species_spectra(*fit.as_species_args()) # PyMORGAN plots

Units & Trace Extraction
------------------------

- **Units:** Signal data is in :math:`\text{mOD}`, delays are in the dataset's own time unit, and
  the probe axis is in its native spectral unit. Fitted lifetimes are reported in the dataset's
  time unit and are never silently converted.
- **Trace extraction:** For a headless single-trace fit, slice the dataset directly
  (``data.Z[:, idx, detector]``) or obtain extracted traces from ``Dataset1D.plot_kinetics``,
  which returns ``(ax, t, Y)``.
