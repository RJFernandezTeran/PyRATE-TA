.. _user_guide_boundary:

The PyMORGAN Boundary
=====================

Ownership of Concerns
---------------------

.. list-table:: Division of Responsibilities Between PyMORGAN and PyRATE-TA
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
   * - Background, chirp, solvent, SVD
     - **PyMORGAN**
     - ``Dataset1D.background_correct``, ``apply_chirp_correction``
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
   * - Analysis-only plots (C(t), K-graph, L-curve)
     - **PyRATE-TA**
     - :mod:`pyrate_ta.plot`

Three Architectural Rules
-------------------------

1. **The dependency is strictly one-way:** ``import pymorgan`` inside PyRATE-TA is expected;
   ``import pyrate_ta`` inside PyMORGAN is never used. ``Dataset1D.plot_species_spectra`` accepts
   plain NumPy arrays via ``*fit.as_species_args()``.
2. **No duplicate file readers:** All raw experimental file loading goes through
   ``pm.load_1D(path, data_type=...)``.
3. **One aesthetics system:** PyRATE-TA calls ``pm.apply_style()`` and reads ``pm.get_settings()``
   for typography, colourmaps, and unit formatting.
