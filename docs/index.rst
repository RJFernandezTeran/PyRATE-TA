PyRATE-TA Documentation
=======================

**Kinetic Analysis of Time-Resolved Spectroscopy: Multi-Exponential, Global, Target & Lifetime Density Analysis**

PyRATE-TA is the companion kinetic-modelling and fitting package to
`PyMORGAN <https://github.com/RJFernandezTeran/PyMORGAN>`_. While PyMORGAN owns data loading,
preprocessing, and publication plotting, PyRATE-TA provides Variable Projection (VARPRO) global
and target analysis, rate-matrix (:math:`\mathbf{K}`) scheme compilation, Lifetime Density
Analysis (LDA), and constrained pure positive species spectrum recovery (GSB / SE de-bleaching).

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: 🚀 Installation & Setup
      :link: installation
      :link-type: doc

      Install via ``pip`` / ``uv``, co-develop with PyMORGAN, or run via the standalone
      **PyMORGAN Suite** launcher.

   .. grid-item-card:: 🔗 The PyMORGAN Boundary
      :link: user_guide/boundary
      :link-type: doc

      Understand the clean one-way architecture between ``pymorgan`` (data, preprocessing,
      aesthetics) and ``pyrate_ta`` (kinetic models, solvers, uncertainties).

   .. grid-item-card:: 🧬 Kinetic Models & Rate Matrices
      :link: user_guide/models
      :link-type: doc

      Parallel (DAS), Sequential (EAS), and Target (SAS) models, text-based reaction scheme
      notation, horizontal compartment diagrams, and analytical EMG IRF convolution.

   .. grid-item-card:: ⚡ Global, Target & LDA Fitting
      :link: user_guide/fitting
      :link-type: doc

      Variable projection, coherent-artefact bases, sign-constrained amplitudes, GSB/SE
      de-bleaching, Tikhonov Lifetime Density Analysis (L-curve / GCV / Morozov), and
      ``.prfit`` session archives.

   .. grid-item-card:: 🖥️ Graphical Interface & Settings
      :link: user_guide/gui_and_settings
      :link-type: doc

      Explore the PyQt6 fitting interface, interactive K-matrix editor, 3-group LDA tab,
      and unified ``settings.toml`` configuration.

   .. grid-item-card:: 📚 Full Python API Reference
      :link: api/index
      :link-type: doc

      Auto-generated API documentation for ``pyrate_ta.models``, ``pyrate_ta.fit``,
      ``pyrate_ta.results``, ``pyrate_ta.plot``, ``pyrate_ta.io``, and ``pyrate_ta.settings``.

.. admonition:: Download Complete LaTeX Manual (PDF)
   :class: tip

   In addition to this interactive web documentation, the complete typeset manual is compiled
   from ``docs/*.tex`` and available as a standalone PDF:
   `Download PyRATE-TA Manual (main.pdf) <main.pdf>`_.

Quick Example
-------------

.. code-block:: python

   import pymorgan as pm
   import pyrate_ta as pr

   # 1. Load & preprocess dataset in PyMORGAN
   data = pm.load_1D("scan.pdat", data_type="PDAT")
   data.background_correct(tmin=-20, tmax=-5)

   # 2. Global sequential fit (EAS) with Gaussian IRF in PyRATE-TA
   fit = pr.fit_global(
       data,
       n_components=3,
       model_type="Sequential",
       irf_mode="Gaussian",
       coherent_artifact=True,
   )
   print(fit.summary())

   # 3. Plot Evolution-Associated Spectra (EAS) and concentration profiles
   data.plot_species_spectra(*fit.as_species_args())
   pr.plot_concentrations(fit)

   # 4. Save reproducible fit session (.prfit)
   pr.save_fit("scan_sequential_fit", fit)


Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index
   user_guide/boundary
   user_guide/models
   user_guide/fitting
   user_guide/gui_and_settings
   user_guide/extending

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
   api/models
   api/fit
   api/results
   api/plot_and_io
