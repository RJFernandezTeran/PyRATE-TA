.. _user_guide_fitting:

Global, Target & Lifetime Density Analysis (LDA)
================================================

Variable Projection (VARPRO)
----------------------------

In the bilinear model :math:`\mathbf{D} \approx \mathbf{C}\,\mathbf{S}^{\mathsf{T}}`, the
component spectra :math:`\mathbf{S}` enter linearly. For any fixed set of non-linear parameters
(lifetimes :math:`\boldsymbol{\tau}`, time-zero :math:`t_0`, and IRF width), the optimal spectra
are obtained in closed form:

.. math::

   \mathbf{S}^{\mathsf{T}} = \left(\mathbf{C}^{\mathsf{T}}\mathbf{C}\right)^{-1}\mathbf{C}^{\mathsf{T}}\mathbf{D}

Only the non-linear parameters enter the iterative optimiser, reducing a 500-pixel, 3-component
fit from ~1500 parameters to 3 and dramatically improving numerical conditioning.

Fitting Engines
---------------

.. list-table:: PyRATE-TA Fitting Engines
   :header-rows: 1
   :widths: 24 34 42

   * - Engine
     - Input Data
     - Output
   * - :func:`~pyrate_ta.fit.engines.fit_kinetics`
     - Single kinetic trace :math:`(t, y)`
     - Lifetimes & amplitudes (:class:`~pyrate_ta.results.fits.KineticFit`)
   * - :func:`~pyrate_ta.fit.engines.fit_global`
     - Full :math:`\mathbf{D}`, Parallel or Sequential
     - Lifetimes, **DAS** or **EAS** (:class:`~pyrate_ta.results.fits.GlobalFit`)
   * - :func:`~pyrate_ta.fit.engines.fit_target`
     - Full :math:`\mathbf{D}`, fixed :math:`\mathbf{K}` topology
     - Rate constants, **SAS** (:class:`~pyrate_ta.results.fits.TargetFit`)
   * - :func:`~pyrate_ta.fit.lda.solve_lda`
     - Full :math:`\mathbf{D}`, dense :math:`\tau` grid
     - Lifetime density map (:class:`~pyrate_ta.results.lda_result.LDAResult`)

.. code-block:: python

   import pyrate_ta as pr

   # Global analysis
   fit = pr.fit_global(data, n_components=3, model_type="Sequential")
   print(fit.taus, fit.tau_err)
   data.plot_species_spectra(*fit.as_species_args())

   # Target analysis
   tfit = pr.fit_target(data, scheme="A_eq_B_to_C", taus=[2.0, 5.0, 20.0, 200.0])

   # Lifetime Density Analysis (LDA)
   lda = pr.fit.lda.solve_lda(data, n_taus=100, alpha="auto", non_negative=True)

Coherent Artefact Basis
-----------------------

Setting ``coherent_artifact=True`` appends three unit-normalised columns to the design matrix:
the Gaussian IRF and its first two derivatives (:math:`g(t), g'(t), g''(t)`), absorbing
cross-phase modulation and two-photon absorption around :math:`t = 0` without distorting the
shortest kinetic lifetime :math:`\tau_1`.

Lifetime Density Analysis (LDA)
-------------------------------

Instead of fitting :math:`N` discrete lifetimes, LDA solves for a continuous distribution of
amplitudes over a dense, logarithmically spaced grid of fixed lifetimes via Tikhonov
regularisation:

.. math::

   \min_{\mathbf{S}^{\mathsf{T}}}
   \left\lVert \mathbf{W} \odot \left(\mathbf{D} - \mathbf{C}(\boldsymbol{\tau}_{\text{grid}})\,
   \mathbf{S}^{\mathsf{T}}\right)\right\rVert^2
   + \alpha^2 \left\lVert \mathbf{L}\,\mathbf{S}^{\mathsf{T}}\right\rVert^2

- **Regularisation operators:** Identity (``ridge``, :math:`\mathbf{L} = \mathbf{I}`), first
  difference (``d1``, :math:`\mathbf{L} = \mathbf{D}_1`), and second difference (``d2``,
  :math:`\mathbf{L} = \mathbf{D}_2`, default).
- **Automatic** :math:`\alpha` **selection:** L-curve maximum Menger curvature (``lcurve``),
  Generalised Cross-Validation (``gcv``), or Morozov discrepancy principle (``morozov``).

Steady-State Spectra & Pure Positive Species Recovery (GSB / SE)
----------------------------------------------------------------

In transient absorption, the measured difference spectrum is a superposition of Excited-State
Absorption (ESA), Ground-State Bleach (GSB), and Stimulated Emission (SE):

.. math::

   \Delta A(t, \lambda) = S_{\mathrm{ESA}}(t, \lambda) - a(t)\, \sigma_A(\lambda) - b(t)\, F(\lambda)

When steady-state absorption (:math:`\sigma_A`) and/or emission (:math:`F`) spectra are provided,
:mod:`pyrate_ta.fit.groundstate` extracts pure positive species absorption spectra
:math:`S_{\mathrm{abs}}(\lambda) \ge 0` via constrained non-negative least squares (NNLS):

.. math::

   \min_{a, b \ge 0} \int_{\lambda \in \Lambda_{\mathrm{neg}}} \left| S_{\mathrm{diff}}(\lambda) + a\, \sigma_A(\lambda) + b\, F(\lambda) \right|^2 \mathrm{d}\lambda
   \quad\text{subject to}\quad
   S_{\mathrm{abs}}(\lambda) \ge 0 \;\forall \lambda
