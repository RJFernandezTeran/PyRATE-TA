.. _user_guide_fitting:

Global, Target & Lifetime Density Analysis (LDA)
================================================

Variable Projection (VARPRO)
----------------------------

In the bilinear model :math:`\mathbf{D} \approx \mathbf{C}\,\mathbf{S}^{\mathsf{T}}`, the
component spectra :math:`\mathbf{S}` enter linearly. For any fixed set of non-linear parameters
(lifetimes :math:`\boldsymbol{\tau}`, time-zero :math:`t_0`, and IRF width), the best-fit spectra
are available in closed form:

.. math::

   \mathbf{S}^{\mathsf{T}} = \left(\mathbf{C}^{\mathsf{T}}\mathbf{C}\right)^{-1}\mathbf{C}^{\mathsf{T}}\mathbf{D}

Only the non-linear parameters are passed to the optimiser. For a 500-pixel, 3-component fit,
Variable Projection reduces the parameter vector from ~1500 entries to 3 and conditions the
problem far better than joint amplitude–lifetime optimisation. In practice, the linear solve is
evaluated via QR / SVD least-squares rather than forming normal equations.

Fitting Engines
---------------

.. list-table:: The Four Fitting Engines in PyRATE-TA
   :header-rows: 1
   :widths: 24 34 42

   * - Engine
     - Input Data
     - Output
   * - :func:`~pyrate_ta.fit.engines.fit_kinetics`
     - Single kinetic trace :math:`(t, y)`
     - Lifetimes & amplitudes (:class:`~pyrate_ta.results.fits.KineticFit`)
   * - :func:`~pyrate_ta.fit.engines.fit_global`
     - Full :math:`\mathbf{D}`, free :math:`\mathbf{K}`
     - Lifetimes, **DAS** or **EAS** (:class:`~pyrate_ta.results.fits.GlobalFit`)
   * - :func:`~pyrate_ta.fit.engines.fit_target`
     - Full :math:`\mathbf{D}`, fixed connectivity
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

   # Preview (evaluates linear spectra at fixed lifetimes without optimising)
   prev = pr.preview_global(data, taus=[1.0, 10.0, 100.0])

   # Target analysis
   tfit = pr.fit_target(data, scheme="A_eq_B_to_C", taus=[2.0, 5.0, 20.0, 200.0])

   # Single-trace fit
   kfit = pr.fit_kinetics(t, y, n_components=2)

   # Lifetime Density Analysis (LDA)
   lda = pr.fit.lda.solve_lda(data, n_taus=100, alpha="auto", non_negative=True)

The Coherent Artefact Basis
---------------------------

Around :math:`t = 0`, ultrafast measurements often contain cross-phase modulation (XPM),
two-photon absorption, and stimulated Raman signals from the solvent and cell windows. Setting
``coherent_artifact=True`` appends three unit-normalised columns to the design matrix — the
Gaussian IRF and its first two derivatives (Kovalenko *et al.*, *Phys. Rev. A* **59**, 2369,
1999):

.. math::

   g(t), \qquad g'(t), \qquad g''(t)

Their amplitudes are solved per probe pixel via Variable Projection so the artefact is absorbed
into its own basis spectra (``fit.artefact_spectra``) rather than distorting the shortest kinetic
lifetime :math:`\tau_1`. ``fit.as_species_args()`` returns only the genuine kinetic species.

Constrained Amplitudes and Sign Masks
-------------------------------------

When physical sign constraints are imposed on species spectra, Variable Projection solves a
bounded linear least-squares problem per probe pixel (:mod:`pyrate_ta.fit.constrained`):

.. math::

   \min_{\mathbf{S}_p}\, \lVert \mathbf{C}_w\,\mathbf{S}_p - \mathbf{d}_{w,p}\rVert^2
   \quad\text{subject to}\quad
   \mathbf{l} \le \mathbf{S}_p \le \mathbf{u}

Controlled by a ``sign_mask`` array of length :math:`N_c`: ``+1`` enforces :math:`S_p \ge 0`,
``-1`` enforces :math:`S_p \le 0`, and ``0`` is unconstrained.

Steady-State Spectra & Pure Positive Species Recovery (GSB and SE)
------------------------------------------------------------------

In transient absorption, the observed differential absorbance :math:`\Delta A(t, \lambda)` is a
linear superposition of three photophysical contributions:

.. math::

   \Delta A(t, \lambda) = S_{\mathrm{ESA}}(t, \lambda) - a(t)\, \sigma_A(\lambda) - b(t)\, F(\lambda)

where :math:`S_{\mathrm{ESA}} \ge 0` is Excited-State Absorption, :math:`\sigma_A \ge 0` is
steady-state ground-state absorption (producing negative **Ground State Bleach**, GSB), and
:math:`F \ge 0` is steady-state fluorescence (producing negative **Stimulated Emission**, SE).

When steady-state absorption (:math:`\sigma_A`) and/or emission (:math:`F`) reference spectra are
provided, :mod:`pyrate_ta.fit.groundstate` extracts pure positive species absorption spectra
:math:`S_{\mathrm{abs}}(\lambda) \ge 0` using constrained non-negative least squares (NNLS):

.. math::

   \min_{a, b \ge 0} \int_{\lambda \in \Lambda_{\mathrm{neg}}} \left| S_{\mathrm{diff}}(\lambda) + a\, \sigma_A(\lambda) + b\, F(\lambda) \right|^2 \mathrm{d}\lambda

subject to the physical positivity constraint:

.. math::

   S_{\mathrm{abs}}(\lambda) = S_{\mathrm{diff}}(\lambda) + a\, \sigma_A(\lambda) + b\, F(\lambda) \ge 0 \quad \forall \lambda

In the GUI, checking **[✓] Recover pure positive spectra (GSB / SE)** switches both the embedded
preview and detached **Plot DAS/EAS/SAS** pop-out from differential spectra to reconstructed
positive absorption spectra alongside dashed :math:`-a\,\sigma_A` and :math:`-b\,F` reference
baselines.

Lifetime Density Analysis (LDA)
-------------------------------

Instead of fitting :math:`N` discrete lifetimes, LDA solves for a continuous distribution of
amplitudes over a dense, logarithmically spaced grid of fixed lifetimes via Tikhonov
regularisation (:mod:`pyrate_ta.fit.lda`):

.. math::

   \min_{\mathbf{S}^{\mathsf{T}}}
   \left\lVert \mathbf{W} \odot \left(\mathbf{D} - \mathbf{C}(\boldsymbol{\tau}_{\text{grid}})\,
   \mathbf{S}^{\mathsf{T}}\right)\right\rVert^2
   + \alpha^2 \left\lVert \mathbf{L}\,\mathbf{S}^{\mathsf{T}}\right\rVert^2

- **Regularisation operators:** Identity (``ridge``, :math:`\mathbf{L} = \mathbf{I}`), first
  difference (``d1``, :math:`\mathbf{L} = \mathbf{D}_1`, penalising slope), and second difference
  (``d2``, :math:`\mathbf{L} = \mathbf{D}_2`, penalising curvature; default).
- **Automatic** :math:`\alpha` **selection (``alpha="auto"``):**
  - ``lcurve``: Corner of the log-residual vs. log-solution-norm curve via maximum Menger
    curvature (Hansen, 1992; default).
  - ``gcv``: Generalised Cross-Validation (Golub *et al.*, 1979).
  - ``morozov``: Morozov discrepancy principle matching the pre-time-zero noise level.
- **Additional options:** ``non_negative`` (per-pixel NNLS), ``svd_components`` (top-:math:`K`
  SVD pre-filtering), ``n_bootstraps`` (Monte Carlo residual resampling), ``find_peaks``
  (automatic peak centroid detection on :math:`A(\tau) = \sum_p |S_{p,i}|`), and
  ``coherent_artifact``.
- **Export:** ``lda_result.to_pdat(path)`` exports the 2D lifetime-density map to PyMORGAN's
  ``.pdat`` format.

Weighting, Reduced :math:`\chi^2`, Uncertainties & Session Files (``.prfit``)
-----------------------------------------------------------------------------

- **Noise weighting &** :math:`\chi^2_\nu`: With weights :math:`w = 1/\sigma` from
  ``Dataset1D.noise_array()``, PyRATE-TA reports the true reduced chi-squared:

  .. math::

     \chi^{2}_{\nu} = \frac{1}{N-p}\sum_{n,k}
     \left(\frac{D_{nk} - \left(\mathbf{C}\mathbf{S}^{\mathsf{T}}\right)_{nk}}{\sigma_{nk}}\right)^{2}

  where :math:`p` counts both non-linear parameters and the :math:`N_p N_c` linear amplitudes
  (with the non-linear-only denominator also recorded on ``FitStatistic.value_nonlinear_dof``).
- **Reproducible Fit Sessions (**\ ``.prfit``\ **):** :func:`pyrate_ta.save_fit` writes a
  compressed archive containing all fitted arrays and a JSON provenance payload (model family,
  scheme text, initial guesses, bounds, fixed mask, delay/probe limits, and convergence report):

  .. code-block:: python

     path = pr.save_fit("scan_fit", fit)     # -> scan_fit.prfit
     prev = pr.load_fit("scan_fit.prfit")
     print(prev.summary())
     print(prev.taus)
