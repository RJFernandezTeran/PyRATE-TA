.. _user_guide_models:

Kinetic Models & Rate Matrices
==============================

The Bilinear Model
------------------

Every model in PyRATE-TA produces a concentration matrix :math:`\mathbf{C}` of shape
:math:`N_{\text{delays}} \times N_{\text{components}}`. The data matrix is then bilinear:

.. math::

   \mathbf{D} \;\approx\; \mathbf{C}\,\mathbf{S}^{\mathsf{T}}

with :math:`\mathbf{S}` of shape :math:`N_{\text{pixels}} \times N_{\text{components}}` holding
the component spectra. The kinetics enter only through :math:`\mathbf{C}`, and the spectra only
through :math:`\mathbf{S}`; this separation is what makes global analysis tractable via Variable
Projection (:ref:`user_guide_fitting`).

Model Families
--------------

.. list-table:: Model Families and the Meaning of the Resulting Spectra
   :header-rows: 1
   :widths: 22 38 40

   * - ``ModelType``
     - Scheme
     - Spectra Obtained
   * - ``Parallel``
     - Independent exponential decays
     - **DAS** (Decay-Associated Spectra)
   * - ``Sequential``
     - Unbranched chain :math:`A \to B \to C \to \cdots`
     - **EAS** (Evolution-Associated Spectra)
   * - ``Target``
     - General :math:`\mathbf{K}` with branching / equilibria
     - **SAS** (Species-Associated Spectra)

The distinction is fundamental:

- A **parallel** model makes no claim about connectivity, and its **DAS** are linear combinations
  of the true species spectra.
- A **sequential** model assumes a single irreversible chain, and its **EAS** are the spectra of
  the successive evolving states.
- Only a **target** model with an explicitly justified rate matrix :math:`\mathbf{K}` yields the
  true **SAS** of actual chemical species.

Parallel and Sequential Models
------------------------------

For a parallel model with lifetimes :math:`\tau_i`:

.. math::

   C_{ni} = \exp\!\left(-t_n / \tau_i\right)

For a sequential model, the concentration profiles follow the analytic solution of the unbranched
chain (a sum of exponentials weighted by :math:`\prod_{j \ne i}(\tau_i^{-1} - \tau_j^{-1})^{-1}`).
Both are special cases of the general rate-matrix propagator below.

Rate Matrices (:math:`\mathbf{K}`) & Degeneracy Handling
--------------------------------------------------------

A target model is defined by its rate matrix :math:`\mathbf{K}`, with off-diagonal element
:math:`K_{ij}` the rate from compartment :math:`j` to compartment :math:`i` and the diagonal the
negative sum of all rates leaving that compartment. The populations obey:

.. math::

   \frac{\mathrm{d}\mathbf{c}(t)}{\mathrm{d}t} = \mathbf{K}\,\mathbf{c}(t), \qquad
   \mathbf{c}(0) = \mathbf{c}_0 \quad\Longrightarrow\quad
   \mathbf{c}(t) = \exp(\mathbf{K} t)\,\mathbf{c}_0

:math:`\mathbf{K}` is built explicitly and :math:`\mathbf{C}` is obtained by eigendecomposition
where :math:`\mathbf{K}` is diagonalisable, and by a matrix-exponential propagator otherwise.

- **Exactly degenerate lifetimes:** When two rates coincide, :math:`\mathbf{K}` is defective and
  the eigenvector matrix is singular, even though the kinetics are smooth (e.g.
  :math:`c_B = k t e^{-kt}` for :math:`A \to B \to` with equal rates). PyRATE-TA splits exact
  degeneracy via an antisymmetric diagonal perturbation of relative size :math:`10^{-6}` so an
  optimiser can pass smoothly through equal lifetimes without aborting.
- **Near-degenerate lifetimes:** When two fitted lifetimes approach within
  ``degeneracy_warn_ratio`` (default ``1.2``), the eigenvector matrix becomes ill-conditioned
  (manifesting as compensating component spectra of opposite sign). PyRATE-TA logs an explicit
  warning when this threshold is crossed.

Predefined Schemes & Writing Reaction Schemes in Text
-----------------------------------------------------

Branched schemes are kept in :data:`pyrate_ta.models.TARGET_SCHEMES` (including equilibria
``A_eq_B``, chains with equilibria ``A_eq_B_to_C`` / ``A_eq_B_eq_C``, branchings
``branch_to_common_D``, and parallel chains ``two_chains_decaying_ends``):

.. code-block:: python

   import pyrate_ta as pr

   model = pr.make_model("Target", scheme="A_eq_B_to_C")
   C = model.concentrations(t, taus=[2.0, 5.0, 20.0, 200.0], irf_fwhm=0.15)

Custom target models are written directly as chemical reactions using
:func:`~pyrate_ta.models.scheme_text.scheme_from_text`:

.. code-block:: text

   A  ->  B   : k1          # irreversible step
   B <->  C   : k2, k3      # equilibrium (forward, backward)
   C  ->      : k4          # decay to ground state
   init A = 1               # optional initial populations

Every reaction contributes :math:`-k` to the reactant's diagonal and :math:`+k` to the product's
row, guaranteeing mass conservation:

- **Species are named:** ``S1``, ``ICT``, ``T1`` appear on the compartment diagram, in the
  lifetime table, and in the fit result.
- **Rates are named and shareable:** Reusing the same rate symbol on two arrows constrains both
  channels to share a single fitted rate parameter.
- **Arbitrary topology:** Chains, branches, equilibria, cycles, disconnected pools, and
  non-decaying photoproducts are supported, and :func:`~pyrate_ta.models.scheme_text.scheme_to_text`
  round-trips the scheme back to text.

Scheme Diagrams
---------------

:func:`pyrate_ta.plot_scheme` draws the horizontal compartment diagram of any model, scheme, or
fit result: one node per compartment, one arrow per rate, the ground state positioned below, and
equilibria drawn as opposed arrows labelled with rate symbols (:math:`k_1, k_2, \dots`) or
fitted lifetimes (``rate_labels="value"``).

Instrument Response Function (IRF)
----------------------------------

The instrument response is convolved into :math:`\mathbf{C}`, not into the data. For exponential
models convolved with a Gaussian IRF of width :math:`\sigma` centred at :math:`t_0`, PyRATE-TA
evaluates the analytical Exponentially Modified Gaussian (EMG):

.. math::

   \left(g \ast e^{-t/\tau}\right)(t)
   = \tfrac{1}{2}\exp\!\left(\frac{\sigma^{2}}{2\tau^{2}} - \frac{t-t_0}{\tau}\right)
   \operatorname{erfc}\!\left(\frac{1}{\sqrt{2}}\left(\frac{\sigma}{\tau} - \frac{t-t_0}{\sigma}\right)\right)

The three IRF handling modes are ``None`` (no IRF), ``Gaussian`` (analytical EMG), and ``Clip``
(drop delays below ``t_min``).
