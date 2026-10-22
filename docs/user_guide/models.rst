.. _user_guide_models:

Kinetic Models & Rate Matrices
==============================

The Bilinear Model
------------------

Every kinetic model in PyRATE-TA produces a concentration matrix :math:`\mathbf{C}` of shape
:math:`N_{\text{delays}} \times N_{\text{components}}`. The transient data matrix
:math:`\mathbf{D}` is factored as:

.. math::

   \mathbf{D} \;\approx\; \mathbf{C}\,\mathbf{S}^{\mathsf{T}}

with :math:`\mathbf{S}` of shape :math:`N_{\text{pixels}} \times N_{\text{components}}` holding
the component spectra.

.. list-table:: Model Families and Resulting Spectra
   :header-rows: 1
   :widths: 22 38 40

   * - ``ModelType``
     - Scheme
     - Resulting Spectra
   * - ``Parallel``
     - Independent exponential decays
     - **DAS** (Decay-Associated Spectra)
   * - ``Sequential``
     - Unbranched chain :math:`A \to B \to C \to \cdots`
     - **EAS** (Evolution-Associated Spectra)
   * - ``Target``
     - General rate matrix :math:`\mathbf{K}` with branching/equilibria
     - **SAS** (Species-Associated Spectra)

Rate Matrices (:math:`\mathbf{K}`)
----------------------------------

A target model is defined by its rate matrix :math:`\mathbf{K}`, where off-diagonal element
:math:`K_{ij}` is the rate constant from compartment :math:`j` to compartment :math:`i` and each
diagonal element is the negative sum of all rates leaving that compartment:

.. math::

   \frac{\mathrm{d}\mathbf{c}(t)}{\mathrm{d}t} = \mathbf{K}\,\mathbf{c}(t), \qquad
   \mathbf{c}(0) = \mathbf{c}_0 \quad\Longrightarrow\quad
   \mathbf{c}(t) = \exp(\mathbf{K} t)\,\mathbf{c}_0

Writing a Reaction Scheme in Text Notation
------------------------------------------

Target models can be selected from :data:`~pyrate_ta.models.schemes.TARGET_SCHEMES` or written
directly as chemical reactions using :func:`~pyrate_ta.models.scheme_text.scheme_from_text`:

.. code-block:: text

   A  ->  B   : k1          # irreversible step
   B <->  C   : k2, k3      # reversible equilibrium (forward, backward)
   C  ->      : k4          # decay to ground state
   init A = 1               # initial population

.. code-block:: python

   import pyrate_ta as pr

   model = pr.make_model("Target", scheme="A_eq_B_to_C")
   C = model.concentrations(t, taus=[2.0, 5.0, 20.0, 200.0], irf_fwhm=0.15)

Instrument Response Function (IRF)
----------------------------------

For exponential models convolved with a Gaussian IRF of width :math:`\sigma` centred at
:math:`t_0`, PyRATE-TA evaluates the analytical Exponentially Modified Gaussian (EMG):

.. math::

   \left(g \ast e^{-t/\tau}\right)(t)
   = \tfrac{1}{2}\exp\!\left(\frac{\sigma^{2}}{2\tau^{2}} - \frac{t-t_0}{\tau}\right)
   \operatorname{erfc}\!\left(\frac{1}{\sqrt{2}}\left(\frac{\sigma}{\tau} - \frac{t-t_0}{\sigma}\right)\right)
