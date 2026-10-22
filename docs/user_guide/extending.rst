.. _user_guide_extending:

Extending PyRATE-TA
===================

Where New Code Goes
-------------------

- ``pyrate_ta/models/``: Kinetic model definitions. A model owns its parameter vector layout, its
  bounds, and the mapping from parameters to :math:`\mathbf{C}`. Models are pure NumPy and must
  be importable without Qt.
- ``pyrate_ta/fit/``: Solvers, one module per engine, plus the shared driver that handles
  parameter packing, fixed/free masks, and convergence reporting. ``lda.py`` is the regularised
  LDA solver; ``groundstate.py`` converts difference spectra to pure positive absorption spectra;
  ``constrained.py`` provides the sign-mask amplitude solver.
- ``pyrate_ta/results/``: Result dataclasses and their serialisation, carrying the
  reproducibility provenance payload. ``lda_result.py`` is the companion to ``fit/lda.py``.
- ``pyrate_ta/plot/``: Only what PyMORGAN cannot express: residual matrices, concentration
  profiles :math:`\mathbf{c}(t)`, :math:`\mathbf{K}`-matrix / compartment diagrams,
  lifetime-density maps, fit-monitor bars, and kinetic/spectral traces with residuals.
- ``pyrate_ta/io/``: Import and export of fit sessions (``.prfit``) and steady-state reference
  spectra — never raw spectroscopy formats, which belong to PyMORGAN.
- ``pyrate_ta/gui/``: The PyQt6 application assembled from per-tab mixins in ``gui/tabs/``
  (``DataTabMixin``, ``ModelTabMixin``, ``FittingTabMixin``, ``LDATabMixin``).

Adding a Model, Public API Export, or GUI Panel
-----------------------------------------------

1. **Adding a model:** Build :math:`\mathbf{K}` explicitly and reuse the shared propagator
   (:mod:`pyrate_ta.models.propagator`) so the degeneracy guard applies automatically. Add a
   synthetic-recovery unit test asserting that known parameters are recovered within tolerance.
2. **Adding to the lazy public API:** Add the symbol to ``_EXPORTS`` and ``__all__`` (plus the
   ``TYPE_CHECKING`` block) in ``src/pyrate_ta/__init__.py``.
3. **Adding a GUI panel:** Declare widgets in ``src/pyrate_ta/gui/main_window.ui`` using Qt
   Designer (``uv run pyrate-ta-edit-gui``) and bind them defensively from a mixin in
   ``src/pyrate_ta/gui/tabs/``. Wrap long-running operations in ``busy_guard``.
