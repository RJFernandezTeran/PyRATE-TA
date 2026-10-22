.. _user_guide_extending:

Extending PyRATE-TA
===================

Package Architecture
--------------------

- ``pyrate_ta.models``: Pure-NumPy kinetic model definitions, parameter bounds, reaction-scheme
  text parser, and matrix-exponential propagators.
- ``pyrate_ta.fit``: Variable projection driver, global/target/single-trace engines, constrained
  NNLS amplitude solver, GSB/SE pure positive spectrum recovery, and Tikhonov LDA solver.
- ``pyrate_ta.results``: Reproducible result dataclasses (:class:`~pyrate_ta.results.fits.KineticFit`,
  :class:`~pyrate_ta.results.fits.GlobalFit`, :class:`~pyrate_ta.results.fits.TargetFit`,
  :class:`~pyrate_ta.results.lda_result.LDAResult`).
- ``pyrate_ta.plot``: Qt-free analysis plotters (concentration profiles, compartment diagrams,
  rate-matrix heatmaps, L-curve/LDA maps, residual traces, and live fit monitor bars).
- ``pyrate_ta.io``: Fit session archive (``.prfit``) and steady-state reference spectrum loaders.
- ``pyrate_ta.gui``: PyQt6 graphical application assembled from modular tab mixins.
