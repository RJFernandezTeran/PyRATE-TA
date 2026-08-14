"""Import/export of fit sessions.

**Not spectroscopy file formats.** Raw data readers (``PDAT``, ``P2DAT``, MESS
directories, ...) belong to PyMORGAN's loader registry; load through
``pymorgan.load_1D`` / ``load_2D``. A format PyRATE needs and PyMORGAN lacks is
contributed upstream, or registered at runtime with
``pymorgan.register_loader`` -- never forked into this package.

What lives here is the analysis session: the model, its parameters, the
fixed/free masks, the ranges used and the results, written so a fit can be
reopened. The format is PyRATE's own ``.prfit`` -- a compressed ``.npz``
carrying both the arrays and a JSON reproducibility payload, so the numbers can
never be separated from the record of how they were produced.
"""

from __future__ import annotations

from .session import (
    FORMAT_VERSION,
    LoadedFit,
    default_session_path,
    load_fit,
    save_fit,
)

__all__ = [
    "save_fit",
    "load_fit",
    "LoadedFit",
    "default_session_path",
    "FORMAT_VERSION",
]
