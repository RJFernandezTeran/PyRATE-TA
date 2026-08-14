"""Per-tab mixins making up :class:`~pyrate_ta.gui.main_window.MainWindow`.

Each mixin is a plain method container with no state of its own: ``self`` is
always the main window, so every widget lookup resolves against the widgets
declared in ``main_window.ui``.
"""

from __future__ import annotations

from .data import DataTabMixin
from .fitting import FitTabMixin
from .lda import LDATabMixin
from .model import ModelTabMixin

__all__ = ["DataTabMixin", "ModelTabMixin", "FitTabMixin", "LDATabMixin"]
