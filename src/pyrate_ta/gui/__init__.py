"""PyQt6 application.

All widgets, layouts, containers, tab pages and controls MUST be declared in Qt
Designer ``.ui`` files. Python code here binds signals, slots,

dynamic styles and data states to widgets defined in the ``.ui`` file, so the
interface stays editable in Qt Designer.

``MainWindow`` is assembled from per-tab mixins in :mod:`pyrate_ta.gui.tabs`;
``main_window.py`` keeps only construction, wiring, menus, exports and theming.
Reuse ``pymorgan.gui.theme`` rather than forking the dark palette, so both
applications look identical side by side.

``app.py`` must not import ``pyrate`` or ``pymorgan`` at module level, so the
splash screen can appear before the pipelines are paid for.
"""

from __future__ import annotations

__all__: list[str] = []
