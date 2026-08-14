"""A draggable crosshair for choosing which cuts to show.

The contour panel carries a vertical and a horizontal guide line. Their
position *is* the selection: the vertical line picks the probe position the
kinetics panel plots, the horizontal one the delay the spectra panel plots.
Dragging either updates both panels, so the traces on screen are always the
ones being pointed at rather than an arbitrary default.

Framework-agnostic on purpose -- it talks to a Matplotlib canvas through
``mpl_connect`` only and knows nothing about Qt -- so it can be exercised with
a headless Agg canvas, exactly like PyMORGAN's :class:`ContourPicker`, which
this deliberately does not duplicate: that one collects clicks and finishes,
this one is a persistent, draggable selection.
"""

from __future__ import annotations

from collections.abc import Callable

from ..log import get_logger

logger = get_logger(__name__)

_LINE_KW = {"color": "0.2", "lw": 1.0, "ls": "--", "alpha": 0.9, "zorder": 5}
#: How close (in axes fraction) the pointer must be to grab a line rather than
#: jump both to the click position.
_GRAB_TOLERANCE = 0.03


class Crosshair:
    """Two draggable guide lines on one axis.

    Parameters
    ----------
    canvas : matplotlib canvas
        Host of ``ax``; used for ``mpl_connect`` and ``draw_idle``.
    ax : matplotlib.axes.Axes
        The contour axis to draw on.
    on_change : callable
        ``on_change(x, y)`` while dragging, with the data coordinates of the
        crosshair. Called on every motion event, so it should be cheap; the
        expensive redraw belongs in :paramref:`on_release`.
    on_release : callable, optional
        ``on_release(x, y)`` once, when the drag ends.
    x, y : float, optional
        Initial position. Defaults to the centre of the current view.
    """

    def __init__(
        self,
        canvas,
        ax,
        on_change,
        on_release=None,
        *,
        x=None,
        y=None,
        x_values=None,
        y_values=None,
        line_kw=None,
    ):
        self.canvas = canvas
        self.ax = ax
        self.on_change: Callable[[float, float], None] = on_change
        self.on_release = on_release
        self.x_values = x_values
        self.y_values = y_values

        style = dict(_LINE_KW)
        style.update(line_kw or {})
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        self.x = float(x if x is not None else 0.5 * (x0 + x1))
        self.y = float(y if y is not None else 0.5 * (y0 + y1))

        self.vline = ax.axvline(self.x, **style)
        self.hline = ax.axhline(self.y, **style)
        self._dragging = None  # None, "v", "h" or "both"
        self._cids = []
        self.connect()

    # ------------------------------------------------------------------ #
    #                            Connection                              #
    # ------------------------------------------------------------------ #
    def connect(self):
        """Start listening for mouse and keyboard events."""
        if self._cids:
            return self
        self._cids = [
            self.canvas.mpl_connect("button_press_event", self._on_press),
            self.canvas.mpl_connect("motion_notify_event", self._on_motion),
            self.canvas.mpl_connect("button_release_event", self._on_release),
            self.canvas.mpl_connect("key_press_event", self._on_key_press),
        ]
        return self

    def disconnect(self):
        """Stop listening and drop the guide lines."""
        for cid in self._cids:
            try:
                self.canvas.mpl_disconnect(cid)
            except Exception:  # pragma: no cover - canvas already gone
                logger.debug("could not disconnect a crosshair callback", exc_info=True)
        self._cids = []
        for line in (self.vline, self.hline):
            try:
                line.remove()
            except (ValueError, AttributeError, NotImplementedError):
                # The usual case, not an error: the figure is rebuilt on every
                # render, so by the time the old crosshair is dropped its lines
                # have already been detached by ``figure.clear()``. Matplotlib
                # then has no remove method left for them and raises
                # NotImplementedError -- the lines are gone either way.
                logger.debug("crosshair line already detached", exc_info=True)

    @property
    def position(self) -> tuple[float, float]:
        return self.x, self.y

    def set_position(self, x=None, y=None, notify: bool = False):
        """Move the crosshair, optionally telling the listener."""
        if x is not None:
            self.x = float(x)
            self.vline.set_xdata([self.x, self.x])
        if y is not None:
            self.y = float(y)
            self.hline.set_ydata([self.y, self.y])
        if notify:
            self.on_change(self.x, self.y)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ #
    #                              Events                                #
    # ------------------------------------------------------------------ #
    def _which_line(self, event) -> str:
        """Whether the press grabbed one line, or neither (so both move).

        Distance is measured in axes fractions rather than data units: the
        delay axis is usually symlog, where a fixed data tolerance would be
        enormous at long delays and unusable near zero.
        """
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        try:
            fx = abs(self.ax.transData.transform((self.x, self.y))[0] - event.x)
            fy = abs(self.ax.transData.transform((self.x, self.y))[1] - event.y)
        except Exception:  # pragma: no cover - degenerate transforms
            logger.debug("crosshair transform unavailable", exc_info=True)
            return "both"
        width = abs(self.ax.bbox.width) or 1.0
        height = abs(self.ax.bbox.height) or 1.0
        near_v = fx / width < _GRAB_TOLERANCE
        near_h = fy / height < _GRAB_TOLERANCE
        if near_v and near_h:
            return "both"
        if near_v:
            return "v"
        if near_h:
            return "h"
        del x0, x1, y0, y1
        return "both"  # a click away from the lines moves the whole crosshair

    def _on_press(self, event):
        if event.inaxes is not self.ax or event.button != 1:
            return
        if hasattr(self.canvas, "setFocus"):
            try:
                self.canvas.setFocus()
            except Exception:
                pass
        self._dragging = self._which_line(event)
        self._move_to(event)

    def _on_motion(self, event):
        if self._dragging is None or event.inaxes is not self.ax:
            return
        self._move_to(event)

    def _on_release(self, event):
        if self._dragging is None:
            return
        self._dragging = None
        if self.on_release is not None:
            self.on_release(self.x, self.y)

    def _on_key_press(self, event):
        key = str(getattr(event, "key", "") or "").lower()
        if key not in ("left", "right", "up", "down"):
            return

        import numpy as np

        new_x, new_y = self.x, self.y
        moved = False

        if key in ("left", "right"):
            if self.x_values is not None and len(self.x_values) > 0:
                vals = np.sort(np.asarray(self.x_values, dtype=float).ravel())
                idx = int(np.argmin(np.abs(vals - self.x)))
                if key == "left":
                    idx = max(0, idx - 1)
                else:
                    idx = min(len(vals) - 1, idx + 1)
                new_x = float(vals[idx])
            else:
                x0, x1 = self.ax.get_xlim()
                step = 0.01 * abs(x1 - x0)
                new_x = self.x - step if key == "left" else self.x + step
            moved = True

        elif key in ("up", "down"):
            if self.y_values is not None and len(self.y_values) > 0:
                vals = np.sort(np.asarray(self.y_values, dtype=float).ravel())
                idx = int(np.argmin(np.abs(vals - self.y)))
                if key == "down":
                    idx = max(0, idx - 1)
                else:
                    idx = min(len(vals) - 1, idx + 1)
                new_y = float(vals[idx])
            else:
                y0, y1 = self.ax.get_ylim()
                step = 0.01 * abs(y1 - y0)
                new_y = self.y - step if key == "down" else self.y + step
            moved = True

        if moved:
            self.set_position(new_x, new_y, notify=True)
            if self.on_release is not None:
                self.on_release(self.x, self.y)

    def _move_to(self, event):
        if event.xdata is None or event.ydata is None:
            return
        x = event.xdata if self._dragging in ("v", "both") else None
        y = event.ydata if self._dragging in ("h", "both") else None
        self.set_position(x, y, notify=True)



def nearest(values, target) -> float:
    """The element of ``values`` closest to ``target`` (the crosshair snaps to data).

    A cut is only meaningful at a measured probe position or delay, so the
    crosshair reports the nearest one rather than an interpolated coordinate.
    """
    import numpy as np

    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0:
        return float(target)
    return float(values[int(np.argmin(np.abs(values - float(target))))])
