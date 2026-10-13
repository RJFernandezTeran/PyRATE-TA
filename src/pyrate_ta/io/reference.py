"""Steady-state reference spectra (absorption and emission) I/O and processing."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from ..log import get_logger

logger = get_logger(__name__)


def load_reference_spectrum(source: str | Path | io.StringIO) -> tuple[np.ndarray, np.ndarray]:
    """Robustly load a 1D reference spectrum from a file or text buffer.

    Handles arbitrary comment markers (``#``, ``%``, ``;``, ``//``, ``*``),
    header lines, multiple delimiter types (comma, tab, whitespace, semicolon),
    duplicate coordinates, and arbitrary probe axis orientation (ascending or descending).

    Parameters
    ----------
    source : str, Path, or StringIO
        Path to file or file-like buffer containing tabular numeric data.

    Returns
    -------
    (x, y) : tuple of numpy.ndarray
        Sorted ascending probe coordinates and corresponding spectral amplitudes.

    Raises
    ------
    ValueError
        If no valid 2-column numeric data can be parsed.
    """
    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"Reference spectrum file not found: {source}")
        with open(p, encoding="utf-8", errors="ignore") as f:
            raw_lines = f.readlines()
    elif hasattr(source, "readlines"):
        raw_lines = source.readlines()
    elif hasattr(source, "read"):
        raw_lines = source.read().splitlines(keepends=True)
    else:
        raise TypeError(f"Expected file path or text buffer, got {type(source).__name__}")

    clean_rows: list[tuple[float, float]] = []
    for line in raw_lines:
        s = line.strip()
        if not s or s.startswith(("#", "%", ";", "//", "*", "!")):
            continue
        # Normalise delimiters: replace commas and semicolons with spaces
        s_norm = s.replace(",", " ").replace(";", " ")
        parts = s_norm.split()
        if len(parts) >= 2:
            try:
                x_val = float(parts[0])
                y_val = float(parts[1])
                if np.isfinite(x_val) and np.isfinite(y_val):
                    clean_rows.append((x_val, y_val))
            except ValueError:
                # Header row or non-numeric label line
                continue

    if len(clean_rows) < 2:
        raise ValueError(
            "Could not parse at least two numeric (probe, intensity) rows from the file. "
            "Please ensure the file has two whitespace-, tab- or comma-separated columns."
        )

    data = np.array(clean_rows, dtype=float)
    # Sort by probe axis ascending
    order = np.argsort(data[:, 0])
    x = data[order, 0]
    y = data[order, 1]

    # Deduplicate x coordinates by taking the mean of duplicate probe points
    unique_x, unique_indices, counts = np.unique(x, return_index=True, return_counts=True)
    if len(unique_x) < len(x):
        unique_y = np.empty_like(unique_x)
        for idx, ux in enumerate(unique_x):
            unique_y[idx] = np.mean(y[x == ux])
        return unique_x, unique_y

    return x, y


def interpolate_reference_spectrum(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    probe_grid: np.ndarray,
    *,
    left: float = 0.0,
    right: float = 0.0,
    clip_non_negative: bool = True,
) -> np.ndarray:
    """Interpolate a reference spectrum onto the dataset's probe grid.

    Parameters
    ----------
    x_raw : numpy.ndarray
        Raw reference probe coordinates (sorted ascending).
    y_raw : numpy.ndarray
        Raw reference intensities.
    probe_grid : numpy.ndarray
        Target probe grid of the transient dataset.
    left, right : float, default 0.0
        Values assigned outside the raw spectrum's spectral coverage window.
    clip_non_negative : bool, default True
        Clamp negative baseline values to zero (physical for absorption and emission).

    Returns
    -------
    numpy.ndarray
        Interpolated spectrum matching ``probe_grid.shape``.

    Raises
    ------
    ValueError
        If the raw spectrum does not overlap with the target probe window at all.
    """
    order = np.argsort(x_raw)
    xs, ys = x_raw[order], y_raw[order]

    p_grid = np.asarray(probe_grid, dtype=float)
    p_min, p_max = float(np.min(p_grid)), float(np.max(p_grid))
    r_min, r_max = float(xs[0]), float(xs[-1])

    if r_max < p_min or r_min > p_max:
        raise ValueError(
            f"Reference spectrum window [{r_min:.4g} - {r_max:.4g}] does not overlap "
            f"with dataset probe window [{p_min:.4g} - {p_max:.4g}]."
        )

    # Evaluate interpolation: NumPy's interp natively handles decreasing probe_grid
    y_interp = np.interp(p_grid, xs, ys, left=left, right=right)

    if clip_non_negative:
        y_interp = np.maximum(0.0, y_interp)

    return y_interp
