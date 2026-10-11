"""Tests for reference spectrum I/O and interpolation."""

from __future__ import annotations

import numpy as np
import pytest

from pyrate_ta.io.reference import (
    interpolate_reference_spectrum,
    load_reference_spectrum,
)


def test_load_reference_spectrum_csv(tmp_path):
    """Test loading comma-separated reference spectrum."""
    file = tmp_path / "abs_spec.csv"
    file.write_text(
        "Wavelength,Absorbance\n"
        "400.0,0.05\n"
        "450.0,0.85\n"
        "500.0,1.20\n"
        "550.0,0.30\n"
        "600.0,0.02\n"
    )
    x, y = load_reference_spectrum(file)
    assert len(x) == 5
    assert len(y) == 5
    assert np.allclose(x, [400, 450, 500, 550, 600])
    assert np.allclose(y, [0.05, 0.85, 1.20, 0.30, 0.02])


def test_load_reference_spectrum_comments_and_reversed(tmp_path):
    """Test loading files with # and % comments and descending wavelength order."""
    file = tmp_path / "em_spec.dat"
    file.write_text(
        "# Fluorescence emission spectrum\n"
        "% Measured on Fluorolog\n"
        "700.0\t0.01\n"
        "650.0\t0.45\n"
        "600.0\t1.50\n"
        "550.0\t0.80\n"
        "500.0\t0.05\n"
    )
    x, y = load_reference_spectrum(file)
    # Must be sorted strictly ascending
    assert np.all(np.diff(x) > 0)
    assert x[0] == 500.0
    assert x[-1] == 700.0
    assert np.isclose(y[x == 600.0][0], 1.50)


def test_load_reference_spectrum_duplicate_wavelengths(tmp_path):
    """Test handling duplicate probe coordinates."""
    file = tmp_path / "dup.txt"
    file.write_text(
        "500 0.1\n"
        "500 0.2\n"
        "520 0.5\n"
        "540 0.8\n"
    )
    x, y = load_reference_spectrum(file)
    assert len(x) == 3
    assert np.all(np.diff(x) > 0)


def test_load_reference_spectrum_invalid(tmp_path):
    """Test error handling for bad files."""
    file = tmp_path / "bad.txt"
    file.write_text("No numerical data here at all\nJust text\n")
    with pytest.raises(ValueError, match="Could not parse at least two numeric"):
        load_reference_spectrum(file)


def test_interpolate_reference_spectrum():
    """Test interpolating reference spectrum onto a target probe grid."""
    x_raw = np.array([400.0, 500.0, 600.0])
    y_raw = np.array([0.0, 1.0, 0.0])

    probe = np.array([350.0, 400.0, 450.0, 500.0, 550.0, 600.0, 650.0])
    y_interp = interpolate_reference_spectrum(x_raw, y_raw, probe)

    assert len(y_interp) == len(probe)
    # Zero outside bounds
    assert y_interp[0] == 0.0
    assert y_interp[-1] == 0.0
    # Peak at 500
    assert np.isclose(y_interp[3], 1.0)
    # Linear midpoint
    assert np.isclose(y_interp[2], 0.5)
    assert np.isclose(y_interp[4], 0.5)
    # All non-negative
    assert np.all(y_interp >= 0.0)
