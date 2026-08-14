"""The ``Dataset1D`` -> ``(t, D, sigma)`` adapter.

This is the **only** PyMORGAN-aware module in the fitting stack: everything
downstream takes plain arrays. It slices one detector, applies the delay and
probe restrictions, clips the early delays if asked, and fetches the per-point
noise from PyMORGAN (:meth:`pymorgan.Dataset1D.noise_array`) -- PyRATE never
estimates noise itself.

Restricting the ranges changes what the fit saw, so every choice made here is
recorded on the returned object and travels into the result's reproducibility
payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..log import get_logger

logger = get_logger(__name__)


@dataclass
class ProblemData:
    """One detector's data, ready for a fit, with its provenance."""

    t: np.ndarray
    D: np.ndarray
    sigma: np.ndarray | None = None
    probe: np.ndarray | None = None
    detector: int = 0
    delay_range: tuple[float, float] | None = None
    probe_range: tuple[float, float] | None = None
    t_min: float | None = None
    noise_source: str | None = None
    source: str | None = None
    time_unit: str | None = None
    n_delays_dropped: int = 0
    n_pixels_dropped: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.D.shape


def _noise_source(dataset) -> str | None:
    """Where PyMORGAN's noise array came from, for the record."""
    z = getattr(dataset, "Zstdv", None)
    if z is not None:
        z = np.asarray(z, dtype=float)
        if z.size and np.any(np.isfinite(z)) and not np.allclose(np.nan_to_num(z), 0.0):
            return "pdatn"
    if getattr(dataset, "has_single_scans", False):
        return "single_scans"
    return None


def prepare(
    dataset,
    *,
    detector: int = 0,
    delay_range: tuple[float, float] | None = None,
    probe_range: tuple[float, float] | None = None,
    t_min: float | None = None,
    need_noise: bool = False,
) -> ProblemData:
    """Slice a :class:`pymorgan.Dataset1D` into the arrays a fit takes.

    Parameters
    ----------
    dataset : pymorgan.Dataset1D
        Loaded (and processed) dataset. Background correction, chirp and
        solvent subtraction all belong upstream, in PyMORGAN, and are assumed
        to have been applied already if they were wanted.
    detector : int
        Which detector channel to fit. One detector at a time: a joint fit over
        detectors is a separate engine, not a reshape.
    delay_range, probe_range : tuple, optional
        Inclusive ``(min, max)`` windows. ``None`` keeps everything.
    t_min : float, optional
        Drop delays below this value -- the coherent-artefact clipping. Applied
        after ``delay_range`` and recorded separately, because it is a
        modelling decision rather than a view.
    need_noise : bool
        Raise if the dataset carries no noise array, instead of returning
        ``sigma = None``.

    Returns
    -------
    ProblemData
    """
    Z = np.asarray(dataset.Z, dtype=float)
    if Z.ndim == 2:
        Z = Z[:, :, None]
    n_det = Z.shape[2]
    if not 0 <= int(detector) < n_det:
        raise ValueError(f"detector {detector} out of range; the dataset has {n_det}")

    delays = np.asarray(dataset.delays, dtype=float).ravel()
    probe = np.asarray(dataset.probe, dtype=float).ravel()
    D = Z[:, :, int(detector)]

    sigma_all = None
    noise = None
    try:
        noise = dataset.noise_array()
    except AttributeError:  # older PyMORGAN, before the accessor was public
        noise = getattr(dataset, "_noise_array", lambda: None)()
    if noise is not None:
        noise = np.asarray(noise, dtype=float)
        if noise.ndim == 2:
            noise = noise[:, :, None]
        sigma_all = noise[:, :, int(detector)]
    elif need_noise:
        raise ValueError(
            "this dataset carries no per-point noise (no .pdatn sibling and no "
            "single scans), so a weighted fit is not possible."
        )

    # --- restrict ---------------------------------------------------------- #
    keep_t = np.ones(delays.size, dtype=bool)
    if delay_range is not None:
        keep_t &= (delays >= float(delay_range[0])) & (delays <= float(delay_range[1]))
    if t_min is not None:
        keep_t &= delays >= float(t_min)
    keep_p = np.ones(probe.size, dtype=bool)
    if probe_range is not None:
        lo, hi = sorted((float(probe_range[0]), float(probe_range[1])))
        keep_p &= (probe >= lo) & (probe <= hi)

    if not np.any(keep_t) or not np.any(keep_p):
        raise ValueError("the requested delay/probe window contains no data")

    n_t_dropped = int(np.count_nonzero(~keep_t))
    n_p_dropped = int(np.count_nonzero(~keep_p))
    if n_t_dropped or n_p_dropped:
        logger.info(
            "Fitting a restricted window: %d delay(s) and %d pixel(s) excluded.",
            n_t_dropped,
            n_p_dropped,
        )

    t = delays[keep_t]
    D = D[np.ix_(keep_t, keep_p)]
    probe_kept = probe[keep_p]
    sigma = sigma_all[np.ix_(keep_t, keep_p)] if sigma_all is not None else None

    units = getattr(dataset, "units", {}) or {}
    return ProblemData(
        t=t,
        D=D,
        sigma=sigma,
        probe=probe_kept,
        detector=int(detector),
        delay_range=(float(t.min()), float(t.max())),
        probe_range=(float(probe_kept.min()), float(probe_kept.max())),
        t_min=None if t_min is None else float(t_min),
        noise_source=_noise_source(dataset) if sigma is not None else None,
        source=getattr(dataset, "source", None),
        time_unit=units.get("unitsT_ltx"),
        n_delays_dropped=n_t_dropped,
        n_pixels_dropped=n_p_dropped,
        meta={"data_type": getattr(dataset, "data_type", None), "n_detectors": n_det},
    )


def is_dataset(obj) -> bool:
    """Duck-type test for a PyMORGAN dataset, without importing ``pymorgan``.

    Keeping the import out of this check is what lets ``fit_global`` accept
    either a dataset or plain arrays while ``import pyrate`` stays cheap.
    """
    return all(hasattr(obj, name) for name in ("Z", "delays", "probe"))
