"""Tabular and 2D surface export of fitted spectra, concentrations, surfaces, and reports.

Exports use a single uncommented header row by default (no ``#`` preamble
lines), so every CSV opens cleanly in Excel, OriginPro, ``pandas.read_csv``,
``numpy.loadtxt(..., delimiter=",", skiprows=1)``, and
``numpy.genfromtxt(..., delimiter=",", names=True)``. Two-dimensional Data,
Fit, and Residual surfaces always follow the PyMORGAN ``.pdat`` orientation
(rows = delays, columns = probe; never transposed) and can be written either as
clean single-header ``.csv``/``.tsv`` matrices or as native ``.pdat`` files via
:func:`pymorgan.write_pdat`. Full run provenance, fitted parameters with
1-sigma uncertainties, solver diagnostics, target rate matrices, GSB/SE scale
factors, and literature citations are written to a companion human-readable
``.txt`` summary sharing the same ``YYYYMMDD-HHMMSS`` timestamp, with optional
single-file ``.zip`` bundling.
"""

from __future__ import annotations

import os
import re
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from ..log import get_logger

logger = get_logger(__name__)

_BANNER_WIDTH = 72
_RULE_HEAVY = "=" * _BANNER_WIDTH
_RULE_LIGHT = "-" * _BANNER_WIDTH


class ExportedSurfacePaths(NamedTuple):
    """Paths written by :func:`export_surfaces`, plus their shared timestamp."""

    data: Path
    fit: Path
    residuals: Path
    summary: Path
    timestamp: str
    archive: Path | None = None


class ExportedFitPaths(NamedTuple):
    """Paths written by :func:`export_fit_results`, plus their shared timestamp."""

    spectra: Path
    concentrations: Path
    summary: Path
    timestamp: str
    data: Path | None = None
    fit: Path | None = None
    residuals: Path | None = None
    archive: Path | None = None


def make_timestamp(dt: datetime | None = None) -> str:
    """Return a filesystem-safe ``YYYYMMDD-HHMMSS`` timestamp string."""
    now = dt if dt is not None else datetime.now()
    return now.strftime("%Y%m%d-%H%M%S")


# --------------------------------------------------------------------------- #
#                        Internal resolution helpers                          #
# --------------------------------------------------------------------------- #
def _is_fit_like(obj: Any) -> bool:
    """True when ``obj`` looks like a ``KineticFit`` or ``LoadedFit``."""
    if obj is None or isinstance(obj, (str, bytes, os.PathLike, Path)):
        return False
    return (hasattr(obj, "taus") and hasattr(obj, "S")) or (
        hasattr(obj, "meta") and hasattr(obj, "_arrays")
    )


def _resolve_fit_and_path(
    first: Any = None,
    second: Any = None,
    *,
    fit: Any = None,
    path: str | os.PathLike[str] | None = None,
    fn_name: str = "export",
) -> tuple[Any, Path | None]:
    """Accept either ``fn(fit, path=None)`` or ``fn(path, fit)`` seamlessly."""
    fit_obj = fit
    target = path

    for arg in (first, second):
        if arg is None:
            continue
        if _is_fit_like(arg):
            if fit_obj is not None and fit_obj is not arg:
                raise TypeError(f"{fn_name}() received multiple fit objects")
            fit_obj = arg
        elif isinstance(arg, (str, os.PathLike, Path)):
            if target is not None:
                raise TypeError(f"{fn_name}() received multiple output paths")
            target = arg
        else:
            raise TypeError(
                f"{fn_name}() expected a fit result and optional path, got {type(arg).__name__}"
            )

    if fit_obj is None:
        raise TypeError(f"{fn_name}() missing required fit result argument")
    return fit_obj, (None if target is None else Path(target))


def _fit_attr(fit: Any, name: str, default: Any = None) -> Any:
    """Read an attribute from either a live ``KineticFit`` or a ``LoadedFit``."""
    val = getattr(fit, name, None)
    if val is not None:
        return val
    meta = getattr(fit, "meta", None)
    if isinstance(meta, dict) and name in meta:
        return meta[name]
    return default


def _fit_settings(fit: Any) -> dict[str, Any]:
    s = _fit_attr(fit, "settings", None)
    return dict(s) if isinstance(s, dict) else {}


def _clean_csv_text(raw: str | None) -> str:
    """Convert LaTeX/Unicode unit or label markup into clean, comma-free ASCII."""
    if not raw:
        return ""
    text = str(raw)
    text = text.replace(r"\Delta", "d").replace(r"\delta", "d")
    text = text.replace("Δ", "d")
    text = text.replace(r"\times", "x").replace("×", "x")
    text = text.replace(r"\mu", "u").replace("µ", "u").replace("μ", "u")
    text = text.replace("±", "+/-").replace("∞", "inf")
    text = text.replace("^{-1}", "-1")
    text = re.sub(r"\^\{([^}]+)\}", r"^\1", text)
    text = text.replace("$", "").replace("{", "").replace("}", "").replace("\\", "")
    text = text.replace(",", " ").replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def _resolve_units(fit: Any, dataset: Any = None) -> tuple[str, str, str, str]:
    """Return ``(time_unit, probe_label, probe_unit, signal_unit)`` as clean ASCII."""
    ds_units = (getattr(dataset, "units", None) or {}) if dataset is not None else {}
    f_settings = _fit_settings(fit)

    time_u = _clean_csv_text(ds_units.get("unitsT_ltx") or f_settings.get("time_unit") or "")
    probe_lbl = _clean_csv_text(ds_units.get("unitsL_lbl") or f_settings.get("probe_label") or "")
    probe_u = _clean_csv_text(ds_units.get("unitsL_ltx") or f_settings.get("probe_unit") or "")
    z_lbl = _clean_csv_text(ds_units.get("unitsZ_lbl") or "")
    z_u = _clean_csv_text(ds_units.get("unitsZ_ltx") or f_settings.get("signal_unit") or "")
    if z_u.startswith("x 10") and z_lbl:
        sig_u = f"{z_lbl} ({z_u})"
    else:
        sig_u = z_u or z_lbl

    if not probe_lbl and probe_u:
        low = probe_u.lower()
        if low == "nm":
            probe_lbl = "Wavelength"
        elif "cm" in low:
            probe_lbl = "Wavenumber"
        elif low == "ev":
            probe_lbl = "Energy"
        elif low == "thz":
            probe_lbl = "Frequency"
    return time_u, probe_lbl, probe_u, sig_u


def _spectra_kind(fit: Any) -> str:
    kind = getattr(fit, "spectra_kind", None)
    if kind:
        return str(kind)
    model_type = str(_fit_attr(fit, "model_type", ""))
    return {"Parallel": "DAS", "Sequential": "EAS", "Target": "SAS"}.get(model_type, "spectra")


def _n_kinetic(fit: Any) -> int:
    nk = getattr(fit, "n_kinetic", None)
    if nk is not None:
        return int(nk)
    S = getattr(fit, "S", None)
    taus = np.atleast_1d(_fit_attr(fit, "taus", []))
    total = int(np.shape(S)[1]) if S is not None and np.ndim(S) == 2 else int(taus.size)
    n_art = int(_fit_attr(fit, "n_artifact", 0) or 0)
    return max(total - n_art, 0)


def _species_names(fit: Any) -> list[str]:
    """Kinetic + artefact column names, matching ``plot_concentrations``."""
    from ..plot.concentrations import species_labels_for

    raw = species_labels_for(fit)
    return [_clean_csv_text(name) or f"S{i + 1}" for i, name in enumerate(raw)]


def _format_tau_tag(
    tau: float,
    err: float | None,
    fixed: bool,
    time_unit: str,
    *,
    show_uncertainties: bool,
    round_uncertainties: bool,
) -> str:
    """Format a compact ASCII lifetime tag such as ``0.45 +/- 0.03 ps`` or ``inf fixed``."""
    from ..helpers import format_lifetime

    unit_suffix = f" {time_unit}" if time_unit else ""
    if not np.isfinite(tau):
        return "inf fixed" if fixed else "inf"

    has_err = err is not None and np.isfinite(err)
    if show_uncertainties and has_err:
        if round_uncertainties:
            try:
                from pymorgan.helpers import format_uncertainty_pair

                v_str, e_str = format_uncertainty_pair(float(tau), float(err))
                if abs(tau) > 0 and v_str == "0":
                    core = f"{format_lifetime(tau)} +/- {float(err):.3g}"
                elif e_str is not None:
                    core = f"{v_str} +/- {e_str}"
                else:
                    core = format_lifetime(tau)
            except Exception:
                core = f"{format_lifetime(tau)} +/- {float(err):.3g}"
        else:
            core = f"{format_lifetime(tau)} +/- {float(err):.3g}"
    else:
        core = format_lifetime(tau)

    tag = f"{core}{unit_suffix}"
    if fixed:
        tag += " fixed"
    return _clean_csv_text(tag)


def _source_stem_and_dir(fit: Any, directory: str | os.PathLike[str] | None = None) -> tuple[str, Path]:
    source = _fit_attr(fit, "source", None)
    if source:
        src_path = Path(str(source))
        stem = src_path.stem or "pyrate_fit"
        base_dir = Path(directory) if directory is not None else src_path.parent
    else:
        stem = "pyrate_fit"
        base_dir = Path(directory) if directory is not None else Path.cwd()
    return stem, base_dir


def _resolve_timestamp(fit: Any, timestamp: str | None = None) -> str:
    if timestamp:
        return str(timestamp)
    fit_ts = _fit_attr(fit, "timestamp", None)
    if fit_ts:
        return str(fit_ts)
    return make_timestamp()


def _delimiter_for(path: Path) -> str:
    if path.suffix.lower() in {".tsv", ".txt", ".dat"}:
        return "\t"
    return ","


def _resolve_gsb_recovery(
    fit: Any,
    probe: np.ndarray,
    *,
    dataset: Any = None,
    recover_gsb: bool = False,
    S_abs: Any = None,
    gs_spectrum: Any = None,
    em_spectrum: Any = None,
) -> tuple[np.ndarray | None, Any]:
    """Return ``(S_abs_kinetic, bleach_result)`` when GSB/SE recovery is requested."""
    n_kin = _n_kinetic(fit)
    if S_abs is not None:
        arr = np.atleast_2d(np.asarray(S_abs, dtype=float))
        if arr.shape[0] != probe.size and arr.shape[1] == probe.size:
            arr = arr.T
        if arr.shape[0] != probe.size:
            raise ValueError(
                f"S_abs has {arr.shape[0]} rows, expected {probe.size} to match probe axis"
            )
        return arr[:, :n_kin], None

    if not recover_gsb:
        return None, None

    # Check if fit already carries pre-computed S_abs (and no new reference spectra override it)
    existing_s_abs = getattr(fit, "S_abs", None)
    if (
        existing_s_abs is not None
        and gs_spectrum is None
        and em_spectrum is None
        and (
            dataset is None
            or (
                getattr(dataset, "gs_spectrum", None) is None
                and getattr(dataset, "em_spectrum", None) is None
            )
        )
    ):
        arr = np.atleast_2d(np.asarray(existing_s_abs, dtype=float))
        if arr.shape[0] == probe.size:
            return arr[:, :n_kin], None

    S_full = np.atleast_2d(np.asarray(fit.S, dtype=float))
    S_kin = S_full[:, :n_kin]
    Np = probe.size

    def _align_ref(ref_arr, full_probe):
        if ref_arr is None:
            return None
        r = np.asarray(ref_arr, dtype=float).ravel()
        if r.size == Np and (full_probe is None or np.allclose(full_probe, probe)):
            return r
        if full_probe is not None and full_probe.size == r.size:
            order = np.argsort(full_probe)
            return np.interp(probe, full_probe[order], r[order], left=0.0, right=0.0)
        if r.size == Np:
            return r
        raise ValueError(
            f"Reference spectrum length ({r.size}) does not match fitted probe axis ({Np})"
        )

    ds_probe = (
        np.asarray(dataset.probe, dtype=float).ravel()
        if dataset is not None and getattr(dataset, "probe", None) is not None
        else None
    )

    gs_vec = _align_ref(
        gs_spectrum if gs_spectrum is not None else getattr(dataset, "gs_spectrum", None),
        ds_probe,
    )
    em_vec = _align_ref(
        em_spectrum if em_spectrum is not None else getattr(dataset, "em_spectrum", None),
        ds_probe,
    )

    # Same early-delay negative transient fallback as FitTabMixin._resolved_species_spectra_data
    has_gs = gs_vec is not None and np.any(gs_vec > 0)
    has_em = em_vec is not None and np.any(em_vec > 0)
    if not has_gs and not has_em and dataset is not None and getattr(dataset, "Z", None) is not None:
        det = int(_fit_attr(fit, "detector", 0) or 0)
        Z = np.asarray(dataset.Z, dtype=float)
        Z_mat = Z[:, :, det] if Z.ndim == 3 else Z
        t_arr = np.asarray(dataset.delays, dtype=float).ravel()
        t_pos = t_arr > 0
        early_full = Z_mat[t_pos, :][0, :] if np.any(t_pos) else Z_mat[0, :]
        early_aligned = _align_ref(early_full, ds_probe)
        est_gs = np.maximum(0.0, -early_aligned)
        if np.any(est_gs > 0):
            gs_vec = est_gs

    from ..fit.groundstate import compute_absolute_spectra

    bleach_res = compute_absolute_spectra(
        S_kin,
        gs_spectrum=gs_vec,
        em_spectrum=em_vec,
        gs_scale="auto",
        em_scale="auto",
    )
    return bleach_res.S_abs[:, :n_kin], bleach_res


def _metadata_comment_lines(fit: Any, timestamp: str, *, kind_title: str) -> list[str]:
    """Optional ``#`` header lines when ``metadata=True`` is explicitly requested."""
    from ..__about__ import __version__

    model_type = _fit_attr(fit, "model_type", "Unknown")
    n_comp = _fit_attr(fit, "n_components", _n_kinetic(fit))
    source = _fit_attr(fit, "source", None) or "array input"
    lines = [
        f"# PyRATE-TA v{__version__} - {kind_title}",
        f"# Exported: {timestamp} | Model: {model_type} ({n_comp} component(s)) | Source: {source}",
    ]
    stat = _fit_attr(fit, "statistic", None)
    if stat is not None:
        if isinstance(stat, dict):
            lines.append(f"# Statistic: {stat.get('kind')} = {stat.get('value'):.5g}")
        else:
            lines.append(f"# Statistic: {stat}")
    return lines


def _write_delimited_table(
    path: Path,
    headers: list[str],
    matrix: np.ndarray,
    *,
    delimiter: str,
    fmt: str = "%.12g",
    comment_lines: list[str] | None = None,
) -> Path:
    """Write a 2D numeric array with a single clean header row in UTF-8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_headers = [h.replace(delimiter, " ").strip() for h in headers]
    header_line = delimiter.join(clean_headers)

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        if comment_lines:
            for line in comment_lines:
                handle.write(f"{line}\n")
        handle.write(f"{header_line}\n")
        np.savetxt(handle, matrix, delimiter=delimiter, fmt=fmt)
    return path


# --------------------------------------------------------------------------- #
#                       Companion .txt fit report                             #
# --------------------------------------------------------------------------- #
def format_fit_report(
    fit: Any,
    *,
    dataset: Any = None,
    timestamp: str | None = None,
    bleach_result: Any = None,
) -> str:
    """Format a complete terminal-style human-readable report for a fit result.

    Covers the run banner, dataset and fitted window provenance, noise
    weighting, solver convergence diagnostics, fitted lifetimes with 1-sigma
    uncertainties and corresponding rate constants ``k = 1/tau``, IRF/time-zero
    and coherent-artefact settings, fit statistics, target scheme & ``K``
    matrix + eigenvalues (for ``TargetFit``), GSB/SE scale factors (when
    applicable), and relevant literature citations.
    """
    from ..__about__ import __version__
    from ..cite import citation_text
    from ..helpers import format_lifetime
    from ..settings import get_settings

    ts = _resolve_timestamp(fit, timestamp)
    s = get_settings()
    time_u, probe_lbl, probe_u, sig_u = _resolve_units(fit, dataset)
    u_sfx = f" {time_u}" if time_u else ""
    p_sfx = f" {probe_u}" if probe_u else ""

    model_type = str(_fit_attr(fit, "model_type", "Unknown"))
    spec_kind = _spectra_kind(fit)
    n_comp = int(_fit_attr(fit, "n_components", _n_kinetic(fit)) or _n_kinetic(fit))
    n_art = int(_fit_attr(fit, "n_artifact", 0) or 0)
    f_settings = _fit_settings(fit)
    is_preview = bool(f_settings.get("preview", False))

    title_mode = f"{model_type} Preview (no optimisation)" if is_preview else f"{model_type} Fit Report"
    lines: list[str] = [
        _RULE_HEAVY,
        f"PyRATE-TA v{__version__} -- {title_mode}",
        f"Timestamp: {ts}",
        _RULE_HEAVY,
        "",
    ]

    # --- Dataset & Fit Window ---------------------------------------------- #
    lines.append("Dataset & Fit Window")
    lines.append(_RULE_LIGHT)
    source = _fit_attr(fit, "source", None)
    lines.append(f"  Source         : {source if source else '(in-memory array)'}")
    lines.append(f"  Detector       : {int(_fit_attr(fit, 'detector', 0) or 0)}")

    t_arr = np.atleast_1d(np.asarray(_fit_attr(fit, "t", []), dtype=float))
    probe_arr = _fit_attr(fit, "probe", None)
    delay_range = _fit_attr(fit, "delay_range", None)
    probe_range = _fit_attr(fit, "probe_range", None)
    t_min = _fit_attr(fit, "t_min", None)

    if delay_range is not None:
        lines.append(
            f"  Delay window   : [{delay_range[0]:.6g}, {delay_range[1]:.6g}]{u_sfx}"
            f" ({t_arr.size} delays)"
        )
    elif t_arr.size:
        lines.append(
            f"  Delay window   : [{float(np.nanmin(t_arr)):.6g}, {float(np.nanmax(t_arr)):.6g}]{u_sfx}"
            f" ({t_arr.size} delays)"
        )
    if t_min is not None:
        lines.append(f"  Early-t clip   : t >= {float(t_min):.6g}{u_sfx}")

    if probe_arr is not None:
        p_vec = np.atleast_1d(np.asarray(probe_arr, dtype=float))
        p_name = probe_lbl or "Probe"
        if probe_range is not None:
            lines.append(
                f"  {p_name:<14} : [{probe_range[0]:.6g}, {probe_range[1]:.6g}]{p_sfx}"
                f" ({p_vec.size} pixels)"
            )
        elif p_vec.size:
            lines.append(
                f"  {p_name:<14} : [{float(np.nanmin(p_vec)):.6g}, {float(np.nanmax(p_vec)):.6g}]{p_sfx}"
                f" ({p_vec.size} pixels)"
            )
    else:
        S_mat = _fit_attr(fit, "S", None)
        if S_mat is not None and np.ndim(S_mat) == 2:
            lines.append(f"  Channels       : {np.shape(S_mat)[0]} trace(s)")

    if sig_u:
        lines.append(f"  Signal unit    : {sig_u}")

    # Weighting summary
    weights = _fit_attr(fit, "weights", None)
    noise_src = f_settings.get("noise_source")
    if weights is not None:
        if isinstance(weights, dict):
            w_on = bool(weights.get("weighted", False))
            w_src = weights.get("source") or noise_src or "supplied"
            w_floor = weights.get("floor")
            n_floored = int(weights.get("n_floored", 0) or 0)
            n_masked = int(weights.get("n_masked", 0) or 0)
        else:
            w_on = bool(getattr(weights, "weighted", False))
            w_src = getattr(weights, "source", None) or noise_src or "supplied"
            w_floor = getattr(weights, "floor", None)
            n_floored = int(getattr(weights, "n_floored", 0) or 0)
            n_masked = int(getattr(weights, "n_masked", 0) or 0)
        if w_on:
            w_desc = f"1/sigma weighted (source: {w_src})"
            if w_floor is not None and n_floored > 0:
                w_desc += f", floor={float(w_floor):.3g} ({n_floored} pts floored)"
            if n_masked > 0:
                w_desc += f", {n_masked} pts masked"
        else:
            w_desc = "unweighted (ordinary least squares)" + (
                f", {n_masked} non-finite pts masked" if n_masked > 0 else ""
            )
        lines.append(f"  Weighting      : {w_desc}")
    lines.append("")

    # --- Model & Convergence ----------------------------------------------- #
    lines.append("Model & Solver")
    lines.append(_RULE_LIGHT)
    art_note = f" + {n_art} coherent-artefact column(s)" if n_art else ""
    taus_for_hdr = np.atleast_1d(np.asarray(_fit_attr(fit, "taus", []), dtype=float))
    if model_type.lower().startswith("target"):
        lines.append(
            f"  Model family   : {model_type} ({n_comp} compartment(s),"
            f" {taus_for_hdr.size} rate constant(s){art_note})"
        )
    else:
        lines.append(f"  Model family   : {model_type} ({n_comp} kinetic component(s){art_note})")
    lines.append(f"  Spectra kind   : {spec_kind}")

    scheme_label = _fit_attr(fit, "scheme_label", None)
    scheme_key = _fit_attr(fit, "scheme_key", None)
    if scheme_label or scheme_key:
        lines.append(f"  Target scheme  : {scheme_label or scheme_key}")

    report = _fit_attr(fit, "report", None)
    if report is None:
        lines.append("  Solver status  : preview evaluation only (no non-linear optimisation run)")
    elif isinstance(report, dict):
        conv = bool(report.get("converged", False))
        status_txt = "CONVERGED" if conv else "DID NOT CONVERGE"
        lines.append(f"  Solver status  : {status_txt} (method={report.get('method', '?')})")
        lines.append(f"  Message        : {report.get('message', '')}")
        lines.append(
            f"  Evaluations    : {report.get('n_evaluations', '?')}"
            f" (max_iterations={report.get('max_iterations', '?')})"
        )
        if report.get("cost") is not None and report.get("optimality") is not None:
            lines.append(
                f"  Cost / Optim.  : cost={float(report['cost']):.6g}, "
                f"optimality={float(report['optimality']):.3e}"
            )
        at_bounds = report.get("at_bounds") or []
        if at_bounds:
            lines.append(f"  WARNING        : parameter(s) at bounds: {', '.join(map(str, at_bounds))}")
        if not conv:
            lines.append(
                "  WARNING        : solver did not converge; parameters below are the last trial values."
            )
    else:
        conv = bool(getattr(report, "converged", False))
        status_txt = "CONVERGED" if conv else "DID NOT CONVERGE"
        lines.append(f"  Solver status  : {status_txt} (method={report.method})")
        lines.append(f"  Message        : {report.message}")
        lines.append(
            f"  Evaluations    : {report.n_evaluations} (max_iterations={report.max_iterations})"
        )
        lines.append(f"  Cost / Optim.  : cost={report.cost:.6g}, optimality={report.optimality:.3e}")
        if report.at_bounds:
            lines.append(f"  WARNING        : parameter(s) at bounds: {', '.join(report.at_bounds)}")
        if not conv:
            lines.append(
                "  WARNING        : solver did not converge; parameters below are the last trial values."
            )
    lines.append("")

    # --- Fitted Kinetic Parameters ----------------------------------------- #
    lines.append("Kinetic Parameters (1-sigma linearised uncertainties)")
    lines.append(_RULE_LIGHT)
    taus = np.atleast_1d(np.asarray(_fit_attr(fit, "taus", []), dtype=float))
    tau_err = np.atleast_1d(
        np.asarray(_fit_attr(fit, "tau_err", np.full(taus.size, np.nan)), dtype=float)
    )
    is_fixed = np.atleast_1d(
        np.asarray(_fit_attr(fit, "is_fixed", np.zeros(taus.size, dtype=bool)), dtype=bool)
    )
    param_names = list(_fit_attr(fit, "param_names", []) or [])
    init_params = _fit_attr(fit, "initial_params", None)
    bounds = _fit_attr(fit, "bounds", None)
    species = _species_names(fit)
    is_target = model_type.lower().startswith("target")

    rate_unit = f"1/{time_u}" if time_u else "1/time"
    for i, tau in enumerate(taus):
        err = float(tau_err[i]) if i < tau_err.size else np.nan
        fixed = bool(is_fixed[i]) if i < is_fixed.size else False
        p_name = param_names[i] if i < len(param_names) else f"tau{i + 1}"
        comp_tag = "" if is_target or i >= len(species) else f" [{species[i]}]"

        tau_str = _format_tau_tag(
            float(tau),
            err,
            fixed,
            time_u,
            show_uncertainties=bool(s.show_uncertainties),
            round_uncertainties=bool(s.round_uncertainties),
        )
        if not np.isfinite(tau) or tau == 0:
            k_str = f"k = 0 {rate_unit}"
        else:
            k_val = 1.0 / float(tau)
            if np.isfinite(err) and bool(s.show_uncertainties):
                k_err = abs(err / (float(tau) ** 2))
                k_str = f"k = {k_val:.4g} +/- {k_err:.3g} {rate_unit}"
            else:
                k_str = f"k = {k_val:.4g} {rate_unit}"

        extra_bits: list[str] = []
        if init_params is not None and not bool(_fit_attr(fit, "sorted_by_lifetime", False)):
            init_arr = np.atleast_1d(np.asarray(init_params, dtype=float))
            if i < init_arr.size:
                extra_bits.append(f"init={format_lifetime(float(init_arr[i]))}")
        if bounds is not None and len(bounds) == 2 and not bool(
            _fit_attr(fit, "sorted_by_lifetime", False)
        ):
            lo_arr = np.atleast_1d(np.asarray(bounds[0], dtype=float))
            hi_arr = np.atleast_1d(np.asarray(bounds[1], dtype=float))
            if i < lo_arr.size and i < hi_arr.size:
                extra_bits.append(
                    f"bounds=[{format_lifetime(float(lo_arr[i]))}, {format_lifetime(float(hi_arr[i]))}]"
                )
        tail = f"  ({', '.join(extra_bits)})" if extra_bits else ""
        lines.append(f"  {p_name:<8}{comp_tag:<6} = {tau_str:<26} | {k_str}{tail}")

    if bool(_fit_attr(fit, "sorted_by_lifetime", False)):
        lines.append("  Note: components sorted in ascending order of lifetime; spectra reordered to match.")

    # IRF & t0
    t0 = _fit_attr(fit, "t0", 0.0)
    t0_err = _fit_attr(fit, "t0_err", None)
    irf_fwhm = _fit_attr(fit, "irf_fwhm", None)
    irf_err = _fit_attr(fit, "irf_err", None)
    lines.append("")
    if t0 is not None:
        if t0_err is not None and np.isfinite(t0_err):
            lines.append(f"  {'t0':<14} = {float(t0):.5g} +/- {float(t0_err):.3g}{u_sfx} (fitted)")
        else:
            lines.append(f"  {'t0':<14} = {float(t0):.5g}{u_sfx} (fixed)")
    if irf_fwhm is not None and np.isfinite(irf_fwhm) and float(irf_fwhm) > 0:
        if irf_err is not None and np.isfinite(irf_err):
            lines.append(
                f"  {'IRF FWHM':<14} = {float(irf_fwhm):.5g} +/- {float(irf_err):.3g}{u_sfx} (Gaussian, fitted)"
            )
        else:
            lines.append(f"  {'IRF FWHM':<14} = {float(irf_fwhm):.5g}{u_sfx} (Gaussian, fixed)")
    else:
        lines.append(f"  {'IRF FWHM':<14} = None (unconvolved step-exponential)")

    if n_art > 0:
        from ..models.irf import ARTIFACT_LABELS

        art_cols = ", ".join(ARTIFACT_LABELS[:n_art])
        lines.append(f"  {'Coherent art.':<14} = {n_art} basis column(s) ({art_cols})")
    lines.append("")

    # --- Fit Statistic ----------------------------------------------------- #
    stat = _fit_attr(fit, "statistic", None)
    if stat is not None:
        lines.append("Fit Statistic")
        lines.append(_RULE_LIGHT)
        if isinstance(stat, dict):
            s_kind = stat.get("kind")
            s_val = float(stat.get("value", np.nan))
            s_nl = stat.get("value_nonlinear_dof")
            n_pts = stat.get("n_points")
            dof = stat.get("dof")
            n_nl = stat.get("n_nonlinear")
            n_amp = stat.get("n_amplitudes")
            n_src = stat.get("noise_source")
            if s_kind == "chi2_red":
                lines.append(
                    f"  reduced chi2   = {s_val:.5g} (full dof={dof}, points={n_pts},"
                    f" n_nonlinear={n_nl}, n_amplitudes={n_amp}, noise={n_src})"
                )
                if s_nl is not None:
                    lines.append(f"  chi2 / dof_nl  = {float(s_nl):.5g} (nonlinear-only dof)")
            else:
                lines.append(
                    f"  SSR            = {s_val:.6g} (unweighted, points={n_pts}, full dof={dof},"
                    f" n_nonlinear={n_nl}, n_amplitudes={n_amp})"
                )
        else:
            lines.append(f"  {stat}")
            if getattr(stat, "kind", None) == "chi2_red" and getattr(
                stat, "value_nonlinear_dof", None
            ) is not None:
                lines.append(
                    f"  chi2 / dof_nl  = {float(stat.value_nonlinear_dof):.5g}"
                    f" (nonlinear-only denominator: N - {stat.n_nonlinear})"
                )
            lines.append(
                f"  Parameter count: N={stat.n_points} points, {stat.n_nonlinear} free nonlinear,"
                f" {stat.n_amplitudes} linear amplitudes, full dof={stat.dof}"
            )
        lines.append("")

    # --- Target Analysis Details (K-matrix, scheme, eigenvalues) ----------- #
    K = _fit_attr(fit, "K", None)
    scheme_text_val = _fit_attr(fit, "scheme_text", None)
    eigenvalues = _fit_attr(fit, "eigenvalues", None)
    if is_target or K is not None or scheme_text_val:
        lines.append("Target Model Details")
        lines.append(_RULE_LIGHT)
        n_kin = _n_kinetic(fit)
        kin_species = species[:n_kin]
        if kin_species:
            lines.append(f"  Compartments   : {', '.join(kin_species)}")
        if scheme_text_val:
            lines.append("  Scheme text    :")
            for sline in str(scheme_text_val).strip().splitlines():
                lines.append(f"    {sline}")
        if K is not None:
            K_arr = np.asarray(K, dtype=float)
            lines.append(f"  Rate matrix K ({rate_unit}):")
            col_hdr = "          " + " ".join(
                f"{(kin_species[j] if j < len(kin_species) else f'S{j+1}'):>12}"
                for j in range(K_arr.shape[1])
            )
            lines.append(col_hdr)
            for r_idx in range(K_arr.shape[0]):
                r_lbl = kin_species[r_idx] if r_idx < len(kin_species) else f"S{r_idx + 1}"
                row_vals = " ".join(f"{K_arr[r_idx, c_idx]:12.5g}" for c_idx in range(K_arr.shape[1]))
                lines.append(f"    {r_lbl:<6} {row_vals}")
        if eigenvalues is not None:
            eig_arr = np.atleast_1d(np.asarray(eigenvalues, dtype=float))
            lines.append("  Eigenvalues & apparent lifetimes (-1/lambda):")
            for j, lam in enumerate(eig_arr):
                app_tau = np.inf if lam == 0 else (-1.0 / float(lam))
                app_str = "inf" if not np.isfinite(app_tau) else f"{format_lifetime(abs(app_tau))}{u_sfx}"
                lines.append(f"    lambda_{j + 1} = {float(lam):12.5g} {rate_unit}  ->  tau_app = {app_str}")
        lines.append(
            "  Note: branched schemes are not generally identifiable from a single dataset;"
            " the eigenvalues are what the data uniquely determine."
        )
        lines.append("")

    # --- GSB / SE Pure Positive Spectra Recovery --------------------------- #
    gs_scale = getattr(bleach_result, "gs_scale", None)
    em_scale = getattr(bleach_result, "em_scale", None)
    if gs_scale is None:
        gs_scale = _fit_attr(fit, "gs_scale", None)
    if bleach_result is not None or gs_scale is not None or getattr(fit, "S_abs", None) is not None:
        lines.append("GSB / SE Pure Positive Recovery")
        lines.append(_RULE_LIGHT)
        if gs_scale is not None:
            gs_arr = np.atleast_1d(np.asarray(gs_scale, dtype=float))
            gs_txt = ", ".join(f"{v:.4g}" for v in gs_arr)
            lines.append(f"  GSB scale (a)  : [{gs_txt}]")
        if em_scale is not None:
            em_arr = np.atleast_1d(np.asarray(em_scale, dtype=float))
            em_txt = ", ".join(f"{v:.4g}" for v in em_arr)
            lines.append(f"  SE scale (b)   : [{em_txt}]")
        a_max = getattr(bleach_result, "a_max", None) or _fit_attr(fit, "a_max", None)
        if a_max is not None:
            lines.append(f"  Max GSB a_max  : {float(a_max):.4g}")
        lines.append("")

    # --- Citations --------------------------------------------------------- #
    cite_keys = ["pyrate_ta", "chameleon2022", "vanstokkum2004"]
    if model_type in ("Sequential", "Target"):
        cite_keys.append("berberansantos1990")
    if n_art > 0:
        cite_keys.append("kovalenko1999")
    lines.append("References")
    lines.append(_RULE_LIGHT)
    lines.append(citation_text(cite_keys))
    lines.append("")
    return "\n".join(lines)


def export_fit_summary(
    fit_or_path: Any = None,
    path_or_fit: Any = None,
    *,
    fit: Any = None,
    path: str | os.PathLike[str] | None = None,
    dataset: Any = None,
    timestamp: str | None = None,
    bleach_result: Any = None,
    directory: str | os.PathLike[str] | None = None,
) -> Path:
    """Write the companion human-readable ``.txt`` fit report to disk.

    Accepts either ``export_fit_summary(fit, path=None)`` or
    ``export_fit_summary(path, fit)``. When ``path`` is omitted, writes
    ``<stem>_fit_summary_<YYYYMMDD-HHMMSS>.txt`` beside ``fit.source`` (or in
    ``directory`` / the current working directory).
    """
    fit_obj, target = _resolve_fit_and_path(
        fit_or_path, path_or_fit, fit=fit, path=path, fn_name="export_fit_summary"
    )
    ts = _resolve_timestamp(fit_obj, timestamp)
    if target is None:
        stem, base_dir = _source_stem_and_dir(fit_obj, directory)
        target = base_dir / f"{stem}_fit_summary_{ts}.txt"
    elif not target.suffix:
        target = target.with_suffix(".txt")

    target.parent.mkdir(parents=True, exist_ok=True)
    report_txt = format_fit_report(
        fit_obj, dataset=dataset, timestamp=ts, bleach_result=bleach_result
    )
    target.write_text(report_txt, encoding="utf-8", newline="\n")
    logger.info("Fit summary written to %s", target)
    return target


# --------------------------------------------------------------------------- #
#                    DAS / EAS / SAS Spectra CSV Export                       #
# --------------------------------------------------------------------------- #
def export_spectra(
    fit_or_path: Any = None,
    path_or_fit: Any = None,
    *,
    fit: Any = None,
    path: str | os.PathLike[str] | None = None,
    dataset: Any = None,
    recover_gsb: bool = False,
    S_abs: Any = None,
    bleach_result: Any = None,
    gs_spectrum: Any = None,
    em_spectrum: Any = None,
    include_artifact: bool = False,
    metadata: bool = False,
    timestamp: str | None = None,
    write_summary: bool = False,
    directory: str | os.PathLike[str] | None = None,
    fmt: str = "%.12g",
) -> Path:
    """Export fitted species-associated spectra (DAS, EAS, or SAS) to CSV.

    Parameters
    ----------
    fit_or_path, path_or_fit
        The fit result (:class:`~pyrate_ta.results.KineticFit` or
        :class:`~pyrate_ta.io.LoadedFit`) and optional destination path, in
        either order. When no path is supplied, defaults to
        ``<stem>_<DAS|EAS|SAS>_<YYYYMMDD-HHMMSS>.csv``.
    dataset : pymorgan.Dataset1D, optional
        Dataset providing axis units and steady-state reference spectra
        (``gs_spectrum`` / ``em_spectrum``) when GSB/SE recovery is enabled.
    recover_gsb : bool, default False
        When ``True`` (or when ``S_abs`` is supplied), append pure-positive
        GSB/SE-recovered columns ``<KIND>_<species>_abs`` side by side after
        the differential ``<KIND>_<species>`` columns in the same CSV.
    S_abs : array_like, optional
        Pre-computed pure-positive spectra ``(Np, n_kinetic)``.
    gs_spectrum, em_spectrum : array_like, optional
        Explicit steady-state absorption / emission reference spectra on the
        dataset or fitted probe grid.
    include_artifact : bool, default False
        Append coherent-artefact amplitude spectra (``Art_IRF``,
        ``Art_dIRF/dt``, ``Art_d2IRF/dt2``) when present on ``fit``.
    metadata : bool, default False
        Keep ``False`` (default) for a single uncommented header row with zero
        ``#`` lines. Set ``True`` only if ``#`` provenance comments are wanted
        at the top of the file.
    timestamp : str, optional
        Shared ``YYYYMMDD-HHMMSS`` timestamp; defaults to ``fit.timestamp``.
    write_summary : bool, default False
        Also write ``<stem>_fit_summary_<timestamp>.txt`` in the same folder.
    directory : str or Path, optional
        Output folder used when ``path`` is omitted.
    fmt : str, default ``"%.12g"``
        Numeric format string passed to :func:`numpy.savetxt`.

    Returns
    -------
    pathlib.Path
        Path to the written CSV/TSV file.
    """
    from ..settings import get_settings

    fit_obj, target = _resolve_fit_and_path(
        fit_or_path, path_or_fit, fit=fit, path=path, fn_name="export_spectra"
    )
    S_raw = _fit_attr(fit_obj, "S", None)
    if S_raw is None:
        raise ValueError("this fit result carries no spectra array (S is None)")
    S_mat = np.atleast_2d(np.asarray(S_raw, dtype=float))
    Np = S_mat.shape[0]
    n_kin = _n_kinetic(fit_obj)
    n_art = int(_fit_attr(fit_obj, "n_artifact", 0) or 0)

    ts = _resolve_timestamp(fit_obj, timestamp)
    kind = _spectra_kind(fit_obj)
    stem, base_dir = _source_stem_and_dir(fit_obj, directory)
    if target is None:
        target = base_dir / f"{stem}_{kind}_{ts}.csv"
    elif not target.suffix:
        target = target.with_suffix(".csv")

    time_u, probe_lbl, probe_u, _ = _resolve_units(fit_obj, dataset)
    fit_probe = _fit_attr(fit_obj, "probe", None)
    if fit_probe is not None:
        probe_vec = np.asarray(fit_probe, dtype=float).ravel()
        if probe_vec.size != Np:
            raise ValueError(f"fit.probe length ({probe_vec.size}) != S rows ({Np})")
        p_header_base = probe_lbl or "Probe"
        p_header = f"{p_header_base} ({probe_u})" if probe_u else p_header_base
    elif dataset is not None and getattr(dataset, "probe", None) is not None and len(dataset.probe) == Np:
        probe_vec = np.asarray(dataset.probe, dtype=float).ravel()
        p_header_base = probe_lbl or "Probe"
        p_header = f"{p_header_base} ({probe_u})" if probe_u else p_header_base
    else:
        probe_vec = np.arange(Np, dtype=float)
        p_header = "Channel"

    s = get_settings()
    taus = np.atleast_1d(np.asarray(_fit_attr(fit_obj, "taus", []), dtype=float))
    tau_err = np.atleast_1d(
        np.asarray(_fit_attr(fit_obj, "tau_err", np.full(taus.size, np.nan)), dtype=float)
    )
    is_fixed = np.atleast_1d(
        np.asarray(_fit_attr(fit_obj, "is_fixed", np.zeros(taus.size, dtype=bool)), dtype=bool)
    )
    species = _species_names(fit_obj)
    model_type = str(_fit_attr(fit_obj, "model_type", ""))
    is_target = model_type.lower().startswith("target")

    headers = [p_header]
    cols = [probe_vec]

    tau_tags: list[str] = []
    for i in range(n_kin):
        if is_target or i >= taus.size:
            tau_tags.append("")
        else:
            tau_tags.append(
                _format_tau_tag(
                    float(taus[i]),
                    float(tau_err[i]) if i < tau_err.size else np.nan,
                    bool(is_fixed[i]) if i < is_fixed.size else False,
                    time_u,
                    show_uncertainties=bool(s.show_uncertainties),
                    round_uncertainties=bool(s.round_uncertainties),
                )
            )

    for i in range(n_kin):
        sp = species[i] if i < len(species) else f"S{i + 1}"
        tag = tau_tags[i]
        headers.append(f"{kind}_{sp} ({tag})" if tag else f"{kind}_{sp}")
        cols.append(S_mat[:, i])

    S_abs_kin, resolved_bleach = _resolve_gsb_recovery(
        fit_obj,
        probe_vec,
        dataset=dataset,
        recover_gsb=bool(recover_gsb),
        S_abs=S_abs,
        gs_spectrum=gs_spectrum,
        em_spectrum=em_spectrum,
    )
    bleach_res = bleach_result if bleach_result is not None else resolved_bleach
    if S_abs_kin is not None:
        for i in range(n_kin):
            sp = species[i] if i < len(species) else f"S{i + 1}"
            tag = tau_tags[i]
            headers.append(f"{kind}_{sp}_abs ({tag})" if tag else f"{kind}_{sp}_abs")
            cols.append(S_abs_kin[:, i])

    if include_artifact and n_art > 0 and S_mat.shape[1] > n_kin:
        from ..models.irf import ARTIFACT_LABELS

        for j in range(S_mat.shape[1] - n_kin):
            art_lbl = ARTIFACT_LABELS[j] if j < len(ARTIFACT_LABELS) else f"g{j}"
            headers.append(f"Art_{_clean_csv_text(art_lbl)}")
            cols.append(S_mat[:, n_kin + j])

    table = np.column_stack(cols)
    comments = (
        _metadata_comment_lines(fit_obj, ts, kind_title=f"{kind} Spectra") if metadata else None
    )
    written = _write_delimited_table(
        target,
        headers,
        table,
        delimiter=_delimiter_for(target),
        fmt=fmt,
        comment_lines=comments,
    )
    logger.info("Exported %s spectra to %s", kind, written)

    if write_summary:
        summary_path = written.parent / f"{stem}_fit_summary_{ts}.txt"
        export_fit_summary(
            fit_obj,
            summary_path,
            dataset=dataset,
            timestamp=ts,
            bleach_result=bleach_res,
        )
    return written


# --------------------------------------------------------------------------- #
#                    Concentration Profiles C(t) CSV Export                   #
# --------------------------------------------------------------------------- #
def export_concentrations(
    fit_or_path: Any = None,
    path_or_fit: Any = None,
    *,
    fit: Any = None,
    path: str | os.PathLike[str] | None = None,
    dataset: Any = None,
    normalise: bool = False,
    include_irf: bool = True,
    include_artifact: bool = False,
    metadata: bool = False,
    timestamp: str | None = None,
    write_summary: bool = False,
    directory: str | os.PathLike[str] | None = None,
    fmt: str = "%.12g",
) -> Path:
    """Export fitted concentration / population profiles ``C(t)`` to CSV.

    Parameters
    ----------
    fit_or_path, path_or_fit
        The fit result (:class:`~pyrate_ta.results.KineticFit` or
        :class:`~pyrate_ta.io.LoadedFit`) and optional destination path, in
        either order. When no path is supplied, defaults to
        ``<stem>_concentrations_<YYYYMMDD-HHMMSS>.csv``.
    dataset : pymorgan.Dataset1D, optional
        Dataset providing the time-axis unit when not already stored on ``fit``.
    normalise : bool, default False
        Scale each species column to unit peak magnitude, matching
        ``plot_concentrations(..., normalise=True)``.
    include_irf : bool, default True
        Append an ``IRF`` Gaussian response column (matching the grey shaded
        IRF envelope in :func:`~pyrate_ta.plot.plot_concentrations`) whenever
        ``fit.irf_fwhm`` is finite and positive.
    include_artifact : bool, default False
        Append the coherent-artefact design basis columns (``Art_IRF``,
        ``Art_dIRF/dt``, ``Art_d2IRF/dt2``) when present in ``fit.C``.
    metadata : bool, default False
        Keep ``False`` (default) for a single uncommented header row with zero
        ``#`` lines.
    timestamp : str, optional
        Shared ``YYYYMMDD-HHMMSS`` timestamp; defaults to ``fit.timestamp``.
    write_summary : bool, default False
        Also write ``<stem>_fit_summary_<timestamp>.txt`` in the same folder.
    directory : str or Path, optional
        Output folder used when ``path`` is omitted.
    fmt : str, default ``"%.12g"``
        Numeric format string passed to :func:`numpy.savetxt`.

    Returns
    -------
    pathlib.Path
        Path to the written CSV/TSV file.
    """
    from ..settings import get_settings

    fit_obj, target = _resolve_fit_and_path(
        fit_or_path, path_or_fit, fit=fit, path=path, fn_name="export_concentrations"
    )
    C_raw = _fit_attr(fit_obj, "C", None)
    if C_raw is None:
        raise ValueError(
            "this fit result carries no concentration profiles (C is None); "
            "refit with an engine that stores C"
        )
    C_mat = np.atleast_2d(np.asarray(C_raw, dtype=float))
    t_vec = np.asarray(_fit_attr(fit_obj, "t", []), dtype=float).ravel()
    if C_mat.shape[0] != t_vec.size:
        raise ValueError(
            f"C has {C_mat.shape[0]} rows but the delay axis has {t_vec.size} points"
        )

    n_art = int(_fit_attr(fit_obj, "n_artifact", 0) or 0)
    n_kin = max(min(_n_kinetic(fit_obj), C_mat.shape[1]), 0)

    ts = _resolve_timestamp(fit_obj, timestamp)
    stem, base_dir = _source_stem_and_dir(fit_obj, directory)
    if target is None:
        target = base_dir / f"{stem}_concentrations_{ts}.csv"
    elif not target.suffix:
        target = target.with_suffix(".csv")

    time_u, _, _, _ = _resolve_units(fit_obj, dataset)
    delay_header = f"Delay ({time_u})" if time_u else "Delay"

    s = get_settings()
    taus = np.atleast_1d(np.asarray(_fit_attr(fit_obj, "taus", []), dtype=float))
    tau_err = np.atleast_1d(
        np.asarray(_fit_attr(fit_obj, "tau_err", np.full(taus.size, np.nan)), dtype=float)
    )
    is_fixed = np.atleast_1d(
        np.asarray(_fit_attr(fit_obj, "is_fixed", np.zeros(taus.size, dtype=bool)), dtype=bool)
    )
    species = _species_names(fit_obj)
    model_type = str(_fit_attr(fit_obj, "model_type", ""))
    is_target = model_type.lower().startswith("target")

    headers = [delay_header]
    cols = [t_vec]

    for i in range(n_kin):
        y = C_mat[:, i].copy()
        if normalise:
            peak = float(np.nanmax(np.abs(y))) if y.size else 0.0
            if peak > 0:
                y = y / peak
        sp = species[i] if i < len(species) else f"S{i + 1}"
        if is_target or i >= taus.size:
            headers.append(sp)
        else:
            tag = _format_tau_tag(
                float(taus[i]),
                float(tau_err[i]) if i < tau_err.size else np.nan,
                bool(is_fixed[i]) if i < is_fixed.size else False,
                time_u,
                show_uncertainties=bool(s.show_uncertainties),
                round_uncertainties=bool(s.round_uncertainties),
            )
            headers.append(f"{sp} ({tag})")
        cols.append(y)

    irf_fwhm = _fit_attr(fit_obj, "irf_fwhm", None)
    if include_irf and irf_fwhm is not None and np.isfinite(irf_fwhm) and float(irf_fwhm) > 0:
        t0 = float(_fit_attr(fit_obj, "t0", 0.0) or 0.0)
        sigma_irf = float(irf_fwhm) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        irf_curve = np.exp(-0.5 * ((t_vec - t0) / sigma_irf) ** 2)
        if normalise:
            irf_peak = 1.0
        else:
            irf_peak = float(np.nanmax(C_mat[:, :n_kin])) if n_kin > 0 else 1.0
        irf_curve = irf_curve * (irf_peak if irf_peak > 0 else 1.0)
        headers.append("IRF")
        cols.append(irf_curve)

    if include_artifact and n_art > 0 and C_mat.shape[1] > n_kin:
        from ..models.irf import ARTIFACT_LABELS

        for j in range(C_mat.shape[1] - n_kin):
            art_lbl = ARTIFACT_LABELS[j] if j < len(ARTIFACT_LABELS) else f"g{j}"
            headers.append(f"Art_{_clean_csv_text(art_lbl)}")
            cols.append(C_mat[:, n_kin + j])

    table = np.column_stack(cols)
    comments = (
        _metadata_comment_lines(fit_obj, ts, kind_title="Concentration Profiles C(t)")
        if metadata
        else None
    )
    written = _write_delimited_table(
        target,
        headers,
        table,
        delimiter=_delimiter_for(target),
        fmt=fmt,
        comment_lines=comments,
    )
    logger.info("Exported concentration profiles to %s", written)

    if write_summary:
        summary_path = written.parent / f"{stem}_fit_summary_{ts}.txt"
        export_fit_summary(
            fit_obj,
            summary_path,
            dataset=dataset,
            timestamp=ts,
        )
    return written


# --------------------------------------------------------------------------- #
#                 2D Surface Export (Data / Fit / Residuals)                  #
# --------------------------------------------------------------------------- #
_SURFACE_ALIASES: dict[str, str] = {
    "data": "data",
    "d": "data",
    "fit": "fit",
    "fitted": "fit",
    "model": "fit",
    "residuals": "residuals",
    "residual": "residuals",
    "res": "residuals",
    "r": "residuals",
}


def compute_surface_matrix(
    fit: Any,
    kind: str = "fit",
    *,
    include_artifact: bool = True,
    weighted_residuals: bool = False,
) -> tuple[str, np.ndarray]:
    """Return ``(canonical_kind, Z)`` on the ``(Nd, Np)`` PDAT grid (delays x probe).

    * ``"fit"``: ``C @ S.T`` (includes coherent-artefact columns when
      ``include_artifact=True`` so ``data - fit == residuals`` holds identically).
    * ``"residuals"``: unweighted ``R = D - C @ S.T`` on ``fit.R`` (or scaled by
      ``1/sigma`` when ``weighted_residuals=True`` and weights are active).
    * ``"data"``: ``C_full @ S_full.T + R`` on the exact fitted ``(t, probe)``
      window, so all three surfaces share identical ``(Nd, Np)`` axes even when
      reconstructed from a reopened ``.prfit`` session without the raw file.
    """
    key = str(kind or "").strip().lower()
    canon = _SURFACE_ALIASES.get(key)
    if canon is None:
        raise ValueError(
            f"unknown surface kind {kind!r}; expected 'data', 'fit', or 'residuals'"
        )

    C_raw = _fit_attr(fit, "C", None)
    S_raw = _fit_attr(fit, "S", None)
    R_raw = _fit_attr(fit, "R", None)

    if canon == "residuals":
        if R_raw is None:
            raise ValueError("this fit result carries no residual matrix (R is None)")
        R_mat = np.asarray(R_raw, dtype=float)
        if R_mat.ndim == 1:
            R_mat = R_mat[:, None]
        if weighted_residuals:
            weights = _fit_attr(fit, "weights", None)
            w_arr = None
            w_on = False
            if isinstance(weights, dict):
                w_on = bool(weights.get("weighted", False))
                w_arr = weights.get("w")
            elif weights is not None:
                w_on = bool(getattr(weights, "weighted", False))
                w_arr = getattr(weights, "w", None)
            if w_arr is None:
                w_arr = getattr(fit, "weights_w", None)
            if not w_on or w_arr is None:
                raise ValueError(
                    "weighted_residuals=True requires a noise-weighted fit carrying "
                    "per-point 1/sigma weights"
                )
            w = np.asarray(w_arr, dtype=float)
            if w.ndim == 1:
                w = w[:, None]
            R_mat = w * R_mat
        return canon, R_mat

    if C_raw is None or S_raw is None:
        raise ValueError(
            f"this fit result carries no concentration/spectra matrices (C or S is None); "
            f"cannot reconstruct the {canon!r} surface"
        )
    C_full = np.atleast_2d(np.asarray(C_raw, dtype=float))
    S_full = np.atleast_2d(np.asarray(S_raw, dtype=float))

    if canon == "fit":
        if not include_artifact:
            n_kin = _n_kinetic(fit)
            return canon, C_full[:, :n_kin] @ S_full[:, :n_kin].T
        return canon, C_full @ S_full.T

    # canon == "data"
    if R_raw is None:
        raise ValueError(
            "this fit result carries no residual matrix (R is None); "
            "cannot reconstruct the 'data' surface"
        )
    R_mat = np.asarray(R_raw, dtype=float)
    if R_mat.ndim == 1:
        R_mat = R_mat[:, None]
    return canon, (C_full @ S_full.T) + R_mat


def _resolve_surface_axes(
    fit: Any,
    dataset: Any,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Return ``(t_vec, probe_vec, delay_header, probe_header)`` for an ``(Nd, Np)`` surface."""
    Nd, Np = shape
    t_raw = _fit_attr(fit, "t", None)
    t_vec = (
        np.asarray(t_raw, dtype=float).ravel()
        if t_raw is not None and np.size(t_raw) == Nd
        else np.arange(Nd, dtype=float)
    )

    time_u, probe_lbl, probe_u, _ = _resolve_units(fit, dataset)
    delay_header = f"Delay ({time_u})" if time_u else "Delay"

    fit_probe = _fit_attr(fit, "probe", None)
    if fit_probe is not None:
        probe_vec = np.asarray(fit_probe, dtype=float).ravel()
        if probe_vec.size != Np:
            raise ValueError(f"fit.probe length ({probe_vec.size}) != surface columns ({Np})")
        p_base = probe_lbl or "Probe"
        probe_header = f"{p_base} ({probe_u})" if probe_u else p_base
    elif (
        dataset is not None
        and getattr(dataset, "probe", None) is not None
        and len(dataset.probe) == Np
    ):
        probe_vec = np.asarray(dataset.probe, dtype=float).ravel()
        p_base = probe_lbl or "Probe"
        probe_header = f"{p_base} ({probe_u})" if probe_u else p_base
    else:
        probe_vec = np.arange(Np, dtype=float)
        probe_header = "Channel"

    return t_vec, probe_vec, delay_header, probe_header


def _resolve_pdat_units(fit: Any, dataset: Any = None) -> tuple[str, str]:
    """Return ``(time_unit, probe_unit)`` formatted for :func:`pymorgan.write_pdat`."""
    ds_units = (getattr(dataset, "units", None) or {}) if dataset is not None else {}
    f_settings = _fit_settings(fit)

    raw_tu = str(ds_units.get("unitsT_ltx") or f_settings.get("time_unit") or "ps")
    time_u = raw_tu.replace("$", "").strip() or "ps"

    lbl = str(ds_units.get("unitsL_lbl") or f_settings.get("probe_label") or "").strip()
    raw_pu = str(ds_units.get("unitsL_ltx") or f_settings.get("probe_unit") or "").replace("$", "").strip()
    if lbl == "Wavenumber" or "cm" in raw_pu.lower():
        probe_u = "cm^{-1}"
    elif lbl == "Energy" or raw_pu.lower() == "ev":
        probe_u = "eV"
    elif lbl == "Wavelength" or raw_pu.lower() == "nm":
        probe_u = "nm"
    else:
        probe_u = raw_pu or "nm"
    return time_u, probe_u


def _normalize_surface_format(
    surface_format: str | None = None,
    target: Path | None = None,
) -> tuple[str, str]:
    """Return ``(canonical_format, extension)`` where format is ``'csv'``, ``'tsv'``, or ``'pdat'``."""
    if target is not None and target.suffix:
        sfx = target.suffix.lower()
        if sfx == ".pdat":
            return "pdat", ".pdat"
        if sfx in {".tsv", ".txt", ".dat"}:
            return "tsv", sfx
        if sfx == ".csv":
            return "csv", ".csv"

    norm = (surface_format or "csv").strip().lstrip(".").lower()
    if norm == "pdat":
        return "pdat", ".pdat"
    if norm == "tsv":
        return "tsv", ".tsv"
    if norm == "csv":
        return "csv", ".csv"
    raise ValueError(
        f"unsupported surface_format {surface_format!r}; expected 'csv', 'tsv', or 'pdat'"
    )


def _pack_zip_archive(zip_path: Path, files: list[Path]) -> Path:
    """Create a deflate-compressed ``.zip`` archive with flat top-level filenames."""
    zip_target = zip_path if zip_path.suffix.lower() == ".zip" else zip_path.with_suffix(".zip")
    zip_target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_target, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in files:
            zf.write(file_path, arcname=file_path.name)
    logger.info("Packed %d export files into %s", len(files), zip_target)
    return zip_target


def _resolve_bundle_destination(
    fit_obj: Any,
    raw_target: Path | None,
    *,
    ts: str,
    bundle_tag: str,
    as_zip: bool,
    zip_path: str | os.PathLike[str] | None = None,
    stem_override: str | None = None,
) -> tuple[str, Path, Path | None]:
    """Resolve ``(stem, out_dir, zip_target)`` for folder or ``.zip`` bundle exports."""
    default_stem, default_dir = _source_stem_and_dir(
        fit_obj, None if (raw_target is not None and raw_target.suffix.lower() == ".zip") else raw_target
    )
    stem = stem_override or default_stem

    if zip_path is not None:
        zt = Path(zip_path)
        if zt.suffix.lower() != ".zip":
            zt = zt.with_suffix(".zip")
        return stem, default_dir, zt

    if raw_target is not None and raw_target.suffix.lower() == ".zip":
        raw_stem = raw_target.stem
        for suffix in (f"_{bundle_tag}_{ts}", f"_{ts}", f"_{bundle_tag}"):
            if raw_stem.endswith(suffix) and len(raw_stem) > len(suffix):
                raw_stem = raw_stem[: -len(suffix)]
                break
        if stem_override is None and raw_stem:
            stem = raw_stem
        return stem, raw_target.parent, raw_target

    if as_zip:
        return stem, default_dir, default_dir / f"{stem}_{bundle_tag}_{ts}.zip"

    return stem, default_dir, None


def export_surface(
    fit_or_path: Any = None,
    path_or_fit: Any = None,
    *,
    kind: str = "fit",
    fit: Any = None,
    path: str | os.PathLike[str] | None = None,
    dataset: Any = None,
    surface_format: str = "csv",
    include_artifact: bool = True,
    weighted_residuals: bool = False,
    metadata: bool = False,
    timestamp: str | None = None,
    write_summary: bool = False,
    directory: str | os.PathLike[str] | None = None,
    fmt: str = "%.12g",
) -> Path:
    """Export one 2D surface (``data``, ``fit``, or ``residuals``) in ``.csv``, ``.tsv``, or ``.pdat``.

    Surfaces always follow the PyMORGAN ``.pdat`` convention (never transposed):
    rows correspond to delays ``t`` (``Nd``) and columns correspond to probe
    coordinates ``probe`` (``Np``).

    * For ``.csv`` / ``.tsv``: writes a single header row
      ``Delay (ps) \\ Wavelength (nm), <probe_1>, <probe_2>, ...`` with zero ``#``
      comment lines by default, followed by ``[t | Z]``.
    * For ``.pdat``: delegates directly to :func:`pymorgan.write_pdat` so the
      exported surface loads straight back into PyMORGAN or PyRATE-TA via
      ``pymorgan.load_1D(...)``.
    """
    fit_obj, target = _resolve_fit_and_path(
        fit_or_path, path_or_fit, fit=fit, path=path, fn_name="export_surface"
    )
    canon_kind, z_mat = compute_surface_matrix(
        fit_obj,
        kind=kind,
        include_artifact=include_artifact,
        weighted_residuals=weighted_residuals,
    )
    t_vec, probe_vec, delay_hdr, probe_hdr = _resolve_surface_axes(fit_obj, dataset, z_mat.shape)

    ts = _resolve_timestamp(fit_obj, timestamp)
    stem, base_dir = _source_stem_and_dir(fit_obj, directory)
    fmt_kind, ext = _normalize_surface_format(surface_format, target)

    if target is None:
        target = base_dir / f"{stem}_{canon_kind}_{ts}{ext}"
    elif not target.suffix:
        target = target.with_suffix(ext)

    if fmt_kind == "pdat":
        import pymorgan as pm

        pdat_time_u, pdat_probe_u = _resolve_pdat_units(fit_obj, dataset)
        written = pm.write_pdat(
            target,
            t_vec,
            probe_vec,
            z_mat,
            time_unit=pdat_time_u,
            probe_unit=pdat_probe_u,
            noise=None,
        )
    else:
        delim = "\t" if fmt_kind == "tsv" else _delimiter_for(target)
        corner = f"{delay_hdr} \\ {probe_hdr}"
        probe_headers = [(fmt % float(x)).strip() for x in probe_vec]
        headers = [corner, *probe_headers]
        table = np.column_stack([t_vec, z_mat])
        comments = (
            _metadata_comment_lines(
                fit_obj, ts, kind_title=f"{canon_kind.capitalize()} 2D Surface"
            )
            if metadata
            else None
        )
        written = _write_delimited_table(
            target,
            headers,
            table,
            delimiter=delim,
            fmt=fmt,
            comment_lines=comments,
        )

    logger.info("Exported %s surface to %s", canon_kind, written)
    if write_summary:
        summary_path = written.parent / f"{stem}_fit_summary_{ts}.txt"
        export_fit_summary(fit_obj, summary_path, dataset=dataset, timestamp=ts)
    return written


def export_surfaces(
    fit_or_dir: Any = None,
    dir_or_fit: Any = None,
    *,
    fit: Any = None,
    directory: str | os.PathLike[str] | None = None,
    dataset: Any = None,
    surface_format: str = "csv",
    as_zip: bool = False,
    zip_path: str | os.PathLike[str] | None = None,
    keep_files: bool = False,
    stem_override: str | None = None,
    include_artifact: bool = True,
    weighted_residuals: bool = False,
    metadata: bool = False,
    timestamp: str | None = None,
    bleach_result: Any = None,
    fmt: str = "%.12g",
) -> ExportedSurfacePaths:
    """Export all 3 surfaces (``data``, ``fit``, ``residuals``) + ``.txt`` summary at once.

    All files share the same ``YYYYMMDD-HHMMSS`` timestamp and ``.pdat``
    orientation (rows = delays, columns = probe; never transposed). Pass
    ``as_zip=True`` (or a ``.zip`` path as ``directory`` / ``zip_path``) to
    bundle everything into a single ``.zip`` archive.
    """
    fit_obj, dir_target = _resolve_fit_and_path(
        fit_or_dir, dir_or_fit, fit=fit, path=directory, fn_name="export_surfaces"
    )
    ts = _resolve_timestamp(fit_obj, timestamp)
    fmt_kind, ext = _normalize_surface_format(surface_format)
    stem, out_dir, zip_target = _resolve_bundle_destination(
        fit_obj,
        dir_target,
        ts=ts,
        bundle_tag="surfaces",
        as_zip=bool(as_zip),
        zip_path=zip_path,
        stem_override=stem_override,
    )

    def _write_into(folder: Path) -> tuple[Path, Path, Path, Path]:
        d_p = export_surface(
            fit_obj,
            folder / f"{stem}_data_{ts}{ext}",
            kind="data",
            dataset=dataset,
            surface_format=fmt_kind,
            include_artifact=include_artifact,
            metadata=metadata,
            timestamp=ts,
            fmt=fmt,
        )
        f_p = export_surface(
            fit_obj,
            folder / f"{stem}_fit_{ts}{ext}",
            kind="fit",
            dataset=dataset,
            surface_format=fmt_kind,
            include_artifact=include_artifact,
            metadata=metadata,
            timestamp=ts,
            fmt=fmt,
        )
        r_p = export_surface(
            fit_obj,
            folder / f"{stem}_residuals_{ts}{ext}",
            kind="residuals",
            dataset=dataset,
            surface_format=fmt_kind,
            weighted_residuals=weighted_residuals,
            metadata=metadata,
            timestamp=ts,
            fmt=fmt,
        )
        s_p = export_fit_summary(
            fit_obj,
            folder / f"{stem}_fit_summary_{ts}.txt",
            dataset=dataset,
            timestamp=ts,
            bleach_result=bleach_result,
        )
        return d_p, f_p, r_p, s_p

    if zip_target is not None and not keep_files:
        with tempfile.TemporaryDirectory() as tmp:
            d_p, f_p, r_p, s_p = _write_into(Path(tmp))
            archive = _pack_zip_archive(zip_target, [d_p, f_p, r_p, s_p])
        return ExportedSurfacePaths(
            data=Path(d_p.name),
            fit=Path(f_p.name),
            residuals=Path(r_p.name),
            summary=Path(s_p.name),
            timestamp=ts,
            archive=archive,
        )

    d_p, f_p, r_p, s_p = _write_into(out_dir)
    archive = _pack_zip_archive(zip_target, [d_p, f_p, r_p, s_p]) if zip_target is not None else None
    return ExportedSurfacePaths(
        data=d_p,
        fit=f_p,
        residuals=r_p,
        summary=s_p,
        timestamp=ts,
        archive=archive,
    )


# --------------------------------------------------------------------------- #
#           Complete Fit Bundle Export (Folder or Single .zip File)           #
# --------------------------------------------------------------------------- #
def export_fit_results(
    fit_or_dir: Any = None,
    dir_or_fit: Any = None,
    *,
    fit: Any = None,
    directory: str | os.PathLike[str] | None = None,
    dataset: Any = None,
    recover_gsb: bool = False,
    S_abs: Any = None,
    bleach_result: Any = None,
    gs_spectrum: Any = None,
    em_spectrum: Any = None,
    normalise_concentrations: bool = False,
    include_irf: bool = True,
    include_artifact: bool = False,
    include_surfaces: bool = True,
    surface_format: str = "csv",
    include_artifact_in_surfaces: bool = True,
    weighted_residuals: bool = False,
    as_zip: bool = False,
    zip_path: str | os.PathLike[str] | None = None,
    keep_files: bool = False,
    stem_override: str | None = None,
    metadata: bool = False,
    timestamp: str | None = None,
    fmt: str = "%.12g",
) -> ExportedFitPaths:
    """Export DAS/EAS/SAS, concentrations, 2D surfaces, and ``.txt`` summary together.

    Accepts ``export_fit_results(fit, directory=None)`` or
    ``export_fit_results(directory, fit)`` (including a ``.zip`` path). Every
    output file shares the same ``YYYYMMDD-HHMMSS`` timestamp:

    * ``<stem>_<DAS|EAS|SAS>_<timestamp>.csv``
    * ``<stem>_concentrations_<timestamp>.csv``
    * ``<stem>_data_<timestamp>.<csv|tsv|pdat>`` (when ``include_surfaces=True``)
    * ``<stem>_fit_<timestamp>.<csv|tsv|pdat>`` (when ``include_surfaces=True``)
    * ``<stem>_residuals_<timestamp>.<csv|tsv|pdat>`` (when ``include_surfaces=True``)
    * ``<stem>_fit_summary_<timestamp>.txt``

    Set ``as_zip=True`` (or pass a ``.zip`` destination path) to bundle all
    files into a single ``<stem>_fit_results_<timestamp>.zip`` archive.
    """
    fit_obj, dir_target = _resolve_fit_and_path(
        fit_or_dir, dir_or_fit, fit=fit, path=directory, fn_name="export_fit_results"
    )
    ts = _resolve_timestamp(fit_obj, timestamp)
    fmt_kind, surf_ext = _normalize_surface_format(surface_format)
    stem, out_dir, zip_target = _resolve_bundle_destination(
        fit_obj,
        dir_target,
        ts=ts,
        bundle_tag="fit_results",
        as_zip=bool(as_zip),
        zip_path=zip_path,
        stem_override=stem_override,
    )
    kind = _spectra_kind(fit_obj)

    S_mat = np.atleast_2d(np.asarray(_fit_attr(fit_obj, "S"), dtype=float))
    Np = S_mat.shape[0]
    fit_probe = _fit_attr(fit_obj, "probe", None)
    if fit_probe is not None:
        probe_vec = np.asarray(fit_probe, dtype=float).ravel()
    elif dataset is not None and getattr(dataset, "probe", None) is not None and len(dataset.probe) == Np:
        probe_vec = np.asarray(dataset.probe, dtype=float).ravel()
    else:
        probe_vec = np.arange(Np, dtype=float)

    S_abs_kin, resolved_bleach = _resolve_gsb_recovery(
        fit_obj,
        probe_vec,
        dataset=dataset,
        recover_gsb=bool(recover_gsb),
        S_abs=S_abs,
        gs_spectrum=gs_spectrum,
        em_spectrum=em_spectrum,
    )
    bleach_res = bleach_result if bleach_result is not None else resolved_bleach

    def _write_bundle(folder: Path):
        spec_p = export_spectra(
            fit_obj,
            folder / f"{stem}_{kind}_{ts}.csv",
            dataset=dataset,
            recover_gsb=False,
            S_abs=S_abs_kin,
            include_artifact=include_artifact,
            metadata=metadata,
            timestamp=ts,
            write_summary=False,
            fmt=fmt,
        )
        conc_p = export_concentrations(
            fit_obj,
            folder / f"{stem}_concentrations_{ts}.csv",
            dataset=dataset,
            normalise=normalise_concentrations,
            include_irf=include_irf,
            include_artifact=include_artifact,
            metadata=metadata,
            timestamp=ts,
            write_summary=False,
            fmt=fmt,
        )
        data_p = fit_p = res_p = None
        if include_surfaces:
            data_p = export_surface(
                fit_obj,
                folder / f"{stem}_data_{ts}{surf_ext}",
                kind="data",
                dataset=dataset,
                surface_format=fmt_kind,
                include_artifact=include_artifact_in_surfaces,
                metadata=metadata,
                timestamp=ts,
                fmt=fmt,
            )
            fit_p = export_surface(
                fit_obj,
                folder / f"{stem}_fit_{ts}{surf_ext}",
                kind="fit",
                dataset=dataset,
                surface_format=fmt_kind,
                include_artifact=include_artifact_in_surfaces,
                metadata=metadata,
                timestamp=ts,
                fmt=fmt,
            )
            res_p = export_surface(
                fit_obj,
                folder / f"{stem}_residuals_{ts}{surf_ext}",
                kind="residuals",
                dataset=dataset,
                surface_format=fmt_kind,
                weighted_residuals=weighted_residuals,
                metadata=metadata,
                timestamp=ts,
                fmt=fmt,
            )
        sum_p = export_fit_summary(
            fit_obj,
            folder / f"{stem}_fit_summary_{ts}.txt",
            dataset=dataset,
            timestamp=ts,
            bleach_result=bleach_res,
        )
        written_list = [spec_p, conc_p]
        if include_surfaces and data_p is not None and fit_p is not None and res_p is not None:
            written_list.extend([data_p, fit_p, res_p])
        written_list.append(sum_p)
        return spec_p, conc_p, sum_p, data_p, fit_p, res_p, written_list

    if zip_target is not None and not keep_files:
        with tempfile.TemporaryDirectory() as tmp:
            spec_p, conc_p, sum_p, data_p, fit_p, res_p, files = _write_bundle(Path(tmp))
            archive = _pack_zip_archive(zip_target, files)
        return ExportedFitPaths(
            spectra=Path(spec_p.name),
            concentrations=Path(conc_p.name),
            summary=Path(sum_p.name),
            timestamp=ts,
            data=Path(data_p.name) if data_p is not None else None,
            fit=Path(fit_p.name) if fit_p is not None else None,
            residuals=Path(res_p.name) if res_p is not None else None,
            archive=archive,
        )

    spec_p, conc_p, sum_p, data_p, fit_p, res_p, files = _write_bundle(out_dir)
    archive = _pack_zip_archive(zip_target, files) if zip_target is not None else None
    return ExportedFitPaths(
        spectra=spec_p,
        concentrations=conc_p,
        summary=sum_p,
        timestamp=ts,
        data=data_p,
        fit=fit_p,
        residuals=res_p,
        archive=archive,
    )
