"""Saving and reopening a fit session.

One file, ``.prfit``, holding both halves of a result: the arrays (spectra,
concentration profiles, residuals, axes) as a compressed ``.npz``, and the
reproducibility payload as a JSON string inside it. Keeping them together means
a saved fit cannot lose its provenance -- a spectra file with no record of the
model, the window fitted or the noise weighting is not reproducible, and the
whole point of the payload is that it travels with the numbers.

What is written
---------------
* model identity: family, component count, and the scheme *in its own
  notation* when there is one, so a target fit reopens as editable text;
* the parameters, their uncertainties, the fixed/free mask and the bounds;
* what the fit actually saw: delay and probe ranges, clipping, detector,
  source file, and where the noise came from;
* the solver's report and the statistic, including which kind it is.

The format is PyRATE's own and deliberately simple: ``numpy`` and a JSON
document, no third-party dependency and nothing that needs this package's
version to read.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..log import get_logger

logger = get_logger(__name__)

#: Bumped when the layout of the JSON payload changes incompatibly.
FORMAT_VERSION = 1

_SUFFIX = ".prfit"
_ARRAY_FIELDS = (
    "taus",
    "tau_err",
    "is_fixed",
    "S",
    "S_abs",
    "C",
    "R",
    "t",
    "probe",
    "covariance",
    "K",
)


def _jsonable(value):
    """Convert a payload value into something ``json`` can write."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _metadata(fit) -> dict:
    """The reproducibility payload, as a plain dictionary."""
    from ..__about__ import __version__

    report = getattr(fit, "report", None)
    statistic = getattr(fit, "statistic", None)
    weights = getattr(fit, "weights", None)

    meta = {
        "format_version": FORMAT_VERSION,
        "pyrate_version": __version__,
        "result_class": type(fit).__name__,
        "model_type": str(fit.model_type),
        "n_components": int(fit.n_components),
        # Without this, a reopened fit cannot tell which columns of S are
        # spectra and which are the coherent artefact.
        "n_artifact": int(getattr(fit, "n_artifact", 0) or 0),
        "param_names": list(fit.param_names),
        "t0": float(fit.t0),
        "t0_err": fit.t0_err,
        "irf_fwhm": fit.irf_fwhm,
        "irf_err": fit.irf_err,
        "delay_range": fit.delay_range,
        "probe_range": fit.probe_range,
        "t_min": fit.t_min,
        "detector": int(fit.detector),
        "source": fit.source,
        "timestamp": getattr(fit, "timestamp", None),
        "sorted_by_lifetime": bool(fit.sorted_by_lifetime),
        "settings": dict(fit.settings or {}),
        "initial_params": fit.initial_params,
        "bounds": fit.bounds,
        "gs_scale": getattr(fit, "gs_scale", None),
        "gs_source": getattr(fit, "gs_source", None),
        "a_max": getattr(fit, "a_max", None),
        # A target fit reopens as editable text, not as an opaque matrix.
        "scheme_key": getattr(fit, "scheme_key", None),
        "scheme_label": getattr(fit, "scheme_label", None),
        "scheme_text": getattr(fit, "scheme_text", None),
        "eigenvalues": getattr(fit, "eigenvalues", None),
    }
    if statistic is not None:
        meta["statistic"] = {
            "kind": statistic.kind,
            "value": float(statistic.value),
            "value_nonlinear_dof": statistic.value_nonlinear_dof,
            "n_points": int(statistic.n_points),
            "n_nonlinear": int(statistic.n_nonlinear),
            "n_amplitudes": int(statistic.n_amplitudes),
            "dof": int(statistic.dof),
            "weighted": bool(statistic.weighted),
            "noise_source": statistic.noise_source,
        }
    if report is not None:
        meta["report"] = {
            "converged": bool(report.converged),
            "status": int(report.status),
            "message": str(report.message),
            "n_evaluations": int(report.n_evaluations),
            "cost": float(report.cost),
            "optimality": float(report.optimality),
            "method": str(report.method),
            "max_iterations": int(report.max_iterations),
            "at_bounds": list(report.at_bounds),
        }
    if weights is not None:
        meta["weights"] = {
            "weighted": bool(weights.weighted),
            "n_points": int(weights.n_points),
            "n_masked": int(weights.n_masked),
            "floor": weights.floor,
            "n_floored": int(weights.n_floored),
            "source": weights.source,
        }
    return _jsonable(meta)


def save_fit(path, fit) -> Path:
    """Write a fit result to a ``.prfit`` file.

    Parameters
    ----------
    path : str or Path
        Destination. The ``.prfit`` suffix is added when missing.
    fit : KineticFit
        Any result from :mod:`pyrate_ta.fit`.

    Returns
    -------
    pathlib.Path
        The file actually written.
    """
    target = Path(path)
    if target.suffix != _SUFFIX:
        target = target.with_suffix(_SUFFIX)

    arrays = {}
    for name in _ARRAY_FIELDS:
        value = getattr(fit, name, None)
        if value is not None:
            arrays[name] = np.asarray(value)
    weights = getattr(fit, "weights", None)
    w_arr = (
        getattr(weights, "w", None)
        if weights is not None and getattr(weights, "weighted", False)
        else getattr(fit, "weights_w", None)
    )
    if w_arr is not None:
        arrays["weights_w"] = np.asarray(w_arr)

    with target.open("wb") as handle:
        np.savez_compressed(handle, meta=json.dumps(_metadata(fit), indent=1), **arrays)
    logger.info("Fit session written to %s", target)
    return target


class LoadedFit:
    """A reopened fit: the arrays as attributes, the payload as ``meta``.

    Deliberately *not* a live :class:`~pyrate_ta.results.KineticFit`. A reopened
    session has no dataset behind it and cannot be refitted as it stands, and
    returning something that looks like a fresh result would invite exactly
    that. Everything needed to set the fit up again is in :attr:`meta`.
    """

    def __init__(self, meta: dict, arrays: dict):
        self.meta = meta
        self._arrays = arrays
        for name, value in arrays.items():
            setattr(self, name, value)

    def __getattr__(self, name):  # only reached for names not set above
        if name in self.meta:
            return self.meta[name]
        raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        kind = self.meta.get("model_type", "?")
        n = self.meta.get("n_components", "?")
        return f"LoadedFit({kind}, {n} component(s), from {self.meta.get('source')!r})"

    @property
    def converged(self) -> bool:
        return bool(self.meta.get("report", {}).get("converged", False))

    @property
    def spectra_kind(self) -> str:
        """What this reopened fit's amplitude spectra are called: DAS, EAS or SAS."""
        return {"Parallel": "DAS", "Sequential": "EAS", "Target": "SAS"}.get(
            str(self.meta.get("model_type", "")), "spectra"
        )

    @property
    def n_kinetic(self) -> int:
        """Columns of ``C``/``S`` that are kinetic components, artefact aside."""
        S = getattr(self, "S", None)
        taus = getattr(self, "taus", [])
        total = int(np.shape(S)[1]) if S is not None else len(taus)
        return max(total - int(self.meta.get("n_artifact", 0) or 0), 0)

    def species_labels(self) -> list[str]:
        """Names of the species/compartments for DAS/EAS/SAS plots and exports."""
        from ..plot.concentrations import species_labels_for

        return species_labels_for(self)

    def summary(self) -> str:
        """The same shape of summary a live result prints."""
        from ..helpers import format_lifetime

        lines = [
            f"{self.meta.get('model_type')} fit, {self.meta.get('n_components')} component(s)"
            + (f" from {self.meta['source']}" if self.meta.get("source") else ""),
        ]
        report = self.meta.get("report")
        if report:
            head = "converged" if report["converged"] else "DID NOT CONVERGE"
            lines.append(f"{head} ({report['method']}): {report['message']}")
        taus = np.atleast_1d(getattr(self, "taus", []))
        errs = np.atleast_1d(getattr(self, "tau_err", np.full(len(taus), np.nan)))
        fixed = np.atleast_1d(getattr(self, "is_fixed", np.zeros(len(taus), bool)))
        for i, tau in enumerate(taus):
            flag = " (fixed)" if bool(fixed[i]) else ""
            err = errs[i] if i < errs.size else np.nan
            if np.isfinite(err):
                lines.append(f"  tau{i + 1} = {format_lifetime(tau)} +/- {err:.3g}{flag}")
            else:
                lines.append(f"  tau{i + 1} = {format_lifetime(tau)}{flag}")
        statistic = self.meta.get("statistic")
        if statistic:
            lines.append(f"  {statistic['kind']} = {statistic['value']:.5g}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #                             Exporting                              #
    # ------------------------------------------------------------------ #
    def format_report(
        self,
        *,
        dataset=None,
        timestamp: str | None = None,
        bleach_result=None,
    ) -> str:
        """Format a detailed terminal-style fit report string."""
        from .export import format_fit_report

        return format_fit_report(
            self, dataset=dataset, timestamp=timestamp, bleach_result=bleach_result
        )

    def export_spectra(
        self,
        path=None,
        *,
        dataset=None,
        recover_gsb: bool = False,
        S_abs=None,
        bleach_result=None,
        gs_spectrum=None,
        em_spectrum=None,
        include_artifact: bool = False,
        metadata: bool = False,
        timestamp: str | None = None,
        write_summary: bool = False,
        directory=None,
        fmt: str = "%.12g",
    ) -> Path:
        """Export DAS/EAS/SAS spectra to a single-header CSV file."""
        from .export import export_spectra

        return export_spectra(
            self,
            path,
            dataset=dataset,
            recover_gsb=recover_gsb,
            S_abs=S_abs,
            bleach_result=bleach_result,
            gs_spectrum=gs_spectrum,
            em_spectrum=em_spectrum,
            include_artifact=include_artifact,
            metadata=metadata,
            timestamp=timestamp,
            write_summary=write_summary,
            directory=directory,
            fmt=fmt,
        )

    def export_concentrations(
        self,
        path=None,
        *,
        dataset=None,
        normalise: bool = False,
        include_irf: bool = True,
        include_artifact: bool = False,
        metadata: bool = False,
        timestamp: str | None = None,
        write_summary: bool = False,
        directory=None,
        fmt: str = "%.12g",
    ) -> Path:
        """Export concentration profiles ``C(t)`` to a single-header CSV file."""
        from .export import export_concentrations

        return export_concentrations(
            self,
            path,
            dataset=dataset,
            normalise=normalise,
            include_irf=include_irf,
            include_artifact=include_artifact,
            metadata=metadata,
            timestamp=timestamp,
            write_summary=write_summary,
            directory=directory,
            fmt=fmt,
        )

    @property
    def fitted_surface(self) -> np.ndarray:
        """Reconstructed 2D fitted surface ``C @ S.T`` of shape ``(Nd, Np)``."""
        from .export import compute_surface_matrix

        return compute_surface_matrix(self, "fit")[1]

    @property
    def data_surface(self) -> np.ndarray:
        """Fitted-window 2D data surface ``C @ S.T + R`` of shape ``(Nd, Np)``."""
        from .export import compute_surface_matrix

        return compute_surface_matrix(self, "data")[1]

    def export_surface(
        self,
        path=None,
        *,
        kind: str = "fit",
        dataset=None,
        surface_format: str = "csv",
        include_artifact: bool = True,
        weighted_residuals: bool = False,
        metadata: bool = False,
        timestamp: str | None = None,
        write_summary: bool = False,
        directory=None,
        fmt: str = "%.12g",
    ) -> Path:
        """Export one 2D surface (``data``, ``fit``, or ``residuals``) as CSV/TSV or PDAT."""
        from .export import export_surface

        return export_surface(
            self,
            path,
            kind=kind,
            dataset=dataset,
            surface_format=surface_format,
            include_artifact=include_artifact,
            weighted_residuals=weighted_residuals,
            metadata=metadata,
            timestamp=timestamp,
            write_summary=write_summary,
            directory=directory,
            fmt=fmt,
        )

    def export_surfaces(
        self,
        directory=None,
        *,
        dataset=None,
        surface_format: str = "csv",
        as_zip: bool = False,
        zip_path=None,
        keep_files: bool = False,
        stem_override: str | None = None,
        include_artifact: bool = True,
        weighted_residuals: bool = False,
        metadata: bool = False,
        timestamp: str | None = None,
        bleach_result=None,
        fmt: str = "%.12g",
    ):
        """Export all 3 surfaces (``data``, ``fit``, ``residuals``) + ``.txt`` summary."""
        from .export import export_surfaces

        return export_surfaces(
            self,
            directory,
            dataset=dataset,
            surface_format=surface_format,
            as_zip=as_zip,
            zip_path=zip_path,
            keep_files=keep_files,
            stem_override=stem_override,
            include_artifact=include_artifact,
            weighted_residuals=weighted_residuals,
            metadata=metadata,
            timestamp=timestamp,
            bleach_result=bleach_result,
            fmt=fmt,
        )

    def export_summary(
        self,
        path=None,
        *,
        dataset=None,
        timestamp: str | None = None,
        bleach_result=None,
        directory=None,
    ) -> Path:
        """Export the companion human-readable ``.txt`` fit report."""
        from .export import export_fit_summary

        return export_fit_summary(
            self,
            path,
            dataset=dataset,
            timestamp=timestamp,
            bleach_result=bleach_result,
            directory=directory,
        )

    def export_results(
        self,
        directory=None,
        *,
        dataset=None,
        recover_gsb: bool = False,
        S_abs=None,
        bleach_result=None,
        gs_spectrum=None,
        em_spectrum=None,
        normalise_concentrations: bool = False,
        include_irf: bool = True,
        include_artifact: bool = False,
        include_surfaces: bool = True,
        surface_format: str = "csv",
        include_artifact_in_surfaces: bool = True,
        weighted_residuals: bool = False,
        as_zip: bool = False,
        zip_path=None,
        keep_files: bool = False,
        stem_override: str | None = None,
        metadata: bool = False,
        timestamp: str | None = None,
        fmt: str = "%.12g",
    ):
        """Export spectra, concentrations, 2D surfaces, and summary TXT (or single ``.zip``)."""
        from .export import export_fit_results

        return export_fit_results(
            self,
            directory,
            dataset=dataset,
            recover_gsb=recover_gsb,
            S_abs=S_abs,
            bleach_result=bleach_result,
            gs_spectrum=gs_spectrum,
            em_spectrum=em_spectrum,
            normalise_concentrations=normalise_concentrations,
            include_irf=include_irf,
            include_artifact=include_artifact,
            include_surfaces=include_surfaces,
            surface_format=surface_format,
            include_artifact_in_surfaces=include_artifact_in_surfaces,
            weighted_residuals=weighted_residuals,
            as_zip=as_zip,
            zip_path=zip_path,
            keep_files=keep_files,
            stem_override=stem_override,
            metadata=metadata,
            timestamp=timestamp,
            fmt=fmt,
        )


def load_fit(path) -> LoadedFit:
    """Reopen a ``.prfit`` file.

    Raises
    ------
    ValueError
        If the file was written by a newer, incompatible format version. A
        silently mis-read session would be worse than a refusal.
    """
    target = Path(path)
    with np.load(target, allow_pickle=False) as data:
        if "meta" not in data:
            raise ValueError(f"{target} is not a PyRATE-TA fit session (no metadata)")
        meta = json.loads(str(data["meta"]))
        arrays = {key: data[key] for key in data.files if key != "meta"}

    version = int(meta.get("format_version", 0))
    if version > FORMAT_VERSION:
        raise ValueError(
            f"{target} was written by a newer PyRATE-TA (format {version}; this build reads "
            f"{FORMAT_VERSION}). Update PyRATE-TA to open it."
        )
    logger.info("Loaded fit session %s (%s)", target.name, meta.get("model_type"))
    return LoadedFit(meta, arrays)


def default_session_path(fit, directory=None) -> Path:
    """Where a session is written when nobody chooses: beside the dataset.

    ``scan.pdat`` gives ``scan_fit.prfit``. With no source (an array fit) the
    name falls back to ``pyrate_fit.prfit`` in ``directory`` or the working
    directory.
    """
    source = getattr(fit, "source", None)
    if source:
        base = Path(source)
        return base.with_name(f"{base.stem}_fit{_SUFFIX}")
    return Path(directory or Path.cwd()) / f"pyrate_fit{_SUFFIX}"
