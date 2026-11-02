"""Tests for tabular DAS/EAS/SAS, C(t), and companion .txt summary exports."""

from __future__ import annotations

import numpy as np
import pymorgan as pm
import pytest

import pyrate_ta as pr
from pyrate_ta.io.reference import load_reference_spectrum

T = np.concatenate([np.linspace(-1.0, 0.5, 12), np.geomspace(0.6, 1500.0, 90)])
PROBE = np.linspace(450.0, 650.0, 40)


def _make_dataset(
    *,
    ir: bool = False,
    with_refs: bool = False,
    noise_std: float = 0.0,
    source: str | None = None,
):
    probe = np.linspace(1900.0, 2050.0, 40) if ir else PROBE.copy()
    units = (
        {
            "unitsT_lbl": "Delay",
            "unitsT_ltx": "ps",
            "unitsL_lbl": "Wavenumber",
            "unitsL_ltx": "cm^{-1}",
            "unitsZ_lbl": r"$\Delta$A",
            "unitsZ_ltx": "mOD",
        }
        if ir
        else {
            "unitsT_lbl": "Delay",
            "unitsT_ltx": "ps",
            "unitsL_lbl": "Wavelength",
            "unitsL_ltx": "nm",
            "unitsZ_lbl": r"$\Delta$A",
            "unitsZ_ltx": "mOD",
        }
    )
    model = pr.make_model("Sequential", 2, irf_fwhm=0.25)
    C = model.concentrations(T, taus=[4.0, 120.0])

    gs = np.exp(-0.5 * ((probe - (1950.0 if ir else 520.0)) / 20.0) ** 2)
    em = np.exp(-0.5 * ((probe - (1990.0 if ir else 585.0)) / 25.0) ** 2)
    s1 = 0.9 * np.exp(-0.5 * ((probe - (1930.0 if ir else 485.0)) / 18.0) ** 2) - 0.5 * gs
    s2 = 0.5 * np.exp(-0.5 * ((probe - (1975.0 if ir else 550.0)) / 22.0) ** 2) - 0.35 * em
    S = np.column_stack([s1, s2])
    D = C @ S.T
    zstdv = None
    if noise_std > 0.0:
        rng = np.random.default_rng(7)
        D = D + rng.normal(0.0, noise_std, size=D.shape)
        zstdv = np.full((*D.shape, 1), noise_std)
    ds = pm.Dataset1D(D[:, :, None], T, probe, units, Zstdv=zstdv, source=source)
    if with_refs:
        ds.gs_spectrum = gs
        ds.em_spectrum = em
    return ds, S, C, gs, em


def test_single_header_csv_parses_everywhere(tmp_path):
    """Default CSV has zero '#' lines and parses in loadtxt, genfromtxt, and load_reference_spectrum."""
    ds, S_true, C_true, _, _ = _make_dataset(source=str(tmp_path / "sample01.pdat"))
    fit = pr.fit_global(ds, taus=[2.0, 80.0], irf_fwhm=0.25, model_type="Sequential")

    paths = fit.export_results(tmp_path)
    assert paths.spectra.exists()
    assert paths.concentrations.exists()
    assert paths.summary.exists()

    # Unified timestamp across all three files
    ts = paths.timestamp
    assert ts == fit.timestamp
    assert paths.spectra.name == f"sample01_EAS_{ts}.csv"
    assert paths.concentrations.name == f"sample01_concentrations_{ts}.csv"
    assert paths.summary.name == f"sample01_fit_summary_{ts}.txt"

    # Zero '#' comment lines in CSVs; single header row
    spec_lines = paths.spectra.read_text(encoding="utf-8").splitlines()
    assert not any(line.startswith("#") for line in spec_lines)
    header = spec_lines[0].split(",")
    assert header[0] == "Wavelength (nm)"
    assert header[1].startswith("EAS_A (")
    assert header[2].startswith("EAS_B (")

    # Standard NumPy parsers succeed without custom comment juggling
    arr_loadtxt = np.loadtxt(paths.spectra, delimiter=",", skiprows=1)
    assert arr_loadtxt.shape == (PROBE.size, 3)
    assert np.allclose(arr_loadtxt[:, 0], PROBE)
    assert np.allclose(arr_loadtxt[:, 1:], fit.S[:, :2], atol=1e-8)

    arr_gen = np.genfromtxt(paths.spectra, delimiter=",", names=True, encoding="utf-8")
    assert len(arr_gen.dtype.names) == 3
    assert arr_gen.shape == (PROBE.size,)

    # PyRATE-TA's own reference loader reads the exported CSV cleanly
    x_ref, y_ref = load_reference_spectrum(paths.spectra)
    assert np.allclose(x_ref, PROBE)
    assert np.allclose(y_ref, fit.S[:, 0], atol=1e-8)

    # Concentrations CSV includes Delay (ps), A, B, and IRF
    conc_lines = paths.concentrations.read_text(encoding="utf-8").splitlines()
    assert not any(line.startswith("#") for line in conc_lines)
    c_header = conc_lines[0].split(",")
    assert c_header[0] == "Delay (ps)"
    assert c_header[-1] == "IRF"
    c_mat = np.loadtxt(paths.concentrations, delimiter=",", skiprows=1)
    assert c_mat.shape == (fit.t.size, 4)
    assert np.allclose(c_mat[:, 1:3], fit.C[:, :2], atol=1e-8)


def test_ir_units_and_parallel_das_with_infinite_offset(tmp_path):
    """LaTeX units like cm^{-1} clean to cm-1 and infinite components label as 'inf fixed'."""
    ds, _, _, _, _ = _make_dataset(ir=True, source=str(tmp_path / "ir_run.pdat"))
    fit = pr.fit_global(ds, taus=[4.0, 120.0, np.inf], irf_fwhm=0.25, model_type="Parallel")

    spec_p = pr.export_spectra(tmp_path / "ir_das.csv", fit)
    header = spec_p.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header[0] == "Wavenumber (cm-1)"
    assert header[1].startswith("DAS_A (")
    assert header[3] == "DAS_C (inf fixed)"


def test_gsb_se_recovery_exports_differential_and_abs_side_by_side(tmp_path):
    """When recover_gsb=True, both differential and _abs columns sit side by side in one CSV."""
    ds, _, _, _, _ = _make_dataset(with_refs=True, source=str(tmp_path / "bleach.pdat"))
    fit = pr.fit_global(ds, taus=[3.0, 100.0], irf_fwhm=0.25, model_type="Sequential")

    paths = fit.export_results(tmp_path, dataset=ds, recover_gsb=True)
    header = paths.spectra.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert len(header) == 5
    assert header[0] == "Wavelength (nm)"
    assert header[1].startswith("EAS_A (")
    assert header[2].startswith("EAS_B (")
    assert header[3].startswith("EAS_A_abs (")
    assert header[4].startswith("EAS_B_abs (")

    mat = np.loadtxt(paths.spectra, delimiter=",", skiprows=1)
    # Differential spectra still carry negative GSB/SE dips
    assert np.min(mat[:, 1:3]) < -0.05
    # Pure positive recovered columns are >= 0 everywhere
    assert np.min(mat[:, 3:5]) >= -1e-12
    assert np.max(mat[:, 3:5]) > 0.2

    # Companion .txt report records GSB/SE scale factors
    report = paths.summary.read_text(encoding="utf-8")
    assert "GSB / SE Pure Positive Recovery" in report
    assert "GSB scale (a)" in report
    assert "SE scale (b)" in report


def test_target_fit_sas_and_k_matrix_in_summary(tmp_path):
    """TargetFit exports SAS_<compartment> columns and writes K-matrix + eigenvalues to .txt."""
    scheme = pr.scheme_from_text("ICT -> T1 : k1\nT1 -> : k2")
    model = pr.make_model("Target", 0, scheme=scheme, irf_fwhm=0.2)
    C = model.concentrations(T, taus=[6.0, 200.0])
    S = np.column_stack(
        [
            np.exp(-0.5 * ((PROBE - 490.0) / 20.0) ** 2),
            np.exp(-0.5 * ((PROBE - 560.0) / 22.0) ** 2),
        ]
    )
    ds = pm.Dataset1D(
        (C @ S.T)[:, :, None],
        T,
        PROBE,
        {"unitsT_ltx": "ps", "unitsL_lbl": "Wavelength", "unitsL_ltx": "nm"},
        source=str(tmp_path / "dyad.pdat"),
    )
    fit = pr.fit_target(ds, scheme=scheme, taus=[3.0, 120.0], irf_fwhm=0.2)

    paths = pr.export_fit_results(tmp_path, fit)
    assert paths.spectra.name.startswith("dyad_SAS_")
    spec_hdr = paths.spectra.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert spec_hdr == ["Wavelength (nm)", "SAS_ICT", "SAS_T1"]

    conc_hdr = paths.concentrations.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert conc_hdr == ["Delay (ps)", "ICT", "T1", "IRF"]

    txt = paths.summary.read_text(encoding="utf-8")
    assert "Target Fit Report" in txt
    assert "Target Model Details" in txt
    assert "ICT -> T1 : k1" in txt
    assert "Rate matrix K" in txt
    assert "Eigenvalues & apparent lifetimes" in txt
    assert "Berberan-Santos" in txt


def test_coherent_artifact_and_normalised_concentrations_options(tmp_path):
    """include_artifact=True adds Art_* columns without colliding with the IRF column."""
    ds, _, _, _, _ = _make_dataset()
    fit = pr.fit_global(
        ds,
        taus=[4.0, 120.0],
        irf_fwhm=0.25,
        coherent_artifact=True,
        model_type="Sequential",
    )

    spec_p = fit.export_spectra(tmp_path / "with_art.tsv", include_artifact=True)
    # .tsv auto-switches to tab delimiter
    first_line = spec_p.read_text(encoding="utf-8").splitlines()[0]
    assert "\t" in first_line and "," not in first_line
    spec_cols = first_line.split("\t")
    assert spec_cols[-3:] == ["Art_IRF", "Art_dIRF/dt", "Art_d2IRF/dt2"]

    conc_p = fit.export_concentrations(
        tmp_path / "conc_norm.csv",
        normalise=True,
        include_irf=True,
        include_artifact=True,
    )
    c_cols = conc_p.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "IRF" in c_cols
    assert c_cols[-3:] == ["Art_IRF", "Art_dIRF/dt", "Art_d2IRF/dt2"]
    assert len(set(c_cols)) == len(c_cols)  # all column names unique

    c_mat = np.loadtxt(conc_p, delimiter=",", skiprows=1)
    assert np.isclose(np.max(np.abs(c_mat[:, 1])), 1.0)
    assert np.isclose(np.max(np.abs(c_mat[:, 2])), 1.0)

    report = fit.format_report()
    assert "Kovalenko" in report


def test_loaded_prfit_session_exports_identically(tmp_path):
    """A saved and reopened .prfit session supports the full export API."""
    ds, _, _, _, _ = _make_dataset(source=str(tmp_path / "saved_scan.pdat"))
    fit = pr.fit_global(ds, taus=[4.0, 120.0], irf_fwhm=0.25, model_type="Sequential")

    session_file = pr.save_fit(tmp_path / "saved_scan", fit)
    reopened = pr.load_fit(session_file)

    out_dir = tmp_path / "from_loaded"
    paths = reopened.export_results(out_dir)
    assert paths.timestamp == fit.timestamp
    assert paths.spectra.exists()
    assert paths.concentrations.exists()
    assert paths.summary.exists()

    mat = np.loadtxt(paths.spectra, delimiter=",", skiprows=1)
    assert np.allclose(mat[:, 1:], fit.S[:, :2])


def test_non_converged_fit_warns_in_summary_report():
    """A non-converged fit clearly flags DID NOT CONVERGE in the .txt report."""
    ds, _, _, _, _ = _make_dataset()
    fit = pr.fit_global(ds, taus=[0.8, 800.0], irf_fwhm=0.25, max_iterations=2)
    assert fit.converged is False
    report = pr.format_fit_report(fit)
    assert "DID NOT CONVERGE" in report
    assert "WARNING" in report


def test_export_rejects_missing_fit_or_missing_concentrations():
    """Clear errors when called without a fit result or when C is None."""
    with pytest.raises(TypeError, match="missing required fit result"):
        pr.export_spectra()

    ds, _, _, _, _ = _make_dataset()
    fit = pr.fit_global(ds, taus=[4.0, 120.0], irf_fwhm=0.25)
    fit.C = None
    with pytest.raises(ValueError, match="carries no concentration profiles"):
        fit.export_concentrations()


def test_branched_target_scheme_with_more_rates_than_species(tmp_path):
    """Branched target fit (3 compartments, 4 rate constants) exports 3 SAS columns and 4 rates."""
    scheme = pr.scheme_from_text("S1 -> CT : k1\nS1 -> T1 : k2\nCT -> : k3\nT1 -> : k4")
    assert scheme.n_species == 3 and scheme.n_rates == 4

    model = pr.make_model("Target", 0, scheme=scheme, irf_fwhm=0.2)
    C = model.concentrations(T, taus=[3.0, 9.0, 45.0, 400.0])
    S = np.column_stack(
        [
            np.exp(-0.5 * ((PROBE - 475.0) / 16.0) ** 2),
            np.exp(-0.5 * ((PROBE - 535.0) / 18.0) ** 2),
            np.exp(-0.5 * ((PROBE - 605.0) / 20.0) ** 2),
        ]
    )
    ds = pm.Dataset1D(
        (C @ S.T)[:, :, None],
        T,
        PROBE,
        {"unitsT_ltx": "ps", "unitsL_lbl": "Wavelength", "unitsL_ltx": "nm"},
        source=str(tmp_path / "branched.pdat"),
    )
    fit = pr.fit_target(ds, scheme=scheme, taus=[2.5, 8.0, 40.0, 350.0], irf_fwhm=0.2)
    paths = fit.export_results(tmp_path, dataset=ds)

    spec_hdr = paths.spectra.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert spec_hdr == ["Wavelength (nm)", "SAS_S1", "SAS_CT", "SAS_T1"]
    spec_mat = np.loadtxt(paths.spectra, delimiter=",", skiprows=1)
    assert spec_mat.shape == (PROBE.size, 4)

    conc_hdr = paths.concentrations.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert conc_hdr == ["Delay (ps)", "S1", "CT", "T1", "IRF"]

    txt = paths.summary.read_text(encoding="utf-8")
    assert "3 compartment(s), 4 rate constant(s)" in txt


def test_restricted_probe_and_delay_window_with_gsb_and_weighted_chi2(tmp_path):
    """Restricted probe/delay fit interpolates full-range GSB/SE refs and reports reduced chi2."""
    ds, _, _, _, _ = _make_dataset(
        with_refs=True,
        noise_std=0.005,
        source=str(tmp_path / "windowed.pdat"),
    )

    fit = pr.fit_global(
        ds,
        taus=[3.0, 90.0],
        irf_fwhm=0.25,
        probe_range=(480.0, 610.0),
        delay_range=(0.5, 900.0),
        use_weights=True,
    )
    assert fit.probe.size < PROBE.size

    paths = fit.export_results(tmp_path, dataset=ds, recover_gsb=True)
    spec_mat = np.loadtxt(paths.spectra, delimiter=",", skiprows=1)
    assert spec_mat.shape == (fit.probe.size, 5)  # probe + 2 diff + 2 _abs
    assert np.min(spec_mat[:, 3:5]) >= -1e-12

    txt = paths.summary.read_text(encoding="utf-8")
    assert "1/sigma weighted" in txt
    assert "chi2 / dof_nl" in txt


def test_single_trace_fit_kinetics_and_preview_global_exports(tmp_path):
    """Both single-trace fit_kinetics (probe=None) and preview_global (report=None) export cleanly."""
    y = 0.7 * np.exp(-np.maximum(T, 0.0) / 5.0) + 0.3 * np.exp(-np.maximum(T, 0.0) / 80.0)
    kfit = pr.fit_kinetics(T, y, taus=[3.0, 60.0], delay_range=(0.2, 1500.0))
    kpaths = kfit.export_results(tmp_path / "kinetics")
    k_hdr = kpaths.spectra.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert k_hdr[0] == "Channel"
    k_mat = np.atleast_2d(np.loadtxt(kpaths.spectra, delimiter=",", skiprows=1))
    assert k_mat.shape == (1, 3)

    ds, _, _, _, _ = _make_dataset(source=str(tmp_path / "prev.pdat"))
    prev = pr.preview_global(ds, taus=[4.0, 120.0], irf_fwhm=0.25)
    ppaths = prev.export_results(tmp_path / "preview", dataset=ds)
    ptxt = ppaths.summary.read_text(encoding="utf-8")
    assert "Preview (no optimisation)" in ptxt
    assert "preview evaluation only" in ptxt


def test_surfaces_csv_pdat_orientation_and_algebraic_identity(tmp_path):
    """CSV surfaces strictly follow PDAT orientation (rows=delays, cols=probe) and data - fit == R."""
    ds, _, _, _, _ = _make_dataset(noise_std=0.004, source=str(tmp_path / "surf_scan.pdat"))
    fit = pr.fit_global(
        ds,
        taus=[3.5, 110.0],
        irf_fwhm=0.25,
        coherent_artifact=True,
        delay_range=(float(T[0]), float(T[-1])),
        model_type="Sequential",
    )

    spaths = fit.export_surfaces(tmp_path / "csv_surfaces", dataset=ds, surface_format="csv")
    assert spaths.data.exists()
    assert spaths.fit.exists()
    assert spaths.residuals.exists()
    assert spaths.summary.exists()
    assert spaths.archive is None

    # Zero '#' comment lines; first header cell has Delay \ Wavelength, rest are probe coordinates
    hdr_cells = spaths.fit.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert hdr_cells[0] == r"Delay (ps) \ Wavelength (nm)"
    probe_from_hdr = np.array([float(c) for c in hdr_cells[1:]])
    assert np.allclose(probe_from_hdr, PROBE, atol=1e-8)

    d_tbl = np.loadtxt(spaths.data, delimiter=",", skiprows=1)
    f_tbl = np.loadtxt(spaths.fit, delimiter=",", skiprows=1)
    r_tbl = np.loadtxt(spaths.residuals, delimiter=",", skiprows=1)

    # Strictly (Nd, Np + 1): rows = delays, columns = [t | probe_1 .. probe_Np]
    assert d_tbl.shape == (T.size, PROBE.size + 1)
    assert f_tbl.shape == (T.size, PROBE.size + 1)
    assert r_tbl.shape == (T.size, PROBE.size + 1)
    assert np.allclose(d_tbl[:, 0], T)
    assert np.allclose(d_tbl[:, 1:] - f_tbl[:, 1:], r_tbl[:, 1:], atol=1e-10)
    assert np.allclose(f_tbl[:, 1:], fit.fitted_surface, atol=1e-10)
    assert np.allclose(d_tbl[:, 1:], fit.data_surface, atol=1e-10)


def test_surfaces_pdat_roundtrip_uvvis_and_ir_via_pymorgan(tmp_path):
    """Exported .pdat surfaces reload directly via pymorgan.load_1D with exact axes and units."""
    for ir_flag, expected_lbl in ((False, "Wavelength"), (True, "Wavenumber")):
        tag = "ir" if ir_flag else "uv"
        ds, _, _, _, _ = _make_dataset(
            ir=ir_flag, noise_std=0.003, source=str(tmp_path / f"{tag}.pdat")
        )
        fit = pr.fit_global(ds, taus=[4.0, 120.0], irf_fwhm=0.25)
        spaths = pr.export_surfaces(
            fit, tmp_path / f"pdat_{tag}", dataset=ds, surface_format="pdat"
        )
        assert spaths.data.suffix == ".pdat"
        assert spaths.fit.suffix == ".pdat"
        assert spaths.residuals.suffix == ".pdat"

        loaded_fit_ds = pm.load_1D(spaths.fit)
        loaded_res_ds = pm.load_1D(spaths.residuals)
        assert loaded_fit_ds.units["unitsL_lbl"] == expected_lbl
        assert np.allclose(loaded_fit_ds.delays, fit.t)
        assert np.allclose(loaded_fit_ds.probe, fit.probe)
        assert np.allclose(loaded_fit_ds.Z[:, :, 0], fit.fitted_surface, atol=1e-12)
        assert np.allclose(loaded_res_ds.Z[:, :, 0], fit.R, atol=1e-12)


def test_export_fit_results_default_surfaces_and_single_zip_bundle(tmp_path):
    """export_fit_results includes all 3 surfaces by default and supports single-file .zip output."""
    import zipfile

    ds, _, _, _, _ = _make_dataset(with_refs=True, source=str(tmp_path / "fullscan.pdat"))
    fit = pr.fit_global(ds, taus=[4.0, 120.0], irf_fwhm=0.25, model_type="Sequential")

    # 1) Folder export: writes all 6 files (EAS, concentrations, data, fit, residuals, summary)
    folder_dir = tmp_path / "folder_bundle"
    paths = fit.export_results(folder_dir, dataset=ds, recover_gsb=True)
    assert paths.archive is None
    for p in (
        paths.spectra,
        paths.concentrations,
        paths.data,
        paths.fit,
        paths.residuals,
        paths.summary,
    ):
        assert p is not None and p.exists()

    # 2) Single .zip archive export (with PDAT surfaces inside): writes ONLY the .zip file
    zip_dir = tmp_path / "zip_only"
    zip_dir.mkdir()
    zpaths = fit.export_results(
        zip_dir,
        dataset=ds,
        recover_gsb=True,
        surface_format="pdat",
        as_zip=True,
    )
    assert zpaths.archive is not None and zpaths.archive.exists()
    assert list(zip_dir.iterdir()) == [zpaths.archive]

    ts = zpaths.timestamp
    with zipfile.ZipFile(zpaths.archive) as zf:
        members = set(zf.namelist())
        expected = {
            f"fullscan_EAS_{ts}.csv",
            f"fullscan_concentrations_{ts}.csv",
            f"fullscan_data_{ts}.pdat",
            f"fullscan_fit_{ts}.pdat",
            f"fullscan_residuals_{ts}.pdat",
            f"fullscan_fit_summary_{ts}.txt",
        }
        assert members == expected
        # Extract the fit .pdat from the .zip and verify PyMORGAN loads it cleanly
        extracted = tmp_path / "unpacked"
        zf.extract(f"fullscan_fit_{ts}.pdat", path=extracted)
        reloaded_pdat = pm.load_1D(extracted / f"fullscan_fit_{ts}.pdat")
        assert np.allclose(reloaded_pdat.Z[:, :, 0], fit.fitted_surface)


def test_loaded_prfit_session_reconstructs_surfaces_and_writes_zip(tmp_path):
    """A reopened .prfit session reconstructs data/fit/residual surfaces without raw data."""
    import zipfile

    ds, _, _, _, _ = _make_dataset(noise_std=0.002, source=str(tmp_path / "session_surf.pdat"))
    fit = pr.fit_global(ds, taus=[4.0, 120.0], irf_fwhm=0.25, model_type="Parallel")
    prfit_path = pr.save_fit(tmp_path / "session_surf", fit)

    reopened = pr.load_fit(prfit_path)
    assert np.allclose(reopened.fitted_surface, fit.fitted_surface)
    assert np.allclose(reopened.data_surface, fit.data_surface)

    zip_out = tmp_path / "from_session.zip"
    bpaths = reopened.export_results(zip_out)
    assert bpaths.archive == zip_out
    assert zip_out.exists()
    with zipfile.ZipFile(zip_out) as zf:
        assert len(zf.namelist()) == 6


def test_weighted_residuals_surface_live_and_reopened_session(tmp_path):
    """weighted_residuals=True scales residuals by 1/sigma (including after .prfit reload) and refuses unweighted fits."""
    ds, _, _, _, _ = _make_dataset(noise_std=0.004, source=str(tmp_path / "weighted_scan.pdat"))

    unweighted = pr.fit_global(
        ds, taus=[4.0, 120.0], irf_fwhm=0.25, model_type="Sequential", use_weights=False
    )
    with pytest.raises(ValueError, match="weighted_residuals=True"):
        unweighted.export_surface(tmp_path / "bad_res.csv", kind="residuals", weighted_residuals=True)

    weighted = pr.fit_global(
        ds, taus=[4.0, 120.0], irf_fwhm=0.25, model_type="Sequential", use_weights=True
    )
    expected_wres = weighted.weights.w * weighted.R

    live_csv = weighted.export_surface(
        tmp_path / "live_wres.csv", kind="residuals", dataset=ds, weighted_residuals=True
    )
    live_mat = np.loadtxt(live_csv, delimiter=",", skiprows=1)[:, 1:]
    assert np.allclose(live_mat, expected_wres, rtol=1e-10, atol=1e-12)

    prfit_path = pr.save_fit(tmp_path / "weighted_scan", weighted)
    reopened = pr.load_fit(prfit_path)
    reopened_csv = reopened.export_surface(
        tmp_path / "reopened_wres.csv", kind="residuals", weighted_residuals=True
    )
    reopened_mat = np.loadtxt(reopened_csv, delimiter=",", skiprows=1)[:, 1:]
    assert np.allclose(reopened_mat, expected_wres, rtol=1e-10, atol=1e-12)


def test_surface_kinetic_only_vs_artifact_and_tsv_format(tmp_path):
    """include_artifact=False exports kinetic-only C_kin @ S_kin.T, and surface_format='tsv' writes clean TSV."""
    ds, _, _, _, _ = _make_dataset(noise_std=0.002, source=str(tmp_path / "art_surf.pdat"))
    fit = pr.fit_global(
        ds,
        taus=[4.0, 120.0],
        irf_fwhm=0.25,
        model_type="Sequential",
        delay_range=(float(T[0]), float(T[-1])),
        coherent_artifact=True,
    )
    assert fit.n_artifact == 3
    nk = fit.n_kinetic

    tsv_kin = fit.export_surface(
        tmp_path / "fit_kinetic_only",
        kind="fit",
        dataset=ds,
        surface_format="tsv",
        include_artifact=False,
    )
    assert tsv_kin.suffix == ".tsv"
    lines = tsv_kin.read_text(encoding="utf-8").splitlines()
    assert not lines[0].startswith("#")
    assert "\t" in lines[0]

    loaded = np.loadtxt(tsv_kin, delimiter="\t", skiprows=1)
    assert np.allclose(loaded[:, 0], fit.t)
    expected_kin = fit.C[:, :nk] @ fit.S[:, :nk].T
    assert np.allclose(loaded[:, 1:], expected_kin, rtol=1e-10, atol=1e-12)
