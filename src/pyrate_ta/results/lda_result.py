"""Result object for Lifetime Density Analysis (LDA)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..log import get_logger

logger = get_logger(__name__)


@dataclass
class LDAResult:
    """Result of Lifetime Density Analysis (LDA).

    Attributes
    ----------
    tau_grid : numpy.ndarray ``(M,)``
        Logarithmically spaced grid of lifetimes in dataset time units.
    S_map : numpy.ndarray ``(Np, M)``
        2D matrix of lifetime amplitudes across probe pixels.
    alpha_opt : float
        Chosen regularisation parameter alpha.
    penalty : str
        Penalty type (``"ridge"`` / ``"d1"`` / ``"d2"``).
    alpha_method : str
        Method used to pick alpha (``"lcurve"`` / ``"gcv"`` / ``"manual"``).
    l_curve_points : numpy.ndarray ``(K, 2)`` or None
        (log_residual, log_norm) points across the alpha scan for L-curve plot.
    alphas : numpy.ndarray ``(K,)`` or None
        Scanned alpha values.
    gcv_scores : numpy.ndarray ``(K,)`` or None
        Scanned GCV score values.
    residuals : numpy.ndarray ``(Nd, Np)`` or None
        Residual matrix D - C S^T.
    t : numpy.ndarray ``(Nd,)``
        Delays fitted.
    probe : numpy.ndarray ``(Np,)`` or None
        Probe wavelengths/wavenumbers.
    irf_fwhm : float or None
        IRF FWHM used in kernel construction.
    t0 : float
        Time zero offset used in kernel construction.
    """

    tau_grid: np.ndarray
    S_map: np.ndarray
    alpha_opt: float
    penalty: str = "d2"
    alpha_method: str = "lcurve"
    l_curve_points: np.ndarray | None = None
    alphas: np.ndarray | None = None
    gcv_scores: np.ndarray | None = None
    residuals: np.ndarray | None = None
    t: np.ndarray | None = None
    probe: np.ndarray | None = None
    irf_fwhm: float | None = None
    t0: float = 0.0
    svd_components: int = 0
    non_negative: bool = False
    n_bootstraps: int = 0
    bootstrap_std: np.ndarray | None = None
    peaks: list[dict] = field(default_factory=list)
    statistic: Any = None
    delay_range: tuple[float, float] | None = None
    probe_range: tuple[float, float] | None = None
    units: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)

    @property
    def n_taus(self) -> int:
        return len(self.tau_grid)

    def integrated_dynamics(self, metric: str = "abs") -> np.ndarray:
        """Integrated dynamics profile across the lifetime grid.

        Parameters
        ----------
        metric : {"abs", "dynamical_content", "sqrt_sq"}
            - ``"abs"``: absolute sum :math:`A(\\tau) = \\int |S(\\lambda, \\tau)| d\\lambda`.
            - ``"dynamical_content"`` / ``"sqrt_sq"``: :math:`D(\\tau) = \\sqrt{\\int S(\\lambda, \\tau)^2 d\\lambda}`.

        Returns
        -------
        numpy.ndarray, shape ``(M,)``
        """
        if metric in ("dynamical_content", "sqrt_sq", "rms", "d"):
            return np.sqrt(np.sum(self.S_map**2, axis=0))
        return np.sum(np.abs(self.S_map), axis=0)

    def find_peaks(self, metric: str = "abs", prominence: float = 0.05) -> list[dict]:
        """Auto-detect major lifetime peak centroids.

        Parameters
        ----------
        metric : {"abs", "dynamical_content"}
            Which profile metric to detect peaks on.
        prominence : float
            Peak prominence as a fraction of peak maximum.

        Returns
        -------
        list of dict with keys ``"tau"``, ``"amplitude"``, ``"index"``.
        """
        from scipy.signal import find_peaks as sc_find_peaks

        profile = self.integrated_dynamics(metric=metric)
        max_val = np.max(profile) if len(profile) > 0 and np.max(profile) > 0 else 1.0
        p_indices, _ = sc_find_peaks(profile, prominence=float(prominence) * max_val)
        return [
            {
                "tau": float(self.tau_grid[i]),
                "amplitude": float(profile[i]),
                "index": int(i),
            }
            for i in p_indices
        ]

    def citations(self) -> list[str]:
        """Return literature citations for the specific algorithms used in this LDA fit."""
        cites = [
            "Slavov, C., Hartmann, H., & Wachtveitl, J. (2015). Implementation and Evaluation of Data Analysis Strategies for Time-Resolved Optical Spectroscopy. Anal. Chem., 87(4), 2328-2336. DOI: 10.1021/ac504348h",
            "Dorlhiac, G. F., Fare, C., & van Thor, J. J. (2017). PyLDM - An open source package for lifetime density analysis of time-resolved spectroscopic data. PLOS Comput. Biol., 13(5), e1005528. DOI: 10.1371/journal.pcbi.1005528",
            "Steinbach, P. J., Ionescu, R., & Champion, P. M. (2002). Analysis of Kinetics Using a Hybrid Maximum-Entropy/Nonlinear-Least-Squares Method: Application to Protein Folding. Biophys. J., 82(4), 2244-2255. DOI: 10.1016/S0006-3495(02)75570-7",
        ]
        if self.svd_components > 0:
            cites.append(
                "Henry, E. R., & Hofrichter, J. (1992). Singular value decomposition: Application to analysis of experimental data. Methods Enzymol., 210, 129-192. DOI: 10.1016/0076-6879(92)10010-B"
            )
        if self.alpha_method == "morozov":
            cites.append(
                "Morozov, V. A. (1966). On the resolution of ill-posed problems and the choice of the regularization parameter. Soviet Math. Dokl., 7, 414-417."
            )
        elif self.alpha_method == "gcv":
            cites.append(
                "Golub, G. H., Heath, M., & Wahba, G. (1979). Generalized cross-validation as a method for choosing a good ridge parameter. Technometrics, 21(2), 215-223. DOI: 10.1080/00401706.1979.10489751"
            )
        elif self.alpha_method == "lcurve":
            cites.append(
                "Hansen, P. C. (1992). Analysis of discrete ill-posed problems by means of the L-curve. SIAM Review, 34(4), 561-580. DOI: 10.1137/1034115"
            )
        if self.non_negative:
            cites.append(
                "Lawson, C. L., & Hanson, R. J. (1995). Solving Least Squares Problems. SIAM, Philadelphia. DOI: 10.1137/1.9781611971217"
            )
        if self.n_bootstraps > 0:
            cites.append(
                "Efron, B., & Tibshirani, R. J. (1994). An Introduction to the Bootstrap. Chapman & Hall/CRC. DOI: 10.1201/9780429246593"
            )
        return cites

    def format_citations(self) -> str:
        """Format citations as a clean readable bulleted list."""
        cites = self.citations()
        lines = ["Recommended Literature Citations for this LDA Analysis:", ""]
        for i, c in enumerate(cites, 1):
            lines.append(f"[{i}] {c}")
        return "\n".join(lines)

    def summary(self) -> str:
        s = (
            f"LDA Result: {self.n_taus} grid points ({self.tau_grid.min():.3g} to "
            f"{self.tau_grid.max():.3g}), penalty={self.penalty!r}, "
            f"alpha={self.alpha_opt:.4g} (via {self.alpha_method})"
        )
        if self.svd_components > 0:
            s += f", SVD filtered ({self.svd_components} comps)"
        if self.non_negative:
            s += ", NNLS (S>=0)"
        if self.n_bootstraps > 0:
            s += f", Bootstrapped ({self.n_bootstraps} iter)"
        return s

    def to_pdat(self, path: str | Any) -> Any:
        """Export the fitted 2D lifetime density map S(tau, lambda) as a PyMORGAN PDAT file.

        Parameters
        ----------
        path : str or Path
            Destination filepath.

        Returns
        -------
        Path
            Path to the saved PDAT file.
        """
        from pathlib import Path

        pdat_path = Path(path)
        if pdat_path.suffix.lower() != ".pdat":
            pdat_path = pdat_path.with_suffix(".pdat")

        probe = np.asarray(self.probe if self.probe is not None else np.arange(self.S_map.shape[0]), dtype=float)
        tau = np.asarray(self.tau_grid, dtype=float)
        S_map = np.asarray(self.S_map, dtype=float)  # shape (Np, M)

        units = self.units or {}
        time_unit = units.get("unitsT_ltx") or units.get("time_unit") or "ps"
        lbl = units.get("unitsL_lbl") or "Probe"
        if lbl == "Wavenumber":
            probe_unit = r"cm^{-1}"
        else:
            probe_unit = units.get("unitsL_ltx") or "nm"

        header_line = f"{time_unit}*{probe_unit}"

        # PDAT table: Row 0 is [0.0, probe_1, ...], Column 0 is [0.0, tau_1, ...]
        pdat_table = np.empty((len(tau) + 1, len(probe) + 1), dtype=float)
        pdat_table[0, 0] = 0.0
        pdat_table[0, 1:] = probe
        pdat_table[1:, 0] = tau
        pdat_table[1:, 1:] = S_map.T

        pdat_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pdat_path, "w", encoding="utf-8", newline="") as f:
            f.write(f"{header_line}\n")
            np.savetxt(f, pdat_table, delimiter=",", fmt="%.18g")

        logger.info(f"Saved LDA map to PDAT: {pdat_path}")
        return pdat_path
