"""Fit a dataset from the command line, and look at the result.

Run::

    uv run python examples/quick_fit.py path/to/scan.pdat --taus 1 20 500
    uv run python examples/quick_fit.py path/to/scan.pdat --taus 1 20 inf --weights
    uv run python examples/quick_fit.py --synthetic          # no data file needed

Everything the GUI will do is available here already: the model, the fit
window, noise weighting, the scheme diagram and the species spectra. Use it to
check that a fit behaves before driving it from the interface.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def synthetic():
    """A two-component sequential dataset with noise, for a dry run."""
    import pyrate_ta as pr

    t = np.concatenate([np.linspace(-2.0, 0.5, 20), np.geomspace(0.6, 3000.0, 140)])
    probe = np.linspace(1900.0, 2010.0, 64)
    model = pr.make_model("Sequential", 2, irf_fwhm=0.3)
    C = model.concentrations(t, taus=[8.0, 300.0])
    S = np.stack(
        [
            a * np.exp(-((probe - c) ** 2) / (2 * w**2))
            for c, w, a in ((1950.0, 15.0, 1.0), (1975.0, 25.0, -0.6))
        ],
        axis=1,
    )
    sigma = 0.02
    D = C @ S.T + sigma * np.random.default_rng(0).standard_normal((t.size, probe.size))
    return t, probe, D, np.full(D.shape, sigma)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", nargs="?", help="dataset to fit (any format PyMORGAN reads)")
    parser.add_argument("--data-type", default="PDAT", help="loader to use (default: PDAT)")
    parser.add_argument(
        "--taus",
        nargs="+",
        default=["1", "20", "500"],
        help="initial lifetimes; 'inf' adds a non-decaying component",
    )
    parser.add_argument("--model", default="Sequential", help="Parallel / Sequential")
    parser.add_argument("--irf", type=float, default=None, help="IRF FWHM (dataset time unit)")
    parser.add_argument("--fit-irf", action="store_true", help="let the IRF width float")
    parser.add_argument(
        "--coherent-artefact",
        action="store_true",
        help="add the IRF and its first two derivatives, to absorb the time-zero artefact",
    )
    parser.add_argument("--t0", type=float, default=0.0)
    parser.add_argument("--fit-t0", action="store_true")
    parser.add_argument("--tmin", type=float, default=None, help="drop delays below this")
    parser.add_argument(
        "--probe-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("MIN", "MAX"),
        help="fit only this probe window (native unit)",
    )
    parser.add_argument("--detector", type=int, default=0)
    parser.add_argument("--weights", action="store_true", help="weight by the per-point noise")
    parser.add_argument("--synthetic", action="store_true", help="use built-in synthetic data")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--save",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help="write the fit session (.prfit); with no path, beside the dataset",
    )
    args = parser.parse_args(argv)

    import pyrate_ta as pr

    if args.synthetic or not args.path:
        t, _probe, D, sigma = synthetic()
        data, extra = D, {"t": t, "sigma": sigma if args.weights else None}
        print("Using synthetic data (2 components: 8 and 300, IRF 0.3).")
    else:
        import pymorgan as pm

        path = Path(args.path)
        if not path.is_file():
            print(f"No such file: {path}", file=sys.stderr)
            return 2
        data, extra = pm.load_1D(str(path), data_type=args.data_type), {}
        print(f"Loaded {path.name}: {data!r}")
        if args.weights and data.noise_array() is None:
            print(
                "This dataset carries no per-point noise (no .pdatn sibling, no single "
                "scans), so --weights cannot be used.",
                file=sys.stderr,
            )
            return 2

    fit = pr.fit_global(
        data,
        extra.pop("t", None),
        taus=args.taus,
        model_type=args.model,
        irf_fwhm=args.irf,
        fit_irf=args.fit_irf,
        t0=args.t0,
        fit_t0=args.fit_t0,
        coherent_artifact=args.coherent_artifact,
        t_min=args.tmin,
        probe_range=tuple(args.probe_range) if args.probe_range else None,
        detector=args.detector,
        use_weights=args.weights,
        **extra,
    )

    print()
    print(fit.summary())

    if args.save is not None:
        target = args.save or pr.io.default_session_path(fit)
        print(f"\nSession written to {pr.save_fit(target, fit)}")
    if not fit.converged:
        print("\nThe fit did not converge: the values above are trial parameters.")

    if args.no_plots:
        return 0 if fit.converged else 1

    import matplotlib.pyplot as plt

    pr.plot_scheme(fit)

    # The species spectra come from PyMORGAN when there is a dataset to supply
    # the probe axis and its units; the concentration profiles are PyRATE-TA's own
    # view, since they are the model's populations rather than measurements.
    if hasattr(data, "plot_species_spectra"):
        data.plot_species_spectra(*fit.as_species_args())
    else:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        x = fit.probe if fit.probe is not None else np.arange(fit.S.shape[0])
        for i, tau in enumerate(fit.taus):
            ax.plot(x, fit.S[:, i], label=f"{pr.format_lifetime(tau)}")
        ax.set_title(f"{fit.spectra_kind} ({fit.model_type})")
        ax.legend(frameon=False)
        ax.axhline(0.0, color="0.6", lw=0.8)
        fig.tight_layout()

    pr.plot_concentrations(fit)
    pr.plot_matrix(fit.t, fit.probe, fit.R, title="Residuals")
    plt.show()
    return 0 if fit.converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
