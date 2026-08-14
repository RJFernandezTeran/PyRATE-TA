"""Lifetime Density Analysis (LDA) CLI example script.

Run::

    uv run python examples/lda_fit.py --synthetic
    uv run python examples/lda_fit.py --synthetic --penalty ridge --alpha-method gcv
    uv run python examples/lda_fit.py path/to/scan.pdat --data-type PDAT

Solves for a continuous distribution of lifetime amplitudes over a dense
grid of 100 fixed lifetimes, and plots the 2D Lifetime Density Map and L-Curve.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def synthetic():
    """Synthetic two-component time-resolved dataset with noise."""
    import pyrate_ta as pr

    t = np.concatenate([np.linspace(-2.0, 0.5, 25), np.geomspace(0.6, 3000.0, 150)])
    probe = np.linspace(1900.0, 2010.0, 64)
    model = pr.make_model("Parallel", 2, irf_fwhm=0.3)
    C = model.concentrations(t, taus=[8.0, 350.0])
    S = np.stack(
        [
            a * np.exp(-((probe - c) ** 2) / (2 * w**2))
            for c, w, a in ((1945.0, 12.0, 1.2), (1980.0, 22.0, -0.8))
        ],
        axis=1,
    )
    sigma = 0.015
    D = C @ S.T + sigma * np.random.default_rng(42).standard_normal((t.size, probe.size))
    return t, probe, D


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", nargs="?", help="dataset to analyse (any format PyMORGAN reads)")
    parser.add_argument("--data-type", default="PDAT", help="loader to use (default: PDAT)")
    parser.add_argument("--n-taus", type=int, default=100, help="number of grid points (default 100)")
    parser.add_argument("--penalty", default="d2", choices=["d2", "d1", "ridge"], help="regularisation penalty")
    parser.add_argument("--alpha-method", default="lcurve", choices=["lcurve", "gcv"], help="alpha selection method")
    parser.add_argument("--alpha", type=float, default=None, help="manual alpha value")
    parser.add_argument("--synthetic", action="store_true", help="use synthetic test dataset")
    parser.add_argument("--no-plots", action="store_true", help="disable plot display")
    args = parser.parse_args(argv)

    import matplotlib.pyplot as plt
    import pymorgan as pm

    from pyrate_ta.fit.lda import solve_lda
    from pyrate_ta.plot.lda import plot_l_curve, plot_lda_map

    if args.synthetic or not args.path:
        t, probe, D = synthetic()
        data = D
        print("Using synthetic dataset (2 true components: 8.0 ps and 350.0 ps, IRF 0.3 ps).")
    else:
        path = Path(args.path)
        if not path.is_file():
            print(f"No such file: {path}", file=sys.stderr)
            return 2
        dataset = pm.load_1D(str(path), data_type=args.data_type)
        data = dataset
        t = dataset.delays
        print(f"Loaded {path.name}: {dataset!r}")

    print(f"Running LDA ({args.n_taus} taus, penalty={args.penalty!r}, method={args.alpha_method!r})...")
    alpha_val = args.alpha if args.alpha is not None else "auto"
    res = solve_lda(
        data,
        t=t if isinstance(data, np.ndarray) else None,
        n_taus=args.n_taus,
        penalty=args.penalty,
        alpha=alpha_val,
        alpha_method=args.alpha_method,
    )

    print(res.summary())

    if not args.no_plots:
        pm.apply_style()
        plot_lda_map(res, discrete_taus=[8.0, 350.0] if args.synthetic else None)
        plot_l_curve(res)
        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
