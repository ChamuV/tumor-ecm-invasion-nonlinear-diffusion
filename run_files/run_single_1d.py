# run_files/run_single_1d.py

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from tumour_ecm.solvers.tumour_ecm_1d import TumourECM1D
from tumour_ecm.plotting.travelling_wave_single import plot_u_and_m_travelling_wave
from tumour_ecm.plotting.speed_plots import plot_wave_speed_fit


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    model = TumourECM1D(
        lam=1.0,
        alpha=0.0,      # alpha=0 gives the no-recovery case
        m0=0.1,
        L=200,
        N=20001,
        T=200,
        dt=0.1,
        init_type="tanh",
        steepness=0.85,
        perc=0.4,
        t_start=60,
        t_end=180,
        num_points=200,
        n0=1.0,
        K=1.0,
        rho=1.0,
        D=1.0,
        Mmax=1.0,
    )

    print("Solving single 1D tumour--ECM run...")
    model.solve()

    print("Plotting travelling-wave profiles...")
    plot_u_and_m_travelling_wave(
        model,
        t_indices=(0, 500, 1000, 1500, 2000),
        ylim=(-0.05, 1.05),
    )

    print("Plotting wave-speed fit...")
    c, intercept, r2 = plot_wave_speed_fit(
        model,
        threshold=0.5,
        band=(0.1, 0.9),
        spline_type="cubic",
        target="N",
    )

    print("\nSingle run complete.")
    print(f"lambda = {model.lam}")
    print(f"alpha  = {model.alpha}")
    print(f"m0     = {model.m0}")
    print(f"speed  = {c}")
    print(f"R^2    = {r2}")


if __name__ == "__main__":
    main()