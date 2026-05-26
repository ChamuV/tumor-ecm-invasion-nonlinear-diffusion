# run_files/run_phase_plane.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from tumour_ecm.analysis.phase_plane import plot_phaseplane_recovery_appendix


def main():
    lam = 0.1
    alpha = 0.01

    c_meas = 0.774551534454284

    mbar = alpha / (alpha + lam)
    eps_up = 0.01

    plot_phaseplane_recovery_appendix(
    lam=0.1,
    #alpha=0.01,
    alpha=0.0,
    m0=mbar,
    U0=0.99,
    P0=-0.001,
    M0=min(1.0 - 1e-9, mbar + eps_up),
    speeds=None,
)


if __name__ == "__main__":
    main()