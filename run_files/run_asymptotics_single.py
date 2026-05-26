# run_files/run_asymptotics_single.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from tumour_ecm.plotting.asymptotic_single import plot_asymptotic_single
from tumour_ecm.plotting.asymptotic_grids import plot_asymptotic_grid


def main():
    base_dir = ROOT / "outputs" / "sweeps" / "tumour_ecm_1d"

    plot_asymptotic_single(
        base_dir=base_dir,
        lam=0.001,
        alpha=0.001,
        m0=0.5,
        t_ref="auto",
        regime="fkpp",
    )

    rows = [
        (0.001, [0.001, 0.01, 0.05]),
        ([0.001, 0.01, 0.05], 0.001),
    ]

    plot_asymptotic_grid(
        base_dir=base_dir,
        rows=rows,
        m0=0.5,
        t_ref="auto",
        regime="fkpp",
        suptitle="Weak degradation/weak regeneration: FKPP limit",
    )


if __name__ == "__main__":
    main()