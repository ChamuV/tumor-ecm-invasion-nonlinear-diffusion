# # run_files/run_asymptotics_all.py

# run_files/run_asymptotics.py

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from tumour_ecm.plotting.asymptotic_single import plot_asymptotic_single
from tumour_ecm.plotting.asymptotic_grids import plot_asymptotic_grid


def main():
    base_dir = ROOT / "outputs" / "sweeps" / "tumour_ecm_1d"

    # =========================
    # 1. FKPP limit
    # weak degradation + weak regeneration
    # lambda small, alpha small
    # =========================
    plot_asymptotic_single(
        base_dir=base_dir,
        lam=0.001,
        alpha=0.001,
        m0=0.5,
        t_ref="auto",
        regime="fkpp",
    )

    rows_fkpp = [
        (0.001, [0.001, 0.01, 0.05]),
        ([0.001, 0.01, 0.05], 0.001),
    ]

    plot_asymptotic_grid(
        base_dir=base_dir,
        rows=rows_fkpp,
        m0=0.5,
        t_ref="auto",
        regime="fkpp",
        suptitle="Weak degradation/weak regeneration: FKPP limit",
    )

    # =========================
    # 2. Logistic limit, strong ratio
    # sigma = lambda / alpha large
    # lambda much larger than alpha
    # =========================
    plot_asymptotic_single(
        base_dir=base_dir,
        lam=100.0,
        alpha=0.001,
        m0=0.5,
        t_ref="auto",
        regime="logistic",
    )

    rows_logistic_strong_ratio = [
        (0.001, [1.0, 10.0, 100.0]),
        (0.01, [1.0, 10.0, 100.0]),
    ]

    plot_asymptotic_grid(
        base_dir=base_dir,
        rows=rows_logistic_strong_ratio,
        m0=0.5,
        t_ref="auto",
        regime="logistic",
        suptitle="Strong ratio: logistic limit",
    )

    # =========================
    # 3. Logistic limit, weak ratio
    # sigma = lambda / alpha small
    # alpha much larger than lambda
    # =========================
    plot_asymptotic_single(
        base_dir=base_dir,
        lam=0.001,
        alpha=100.0,
        m0=0.5,
        t_ref="auto",
        regime="logistic",
    )

    rows_logistic_weak_ratio = [
        (100.0, [0.001, 0.01, 0.05]),
        (10.0, [0.001, 0.01, 0.05]),
    ]

    plot_asymptotic_grid(
        base_dir=base_dir,
        rows=rows_logistic_weak_ratio,
        m0=0.5,
        t_ref="auto",
        regime="logistic",
        suptitle="Weak ratio: logistic limit",
    )

    # =========================
    # 4. Balanced strong/strong limit
    # lambda and alpha both large, comparable scale
    # =========================
    plot_asymptotic_single(
        base_dir=base_dir,
        lam=100.0,
        alpha=100.0,
        m0=0.5,
        t_ref="auto",
        regime="sharp_front",
    )

    rows_sharp = [
        ([10.0, 100.0, 1000.0], 10.0),
        ([10.0, 100.0, 1000.0], 100.0),
    ]

    plot_asymptotic_grid(
        base_dir=base_dir,
        rows=rows_sharp,
        m0=0.5,
        t_ref="auto",
        regime="sharp_front",
        suptitle="Strong degradation/strong regeneration: sharp-front limit",
    )


if __name__ == "__main__":
    main()