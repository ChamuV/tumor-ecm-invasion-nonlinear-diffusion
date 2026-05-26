# src/tumour_ecm/plotting/mass_plots.py

import numpy as np
import matplotlib.pyplot as plt

from tumour_ecm.utils.io import find_run_dir, load_snapshots


def compute_mass(x, arr):
    """
    Compute spatial integral over time using trapezoidal rule.

    arr shape:
        (Nt, Nx)
    """
    return np.trapz(arr, x=x, axis=1)


def plot_ecm_mass_from_runs(
    base_dir,
    parameter_pairs,
    m0_vals,
    alpha=None,
    which="M",
    figsize=(11, 4.5),
    cmap_name="viridis",
):
    """
    Plot total tumour or ECM mass over time for selected parameter pairs.

    parameter_pairs example:
        [
            {"lam": 1e6, "alpha": 1e3, "title": r"$\lambda=10^6,\alpha=10^3$"},
            {"lam": 1e3, "alpha": 1e6, "title": r"$\lambda=10^3,\alpha=10^6$"},
        ]

    which:
        "M" for ECM mass
        "N" or "U" for tumour mass
    """
    fig, axes = plt.subplots(1, len(parameter_pairs), figsize=figsize, sharey=True)

    if len(parameter_pairs) == 1:
        axes = [axes]

    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i / max(1, len(m0_vals) - 1)) for i in range(len(m0_vals))]

    for ax, pair in zip(axes, parameter_pairs):
        lam = pair["lam"]
        alpha_pair = pair.get("alpha", alpha)
        title = pair.get("title", rf"$\lambda={lam:g},\ \alpha={alpha_pair:g}$")

        for k, m0 in enumerate(m0_vals):
            run_dir = find_run_dir(
                base_dir=base_dir,
                lam=lam,
                m0=m0,
                alpha=alpha_pair,
            )

            snap = load_snapshots(run_dir)

            if snap is None:
                continue

            x = snap["x"]
            times = snap["times"]

            if which.upper() in ["M", "ECM"]:
                arr = snap["M_arr"]
            else:
                arr = snap["N_arr"]

            mass = compute_mass(x, arr)

            ax.plot(
                times,
                mass,
                "-o",
                markersize=3,
                linewidth=1.8,
                color=colors[k],
                label=rf"{m0:g}",
            )

        ax.set_title(title, fontsize=14)
        ax.set_xlabel(r"Time $t$", fontsize=13)
        ax.grid(True, linestyle="--", alpha=0.25)

    axes[0].set_ylabel(
        r"Area under $m(x,t)$" if which.upper() in ["M", "ECM"] else r"Area under $u(x,t)$",
        fontsize=13,
    )

    axes[-1].legend(
        title=r"$m_0$",
        fontsize=9,
        title_fontsize=10,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )

    plt.tight_layout()
    plt.show()

    return fig, axes