# src/tumour_ecm/plotting/travelling_wave_grid.py

import numpy as np
import matplotlib.pyplot as plt

from tumour_ecm.utils.io import find_run_dir, load_snapshots, load_summary


def plot_travelling_wave_grid(
    base_dir,
    lambda_vals,
    m0_vals,
    alpha=None,
    times=(0, 100, 200, 300, 400, 500),
    xlim=None,
    ylim=(-0.05, 1.05),
    figsize=None,
    show_initial_as_dashed=True,
    show_legend=True,
    title=None,
):
    """
    Grid of travelling-wave profiles.

    Rows: lambda values.
    Columns: m0 values.

    Orange: tumour u(x,t)
    Blue: ECM m(x,t)
    """

    nrows = len(lambda_vals)
    ncols = len(m0_vals)

    if figsize is None:
        figsize = (4.0 * ncols, 2.0 * nrows)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for i, lam in enumerate(lambda_vals):
        for j, m0 in enumerate(m0_vals):
            ax = axes[i, j]

            run_dir = find_run_dir(base_dir, lam=lam, m0=m0, alpha=alpha)
            snap = load_snapshots(run_dir)

            if snap is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue

            x = snap["x"]
            t_arr = snap["times"]
            U = snap["N_arr"]
            M = snap["M_arr"]

            for t_target in times:
                idx = int(np.argmin(np.abs(t_arr - t_target)))
                linestyle = "--" if show_initial_as_dashed and idx == 0 else "-"

                ax.plot(
                    x,
                    U[idx],
                    color="tab:orange",
                    linestyle=linestyle,
                    linewidth=1.5,
                )

                ax.plot(
                    x,
                    M[idx],
                    color="tab:blue",
                    linestyle=linestyle,
                    linewidth=1.5,
                )

            if i == 0:
                ax.set_title(rf"$m_0 = {m0:g}$", fontsize=13)

            if j == 0:
                ax.set_ylabel(rf"$\lambda = {lam:g}$" + "\n" + r"$u,m$", fontsize=12)

            if i == nrows - 1:
                ax.set_xlabel(r"$x$", fontsize=12)

            if xlim is not None:
                ax.set_xlim(xlim)
            else:
                ax.set_xlim([x.min(), x.max()])

            ax.set_ylim(ylim)
            ax.tick_params(labelsize=9)

    if show_legend:
        handles = [
            plt.Line2D([0], [0], color="tab:orange", lw=2, label=r"$u(x,t)$ tumour"),
            plt.Line2D([0], [0], color="tab:blue", lw=2, label=r"$m(x,t)$ ECM"),
        ]

        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=2,
            fontsize=12,
            frameon=False,
            bbox_to_anchor=(0.5, -0.01),
        )

    if title:
        fig.suptitle(title, fontsize=16, y=1.01)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.show()

    return fig, axes