# src/tumour_ecm/plotting/travelling_wave_grid.py

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from tumour_ecm.utils.io import find_run_dir, load_snapshots, load_summary


def _nearest_indices(times, t_points):
    times = np.asarray(times, dtype=float)

    return [
        int(np.argmin(np.abs(times - t)))
        for t in t_points
    ]


def _resolve_colors(color_scheme):
    if color_scheme == "opt1":
        return "#1f77b4", "#d62728"      # ECM blue, tumour red

    if color_scheme == "opt2":
        return "#1f77b4", "#ff8c00"      # ECM blue, tumour orange

    if color_scheme == "opt3":
        return "#6a00a8", "#ff8c00"      # ECM purple, tumour orange

    return "#1f77b4", "#ff8c00"


def _load_speed(run_dir, which="N"):
    summary = load_summary(run_dir)

    if not summary:
        return np.nan

    key = "wave_speed" if which.upper() == "N" else "m_wave_speed"
    value = summary.get(key, np.nan)

    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except Exception:
        return np.nan


def _format_param(value):
    value = float(value)

    if value == 0:
        return "0"

    if abs(value) >= 1000 or abs(value) <= 0.001:
        exponent = int(np.round(np.log10(abs(value))))
        mantissa = value / (10 ** exponent)

        if np.isclose(mantissa, 1.0):
            return f"10^{exponent}"

        return f"{mantissa:.2g}×10^{exponent}"

    return f"{value:g}"


def plot_travelling_wave_alpha_lambda_grid(
    base_dir,
    alpha_vals,
    lambda_vals,
    m0,
    t_points=(0, 100, 200, 300, 400),
    yticks_mode="basic",
    show_arrows=True,
    show_speed_text=True,
    color_scheme="opt2",
    figsize=(13.6, 10.0),
    save=False,
    out_path=None,
    dpi=600,
):
    """
    Dissertation-style travelling-wave grid.

    Columns:
        alpha values.

    Rows:
        lambda values.

    Fixed:
        m0.

    Uses saved outputs from:

        outputs/sweeps/tumour_ecm_1d/
            alpha_*/
                lambda_*/
                    m0_*/
                        snapshots.npz
                        summary.json
    """
    base_dir = Path(base_dir)

    alpha_vals = list(alpha_vals)
    lambda_vals = list(lambda_vals)

    nrows = len(lambda_vals)
    ncols = len(alpha_vals)

    m_color, u_color = _resolve_colors(color_scheme)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    fig.patch.set_facecolor("white")

    # Column titles: alpha values
    for j, alpha in enumerate(alpha_vals):
        axes[0, j].set_title(
            f"alpha = {_format_param(alpha)}",
            fontsize=18,
            pad=8,
        )

    legend_handles = None

    if yticks_mode == "basic":
        shared_ylim = (0.0, 1.05)
        yticks = [0.0, 0.5, 1.0]

    elif yticks_mode == "split":
        shared_ylim = (0.0, 1.05)
        yticks = np.arange(0.0, 1.01, 0.2)

    elif yticks_mode == "splitplus":
        shared_ylim = (0.0, 1.25)
        yticks = np.arange(0.0, 1.21, 0.2)

    else:
        shared_ylim = (0.0, 1.05)
        yticks = [0.0, 0.5, 1.0]

    for i, lam in enumerate(lambda_vals):
        # Row label on far left
        axes[i, 0].text(
            -0.25,
            0.5,
            f"lambda = {_format_param(lam)}",
            transform=axes[i, 0].transAxes,
            ha="center",
            va="center",
            fontsize=22,
            fontweight="bold",
            rotation=90,
        )

        for j, alpha in enumerate(alpha_vals):
            ax = axes[i, j]

            ax.grid(False)
            ax.set_ylim(shared_ylim)
            ax.set_yticks(yticks)

            if j > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel("u, m", fontsize=18)

            if i < nrows - 1:
                ax.set_xticklabels([])

            run_dir = find_run_dir(
                base_dir=base_dir,
                lam=lam,
                m0=m0,
                alpha=alpha,
            )

            snap = load_snapshots(run_dir)

            if snap is None:
                ax.text(
                    0.5,
                    0.5,
                    "missing run",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=13,
                )
                continue

            x = snap["x"]
            times = snap["times"]
            U = snap["N_arr"]
            M = snap["M_arr"]

            L = float(x[-1])

            ax.set_xlim(0.0, L)
            ax.tick_params(axis="x", labelsize=14)
            ax.tick_params(axis="y", labelsize=14)

            t_indices = _nearest_indices(times, t_points)

            h_u = None
            h_m = None

            for k, idx in enumerate(t_indices):
                linestyle = "--" if k == 0 else "-"

                h_u, = ax.plot(
                    x,
                    U[idx],
                    color=u_color,
                    linestyle=linestyle,
                    linewidth=2.0,
                )

                h_m, = ax.plot(
                    x,
                    M[idx],
                    color=m_color,
                    linestyle=linestyle,
                    linewidth=2.0,
                )

            if legend_handles is None and h_u is not None and h_m is not None:
                legend_handles = [h_u, h_m]

            if show_arrows:
                ax.annotate(
                    "",
                    xy=(0.85 * L, 0.8),
                    xytext=(0.70 * L, 0.8),
                    arrowprops=dict(
                        arrowstyle="->",
                        lw=2.3,
                        color=u_color,
                    ),
                )

                ax.annotate(
                    "",
                    xy=(0.85 * L, 0.25),
                    xytext=(0.70 * L, 0.25),
                    arrowprops=dict(
                        arrowstyle="->",
                        lw=2.3,
                        color=m_color,
                    ),
                )

            if show_speed_text:
                c = _load_speed(run_dir, which="N")

                if np.isfinite(c):
                    ax.text(
                        0.03,
                        0.86,
                        f"c = {c:.3g}",
                        transform=ax.transAxes,
                        fontsize=18,
                        ha="left",
                        va="top",
                    )

    for ax in axes[-1, :]:
        ax.set_xlabel("x", fontsize=18)

    fig.suptitle(
        f"Numerical solutions at m0 = {m0}",
        fontsize=20,
        y=0.98,
    )

    plt.subplots_adjust(
        left=0.12,
        right=0.96,
        top=0.90,
        bottom=0.12,
        wspace=0.08,
        hspace=0.24,
    )

    if legend_handles is not None:
        fig.legend(
            legend_handles,
            ["u(x,t) tumour", "m(x,t) ECM"],
            loc="lower center",
            ncol=2,
            frameon=False,
            fontsize=16,
            bbox_to_anchor=(0.5, 0.02),
        )

    if save:
        if out_path is None:
            out_path = f"plots/travelling_wave_grid_m0_{str(m0).replace('.', 'p')}"

        out_path = Path(out_path)
        os.makedirs(out_path.parent, exist_ok=True)

        fig.savefig(str(out_path) + ".pdf", bbox_inches="tight")
        fig.savefig(str(out_path) + ".png", dpi=dpi, bbox_inches="tight")

    return fig, axes