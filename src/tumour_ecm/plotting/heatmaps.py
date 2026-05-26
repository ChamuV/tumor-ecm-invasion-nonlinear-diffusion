# src/tumour_ecm/plotting/heatmaps.py

import numpy as np
import matplotlib.pyplot as plt

from tumour_ecm.utils.io import load_speed_from_run


def plot_wavespeed_heatmap(
    base_dir,
    lambda_vals,
    m0_vals,
    alpha=None,
    which_speed="N",
    cmap="viridis",
    levels=20,
    fig_size=(8.6, 6.9),
    title="Wavespeed heatmap",
):
    lambda_vals = np.asarray(lambda_vals, dtype=float)
    m0_vals = np.asarray(m0_vals, dtype=float)

    H = np.full((len(m0_vals), len(lambda_vals)), np.nan)

    for i, m0 in enumerate(m0_vals):
        for j, lam in enumerate(lambda_vals):
            H[i, j] = load_speed_from_run(
                base_dir=base_dir,
                lam=lam,
                m0=m0,
                alpha=alpha,
                which=which_speed,
            )

    if not np.isfinite(H).any():
        fig, ax = plt.subplots(figsize=fig_size)
        ax.text(0.5, 0.5, "No speeds found.", ha="center", va="center")
        ax.axis("off")
        plt.show()
        return H, (fig, ax)

    X, Y = np.meshgrid(np.log10(lambda_vals), m0_vals)

    vmin = float(np.nanmin(H))
    vmax = float(np.nanmax(H))

    fig, ax = plt.subplots(figsize=fig_size)

    cf = ax.contourf(
        X,
        Y,
        H,
        levels=levels,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xlabel(r"$\log_{10}(\lambda)$", fontsize=18)
    ax.set_ylabel(r"Initial ECM density $m_0$", fontsize=18)
    ax.set_title(title, fontsize=22)
    ax.tick_params(axis="both", labelsize=14)

    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label("Numerically estimated wave speed", fontsize=16)
    cbar.ax.tick_params(labelsize=13)

    plt.tight_layout()
    plt.show()

    return H, (fig, ax)


def plot_alpha_heatmap_grid(
    base_dir,
    alpha_vals,
    lambda_vals,
    m0_vals,
    which_speed="N",
    nrows=4,
    ncols=3,
    cmap="viridis",
    levels=20,
    figsize=(12, 13),
    suptitle=None,
):
    """
    Multi-panel heatmap grid like your 4x3 alpha figure.
    """

    alpha_vals = list(alpha_vals)
    lambda_vals = np.asarray(lambda_vals, dtype=float)
    m0_vals = np.asarray(m0_vals, dtype=float)

    X, Y = np.meshgrid(np.log10(lambda_vals), m0_vals)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, alpha in zip(axes, alpha_vals):
        H = np.full((len(m0_vals), len(lambda_vals)), np.nan)

        for i, m0 in enumerate(m0_vals):
            for j, lam in enumerate(lambda_vals):
                H[i, j] = load_speed_from_run(
                    base_dir=base_dir,
                    lam=lam,
                    m0=m0,
                    alpha=alpha,
                    which=which_speed,
                )

        H_masked = np.ma.masked_invalid(H)

        if np.isfinite(H).any():
            vmin = float(np.nanmin(H))
            vmax = float(np.nanmax(H))
        else:
            vmin, vmax = 0.0, 1.0

        cf = ax.contourf(
            X,
            Y,
            H_masked,
            levels=levels,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(rf"$\alpha = {alpha:g}$", fontsize=14)
        ax.tick_params(labelsize=10)

        cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Estimated wave speed", fontsize=9)

    for ax in axes[len(alpha_vals):]:
        ax.axis("off")

    for ax in axes[-ncols:]:
        ax.set_xlabel(r"$\log_{10}(\lambda)$", fontsize=12)

    for ax in axes[::ncols]:
        ax.set_ylabel(r"$m_0$", fontsize=12)

    if suptitle:
        fig.suptitle(suptitle, fontsize=18, y=1.01)

    plt.tight_layout()
    plt.show()

    return fig, axes