# src/tumour_ecm/plotting/sweep_plots.py

import numpy as np
import matplotlib.pyplot as plt

from tumour_ecm.utils.io import load_speed_from_run


def plot_speed_vs_log_lambda(
    base_dir,
    lambda_vals,
    m0_vals,
    alpha=None,
    which="N",
):
    """
    Plot wave speed against log10(lambda), with one curve per m0.
    """
    plt.figure(figsize=(7, 5))

    log_lambda = np.log10(np.asarray(lambda_vals, dtype=float))

    for m0 in m0_vals:
        speeds = [
            load_speed_from_run(
                base_dir=base_dir,
                lam=lam,
                m0=m0,
                alpha=alpha,
                which=which,
            )
            for lam in lambda_vals
        ]

        plt.plot(
            log_lambda,
            speeds,
            "-o",
            label=rf"$m_0 = {m0:g}$",
        )

    plt.xlabel(r"$\log_{10}(\lambda)$", fontsize=14)
    plt.ylabel(r"Wave speed $c$", fontsize=14)

    if alpha is None:
        title = r"Wave speed vs $\log_{10}(\lambda)$"
    else:
        title = rf"Wave speed vs $\log_{{10}}(\lambda)$, $\alpha = {alpha:g}$"

    plt.title(title, fontsize=15)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_speed_vs_m0(
    base_dir,
    m0_vals,
    lambda_fixed,
    alpha=None,
    which="N",
):
    """
    Plot wave speed against m0 for one fixed lambda.
    """
    speeds = [
        load_speed_from_run(
            base_dir=base_dir,
            lam=lambda_fixed,
            m0=m0,
            alpha=alpha,
            which=which,
        )
        for m0 in m0_vals
    ]

    plt.figure(figsize=(7, 5))
    plt.plot(m0_vals, speeds, "-o", linewidth=2)

    plt.xlabel(r"Initial ECM density $m_0$", fontsize=14)
    plt.ylabel(r"Wave speed $c$", fontsize=14)

    if alpha is None:
        title = rf"Wave speed vs $m_0$, $\lambda = {lambda_fixed:g}$"
    else:
        title = (
            rf"Wave speed vs $m_0$, "
            rf"$\lambda = {lambda_fixed:g}$, "
            rf"$\alpha = {alpha:g}$"
        )

    plt.title(title, fontsize=15)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return np.asarray(m0_vals), np.asarray(speeds)


def plot_speed_vs_log_lambda_two_panels(
    base_dir,
    lambda_vals,
    m0_list_left,
    m0_list_right,
    alpha=None,
    which="N",
    left_title="Group 1",
    right_title="Group 2",
):
    """
    Two-panel version of speed vs log10(lambda), useful when there are many m0 curves.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    x = np.log10(np.asarray(lambda_vals, dtype=float))

    def _plot_panel(ax, m0_list, title):
        for m0 in m0_list:
            speeds = [
                load_speed_from_run(
                    base_dir=base_dir,
                    lam=lam,
                    m0=m0,
                    alpha=alpha,
                    which=which,
                )
                for lam in lambda_vals
            ]

            ax.plot(
                x,
                speeds,
                marker="o",
                label=rf"$m_0 = {m0:g}$",
            )

        ax.set_title(title, fontsize=16)
        ax.set_xlabel(r"$\log_{10}(\lambda)$", fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=10, frameon=False)

    _plot_panel(axes[0], m0_list_left, left_title)
    _plot_panel(axes[1], m0_list_right, right_title)

    axes[0].set_ylabel(r"Wave speed $c$", fontsize=14)

    if alpha is None:
        title = r"Wave speed vs $\log_{10}(\lambda)$"
    else:
        title = rf"Wave speed vs $\log_{{10}}(\lambda)$, $\alpha = {alpha:g}$"

    fig.suptitle(title, fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()