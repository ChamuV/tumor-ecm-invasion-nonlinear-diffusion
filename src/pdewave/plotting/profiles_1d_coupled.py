# src/pdewave/plotting/profiles_1d_coupled.py

import os
import numpy as np
import matplotlib.pyplot as plt


def _nearest_time_indices(t_hist, time_values):
    return [int(np.argmin(np.abs(t_hist - t))) for t in time_values]


def plot_profiles_1d_coupled(
    x,
    u_hist,
    m_hist,
    t_hist,
    time_values=(0, 100, 200, 300),
    title="Coupled 1D travelling wave profiles",
    xlabel=r"$x$",
    ylabel=r"$u(x,t),\, m(x,t)$",
    xlim=None,
    ylim=(-0.05, 1.05),
    u_color="red",
    m_color="blue",
    show_arrows=True,
    arrow_len=None,
    arrow_x_frac=0.65,
    u_arrow_y=0.82,
    m_arrow_y=0.25,
    arrow_lw=2.5,
    head_length=1.5,
    head_width=0.8,
    text_items=None,
    text_x_frac=0.03,
    text_fontsize=17,
    legend=True,
    grid=False,
    save=False,
    folder="plots",
    filename="profiles_1d_coupled.png",
):
    x = np.asarray(x)
    u_hist = np.asarray(u_hist)
    m_hist = np.asarray(m_hist)
    t_hist = np.asarray(t_hist)

    L = x[-1] - x[0]
    indices = _nearest_time_indices(t_hist, time_values)

    fig, ax = plt.subplots(figsize=(8, 6))

    for t_target, idx in zip(time_values, indices):
        ls = "--" if np.isclose(t_hist[idx], 0.0) else "-"

        ax.plot(
            x,
            u_hist[idx],
            color=u_color,
            linestyle=ls,
            label=rf"$u(x,{int(round(t_target))})$",
        )

        ax.plot(
            x,
            m_hist[idx],
            color=m_color,
            linestyle=ls,
            label=rf"$m(x,{int(round(t_target))})$",
        )

    if show_arrows:
        if arrow_len is None:
            arrow_len = 0.13 * L

        arrow_x_start = x[0] + arrow_x_frac * L
        arrow_x_end = min(arrow_x_start + arrow_len, x[-1])

        if arrow_x_end <= arrow_x_start:
            arrow_x_start = x[-1] - arrow_len
            arrow_x_end = x[-1]

        u_arrow_style = dict(
            arrowstyle=f"->,head_length={head_length},head_width={head_width}",
            color=u_color,
            lw=arrow_lw,
        )

        m_arrow_style = dict(
            arrowstyle=f"->,head_length={head_length},head_width={head_width}",
            color=m_color,
            lw=arrow_lw,
        )

        ax.annotate(
            "",
            xy=(arrow_x_end, u_arrow_y),
            xytext=(arrow_x_start, u_arrow_y),
            arrowprops=u_arrow_style,
        )

        ax.annotate(
            "",
            xy=(arrow_x_end, m_arrow_y),
            xytext=(arrow_x_start, m_arrow_y),
            arrowprops=m_arrow_style,
        )

    if text_items is not None:
        x_text = x[0] + text_x_frac * L
        for y, text in text_items:
            ax.text(
                x_text,
                y,
                text,
                fontsize=text_fontsize,
                ha="left",
                va="top",
            )

    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)

    if xlim is None:
        ax.set_xlim([x[0], x[-1]])
    else:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(axis="both", labelsize=16)
    ax.set_title(title, fontsize=20)

    if grid:
        ax.grid(True, alpha=0.3)
    else:
        ax.grid(False)

    if legend:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=4,
            frameon=False,
            fontsize=10,
        )

    fig.tight_layout()

    if save:
        os.makedirs(folder, exist_ok=True)
        outpath = os.path.join(folder, filename)
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure to {outpath}")
        return outpath

    plt.show()
    return fig, ax