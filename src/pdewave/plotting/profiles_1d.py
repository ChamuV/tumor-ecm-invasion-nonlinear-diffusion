# src/pdewave/plotting/profiles_1d.py

import os
import numpy as np
import matplotlib.pyplot as plt


def _nearest_time_indices(t_hist, time_values):
    return [int(np.argmin(np.abs(t_hist - t))) for t in time_values]


def _auto_time_indices(t_hist, max_profiles):
    step = max(1, len(t_hist) // max_profiles)
    indices = list(range(0, len(t_hist), step))

    if indices[-1] != len(t_hist) - 1:
        indices.append(len(t_hist) - 1)

    return indices


def plot_profiles_1d(
    x,
    u_hist,
    t_hist,
    time_values=None,
    max_profiles=8,
    title="1D travelling wave profiles",
    xlabel=r"$x$",
    ylabel=r"$u(x,t)$",
    xlim=None,
    ylim=(-0.05, 1.05),
    line_color="red",
    arrow=True,
    arrow_len=None,
    arrow_start_frac=0.58,
    arrow_y=0.84,
    arrow_lw=2.5,
    head_length=3,
    head_width=1.5,
    text_items=None,
    text_x_frac=0.035,
    text_fontsize=17,
    legend=True,
    grid=False,
    save=False,
    folder="plots",
    filename="profiles_1d.png",
):
    """
    Dissertation-style 1D profile plot.

    Example text_items:
        [(0.88, "$c_* = 2.00$")]
    """

    x = np.asarray(x)
    u_hist = np.asarray(u_hist)
    t_hist = np.asarray(t_hist)

    L = x[-1] - x[0]

    if time_values is not None:
        indices = _nearest_time_indices(t_hist, time_values)
    else:
        indices = _auto_time_indices(t_hist, max_profiles)

    fig, ax = plt.subplots(figsize=(8, 6))

    for idx in indices:
        t_label = t_hist[idx]
        linestyle = "--" if np.isclose(t_label, 0.0) else "-"

        ax.plot(
            x,
            u_hist[idx],
            color=line_color,
            linestyle=linestyle,
            label=rf"$u(x,{t_label:.0f})$",
        )

    if arrow:
        if arrow_len is None:
            arrow_len = 0.13 * L

        arrow_x_start = x[0] + arrow_start_frac * L
        arrow_x_end = min(arrow_x_start + arrow_len, x[-1])

        if arrow_x_end <= arrow_x_start:
            arrow_x_start = x[-1] - arrow_len
            arrow_x_end = x[-1]

        arrow_style = dict(
            arrowstyle=f"->,head_length={head_length},head_width={head_width}",
            color=line_color,
            lw=arrow_lw,
        )

        ax.annotate(
            "",
            xy=(arrow_x_end, arrow_y),
            xytext=(arrow_x_start, arrow_y),
            arrowprops=arrow_style,
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
            ncol=min(4, len(indices)),
            frameon=False,
            fontsize=11,
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