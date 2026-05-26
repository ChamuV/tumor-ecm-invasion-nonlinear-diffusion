# src/tumour_ecm/plotting/travelling_wave_single.py

import numpy as np
import matplotlib.pyplot as plt


def plot_u_and_m_travelling_wave(
    model,
    t_indices=(0, 500, 750, 1000),
    arrow_len=None,
    arrow_lw=2.5,
    arrow_start_frac=0.5,
    head_length=1.5,
    head_width=1,
    bottom_y=0.2,
    ylim=None,
    show_speed=True,
):
    """
    Plot tumour and ECM travelling-wave profiles for a single simulation.
    """
    x = model.x
    u_arr = model.N_arr
    m_arr = model.M_arr
    t_vec = model.times

    if show_speed and getattr(model, "wave_speed", None) is None:
        model.wave_speed, _, _ = model.estimate_wave_speed(
            plot=False,
            target="N",
            threshold=0.5,
            band=(0.1, 0.9),
            spline_type="cubic",
        )

    if arrow_len is None:
        arrow_len = 0.15 * model.L

    arrow_x_start = np.clip(arrow_start_frac * model.L, 0.0, model.L)
    arrow_x_end = np.clip(arrow_x_start + arrow_len, 0.0, model.L)

    if arrow_x_end <= arrow_x_start:
        arrow_x_start = np.clip(model.L - arrow_len, 0.0, model.L)
        arrow_x_end = model.L

    if ylim is None:
        ylim = (-0.05, 1.55 if model.alpha > 0 else 1.05)

    plt.figure(figsize=(8, 6))

    for tidx in t_indices:
        tidx = int(tidx)

        if tidx < 0 or tidx >= len(t_vec):
            continue

        t_label = int(t_vec[tidx])
        linestyle = "--" if t_label == 0 else "-"

        plt.plot(
            x,
            u_arr[tidx],
            color="red",
            linestyle=linestyle,
            label=rf"$u(x,{t_label})$",
        )

        plt.plot(
            x,
            m_arr[tidx],
            color="blue",
            linestyle=linestyle,
            label=rf"$m(x,{t_label})$",
        )

    arrow_style_red = dict(
        arrowstyle=f"->,head_length={head_length},head_width={head_width}",
        color="red",
        lw=arrow_lw,
    )

    arrow_style_blue = dict(
        arrowstyle=f"->,head_length={head_length},head_width={head_width}",
        color="blue",
        lw=arrow_lw,
    )

    plt.annotate(
        "",
        xy=(arrow_x_end, 0.9),
        xytext=(arrow_x_start, 0.9),
        arrowprops=arrow_style_red,
    )

    plt.annotate(
        "",
        xy=(arrow_x_end, bottom_y),
        xytext=(arrow_x_start, bottom_y),
        arrowprops=arrow_style_blue,
    )

    x_text = x[0] + 0.02 * model.L
    c_str = f"{model.wave_speed:.3g}" if show_speed and model.wave_speed is not None else "—"

    text_y_top = ylim[1] - 0.13 * (ylim[1] - ylim[0])
    text_y_bottom = ylim[1] - 0.23 * (ylim[1] - ylim[0])

    plt.text(x_text, text_y_top, rf"$\overline{{m}} = {model.m0}$", fontsize=18)
    plt.text(x_text, text_y_bottom, rf"$c = {c_str}$", fontsize=18)

    plt.xlabel(r"$x$", fontsize=18)
    plt.ylabel(r"$u(x,t),\, m(x,t)$", fontsize=18)
    plt.ylim(ylim)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(rf"$\lambda = {model.lam},\ \alpha = {model.alpha}$", fontsize=20)
    plt.xlim([0.0, model.L])
    plt.grid(False)
    plt.tight_layout()
    plt.show()