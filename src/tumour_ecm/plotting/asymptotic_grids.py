# src/tumour_ecm/plotting/asymptotic_grids.py

import numpy as np
import matplotlib.pyplot as plt

from tumour_ecm.utils.asymptotic_helpers import (
    fmt_pow10,
    load_run_for_asymptotics,
    centre_snapshot_at_front,
)

from tumour_ecm.plotting.asymptotic_single import (
    get_asymptotic_profile,
    regime_title,
)


def _is_scalar(value):
    try:
        float(value)
        return True
    except Exception:
        return False


def _row_length(row):
    A, B = row

    if _is_scalar(A) and not _is_scalar(B):
        return len(B)

    if not _is_scalar(A) and _is_scalar(B):
        return len(A)

    raise ValueError("Invalid row format.")


def plot_asymptotic_grid(
    *,
    base_dir,
    rows,
    m0,
    t_ref,
    regime,
    figsize=(12, 6.5),
    tumour_color="#ff8c00",
    asymptotic_color="black",
    sharex=True,
    sharey=True,
    suptitle=None,
    xi_span_factor=12.0,
    xi_min_span=15.0,
    show=True,
):
    nrows = len(rows)
    ncols = max(_row_length(row) for row in rows)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        squeeze=False,
        sharex=sharex,
        sharey=sharey,
    )

    legend_handles = None

    for i, row in enumerate(rows):
        A, B = row

        if _is_scalar(A) and not _is_scalar(B):
            fixed_alpha = float(A)
            lambda_vals = list(B)
            alpha_vals = None

        elif not _is_scalar(A) and _is_scalar(B):
            alpha_vals = list(A)
            fixed_lambda = float(B)
            lambda_vals = None

        else:
            raise ValueError(
                "Each row must be either (alpha_scalar, [lambda values]) "
                "or ([alpha values], lambda_scalar)."
            )

        for j in range(ncols):
            ax = axes[i, j]

            if lambda_vals is not None:
                if j >= len(lambda_vals):
                    ax.set_axis_off()
                    continue

                lam = lambda_vals[j]
                alpha = fixed_alpha

            else:
                if j >= len(alpha_vals):
                    ax.set_axis_off()
                    continue

                alpha = alpha_vals[j]
                lam = fixed_lambda

            try:
                data = load_run_for_asymptotics(
                    base_dir=base_dir,
                    lam=lam,
                    alpha=alpha,
                    m0=m0,
                )

                c = data["c"]

                xi_num, U_num, _, _ = centre_snapshot_at_front(
                    x=data["x"],
                    U=data["U"],
                    times=data["times"],
                    t_ref=t_ref,
                    strict=True,
                )

                xi_half = max(xi_span_factor * abs(c), xi_min_span)
                xi_dense = np.linspace(-xi_half, xi_half, 1000)

                xi_ref, U_ref = get_asymptotic_profile(
                    regime=regime,
                    c=c,
                    m0=m0,
                    xi_dense=xi_dense,
                )

                line_num, = ax.plot(
                    xi_num,
                    U_num,
                    color=tumour_color,
                    lw=2.2,
                    label="Numerical simulation",
                )

                line_ref, = ax.plot(
                    xi_ref,
                    U_ref,
                    color=asymptotic_color,
                    linestyle="--",
                    lw=2.2,
                    label="Asymptotic profile",
                )

                if legend_handles is None:
                    legend_handles = [line_num, line_ref]

                ax.set_xlim(-xi_half, xi_half)
                ax.set_ylim(-0.05, 1.05)

                ax.set_title(
                    rf"$\alpha={fmt_pow10(alpha)},\ "
                    rf"\lambda={fmt_pow10(lam)},\ "
                    rf"c={c:.3g}$",
                    fontsize=12,
                    fontweight="normal",
                )

            except Exception as exc:
                ax.text(
                    0.5,
                    0.5,
                    f"not plotted\n{type(exc).__name__}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    transform=ax.transAxes,
                )
                ax.set_axis_off()
                continue

            if i == nrows - 1:
                ax.set_xlabel(r"$\xi$", fontsize=14)

            if j == 0:
                ax.set_ylabel(r"$U(\xi)$", fontsize=14)

            ax.grid(True, ls="--", alpha=0.25)

    if legend_handles is not None:
        fig.legend(
            legend_handles,
            [h.get_label() for h in legend_handles],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=2,
            frameon=False,
            fontsize=13,
        )

    if suptitle is None:
        suptitle = regime_title(regime)

    fig.suptitle(suptitle, fontsize=16, fontweight="normal", y=0.95)
    fig.tight_layout(rect=[0.06, 0.06, 0.87, 0.94])

    if show:
        plt.show()

    return fig, axes