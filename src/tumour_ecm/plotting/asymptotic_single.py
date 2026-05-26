# src/tumour_ecm/plotting/asymptotic_single.py

# src/tumour_ecm/plotting/asymptotic_single.py

import numpy as np
import matplotlib.pyplot as plt

from tumour_ecm.utils.asymptotic_helpers import (
    fmt_pow10,
    load_run_for_asymptotics,
    centre_snapshot_at_front,
    centre_profile_at_half,
)

from tumour_ecm.analysis.asymptotics.logistic_limit import logistic_profile
from tumour_ecm.analysis.asymptotics.fkpp_limit import solve_fkpp_ab2am2
from tumour_ecm.analysis.asymptotics.sharp_front_limit import compute_sharp_front_wave


def get_asymptotic_profile(
    *,
    regime,
    c,
    m0,
    xi_dense,
    fkpp_L=40.0,
    fkpp_h=0.02,
    sharp_L=60.0,
):
    regime = regime.lower()

    if regime == "logistic":
        return xi_dense, logistic_profile(xi_dense, c)

    if regime in ["fkpp", "fkpp_constant_diffusion"]:
        xi_ref, U_ref = solve_fkpp_ab2am2(
            c=c,
            m0=m0,
            L=fkpp_L,
            h=fkpp_h,
        )
        return centre_profile_at_half(xi_ref, U_ref)

    if regime in ["sharp_front", "balanced", "strong_strong"]:
        xi_ref, U_ref = compute_sharp_front_wave(
            c=c,
            L=sharp_L,
        )
        return centre_profile_at_half(xi_ref, U_ref)

    raise ValueError("regime must be 'logistic', 'fkpp', or 'sharp_front'.")


def regime_title(regime):
    regime = regime.lower()

    if regime == "logistic":
        return "Logistic limit"

    if regime in ["fkpp", "fkpp_constant_diffusion"]:
        return "Weak degradation/weak regeneration: FKPP limit"

    if regime in ["sharp_front", "balanced", "strong_strong"]:
        return "Strong degradation/strong regeneration: sharp-front limit"

    return regime


def plot_asymptotic_single(
    *,
    base_dir,
    lam,
    alpha,
    m0,
    t_ref,
    regime,
    threshold=0.5,
    band=(0.1, 0.9),
    spline_type="pchip",
    xi_half_width=None,
    xi_span_factor=12.0,
    xi_min_span=15.0,
    tumour_color="#ff8c00",
    asymptotic_color="black",
    asymptotic_linestyle="--",
    figsize=(8.5, 5.2),
    title=None,
    show=True,
):
    data = load_run_for_asymptotics(
        base_dir=base_dir,
        lam=lam,
        alpha=alpha,
        m0=m0,
    )

    c = data["c"]

    xi_num, U_num, idx, x_front = centre_snapshot_at_front(
        x=data["x"],
        U=data["U"],
        times=data["times"],
        t_ref=t_ref,
        threshold=threshold,
        band=band,
        spline_type=spline_type,
        strict=True,
    )

    if xi_half_width is None:
        xi_half_width = max(
            xi_span_factor * abs(c),
            xi_min_span,
        )

    xi_dense = np.linspace(
        -xi_half_width,
        xi_half_width,
        1000,
    )

    xi_ref, U_ref = get_asymptotic_profile(
        regime=regime,
        c=c,
        m0=m0,
        xi_dense=xi_dense,
    )

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        xi_num,
        U_num,
        color=tumour_color,
        lw=2.4,
        alpha=0.95,
        label="Numerical simulation",
    )

    ax.plot(
        xi_ref,
        U_ref,
        color=asymptotic_color,
        linestyle=asymptotic_linestyle,
        lw=2.4,
        label="Asymptotic profile",
    )

    ax.set_xlim(-xi_half_width, xi_half_width)
    ax.set_ylim(-0.05, 1.05)

    ax.set_xlabel(r"$\xi$", fontsize=18)
    ax.set_ylabel(r"$U(\xi)$", fontsize=18)

    if title is None:
        title = (
            rf"{regime_title(regime)}"
            "\n"
            rf"$\lambda={fmt_pow10(lam)},\ "
            rf"\alpha={fmt_pow10(alpha)},\ "
            rf"m_0={m0:g},\ "
            rf"c={c:.3g}$"
        )

    ax.set_title(
        title,
        fontsize=16,
        fontweight="normal",
        pad=10,
    )

    ax.grid(True, ls="--", alpha=0.25)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
        fontsize=13,
    )

    fig.tight_layout(rect=[0.06, 0.16, 0.98, 0.93])

    if show:
        plt.show()

    return fig, ax