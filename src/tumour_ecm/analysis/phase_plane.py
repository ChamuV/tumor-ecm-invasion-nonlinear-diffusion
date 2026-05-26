# src/tumour_ecm/analysis/phase_plane.py

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from tumour_ecm.utils.io import load_speed_from_run


def rhs_design(z, X, c, lam, alpha):
    U, P, M = X

    D = M * (1.0 - M)

    dU = P
    dP = -c * P - U * (1.0 - U) * D
    dM = (D / c) * (lam * U * M - alpha * (1.0 - M))

    return [dU, dP, dM]


def make_stop_event_generic(blowU=5.0, blowP=5.0):
    def event(t, X, *args):
        U, P, M = X

        badM = (M < -1e-6) or (M > 1.0 + 1e-6)

        blowUP = (
            (not np.isfinite(U))
            or (not np.isfinite(P))
            or (abs(U) > blowU)
            or (abs(P) > blowP)
        )

        return 0.0 if (badM or blowUP) else 1.0

    event.terminal = True
    event.direction = 0

    return event


def get_reference_speed(
    *,
    lam,
    alpha,
    m0,
    c_calc=None,
    base_dir=None,
    which_speed="N",
):
    """
    Choose the reference speed used in the phase-plane comparison.

    alpha = 0:
        use c_min = 2 sqrt(m0(1-m0)).

    alpha != 0:
        use c_calc if provided.
        otherwise try to load wave speed from saved simulation data.
        if loading fails, fall back to 2 sqrt(m0(1-m0)).
    """
    if np.isclose(alpha, 0.0):
        c_ref = 2.0 * np.sqrt(m0 * (1.0 - m0))
        c_name = r"c_{\min}"
        source = "formula"

        return c_ref, c_name, source

    c_name = r"c_{\mathrm{calc}}"

    if c_calc is not None:
        return float(c_calc), c_name, "provided"

    if base_dir is not None:
        loaded = load_speed_from_run(
            base_dir=base_dir,
            lam=lam,
            m0=m0,
            alpha=alpha,
            which=which_speed,
        )

        if np.isfinite(loaded):
            return float(loaded), c_name, "loaded"

    fallback = 2.0 * np.sqrt(m0 * (1.0 - m0))

    return fallback, c_name, "fallback_formula"


def plot_phaseplane_recovery_appendix(
    *,
    lam,
    alpha,
    m0,
    U0=0.99,
    P0=-0.001,
    M0=None,
    speeds=None,
    c_calc=None,
    base_dir=None,
    which_speed="N",
    z_end=400.0,
    dz=0.01,
    rtol=1e-12,
    atol=1e-14,
    max_step=0.01,
    title=None,
    legend_outside=True,
    show=True,
):
    if M0 is None:
        M0 = m0

    c_ref, c_name, c_source = get_reference_speed(
        lam=lam,
        alpha=alpha,
        m0=m0,
        c_calc=c_calc,
        base_dir=base_dir,
        which_speed=which_speed,
    )

    if speeds is None:
        speeds = [0.2 * c_ref, c_ref, 2.0 * c_ref]

    t_eval = np.arange(0.0, z_end + dz, dz)
    guard_evt = make_stop_event_generic()

    sols = []

    for c in speeds:
        sol = solve_ivp(
            rhs_design,
            (0.0, z_end),
            [U0, P0, M0],
            args=(c, lam, alpha),
            method="BDF",
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            t_eval=t_eval,
            events=guard_evt,
        )

        sols.append((c, sol))

    legend_labels = [
        rf"$c < {c_name}$",
        rf"$c = {c_name}$",
        rf"$c > {c_name}$",
    ]

    fig, ax = plt.subplots(figsize=(7.4, 3.2))

    colors = ["#4b006e", "#1f9e9a", "#f2d600"]

    for (c, sol), col, label in zip(sols, colors, legend_labels):
        U = sol.y[0]
        P = sol.y[1]

        ax.plot(U, P, lw=1.8, color=col, label=label)
        ax.plot(U[0], P[0], marker="o", ms=4, mfc="white", mec=col)
        ax.plot(U[-1], P[-1], marker="o", ms=4, mfc=col, mec=col)

    ax.axhline(0, ls="--", lw=0.9, color="0.55")
    ax.axvline(0, ls="--", lw=0.9, color="0.75")
    ax.grid(True, ls=":", alpha=0.45)

    ax.set_xlabel(r"$U$", fontsize=16)
    ax.set_ylabel(r"$P$", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)

    if title is None:
        title = (
            rf"$m_0 = {m0:.3g},\ "
            rf"\lambda = {lam:g},\ "
            rf"\alpha = {alpha:g},\ "
            rf"{c_name} = {c_ref:.3g}$"
        )

    ax.set_title(title, fontsize=16, fontweight="normal", pad=10)

    if legend_outside:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=13,
        )

        plt.tight_layout(rect=[0, 0, 0.82, 1])

    else:
        ax.legend(frameon=False, fontsize=13)
        plt.tight_layout()

    if show:
        plt.show()

    return {
        "c_ref": c_ref,
        "c_name": c_name,
        "c_source": c_source,
        "speeds": speeds,
        "solutions": sols,
        "fig": fig,
        "ax": ax,
    }