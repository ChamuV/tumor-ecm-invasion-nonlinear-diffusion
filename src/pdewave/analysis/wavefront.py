# src/pdewave/analysis/wavefront.py

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import (
    CubicSpline,
    PchipInterpolator,
    Akima1DInterpolator,
    interp1d,
)
from scipy.optimize import root_scalar
from scipy.stats import linregress


def _get_spline(method, x, y):
    if method == "cubic":
        return CubicSpline(x, y)
    if method == "pchip":
        return PchipInterpolator(x, y)
    if method == "akima":
        return Akima1DInterpolator(x, y)
    if method == "linear":
        return interp1d(x, y, kind="linear", fill_value="extrapolate")

    raise ValueError(f"Unsupported spline_type: {method}")


def track_wavefront(
    x,
    u_hist,
    t_hist,
    threshold=0.5,
    band=(0.1, 0.9),
    spline_type="cubic",
    t_start=None,
    t_end=None,
    num_points=200,
    front="right",
):
    """
    Track the wavefront by finding where u crosses a chosen threshold.

    This follows the dissertation method:
    local band selection -> interpolation -> root finding.
    """

    x = np.asarray(x)
    u_hist = np.asarray(u_hist)
    t_hist = np.asarray(t_hist)

    if t_start is None:
        t_start = t_hist[0]

    if t_end is None:
        t_end = t_hist[-1]

    t_targets = np.linspace(t_start, t_end, num_points)

    x_fronts = []
    t_fronts = []

    for t_target in t_targets:
        idx = int(np.argmin(np.abs(t_hist - t_target)))
        u = u_hist[idx]

        mask = (u > band[0]) & (u < band[1])
        if np.sum(mask) < 5:
            continue

        x_local = x[mask]
        u_local = u[mask]

        sort_idx = np.argsort(x_local)
        x_local = x_local[sort_idx]
        u_local = u_local[sort_idx]

        crossings = np.where(
            np.sign(u_local[:-1] - threshold)
            != np.sign(u_local[1:] - threshold)
        )[0]

        if len(crossings) == 0:
            continue

        if front == "right":
            i = crossings[-1]
        elif front == "left":
            i = crossings[0]
        else:
            raise ValueError("front must be 'right' or 'left'.")

        xl = x_local[i]
        xr = x_local[i + 1]

        try:
            spline = _get_spline(spline_type, x_local, u_local)

            sol = root_scalar(
                lambda xv: spline(xv) - threshold,
                bracket=[xl, xr],
            )

            if sol.converged:
                x_fronts.append(sol.root)
                t_fronts.append(t_target)

        except Exception:
            continue

    return np.array(t_fronts), np.array(x_fronts)


def estimate_wave_speed(
    x,
    u_hist,
    t_hist,
    threshold=0.5,
    band=(0.1, 0.9),
    spline_type="cubic",
    t_start=None,
    t_end=None,
    num_points=200,
    analytical_speed=None,
    front="right",
    plot=True,
):
    """
    Estimate wave speed by fitting:

        x_front(t) = c t + b

    The slope c is the numerical wave speed.
    """

    t_fronts, x_fronts = track_wavefront(
        x=x,
        u_hist=u_hist,
        t_hist=t_hist,
        threshold=threshold,
        band=band,
        spline_type=spline_type,
        t_start=t_start,
        t_end=t_end,
        num_points=num_points,
        front=front,
    )

    if len(t_fronts) < 2:
        raise ValueError("Not enough valid front points.")

    slope, intercept, r_value, _, _ = linregress(t_fronts, x_fronts)
    r2 = r_value**2

    if plot:
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            t_fronts,
            x_fronts,
            "o",
            markersize=5,
            label="Interpolated front",
        )

        ax.plot(
            t_fronts,
            slope * t_fronts + intercept,
            "k--",
            linewidth=2,
            label=rf"$c_{{num}}={slope:.4f}$",
        )

        ax.text(
            0.04,
            0.86,
            rf"$c_{{num}} = {slope:.4f}$",
            transform=ax.transAxes,
            fontsize=14,
            ha="left",
            va="top",
        )

        if analytical_speed is not None:
            ax.text(
                0.04,
                0.76,
                rf"$c_* = {analytical_speed:.4f}$",
                transform=ax.transAxes,
                fontsize=14,
                ha="left",
                va="top",
            )

            r2_y = 0.66
        else:
            r2_y = 0.76

        ax.text(
            0.04,
            r2_y,
            rf"$R^2 = {r2:.4f}$",
            transform=ax.transAxes,
            fontsize=14,
            ha="left",
            va="top",
        )

        ax.set_xlabel(r"$t$", fontsize=16)
        ax.set_ylabel(r"$x_{\mathrm{front}}(t)$", fontsize=16)
        ax.set_title("Wavefront tracking and speed estimation", fontsize=18)

        ax.tick_params(axis="both", labelsize=14)
        ax.grid(False)

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=2,
            frameon=False,
            fontsize=12,
        )

        fig.tight_layout()
        plt.show()

    return slope, intercept, r2