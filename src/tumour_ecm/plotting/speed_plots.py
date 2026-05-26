# src/tumour_ecm/plotting/speed_plots.py

import matplotlib.pyplot as plt
from scipy.stats import linregress


def plot_wave_speed_fit(
    model,
    threshold=0.5,
    band=(0.1, 0.9),
    spline_type="cubic",
    target="N",
):
    """
    Plot tracked front position and fitted wave speed for one solved model.
    """
    t_fronts, x_fronts = model.track_wavefront_local_interpolation(
        threshold=threshold,
        band=band,
        spline_type=spline_type,
        target=target,
    )

    if len(t_fronts) < 2:
        print("Not enough valid front points.")
        return None, None, None

    slope, intercept, r_value, _, _ = linregress(t_fronts, x_fronts)

    plt.figure(figsize=(8, 4))
    plt.plot(t_fronts, x_fronts, "o", label="Tracked front")
    plt.plot(
        t_fronts,
        slope * t_fronts + intercept,
        "k--",
        label=rf"$c = {slope:.3f}$",
    )

    plt.xlabel(r"Time $t$", fontsize=14)
    plt.ylabel(r"Wavefront position $x(t)$", fontsize=14)
    plt.title(rf"Wave speed fit, $R^2 = {r_value**2:.4f}$", fontsize=15)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return slope, intercept, r_value**2


def plot_front_position(
    model,
    threshold=0.5,
    band=(0.1, 0.9),
    spline_type="cubic",
    target="N",
):
    """
    Plot tracked wavefront position x(t) for one solved model.
    """
    t_fronts, x_fronts = model.track_wavefront_local_interpolation(
        threshold=threshold,
        band=band,
        spline_type=spline_type,
        target=target,
    )

    plt.figure(figsize=(8, 4))
    plt.plot(t_fronts, x_fronts, "o-")

    plt.xlabel(r"Time $t$", fontsize=14)
    plt.ylabel(r"Wavefront position $x(t)$", fontsize=14)
    plt.title("Tracked travelling-wave front", fontsize=15)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return t_fronts, x_fronts