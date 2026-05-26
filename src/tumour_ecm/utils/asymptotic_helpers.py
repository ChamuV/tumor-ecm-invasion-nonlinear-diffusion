# src/tumour_ecm/utils/asymptotic_helpers.py

from pathlib import Path

import numpy as np

from scipy.interpolate import PchipInterpolator, interp1d
from scipy.optimize import root_scalar

from tumour_ecm.utils.io import find_run_dir, load_summary, load_snapshots


def fmt_pow10(value):
    value = float(value)

    if np.isclose(value, 0.0):
        return "0"

    if np.isclose(value, 1.0):
        return "1"

    if np.isclose(value, 10.0):
        return "10"

    exponent = int(np.round(np.log10(abs(value))))

    if np.isclose(value, 10.0**exponent):
        return rf"10^{{{exponent}}}"

    mantissa = value / (10.0**exponent)
    return rf"{mantissa:.2g}\times 10^{{{exponent}}}"


def load_run_for_asymptotics(base_dir, lam, alpha, m0):
    run_dir = find_run_dir(
        base_dir=base_dir,
        lam=lam,
        alpha=alpha,
        m0=m0,
    )

    if run_dir is None:
        raise FileNotFoundError(
            f"No run found for lambda={lam}, alpha={alpha}, m0={m0}"
        )

    summary = load_summary(run_dir)
    snapshots = load_snapshots(run_dir)

    if not summary:
        raise FileNotFoundError(f"No summary.json found in {run_dir}")

    if snapshots is None:
        raise FileNotFoundError(f"No snapshots.npz found in {run_dir}")

    return {
        "run_dir": Path(run_dir),
        "summary": summary,
        "x": snapshots["x"],
        "times": snapshots["times"],
        "U": snapshots["N_arr"],
        "M": snapshots["M_arr"],
        "c": float(summary["wave_speed"]),
        "lambda": float(summary.get("lambda", summary.get("lambda_val", lam))),
        "alpha": float(summary.get("alpha", alpha)),
        "m0": float(summary.get("m0", m0)),
    }


def front_location_at_time(
    x,
    u_row,
    threshold=0.5,
    band=(0.1, 0.9),
    spline_type="pchip",
):
    mask = (u_row > band[0]) & (u_row < band[1])

    if np.sum(mask) < 5:
        return None

    x_local = x[mask]
    u_local = u_row[mask]

    order = np.argsort(x_local)
    x_local = x_local[order]
    u_local = u_local[order]

    if spline_type == "pchip":
        spline = PchipInterpolator(x_local, u_local, extrapolate=True)
    elif spline_type == "linear":
        spline = interp1d(
            x_local,
            u_local,
            kind="linear",
            fill_value="extrapolate",
        )
    else:
        raise ValueError("Use spline_type='pchip' or spline_type='linear'.")

    sign = np.sign(u_local - threshold)
    crossing_idx = np.where(sign[:-1] * sign[1:] < 0)[0]

    if len(crossing_idx) == 0:
        return None

    i = int(crossing_idx[0])
    x_left = x_local[i]
    x_right = x_local[i + 1]

    try:
        sol = root_scalar(
            lambda xv: spline(xv) - threshold,
            bracket=[x_left, x_right],
            method="brentq",
        )

        if sol.converged:
            return float(sol.root)

    except Exception:
        return None

    return None


def best_center_for_time(
    x,
    U,
    times,
    t_ref,
    threshold=0.5,
    band=(0.1, 0.9),
    spline_type="pchip",
    fallback_thresholds=(0.45, 0.55),
):
    idx = int(np.argmin(np.abs(times - t_ref)))

    x_front = front_location_at_time(
        x=x,
        u_row=U[idx],
        threshold=threshold,
        band=band,
        spline_type=spline_type,
    )

    if x_front is not None:
        return idx, x_front

    for threshold_try in fallback_thresholds:
        x_front = front_location_at_time(
            x=x,
            u_row=U[idx],
            threshold=threshold_try,
            band=(0.05, 0.95),
            spline_type=spline_type,
        )

        if x_front is not None:
            return idx, x_front

    return idx, 0.5 * (x[0] + x[-1])


def centre_snapshot_at_front(
    x,
    U,
    times,
    t_ref,
    threshold=0.5,
    band=(0.1, 0.9),
    spline_type="pchip",
):
    idx, x_front = best_center_for_time(
        x=x,
        U=U,
        times=times,
        t_ref=t_ref,
        threshold=threshold,
        band=band,
        spline_type=spline_type,
    )

    xi = x - x_front
    return xi, U[idx], idx, x_front


def x_at_half(x, u):
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)

    if u[0] < u[-1]:
        x = x[::-1]
        u = u[::-1]

    f = interp1d(
        u,
        x,
        bounds_error=False,
        fill_value="extrapolate",
    )

    return float(f(0.5))


def centre_profile_at_half(xi, U):
    x0 = x_at_half(xi, U)
    return xi - x0, U