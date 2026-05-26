# src/tumour_ecm/operators/interpolation.py

from scipy.interpolate import (
    CubicSpline,
    PchipInterpolator,
    Akima1DInterpolator,
    interp1d,
)


def make_spline(method, x, y):
    if method == "cubic":
        return CubicSpline(x, y)

    if method == "pchip":
        return PchipInterpolator(x, y)

    if method == "akima":
        return Akima1DInterpolator(x, y)

    if method == "linear":
        return interp1d(
            x,
            y,
            kind="linear",
            fill_value="extrapolate",
        )

    raise ValueError(f"Unsupported spline_type: {method}")