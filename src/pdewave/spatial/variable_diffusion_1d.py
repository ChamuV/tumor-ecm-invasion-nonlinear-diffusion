# src/pdewave/spatial/variable_diffusion_1d.py

import numpy as np
from scipy.sparse import diags


def variable_diffusion_1d(a, dx, bc, eps=1e-6, fmt="csr"):
    """
    Builds:

        ∂x( a(x) ∂x u )

    using edge-averaged coefficients.
    """
    a = np.asarray(a).reshape(-1)
    a = np.maximum(a, eps)

    N = len(a)

    lower = np.zeros(N)
    center = np.zeros(N)
    upper = np.zeros(N)

    for i in range(1, N - 1):
        a_left = 0.5 * (a[i - 1] + a[i])
        a_right = 0.5 * (a[i] + a[i + 1])

        lower[i] = a_left
        upper[i] = a_right
        center[i] = -(a_left + a_right)

    lower, center, upper = bc.apply_to_variable_diffusion_diagonals(
        a=a,
        lower=lower,
        center=center,
        upper=upper,
    )

    return diags(
        [lower[1:], center, upper[:-1]],
        [-1, 0, 1],
        shape=(N, N),
        format=fmt,
    ) / dx**2