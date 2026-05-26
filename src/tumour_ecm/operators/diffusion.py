# src/tumour_ecm/operators/diffusion.py

import numpy as np

from numba import njit
from scipy.sparse import diags


@njit
def build_laplacian_diagonals_avg(m, D, dx, eps=1e-6):
    """
    Build diagonals for:

        d/dx( D m(1-m) du/dx )

    with Neumann BCs.
    """
    n = len(m)

    lower = np.zeros(n)
    center = np.zeros(n)
    upper = np.zeros(n)

    for i in range(1, n - 1):

        ml = 0.5 * (m[i - 1] + m[i])
        mr = 0.5 * (m[i] + m[i + 1])

        Dl = max(eps, D * ml * (1.0 - ml))
        Dr = max(eps, D * mr * (1.0 - mr))

        lower[i] = Dl
        upper[i] = Dr
        center[i] = -(Dl + Dr)

    # Left BC
    mr = 0.5 * (m[0] + m[1])
    Dr = max(eps, D * mr * (1.0 - mr))

    center[0] = -2.0 * Dr
    upper[0] = 2.0 * Dr

    # Right BC
    ml = 0.5 * (m[-2] + m[-1])
    Dl = max(eps, D * ml * (1.0 - ml))

    center[-1] = -2.0 * Dl
    lower[-1] = 2.0 * Dl

    invdx2 = 1.0 / dx**2

    return (
        invdx2 * lower,
        invdx2 * center,
        invdx2 * upper,
    )


def build_sparse_laplacian(m, D, dx, eps=1e-6):
    lower, center, upper = build_laplacian_diagonals_avg(
        m,
        D,
        dx,
        eps,
    )

    return diags(
        [lower[1:], center, upper[:-1]],
        offsets=[-1, 0, 1],
        format="csr",
    )