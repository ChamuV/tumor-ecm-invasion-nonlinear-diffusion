# src/pdewave/spatial/laplacian_1d.py

import numpy as np
from scipy.sparse import diags


def laplacian_1d(N, dx, bc, fmt="csr"):
    main = -2.0 * np.ones(N)
    off = np.ones(N - 1)

    L = diags([off, main, off], [-1, 0, 1], shape=(N, N)).tolil()

    L = bc.apply_to_matrix(L)

    L = L / dx**2

    if fmt == "dense":
        return L.toarray()

    return L.asformat(fmt)