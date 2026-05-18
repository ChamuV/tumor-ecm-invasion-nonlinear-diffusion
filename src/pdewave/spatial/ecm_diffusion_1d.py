# src/pdewave/spatial/ecm_diffusion_1d.py

import numpy as np
from scipy.sparse import diags


class ECMDiffusionOperator1D:
    """
    Builds the finite-difference operator for

        ∂x( D m(1-m) ∂x u )

    using the edge-averaged m method as the dissertation code:

        m_edge = 0.5 * (m_i + m_{i+1})
        D_edge = D * m_edge * (1 - m_edge).
    """

    def __init__(self, D=1.0, eps=1e-6):
        self.D = D
        self.eps = eps

    def coefficient(self, m_edge):
        return np.maximum(
            self.eps,
            self.D * m_edge * (1.0 - m_edge),
        )

    def build(self, m, dx, bc, fmt="csr"):
        m = np.asarray(m).reshape(-1)
        N = len(m)

        lower = np.zeros(N)
        center = np.zeros(N)
        upper = np.zeros(N)

        for i in range(1, N - 1):
            m_left = 0.5 * (m[i - 1] + m[i])
            m_right = 0.5 * (m[i] + m[i + 1])

            D_left = self.coefficient(m_left)
            D_right = self.coefficient(m_right)

            lower[i] = D_left
            upper[i] = D_right
            center[i] = -(D_left + D_right)

        lower, center, upper = bc.apply_to_ecm_diffusion_diagonals(
            m=m,
            coefficient=self.coefficient,
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