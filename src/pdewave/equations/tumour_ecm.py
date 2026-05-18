# src/pdewave/equations/tumour_ecm.py

import numpy as np

from pdewave.spatial.ecm_diffusion_1d import ECMDiffusionOperator1D


class TumourECMModel:
    """
    Tumour--ECM system:

        u_t = ∂x(D m(1-m) u_x) + rho*u*(1-u/K)

        m_t = alpha*(1-m) - lam*u*m
    """

    def __init__(
        self,
        D=1.0,
        rho=1.0,
        K=1.0,
        lam=1.0,
        alpha=0.0,
        m0=None,
        eps=1e-6,
    ):
        self.D = D
        self.rho = rho
        self.K = K

        self.lam = lam
        self.alpha = alpha

        self.m0 = m0
        self.eps = eps

        self.diffusion_builder = ECMDiffusionOperator1D(
            D=D,
            eps=eps,
        )

    def reaction_u(self, u):
        return self.rho * u * (1.0 - u / self.K)

    def reaction_m(self, u, m):
        return self.alpha * (1.0 - m) - self.lam * u * m

    def diffusion_operator(self, m, dx, bc, fmt="csr"):
        return self.diffusion_builder.build(
            m=m,
            dx=dx,
            bc=bc,
            fmt=fmt,
        )

    def analytical_speed(self):
        """
        Approximate Fisher--KPP travelling-wave speed.

        If alpha = 0 and m is initially constant:

            c* = 2 sqrt(rho * D*m0*(1-m0))
        """
        if self.m0 is None:
            return None

        effective_D = self.D * self.m0 * (1.0 - self.m0)

        if effective_D <= 0:
            return None

        return 2.0 * np.sqrt(
            self.rho * effective_D
        )