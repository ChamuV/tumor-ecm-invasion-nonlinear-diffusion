# src/pdewave/equations/fisher_kpp.py

import numpy as np

from pdewave.spatial.laplacian_1d import laplacian_1d


class FisherKPPModel:
    def __init__(self, D=1.0, r=1.0, K=1.0):
        self.D = D
        self.r = r
        self.K = K

    def reaction(self, u):
        return self.r * u * (1.0 - u / self.K)

    def diffusion_operator(self, u, dx, bc, fmt="csr"):
        N = len(u)
        return self.D * laplacian_1d(N=N, dx=dx, bc=bc, fmt=fmt)

    def analytical_speed(self):
        return 2.0 * np.sqrt(self.r * self.D)