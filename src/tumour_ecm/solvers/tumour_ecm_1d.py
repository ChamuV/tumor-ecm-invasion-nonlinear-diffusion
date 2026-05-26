# src/tumour_ecm/solvers/tumour_ecm_1d.py

import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import eye
from scipy.sparse.linalg import spsolve
from scipy.optimize import root_scalar
from scipy.stats import linregress

from tumour_ecm.operators.reaction import tumour_growth
from tumour_ecm.operators.diffusion import build_sparse_laplacian
from tumour_ecm.operators.interpolation import make_spline


class TumourECM1D:
    """
    One-dimensional tumour--ECM invasion model.

    The model is

        u_t = d/dx( D m(1 - m) u_x ) + rho u(1 - u/K),

        m_t = alpha(1 - m) - lambda u m.

    Special cases
    -------------
    alpha = 0:
        No ECM recovery. This recovers the plain degradation model

            m_t = -lambda u m.

    alpha > 0:
        ECM recovery / functional source model.
    """
    def __init__(
        self,
        D=1.0,
        rho=1.0,
        K=1.0,
        lam=1.0,
        alpha=0.0,
        n0=1.0,
        m0=0.5,
        Mmax=1.0,
        perc=0.2,
        L=1000.0,
        N=5001,
        T=1000.0,
        dt=0.1,
        init_type="step",
        steepness=0.1,
        t_start=50.0,
        t_end=500.0,
        num_points=200,
        diffusion_eps=1e-6,
    ):
        self.D = D
        self.rho = rho
        self.K = K
        self.lam = lam
        self.alpha = alpha
        self.n0 = n0
        self.m0 = m0
        self.Mmax = Mmax
        self.perc = perc
        self.steepness = steepness
        self.diffusion_eps = diffusion_eps

        self.L = L
        self.N = N
        self.dx = L / (N - 1)
        self.x = np.linspace(0.0, L, N)

        self.T = T
        self.dt = dt
        self.Nt = int(T / dt) + 1
        self.times = np.linspace(0.0, T, self.Nt)

        self.N_arr = np.zeros((self.Nt, self.N))
        self.M_arr = np.zeros((self.Nt, self.N))

        self.wave_speed = None

        self.t_start = t_start
        self.t_end = t_end
        self.num_points = num_points

        self.init_type = init_type

    def initial_condition(self):
        if self.init_type == "step":
            u0 = self.n0 * np.where(self.x < self.perc * self.L, 0.7, 0.0)

        elif self.init_type == "tanh":
            u0 = self.n0 * 0.5 * (
                1.0 - np.tanh(self.steepness * (self.x - self.perc * self.L))
            )

        else:
            raise ValueError(
                f"Unknown init_type '{self.init_type}'. "
                "Use init_type='step' or init_type='tanh'."
            )

        m0 = self.m0 * self.Mmax * np.ones_like(self.x)
        return u0, m0

    def update_laplacian(self, m):
        return build_sparse_laplacian(
            m=m,
            D=self.D,
            dx=self.dx,
            eps=self.diffusion_eps,
        )

    def solve(self):
        u_prev, m_prev = self.initial_condition()

        f_prev = tumour_growth(u_prev, self.rho, self.K)
        L_prev = self.update_laplacian(m_prev)

        A0 = eye(self.N, format="csr") - self.dt * L_prev
        u_curr = spsolve(A0.tocsc(), u_prev + self.dt * f_prev)

        m_curr = self._update_m(m_prev, u_curr)

        self.N_arr[0] = u_prev
        self.M_arr[0] = m_prev
        self.N_arr[1] = u_curr
        self.M_arr[1] = m_curr

        I = eye(self.N, format="csr")

        for i in range(2, self.Nt):
            L_curr = self.update_laplacian(m_curr)
            f_curr = tumour_growth(u_curr, self.rho, self.K)

            rhs = (
                (I + 0.5 * self.dt * L_prev) @ u_curr
                + self.dt * (1.5 * f_curr - 0.5 * f_prev)
            )

            A = I - 0.5 * self.dt * L_curr
            u_next = spsolve(A.tocsc(), rhs)

            u_next[0] = u_next[1]
            u_next[-1] = u_next[-2]

            m_next = self._update_m(m_curr, u_next)

            self.N_arr[i] = u_next
            self.M_arr[i] = m_next

            u_curr = u_next
            m_curr = m_next
            f_prev = f_curr
            L_prev = L_curr

        return self

    def _update_m(self, m_old, u_new):
        u_nonnegative = np.maximum(u_new, 0.0)

        denominator = 1.0 + self.dt * (
            self.alpha + self.lam * u_nonnegative
        )

        m_new = (m_old + self.alpha * self.dt) / denominator
        np.clip(m_new, 0.0, self.Mmax, out=m_new)

        return m_new

    def track_wavefront_local_interpolation(
        self,
        threshold=0.5,
        band=(0.1, 0.9),
        spline_type="cubic",
        target="N",
    ):
        x = self.x
        t_vec = self.times

        if target.lower() in ["n", "u", "tumour", "tumor"]:
            u_arr = self.N_arr
        elif target.lower() in ["m", "ecm"]:
            u_arr = self.M_arr
        else:
            raise ValueError("target must be 'N'/'u' or 'M'/'m'.")

        t_list = np.linspace(self.t_start, self.t_end, self.num_points)

        x_fronts = []
        t_fronts = []

        for t_target in t_list:
            idx = int(np.argmin(np.abs(t_vec - t_target)))
            u = u_arr[idx]

            mask = (u > band[0]) & (u < band[1])
            if np.sum(mask) < 5:
                continue

            x_local = x[mask]
            u_local = u[mask]

            sort_idx = np.argsort(x_local)
            x_local = x_local[sort_idx]
            u_local = u_local[sort_idx]

            spline = make_spline(spline_type, x_local, u_local)

            crossing_idx = np.where(
                np.sign(u_local[:-1] - threshold)
                != np.sign(u_local[1:] - threshold)
            )[0]

            if len(crossing_idx) == 0:
                continue

            i = crossing_idx[0]
            x_left = x_local[i]
            x_right = x_local[i + 1]

            try:
                sol = root_scalar(
                    lambda xv: spline(xv) - threshold,
                    bracket=[x_left, x_right],
                )

                if sol.converged:
                    x_fronts.append(sol.root)
                    t_fronts.append(t_target)

            except Exception:
                continue

        return np.array(t_fronts), np.array(x_fronts)

    def estimate_wave_speed(
        self,
        threshold=0.5,
        band=(0.1, 0.9),
        spline_type="cubic",
        plot=True,
        target="N",
    ):
        t_fronts, x_fronts = self.track_wavefront_local_interpolation(
            threshold=threshold,
            band=band,
            spline_type=spline_type,
            target=target,
        )

        if len(t_fronts) < 2:
            print("Not enough valid front points.")
            return None, None, None

        slope, intercept, r_value, _, _ = linregress(t_fronts, x_fronts)

        if plot:
            plt.figure(figsize=(8, 4))
            plt.plot(t_fronts, x_fronts, "o", label="Front")
            plt.plot(
                t_fronts,
                slope * t_fronts + intercept,
                "k--",
                label=f"Slope = {slope:.3f}",
            )
            plt.xlabel("Time t")
            plt.ylabel("Wavefront x(t)")
            plt.title("Wave Speed via Linear Fit")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        print(f"Estimated speed = {slope:.4f}, R² = {r_value**2:.4f}")

        return slope, intercept, r_value**2