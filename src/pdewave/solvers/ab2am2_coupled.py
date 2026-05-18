# src/pdewave/solvers/ab2am2_coupled.py

import numpy as np
from scipy.sparse import identity
from scipy.sparse.linalg import factorized


class AB2AM2Coupled1D:
    """
    AB2--AM2 solver for the coupled tumour--ECM system.

    u is advanced using IMEX AB2--AM2:

        u_t = L(m)u + f(u)

    m is advanced using implicit Euler:

        m_t = alpha(1-m) - lam*u*m

    using u_next.
    """

    def __init__(
        self,
        model,
        initial_condition,
        boundary,
        Lx=1000.0,
        N=5001,
        T=1000.0,
        dt=0.1,
        save_every=1,
    ):
        self.model = model
        self.initial_condition = initial_condition
        self.boundary = boundary

        self.Lx = Lx
        self.N = N
        self.dx = Lx / (N - 1)
        self.x = np.linspace(0.0, Lx, N)

        self.T = T
        self.dt = dt
        self.Nt = int(T / dt) + 1
        self.save_every = save_every

        self.I = identity(self.N, format="csc")

        self.u_hist = []
        self.m_hist = []
        self.t_hist = []

    def f_u(self, u):
        return self.model.reaction_u(u)

    def L(self, m):
        return self.model.diffusion_operator(
            m,
            self.dx,
            bc=self.boundary,
            fmt="csc",
        )

    def save_snapshot(self, t, u, m):
        self.u_hist.append(u.copy())
        self.m_hist.append(m.copy())
        self.t_hist.append(t)

    def update_m_implicit(self, m, u_next):
        denom = 1.0 + self.dt * (
            self.model.alpha + self.model.lam * np.maximum(u_next, 0.0)
        )

        m_next = (m + self.model.alpha * self.dt) / denom

        return np.clip(m_next, 0.0, 1.0)

    def euler_step(self, u_old, m_old):
        L_old = self.L(m_old)

        solve = factorized(
            (self.I - self.dt * L_old).tocsc()
        )

        u = solve(
            u_old + self.dt * self.f_u(u_old)
        )

        u = self.boundary.apply(u)

        m = self.update_m_implicit(m_old, u)

        return u, m

    def step(self, u_old, m_old, u, m):
        L_old = self.L(m_old)
        L_now = self.L(m)

        P = (
            self.I + 0.5 * self.dt * L_old
        ).tocsc()

        M = factorized(
            (self.I - 0.5 * self.dt * L_now).tocsc()
        )

        reaction_ab2 = self.dt * (
            1.5 * self.f_u(u)
            - 0.5 * self.f_u(u_old)
        )

        rhs = P @ u + reaction_ab2

        u_next = M(rhs)
        u_next = self.boundary.apply(u_next)

        m_next = self.update_m_implicit(m, u_next)

        return u_next, m_next

    def run(self, euler_start=True):
        self.u_hist = []
        self.m_hist = []
        self.t_hist = []

        u_old, m_old = self.initial_condition(self.x, self.Lx)

        if euler_start:
            u, m = self.euler_step(u_old, m_old)
        else:
            u, m = u_old.copy(), m_old.copy()

        self.save_snapshot(0.0, u_old, m_old)
        self.save_snapshot(self.dt, u, m)

        for i in range(2, self.Nt):
            u_next, m_next = self.step(u_old, m_old, u, m)

            if i % self.save_every == 0:
                self.save_snapshot(i * self.dt, u_next, m_next)

            u_old = u.copy()
            m_old = m.copy()

            u = u_next.copy()
            m = m_next.copy()

        return (
            np.array(self.t_hist),
            np.array(self.u_hist),
            np.array(self.m_hist),
            self.x,
        )