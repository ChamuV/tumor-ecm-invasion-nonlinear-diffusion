# src/pdewave/solvers/ab2am2_single.py

import numpy as np
from scipy.sparse import identity
from scipy.sparse.linalg import factorized


class AB2AM2SingleSpecies1D:
    """
    AB2--AM2 IMEX solver for scalar 1D reaction--diffusion equations:

        u_t = L(u) + f(u)

    The spatial/diffusion part is treated with AM2.
    The reaction/source part is treated with AB2.

    For constant diffusion operators, set constant_operator=True.
    For state-dependent operators, set constant_operator=False.
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
        constant_operator=True,
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

        self.constant_operator = constant_operator

        self.I = identity(self.N, format="csc")

        self.u_hist = []
        self.t_hist = []

        self._L_const = None
        self._P_const = None
        self._M_const = None
        self._BE_const = None

    def f(self, u):
        return self.model.reaction(u)

    def L(self, u):
        return self.model.diffusion_operator(
            u,
            self.dx,
            bc=self.boundary,
            fmt="csc",
        )

    def save_snapshot(self, t, u):
        self.u_hist.append(u.copy())
        self.t_hist.append(t)

    def build_constant_matrices(self, u0):
        self._L_const = self.L(u0)

        self._BE_const = factorized(
            (self.I - self.dt * self._L_const).tocsc()
        )

        self._P_const = (
            self.I + 0.5 * self.dt * self._L_const
        ).tocsc()

        self._M_const = factorized(
            (self.I - 0.5 * self.dt * self._L_const).tocsc()
        )

    def euler_step(self, u_old):
        if self.constant_operator:
            u = self._BE_const(
                u_old + self.dt * self.f(u_old)
            )
        else:
            L_old = self.L(u_old)
            solve = factorized(
                (self.I - self.dt * L_old).tocsc()
            )
            u = solve(
                u_old + self.dt * self.f(u_old)
            )

        return self.boundary.apply(u)

    def step(self, u_old, u):
        reaction_ab2 = self.dt * (
            1.5 * self.f(u) - 0.5 * self.f(u_old)
        )

        if self.constant_operator:
            rhs = self._P_const @ u + reaction_ab2
            u_next = self._M_const(rhs)

        else:
            L_old = self.L(u_old)
            L_now = self.L(u)

            P = (
                self.I + 0.5 * self.dt * L_old
            ).tocsc()

            M = factorized(
                (self.I - 0.5 * self.dt * L_now).tocsc()
            )

            rhs = P @ u + reaction_ab2
            u_next = M(rhs)

        return self.boundary.apply(u_next)

    def run(self, euler_start=True):
        self.u_hist = []
        self.t_hist = []

        u_old = self.initial_condition(self.x, self.Lx)

        if self.constant_operator:
            self.build_constant_matrices(u_old)

        if euler_start:
            u = self.euler_step(u_old)
        else:
            u = u_old.copy()

        self.save_snapshot(0.0, u_old)
        self.save_snapshot(self.dt, u)

        for i in range(2, self.Nt):
            u_next = self.step(u_old, u)

            if i % self.save_every == 0:
                self.save_snapshot(i * self.dt, u_next)

            u_old = u.copy()
            u = u_next.copy()

        return (
            np.array(self.t_hist),
            np.array(self.u_hist),
            self.x,
        )