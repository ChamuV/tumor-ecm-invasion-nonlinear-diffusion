# src/tumour_ecm/analysis/asymptotics/sharp_front_limit.py

import numpy as np
from scipy.integrate import solve_ivp


def sharp_front_diffusion(U):
    return U / (1.0 + U) ** 2


def sharp_front_rhs(xi, y, c, delta):
    U, J = y

    D = sharp_front_diffusion(U) + delta

    dU = J / D
    dJ = -c * J / D - U * (1.0 - U)

    return [dU, dJ]


def sharp_front_jacobian(xi, y, c, delta):
    U, J = y

    D = sharp_front_diffusion(U) + delta
    dD = (1.0 - U) / (1.0 + U) ** 3

    df1dU = -J * dD / (D * D)
    df1dJ = 1.0 / D

    df2dU = (c * J * dD) / (D * D) - (1.0 - 2.0 * U)
    df2dJ = -c / D

    return np.array(
        [
            [df1dU, df1dJ],
            [df2dU, df2dJ],
        ]
    )


def make_tip_event(epsR):
    def event(xi, y):
        return y[0] - epsR

    event.terminal = True
    event.direction = -1

    return event


def compute_sharp_front_wave(
    c,
    L=60.0,
    epsL=1e-4,
    epsR=1e-4,
    delta=1e-8,
    rtol=1e-7,
    atol=1e-9,
    max_step=0.5,
):
    U0 = 1.0 - epsL
    U0p = -(1.0 / c) * U0 * (1.0 - U0)
    J0 = (sharp_front_diffusion(U0) + delta) * U0p

    sol = solve_ivp(
        fun=lambda xi, y: sharp_front_rhs(xi, y, c, delta),
        t_span=(-L, L),
        y0=[U0, J0],
        method="Radau",
        jac=lambda xi, y: sharp_front_jacobian(xi, y, c, delta),
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        events=make_tip_event(epsR),
    )

    return sol.t, sol.y[0]