# src/tumour_ecm/analysis/asymptotics/fkpp_limit.py

import numpy as np


def fkpp_diffusion(m0):
    return float(m0 * (1.0 - m0))


def fkpp_min_speed(m0):
    return 2.0 * np.sqrt(fkpp_diffusion(m0))


def solve_fkpp_ab2am2(
    c,
    m0,
    L=40.0,
    h=0.02,
    eps=1e-5,
):
    D = fkpp_diffusion(m0)

    if D <= 0:
        raise ValueError("FKPP diffusion m0(1-m0) must be positive.")

    n_steps = int(np.ceil(2.0 * L / abs(h))) + 1
    h = abs(h)

    xi = np.linspace(-L, L, n_steps)

    r = 0.5 * (-c + np.sqrt(c * c + 4.0 * D)) / D

    U0 = 1.0 - eps
    P0 = -r * eps

    Y = np.zeros((2, n_steps), dtype=float)
    Y[:, 0] = [U0, P0]

    def fP(U):
        return -(1.0 / D) * U * (1.0 - U)

    U1 = U0 + h * P0
    P1 = P0 + h * (fP(U0) - (c / D) * P0)

    Ue = U0 + 0.5 * h * (P0 + P1)
    Pe = P0 + 0.5 * h * (
        (fP(U0) - (c / D) * P0)
        + (fP(U1) - (c / D) * P1)
    )

    Y[:, 1] = [Ue, Pe]

    fU_prev = P0
    fP_prev = fP(U0)

    fU_curr = Pe
    fP_curr = fP(Ue)

    for n in range(1, n_steps - 1):
        U_n = Y[0, n]
        P_n = Y[1, n]

        U_next = U_n + h * (1.5 * fU_curr - 0.5 * fU_prev)

        P_ab2 = P_n + h * (1.5 * fP_curr - 0.5 * fP_prev)
        P_next = (P_ab2 - 0.5 * h * (c / D) * P_n) / (
            1.0 + 0.5 * h * (c / D)
        )

        Y[:, n + 1] = [U_next, P_next]

        fU_prev = fU_curr
        fP_prev = fP_curr

        fU_curr = P_next
        fP_curr = fP(U_next)

    return xi, Y[0]