# src/pdewave/initial_conditions/system_initial_conditions/tumour_ecm_ic.py

import numpy as np


class TumourECMIC1D:
    """
    Initial condition builder for the coupled tumour--ECM system.

    It takes a scalar tumour profile generator u_ic and combines it
    with a constant initial ECM field.

    Returns
    -------
    u0 : array
        Initial tumour density.

    m0 : array
        Initial ECM density.
    """

    def __init__(self, u_ic, m0=0.5, Mmax=1.0):
        self.u_ic = u_ic
        self.m0 = m0
        self.Mmax = Mmax

    def __call__(self, x, L):
        u0 = self.u_ic(x, L)
        m0 = self.m0 * self.Mmax * np.ones_like(x)

        return u0, m0