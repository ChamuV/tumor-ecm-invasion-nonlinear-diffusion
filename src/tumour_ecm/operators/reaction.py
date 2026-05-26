# src/tumour_ecm/operators/reaction.py

import numpy as np
from numba import njit


@njit
def tumour_growth(u, rho, K):
    """
    Logistic tumour growth term:

        f(u) = rho u (1 - u/K)
    """
    return rho * u * (1.0 - u / K)