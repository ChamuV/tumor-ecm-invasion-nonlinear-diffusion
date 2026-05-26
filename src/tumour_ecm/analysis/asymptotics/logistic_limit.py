# src/tumour_ecm/analysis/asymptotics/logistic_limit.py

import numpy as np


def logistic_profile(xi, c):
    xi = np.asarray(xi, dtype=float)
    return 1.0 / (1.0 + np.exp(xi / c))