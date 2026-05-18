# src/pdewave/initial_conditions/tanh_1d.py

import numpy as np


class TanhIC1D:
    """
    Smooth travelling-front initial condition:

        u0(x) = height/2 * (1 - tanh(steepness * (x - perc*L)))
    """

    def __init__(self, height=1.0, perc=0.2, steepness=0.1):
        self.height = height
        self.perc = perc
        self.steepness = steepness

    def __call__(self, x, L):
        return self.height * 0.5 * (
            1.0 - np.tanh(self.steepness * (x - self.perc * L))
        )