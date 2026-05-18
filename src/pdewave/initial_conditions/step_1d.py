# src/pdewave/intial_conditions/step_1d.py

import numpy as np


class StepIC1D:
    def __init__(self, height=1.0, perc=0.2):
        self.height = height
        self.perc = perc

    def __call__(self, x, L):
        return self.height * np.where(x < self.perc * L, 1.0, 0.0)