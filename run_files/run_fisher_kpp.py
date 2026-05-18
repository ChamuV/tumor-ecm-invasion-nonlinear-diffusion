# run_files/run_fisher_kpp.py

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from pdewave.equations.fisher_kpp import FisherKPPModel
from pdewave.initial_conditions.step_1d import StepIC1D
from pdewave.boundaries.neumann import NeumannBC1D
from pdewave.solvers.ab2am2_single import AB2AM2SingleSpecies1D
from pdewave.plotting.profiles_1d import plot_profiles_1d
from pdewave.analysis.wavefront import estimate_wave_speed


model = FisherKPPModel(D=1.0, r=1.0, K=1.0)
ic = StepIC1D(height=1.0, perc=0.2)
bc = NeumannBC1D()

solver = AB2AM2SingleSpecies1D(
    model=model,
    initial_condition=ic,
    boundary=bc,
    Lx=200.0,
    N=10001,
    T=200.0,
    dt=0.1,
    save_every=10,
    constant_operator=True,
)

t, u, x = solver.run()

c_theory = model.analytical_speed()

print(f"Analytical speed: {c_theory:.6f}")

plot_profiles_1d(
    x=x,
    u_hist=u,
    t_hist=t,
    time_values=[0, 40, 80, 120, 160, 200],
    title=rf"Fisher--KPP: $D={model.D},\ r={model.r}$",
    arrow_start_frac=0.55,
    arrow_y=0.84,
    head_length=1.5,
    head_width=1.0,
    text_items=[
        (0.86, rf"$c_* = {c_theory:.2f}$"),
    ],
)

c_num, _, r2 = estimate_wave_speed(
    x=x,
    u_hist=u,
    t_hist=t,
    threshold=0.5,
    band=(0.1, 0.9),
    spline_type="cubic",
    t_start=40,
    t_end=140,
    num_points=200,
    analytical_speed=c_theory,
    front="right",
    plot=True,
)

print(f"Numerical speed: {c_num:.6f}")
print(f"R^2: {r2:.6f}")