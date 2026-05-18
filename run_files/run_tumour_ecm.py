# run_files/run_tumour_ecm.py

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from pdewave.equations.tumour_ecm import TumourECMModel
from pdewave.initial_conditions.tanh_1d import TanhIC1D
from pdewave.initial_conditions.systems.tumour_ecm_ic import TumourECMIC1D
from pdewave.boundaries.neumann import NeumannBC1D
from pdewave.solvers.ab2am2_coupled import AB2AM2Coupled1D
from pdewave.plotting.profiles_1d_coupled import plot_profiles_1d_coupled
from pdewave.analysis.wavefront import estimate_wave_speed


m0 = 0.5

model = TumourECMModel(
    D=1.0,
    rho=1.0,
    K=1.0,
    lam=0.1,
    alpha=0.0,
    m0=m0,
)

u_ic = TanhIC1D(
    height=0.7,
    perc=0.2,
    steepness=0.05,
)

ic = TumourECMIC1D(
    u_ic=u_ic,
    m0=m0,
    Mmax=1.0,
)

bc = NeumannBC1D()

solver = AB2AM2Coupled1D(
    model=model,
    initial_condition=ic,
    boundary=bc,
    Lx=200.0,
    N=20001,
    T=200.0,
    dt=0.1,
    save_every=10,
)

t, u, m, x = solver.run()

c_theory = model.analytical_speed()

if c_theory is not None:
    print(f"Approx analytical speed: {c_theory:.6f}")

plot_profiles_1d_coupled(
    x=x,
    u_hist=u,
    m_hist=m,
    t_hist=t,
    time_values=[0, 40, 80, 120, 160, 200],
    title=rf"Tumour--ECM: $\lambda={model.lam},\ \alpha={model.alpha}$",
    text_items=[
        (0.90, rf"$\overline{{m}}={m0}$"),
        (0.80, rf"$c_*={c_theory:.2f}$" if c_theory is not None else r"$c_*=--$"),
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
    t_end=160,
    num_points=200,
    analytical_speed=c_theory,
    front="right",
    plot=True,
)

print(f"Numerical speed: {c_num:.6f}")
print(f"R^2: {r2:.6f}")