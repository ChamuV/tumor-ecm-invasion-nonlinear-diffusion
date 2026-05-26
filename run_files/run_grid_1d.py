# run_files/run_grid_1d.py

import os
import sys
from pathlib import Path

from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from tumour_ecm.solvers.tumour_ecm_1d import TumourECM1D
from tumour_ecm.utils.io import (
    safe_tag,
    ensure_dir,
    save_json,
    save_snapshots,
    save_fronts,
)


def run_one(lam, m0, alpha, base_dir, model_kwargs, overwrite=False):
    try:
        run_dir = (
            Path(base_dir)
            / f"alpha_{safe_tag(alpha)}"
            / f"lambda_{safe_tag(lam)}"
            / f"m0_{safe_tag(m0)}"
        )

        summary_path = run_dir / "summary.json"

        if summary_path.exists() and not overwrite:
            return ("skipped", lam, m0, alpha)

        ensure_dir(run_dir)

        model = TumourECM1D(
            lam=lam,
            m0=m0,
            alpha=alpha,
            **model_kwargs,
        )

        print(f"Running alpha={alpha}, lambda={lam}, m0={m0}")
        model.solve()

        cN, bN, r2N = model.estimate_wave_speed(
            threshold=0.5,
            band=(0.1, 0.9),
            spline_type="cubic",
            plot=False,
            target="N",
        )

        model.wave_speed = cN

        m_threshold = 0.5 * float(m0)
        cM, r2M = None, None

        if m_threshold > 0:
            try:
                cM, bM, r2M = model.estimate_wave_speed(
                    threshold=m_threshold,
                    band=(0.1, 0.9),
                    spline_type="cubic",
                    plot=False,
                    target="M",
                )
            except Exception:
                pass

        summary = {
            "lambda": float(lam),
            "m0": float(m0),
            "alpha": float(alpha),

            "wave_speed": float(cN) if cN is not None else None,
            "r2": float(r2N) if r2N is not None else None,

            "m_threshold": float(m_threshold),
            "m_wave_speed": float(cM) if cM is not None else None,
            "m_r2": float(r2M) if r2M is not None else None,

            "L": model.L,
            "N": model.N,
            "T": model.T,
            "dt": model.dt,
            "init_type": model.init_type,
            "steepness": model.steepness,
            "perc": model.perc,
            "saved_stride": 150,
        }

        save_json(summary_path, summary)
        save_snapshots(run_dir, model, stride=150)
        save_fronts(run_dir, model, target="N", threshold=0.5)

        if m_threshold > 0:
            save_fronts(run_dir, model, target="M", threshold=m_threshold)

        return ("done", lam, m0, alpha, cN, r2N, cM, r2M)

    except Exception as e:
        return ("failed", lam, m0, alpha, str(e))


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    base_dir = ROOT / "outputs" / "sweeps" / "tumour_ecm_1d"

    lambda_vals = [0.001, 0.01, 0.05, 1, 5, 10, 100]
    m0_vals = [0.05, 0.1, 0.2, 0.5, 0.8, 0.9]
    alpha_vals = [0.0, 1.0]

    model_kwargs = dict(
        L=200,
        N=20001,
        T=200,
        dt=0.1,
        D=1.0,
        rho=1.0,
        K=1.0,
        n0=1.0,
        Mmax=1.0,
        init_type="tanh",
        steepness=0.85,
        perc=0.4,
        t_start=60,
        t_end=180,
        num_points=200,
    )

    tasks = [
        (lam, m0, alpha)
        for alpha in alpha_vals
        for lam in lambda_vals
        for m0 in m0_vals
    ]

    print(f"Running {len(tasks)} simulations...")
    print(f"Saving to: {base_dir}")

    results = Parallel(n_jobs=4, verbose=10)(
        delayed(run_one)(
            lam=lam,
            m0=m0,
            alpha=alpha,
            base_dir=base_dir,
            model_kwargs=model_kwargs,
            overwrite=False,
        )
        for lam, m0, alpha in tasks
    )

    done = [r for r in results if r[0] == "done"]
    skipped = [r for r in results if r[0] == "skipped"]
    failed = [r for r in results if r[0] == "failed"]

    save_json(
        base_dir / "grid_run_report.json",
        {
            "done": len(done),
            "skipped": len(skipped),
            "failed": len(failed),
            "failed_runs": failed,
            "lambda_vals": lambda_vals,
            "m0_vals": m0_vals,
            "alpha_vals": alpha_vals,
            "model_kwargs": model_kwargs,
        },
    )

    print("\nGrid run complete.")
    print(f"Done:    {len(done)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Failed:  {len(failed)}")

    if failed:
        print("\nFailed examples:")
        for item in failed[:10]:
            print(item)


if __name__ == "__main__":
    main()