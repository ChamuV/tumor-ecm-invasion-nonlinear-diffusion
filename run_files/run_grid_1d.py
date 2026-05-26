# run_files/run_grid_1d.py

import os
import sys
import math
import json
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


def is_missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def append_jsonl(path, item):
    """
    Append one JSON object as one line.

    This is safer for long sweeps because results are written immediately,
    rather than only at the end of the whole Parallel call.
    """
    path = Path(path)
    ensure_dir(path.parent)

    with open(path, "a") as f:
        f.write(json.dumps(item) + "\n")


def run_one(
    lam,
    m0,
    alpha,
    base_dir,
    model_kwargs,
    overwrite=False,
    r2_cutoff=0.999,
):
    base_dir = Path(base_dir)

    live_completed_path = base_dir / "live_completed_runs.jsonl"
    live_untracked_path = base_dir / "live_untracked_runs.jsonl"
    live_failed_path = base_dir / "live_failed_runs.jsonl"

    try:
        alpha_dir = base_dir / f"alpha_{safe_tag(alpha)}"
        lambda_dir = alpha_dir / f"lambda_{safe_tag(lam)}"
        run_dir = lambda_dir / f"m0_{safe_tag(m0)}"

        summary_path = run_dir / "summary.json"

        if summary_path.exists() and not overwrite:
            skipped_item = {
                "status": "skipped",
                "lambda": float(lam),
                "m0": float(m0),
                "alpha": float(alpha),
                "run_dir": str(run_dir),
                "reason": "summary_exists",
            }
            return ("skipped", skipped_item)

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
        cM, bM, r2M = None, None, None

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
                cM, bM, r2M = None, None, None

        no_n_speed = is_missing(cN)
        low_r2_n = (
            not no_n_speed
            and (is_missing(r2N) or float(r2N) < r2_cutoff)
        )

        no_m_speed = m_threshold > 0 and is_missing(cM)
        low_r2_m = (
            m_threshold > 0
            and not no_m_speed
            and (is_missing(r2M) or float(r2M) < r2_cutoff)
        )

        untracked_reasons = []

        if no_n_speed:
            untracked_reasons.append("NO_N_SPEED")

        if low_r2_n:
            untracked_reasons.append("LOW_N_R2")

        if no_m_speed:
            untracked_reasons.append("NO_M_SPEED")

        if low_r2_m:
            untracked_reasons.append("LOW_M_R2")

        summary = {
            "lambda": float(lam),
            "m0": float(m0),
            "alpha": float(alpha),

            "wave_speed": float(cN) if not no_n_speed else None,
            "r2": float(r2N) if not is_missing(r2N) else None,

            "m_threshold": float(m_threshold),
            "m_wave_speed": float(cM) if not is_missing(cM) else None,
            "m_r2": float(r2M) if not is_missing(r2M) else None,

            "untracked": bool(untracked_reasons),
            "untracked_reasons": untracked_reasons,
            "r2_cutoff": float(r2_cutoff),

            "L": model.L,
            "N": model.N,
            "T": model.T,
            "dt": model.dt,
            "init_type": model.init_type,
            "steepness": model.steepness,
            "perc": model.perc,
            "saved_stride": 150,
            "run_dir": str(run_dir),
        }

        save_json(summary_path, summary)
        save_snapshots(run_dir, model, stride=150)

        if not no_n_speed:
            save_fronts(run_dir, model, target="N", threshold=0.5)

        if m_threshold > 0 and not no_m_speed:
            save_fronts(run_dir, model, target="M", threshold=m_threshold)

        completed_item = {
            "status": "done",
            "lambda": float(lam),
            "m0": float(m0),
            "alpha": float(alpha),
            "wave_speed": float(cN) if not is_missing(cN) else None,
            "r2": float(r2N) if not is_missing(r2N) else None,
            "m_wave_speed": float(cM) if not is_missing(cM) else None,
            "m_r2": float(r2M) if not is_missing(r2M) else None,
            "untracked": bool(untracked_reasons),
            "untracked_reasons": untracked_reasons,
            "run_dir": str(run_dir),
        }

        append_jsonl(live_completed_path, completed_item)

        if untracked_reasons:
            append_jsonl(live_untracked_path, completed_item)

        return ("done", completed_item)

    except Exception as e:
        failed_item = {
            "status": "failed",
            "lambda": float(lam),
            "m0": float(m0),
            "alpha": float(alpha),
            "error": str(e),
        }

        append_jsonl(live_failed_path, failed_item)

        return ("failed", failed_item)


def read_jsonl(path):
    path = Path(path)

    if not path.exists():
        return []

    items = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return items


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    base_dir = ROOT / "outputs" / "sweeps" / "tumour_ecm_1d"
    ensure_dir(base_dir)

    lambda_vals = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0, 1000.0]
    m0_vals = [0.05, 0.1, 0.2, 0.5, 0.8, 0.9]
    alpha_vals = [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0, 100.0, 1000.0]

    r2_cutoff = 0.999

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
    print(f"Live untracked log: {base_dir / 'live_untracked_runs.jsonl'}")
    print(f"Live failed log: {base_dir / 'live_failed_runs.jsonl'}")

    try:
        results = Parallel(n_jobs=4, verbose=10)(
            delayed(run_one)(
                lam=lam,
                m0=m0,
                alpha=alpha,
                base_dir=base_dir,
                model_kwargs=model_kwargs,
                overwrite=False,
                r2_cutoff=r2_cutoff,
            )
            for lam, m0, alpha in tasks
        )

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        print("Completed/untracked/failed runs already finished are still in the live .jsonl logs.")
        raise

    done = []
    skipped = []
    failed = []
    untracked = []

    for tag, item in results:
        if tag == "done":
            done.append(item)

            if item.get("untracked", False):
                untracked.append(item)

        elif tag == "skipped":
            skipped.append(item)

        elif tag == "failed":
            failed.append(item)

    report = {
        "counts": {
            "done": len(done),
            "skipped": len(skipped),
            "failed": len(failed),
            "untracked": len(untracked),
        },
        "r2_cutoff": r2_cutoff,
        "lambda_vals": lambda_vals,
        "m0_vals": m0_vals,
        "alpha_vals": alpha_vals,
        "model_kwargs": model_kwargs,
        "done_runs": done,
        "skipped_runs": skipped,
        "failed_runs": failed,
        "untracked_runs": untracked,
    }

    save_json(base_dir / "grid_run_report.json", report)
    save_json(base_dir / "failed_runs.json", failed)
    save_json(base_dir / "untracked_runs.json", untracked)

    print("\nGrid run complete.")
    print(f"Done:       {len(done)}")
    print(f"Skipped:    {len(skipped)}")
    print(f"Failed:     {len(failed)}")
    print(f"Untracked:  {len(untracked)}")

    print("\nLive logs written to:")
    print(base_dir / "live_completed_runs.jsonl")
    print(base_dir / "live_untracked_runs.jsonl")
    print(base_dir / "live_failed_runs.jsonl")

    if untracked:
        print("\nUntracked examples:")
        for item in untracked[:10]:
            print(item)

    if failed:
        print("\nFailed examples:")
        for item in failed[:10]:
            print(item)


if __name__ == "__main__":
    main()