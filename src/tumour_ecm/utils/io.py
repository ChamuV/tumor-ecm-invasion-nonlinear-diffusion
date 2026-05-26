# src/tumour_ecm/utils/io.py

import re
import json
from pathlib import Path

import numpy as np


# ----- Helpers for saving runs -----

def safe_tag(x):
    return f"{float(x):.6g}".replace(".", "p").replace("-", "m")


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)


def save_json(path, obj):
    path = Path(path)
    ensure_dir(path.parent)

    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_snapshots(run_dir, model, stride=150):
    run_dir = Path(run_dir)
    ensure_dir(run_dir)

    idx = np.unique(
        np.concatenate(
            [
                np.arange(0, model.Nt, stride),
                np.array([model.Nt - 1]),
            ]
        )
    )

    np.savez_compressed(
        run_dir / "snapshots.npz",
        x=model.x,
        times=model.times[idx],
        N_arr=model.N_arr[idx],
        M_arr=model.M_arr[idx],
    )


def save_fronts(
    run_dir,
    model,
    target="N",
    threshold=0.5,
    band=(0.1, 0.9),
    spline_type="cubic",
):
    run_dir = Path(run_dir)
    ensure_dir(run_dir)

    t_fronts, x_fronts = model.track_wavefront_local_interpolation(
        threshold=threshold,
        band=band,
        spline_type=spline_type,
        target=target,
    )

    np.savez_compressed(
        run_dir / f"fronts_{target}.npz",
        t_fronts=t_fronts,
        x_fronts=x_fronts,
    )


# ----- Helpers for loading/finding saved runs -----

_NUM_PATTERN = re.compile(r"([0-9eE\.\+\-p]+)$")


def token_variants(value):
    value = float(value)

    decimal = f"{value:.12f}".rstrip("0").rstrip(".")
    plain = f"{value:g}"
    sci = f"{value:.0e}"

    variants = {decimal, plain, sci}

    if "e" in sci:
        mant, exp = sci.split("e")
        exp_int = int(exp)
        variants.add(f"{mant}e{exp_int}")
        variants.add(f"{mant}e{abs(exp_int):02d}")
        variants.add(f"{mant}e{'+' if exp_int >= 0 else '-'}{abs(exp_int):02d}")

    variants |= {v.replace("e", "E") for v in list(variants)}
    variants |= {v.replace(".", "p") for v in list(variants)}

    return variants


def exactish_dir(parent, prefix, value):
    parent = Path(parent)

    if not parent.exists():
        return None

    for token in token_variants(value):
        path = parent / f"{prefix}_{token}"
        if path.is_dir():
            return path

    return None


def closest_dir_by_number(parent, prefix, target):
    parent = Path(parent)

    if not parent.exists():
        return None

    candidates = [
        p for p in parent.iterdir()
        if p.is_dir() and p.name.startswith(prefix + "_")
    ]

    if not candidates:
        return None

    pairs = []

    for path in candidates:
        match = _NUM_PATTERN.search(path.name)
        if match is None:
            continue

        token = match.group(1).replace("p", ".")

        try:
            value = float(token)
            pairs.append((path, value))
        except Exception:
            continue

    if not pairs:
        return None

    return min(pairs, key=lambda pair: abs(pair[1] - float(target)))[0]


def find_run_dir(base_dir, lam, m0, alpha=None):
    """
    Supports both layouts:

    No alpha:
        base_dir/lambda_*/m0_*

    With alpha:
        base_dir/alpha_*/lambda_*/m0_*

    Also tolerates:
        base_dir/lambda_*/alpha_*/m0_*
    """

    base = Path(base_dir)

    if alpha is None:
        lam_dir = exactish_dir(base, "lambda", lam) or closest_dir_by_number(base, "lambda", lam)
        if lam_dir is None:
            return None

        return exactish_dir(lam_dir, "m0", m0) or closest_dir_by_number(lam_dir, "m0", m0)

    # Layout 1: alpha/lambda/m0
    alpha_dir = exactish_dir(base, "alpha", alpha) or closest_dir_by_number(base, "alpha", alpha)
    if alpha_dir is not None:
        lam_dir = exactish_dir(alpha_dir, "lambda", lam) or closest_dir_by_number(alpha_dir, "lambda", lam)
        if lam_dir is not None:
            m_dir = exactish_dir(lam_dir, "m0", m0) or closest_dir_by_number(lam_dir, "m0", m0)
            if m_dir is not None:
                return m_dir

    # Layout 2: lambda/alpha/m0
    lam_dir = exactish_dir(base, "lambda", lam) or closest_dir_by_number(base, "lambda", lam)
    if lam_dir is not None:
        alpha_dir = exactish_dir(lam_dir, "alpha", alpha) or closest_dir_by_number(lam_dir, "alpha", alpha)
        if alpha_dir is not None:
            return exactish_dir(alpha_dir, "m0", m0) or closest_dir_by_number(alpha_dir, "m0", m0)

    return None


def load_summary(run_dir):
    if run_dir is None:
        return {}

    path = Path(run_dir) / "summary.json"

    if not path.exists():
        return {}

    with open(path, "r") as f:
        return json.load(f)


def load_snapshots(run_dir):
    if run_dir is None:
        return None

    path = Path(run_dir) / "snapshots.npz"

    if not path.exists():
        return None

    data = np.load(path)

    x = data["x"]
    times = data["times"]
    N_arr = data["N_arr"]
    M_arr = data["M_arr"]

    if N_arr.shape[0] == x.size and N_arr.shape[-1] == times.size:
        N_arr = N_arr.T

    if M_arr.shape[0] == x.size and M_arr.shape[-1] == times.size:
        M_arr = M_arr.T

    return {
        "x": x,
        "times": times,
        "N_arr": N_arr,
        "M_arr": M_arr,
    }


def load_speed_from_run(base_dir, lam, m0, alpha=None, which="N"):
    run_dir = find_run_dir(base_dir, lam=lam, m0=m0, alpha=alpha)
    summary = load_summary(run_dir)

    if not summary:
        return np.nan

    key = "wave_speed" if which.upper() == "N" else "m_wave_speed"
    value = summary.get(key, np.nan)

    try:
        return float(value)
    except Exception:
        return np.nan