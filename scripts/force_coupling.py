"""Quantify coupling between MMNet modes and force using mutual information."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io
from scipy.signal import butter, filtfilt, resample_poly
from sklearn.feature_selection import mutual_info_regression


def lowpass(x: np.ndarray, fs_hz: int, cutoff_hz: float) -> np.ndarray:
    b, a = butter(4, cutoff_hz / (0.5 * fs_hz), btype="low")
    return filtfilt(b, a, x, axis=0)


def align_for_lag(modes: np.ndarray, force: np.ndarray, lag_samples: int):
    """Align signals; a positive lag means modes lead force."""
    if lag_samples > 0:
        return modes[:-lag_samples], force[lag_samples:]
    if lag_samples < 0:
        lag_samples = -lag_samples
        return modes[lag_samples:], force[:-lag_samples]
    return modes, force


def summed_mutual_information(
    modes: np.ndarray,
    force: np.ndarray,
    n_neighbors: int = 7,
    seed: int = 42,
) -> float:
    """Sum feature-wise k-nearest-neighbor MI estimates across modes."""
    values = mutual_info_regression(
        modes,
        force,
        n_neighbors=n_neighbors,
        random_state=seed,
    )
    return float(np.sum(values))


def force_mode_coupling(
    modes: np.ndarray,
    force: np.ndarray,
    fs_hz: int = 2000,
    fs_downsampled_hz: int = 50,
    cutoff_hz: float = 2.5,
    lags_ms=range(-150, 151, 25),
    n_neighbors: int = 7,
    n_surrogates: int = 1000,
    seed: int = 42,
) -> dict:
    length = min(len(modes), len(force))
    modes = np.asarray(modes[:length], dtype=float)
    force = np.asarray(force[:length], dtype=float).reshape(-1)
    if modes.ndim == 1:
        modes = modes[:, None]

    modes = lowpass(modes - modes.mean(axis=0), fs_hz, cutoff_hz)
    force = lowpass(force - force.mean(), fs_hz, cutoff_hz)
    modes = resample_poly(modes, fs_downsampled_hz, fs_hz, axis=0)
    force = resample_poly(force, fs_downsampled_hz, fs_hz)
    modes = (modes - modes.mean(axis=0)) / (modes.std(axis=0) + 1e-12)
    force = (force - force.mean()) / (force.std() + 1e-12)

    lags_ms = np.asarray(list(lags_ms), dtype=int)
    lag_samples = np.rint(lags_ms * fs_downsampled_hz / 1000).astype(int)
    mi_by_lag = []
    for lag in lag_samples:
        x, y = align_for_lag(modes, force, int(lag))
        mi_by_lag.append(summed_mutual_information(x, y, n_neighbors, seed))
    mi_by_lag = np.asarray(mi_by_lag)
    best_index = int(np.argmax(mi_by_lag))
    best_lag = int(lag_samples[best_index])
    observed = float(mi_by_lag[best_index])

    rng = np.random.default_rng(seed)
    surrogate = np.empty(n_surrogates, dtype=float)
    for index in range(n_surrogates):
        shifted = np.roll(modes, int(rng.integers(0, len(modes))), axis=0)
        x, y = align_for_lag(shifted, force, best_lag)
        surrogate[index] = summed_mutual_information(x, y, n_neighbors, seed)

    surrogate_mean = float(np.mean(surrogate))
    surrogate_sd = float(np.std(surrogate, ddof=1))
    z_score = (observed - surrogate_mean) / (surrogate_sd + 1e-12)
    p_value = (np.count_nonzero(surrogate >= observed) + 1) / (n_surrogates + 1)
    return {
        "observed_mi": observed,
        "optimal_lag_ms": int(lags_ms[best_index]),
        "surrogate_mean": surrogate_mean,
        "surrogate_sd": surrogate_sd,
        "mi_z": float(z_score),
        "p_value": float(p_value),
        "lags_ms": lags_ms.tolist(),
        "mi_by_lag": mi_by_lag.tolist(),
        "settings": {
            "fs_hz": fs_hz,
            "fs_downsampled_hz": fs_downsampled_hz,
            "cutoff_hz": cutoff_hz,
            "n_neighbors": n_neighbors,
            "n_surrogates": n_surrogates,
            "seed": seed,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", type=Path, required=True, help="NumPy file with shape [time, modes]")
    parser.add_argument("--force_mat", type=Path, required=True)
    parser.add_argument("--force_key", default="force_filt")
    parser.add_argument("--trim_s", type=float, default=0.4)
    parser.add_argument("--output", type=Path, default=Path("force_coupling.json"))
    args = parser.parse_args()

    modes = np.load(args.modes)
    force_data = scipy.io.loadmat(args.force_mat)
    force = np.asarray(force_data[args.force_key]).squeeze()
    trim = int(round(args.trim_s * 2000))
    if trim:
        force = force[trim:-trim]
    result = force_mode_coupling(modes, force)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"lags_ms", "mi_by_lag"}}, indent=2))


if __name__ == "__main__":
    main()
