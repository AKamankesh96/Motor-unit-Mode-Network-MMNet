"""Five-fold blocked temporal cross-validation for MMNet-VAE.

Ten contiguous blocks are rotated through 70% training, 10% validation, and
20% testing partitions. Whole-record normalization reproduces the supplied
analysis notebook; training-only normalization is available as a sensitivity
analysis. In both cases,
250 ms is removed from both sides of partition boundaries to prevent leakage
from smoothing and overlapping windows. The primary result is pooled out-of-
fold explained variance, not the average of fold-specific values.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mmnet import (  # noqa: E402
    Config,
    SequenceLatentVAE,
    fit,
    load_mat_concatenated_data,
    make_overlapping_windows,
    make_tf_datasets,
    muscle_groups_from_counts,
    set_global_determinism,
    trim_edges,
    variance_explained_percent,
    window_starts,
)


def fold_assignment(fold: int, n_blocks: int = 10) -> dict[str, list[int]]:
    """Assign every block to testing exactly once across five folds."""
    half = n_blocks // 2
    test = [fold, fold + half]
    validation = [(fold + 1) % half]
    training = [index for index in range(n_blocks) if index not in test + validation]
    return {"train": training, "validation": validation, "test": test}


def partition_runs(blocks: list[np.ndarray], assignment: dict[str, list[int]], gap: int):
    """Return contiguous runs trimmed only where adjacent partitions differ."""
    labels = {}
    for partition, indices in assignment.items():
        labels.update({index: partition for index in indices})
    output: dict[str, list[np.ndarray]] = {key: [] for key in assignment}
    for partition, indices in assignment.items():
        indices = sorted(indices)
        groups = [[indices[0]]]
        for index in indices[1:]:
            if index == groups[-1][-1] + 1:
                groups[-1].append(index)
            else:
                groups.append([index])
        for group in groups:
            run = np.concatenate([blocks[index] for index in group])
            left = group[0] > 0 and labels[group[0] - 1] != partition
            right = group[-1] < len(blocks) - 1 and labels[group[-1] + 1] != partition
            start = gap if left else 0
            stop = len(run) - gap if right else len(run)
            if stop > start:
                output[partition].append(run[start:stop])
    return output


def windows_from_runs(signal: np.ndarray, runs: list[np.ndarray], window: int, step: int):
    windows = []
    for run in runs:
        segment = signal[run]
        if len(segment) >= window:
            windows.append(make_overlapping_windows(segment, window, step))
    if not windows:
        raise ValueError("No complete windows remain after partition trimming.")
    return np.concatenate(windows)


def reconstruct_runs(model, signal, runs, window, step):
    prediction = np.zeros_like(signal, dtype=np.float32)
    weights = np.zeros((len(signal), 1), dtype=np.float32)
    for run in runs:
        for offset in window_starts(len(run), window, step):
            indices = run[offset : offset + window]
            segment = signal[indices][None].astype(np.float32)
            _, _, reconstruction = model.deterministic(segment)
            prediction[indices] += reconstruction.numpy()[0]
            weights[indices] += 1
    covered = weights[:, 0] > 0
    prediction[covered] /= weights[covered]
    return prediction, covered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat_path", type=Path, required=True)
    parser.add_argument("--mat_key", default="concatenated_data")
    parser.add_argument("--latent_dim", type=int, required=True)
    parser.add_argument("--muscle_counts", type=int, nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--step", type=int, default=250)
    parser.add_argument("--train_step", type=int, default=100)
    parser.add_argument("--gap_ms", type=float, default=250.0)
    parser.add_argument(
        "--normalization",
        choices=("global", "training"),
        default="global",
        help="Use 'global' to reproduce the supplied analysis notebook.",
    )
    parser.add_argument("--start_s", type=float, default=None)
    parser.add_argument("--end_s", type=float, default=None)
    parser.add_argument("--out_dir", type=Path, default=Path("temporal_cv"))
    args = parser.parse_args()

    cfg = Config(
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        window=args.window,
        step=args.step,
    )
    set_global_determinism(cfg.seed)
    raw = load_mat_concatenated_data(args.mat_path, args.mat_key).T
    raw = trim_edges(raw, cfg.fs_hz, cfg.trim_s)
    global_mean = raw.mean(axis=0, keepdims=True)
    global_std = raw.std(axis=0, keepdims=True) + cfg.eps
    global_normalized = (raw - global_mean) / global_std
    if args.start_s is not None or args.end_s is not None:
        start = 0 if args.start_s is None else int(round(args.start_s * cfg.fs_hz))
        stop = len(raw) if args.end_s is None else int(round(args.end_s * cfg.fs_hz))
        raw = raw[start:stop]
        global_normalized = global_normalized[start:stop]
    if sum(args.muscle_counts) != raw.shape[1]:
        raise ValueError("muscle_counts must sum to the number of input motor units.")

    blocks = list(np.array_split(np.arange(len(raw)), 10))
    gap = int(round(args.gap_ms * cfg.fs_hz / 1000))
    groups = muscle_groups_from_counts(args.muscle_counts)
    oof_prediction = np.full_like(raw, np.nan, dtype=float)
    fold_results = []
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for fold in range(5):
        assignment = fold_assignment(fold)
        runs = partition_runs(blocks, assignment, gap)
        train_indices = np.concatenate(runs["train"])
        if args.normalization == "global":
            mean, std = global_mean, global_std
            normalized = global_normalized
        else:
            mean = raw[train_indices].mean(axis=0, keepdims=True)
            std = raw[train_indices].std(axis=0, keepdims=True) + cfg.eps
            normalized = (raw - mean) / std

        train_x = windows_from_runs(normalized, runs["train"], cfg.window, args.train_step)
        validation_x = windows_from_runs(normalized, runs["validation"], cfg.window, args.train_step)
        train_ds, validation_ds = make_tf_datasets(train_x, validation_x, cfg.batch_size)
        tf.keras.backend.clear_session()
        set_global_determinism(cfg.seed + fold)
        model = SequenceLatentVAE(raw.shape[1], cfg.window, cfg.latent_dim, cfg.dropout)
        model(tf.zeros((1, cfg.window, raw.shape[1]), tf.float32))
        checkpoint = args.out_dir / f"fold_{fold + 1}.weights.h5"
        fit(
            model,
            train_ds,
            validation_ds,
            cfg,
            groups,
            checkpoint,
            early_stopping_patience=100,
            min_epochs=50,
        )
        model.load_weights(checkpoint)

        prediction_z, covered = reconstruct_runs(
            model, normalized, runs["test"], cfg.window, cfg.step
        )
        prediction = prediction_z * std + mean
        oof_prediction[covered] = prediction[covered]
        fold_ev = variance_explained_percent(raw[covered], prediction[covered])
        fold_results.append(
            {"fold": fold + 1, "explained_variance": fold_ev, "n_samples": int(covered.sum())}
        )
        print(f"Fold {fold + 1}: EV={fold_ev:.2f}% ({covered.sum()} covered samples)")

    covered = np.all(np.isfinite(oof_prediction), axis=1)
    pooled_ev = variance_explained_percent(raw[covered], oof_prediction[covered])
    result = {
        "pooled_out_of_fold_explained_variance": pooled_ev,
        "covered_samples": int(covered.sum()),
        "fold_diagnostics": fold_results,
        "settings": {
            "n_blocks": 10,
            "n_folds": 5,
            "train_validation_test_percent": [70, 10, 20],
            "gap_each_side_ms": args.gap_ms,
            "normalization": args.normalization,
            "training_window_step": args.train_step,
            "deterministic_inference": "posterior_mean",
        },
    }
    np.save(args.out_dir / "oof_prediction.npy", oof_prediction)
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Pooled out-of-fold EV: {pooled_ev:.2f}%")


if __name__ == "__main__":
    main()
