"""Compute motor-unit contributions to MMNet modes with Integrated Gradients."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mmnet import (  # noqa: E402
    SequenceLatentVAE,
    load_mat_concatenated_data,
    muscle_groups_from_counts,
    trim_edges,
    window_starts,
    zscore_per_mu,
)


def integrated_gradients_for_segment(
    model: SequenceLatentVAE,
    segment: np.ndarray,
    mode_index: int,
    steps: int = 16,
) -> np.ndarray:
    """Return absolute, time-summed IG attribution for one segment and mode."""
    x = tf.convert_to_tensor(segment[None], dtype=tf.float32)
    baseline = tf.zeros_like(x)
    gradients = []
    for alpha in tf.linspace(0.0, 1.0, steps):
        interpolated = baseline + alpha * (x - baseline)
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            hidden = model.encoder(interpolated, training=False)
            mean = model.to_mean(hidden)
            target = tf.reduce_mean(mean[:, :, mode_index])
        gradients.append(tape.gradient(target, interpolated))
    average_gradient = tf.reduce_mean(tf.stack(gradients), axis=0)
    attribution = (x - baseline) * average_gradient
    return np.sum(np.abs(attribution.numpy()[0]), axis=0)


def attribute_modes(
    model: SequenceLatentVAE,
    signal: np.ndarray,
    window: int,
    step: int,
    steps: int = 16,
) -> np.ndarray:
    """Return median attribution across windows with shape [motor units, modes]."""
    starts = window_starts(len(signal), window, step)
    per_window = []
    for start in starts:
        segment = signal[start : start + window]
        per_window.append(
            np.stack(
                [
                    integrated_gradients_for_segment(model, segment, mode, steps)
                    for mode in range(model.latent_dim)
                ],
                axis=1,
            )
        )
    return np.median(np.stack(per_window), axis=0)


def muscle_contributions(attribution: np.ndarray, counts: list[int]) -> pd.DataFrame:
    groups = muscle_groups_from_counts(counts)
    values = np.stack(
        [np.mean(attribution[list(indices)], axis=0) for indices in groups.values()],
        axis=0,
    )
    values /= np.sum(values, axis=0, keepdims=True) + 1e-12
    return pd.DataFrame(
        values,
        index=list(groups),
        columns=[f"mode_{index + 1}" for index in range(attribution.shape[1])],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat_path", type=Path, required=True)
    parser.add_argument("--mat_key", default="concatenated_data")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--muscle_counts", type=int, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--out_dir", type=Path, default=Path("attribution"))
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    data = zscore_per_mu(load_mat_concatenated_data(args.mat_path, args.mat_key))
    signal = trim_edges(data.T, cfg["fs_hz"], cfg["trim_s"])
    if sum(args.muscle_counts) != signal.shape[1]:
        raise ValueError("muscle_counts must sum to the number of input motor units.")

    model = SequenceLatentVAE(
        n_mus=signal.shape[1],
        window=cfg["window"],
        latent_dim=cfg["latent_dim"],
        dropout=cfg["dropout"],
    )
    model(tf.zeros((1, cfg["window"], signal.shape[1]), tf.float32))
    model.load_weights(args.weights)
    attribution = attribute_modes(model, signal, cfg["window"], cfg["step"], args.steps)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    columns = [f"mode_{index + 1}" for index in range(model.latent_dim)]
    pd.DataFrame(attribution, columns=columns).to_csv(
        args.out_dir / "motor_unit_attribution.csv", index_label="motor_unit"
    )
    muscle_contributions(attribution, args.muscle_counts).to_csv(
        args.out_dir / "muscle_contributions.csv", index_label="muscle"
    )
    print(f"Saved attribution results to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
