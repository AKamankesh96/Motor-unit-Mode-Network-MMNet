"""Individualized selection of the number of motor-unit modes.

The implementation follows the manuscript: three elbow heuristics vote on an
explained-variance curve, and the Knee Locator result breaks a three-way tie.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from kneed import KneeLocator


def knee_locator_vote(dimensions: np.ndarray, explained_variance: np.ndarray) -> int:
    knee = KneeLocator(
        dimensions,
        explained_variance,
        curve="concave",
        direction="increasing",
    ).knee
    if knee is None:
        raise ValueError("Knee Locator did not identify an elbow point.")
    return int(knee)


def first_derivative_vote(dimensions: np.ndarray, explained_variance: np.ndarray) -> int:
    """Return the dimension reached by the largest successive EV gain."""
    gains = np.diff(explained_variance)
    if gains.size == 0:
        raise ValueError("At least two dimensions are required.")
    return int(dimensions[np.argmax(gains) + 1])


def maximum_distance_vote(dimensions: np.ndarray, explained_variance: np.ndarray) -> int:
    """Return the point farthest from the line joining the curve endpoints."""
    x = dimensions.astype(float)
    y = explained_variance.astype(float)
    x1, y1, x2, y2 = x[0], y[0], x[-1], y[-1]
    denominator = np.hypot(y2 - y1, x2 - x1)
    if denominator == 0:
        raise ValueError("The explained-variance curve endpoints are identical.")
    distances = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / denominator
    return int(dimensions[np.argmax(distances)])


def select_mode_number(dimensions: np.ndarray, explained_variance: np.ndarray) -> dict:
    dimensions = np.asarray(dimensions, dtype=int)
    explained_variance = np.asarray(explained_variance, dtype=float)
    if dimensions.ndim != 1 or explained_variance.ndim != 1:
        raise ValueError("dimensions and explained_variance must be one-dimensional.")
    if len(dimensions) != len(explained_variance) or len(dimensions) < 2:
        raise ValueError("Inputs must have the same length and contain at least two values.")
    if np.any(np.diff(dimensions) <= 0):
        raise ValueError("dimensions must be strictly increasing.")

    knee_vote = knee_locator_vote(dimensions, explained_variance)
    votes = {
        "knee_locator": knee_vote,
        "first_derivative": first_derivative_vote(dimensions, explained_variance),
        "maximum_distance": maximum_distance_vote(dimensions, explained_variance),
    }
    counts = Counter(votes.values())
    dimension, count = counts.most_common(1)[0]
    selected = dimension if count >= 2 else knee_vote
    return {"selected_dimension": int(selected), "votes": votes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV containing dimension and explained-variance columns")
    parser.add_argument("--dimension_column", default="latent_dim")
    parser.add_argument("--ev_column", default="explained_variance")
    parser.add_argument("--output", type=Path, default=Path("mode_selection.json"))
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    result = select_mode_number(
        frame[args.dimension_column].to_numpy(),
        frame[args.ev_column].to_numpy(),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
