# Motor-unit Mode Network (MMNet)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18895983.svg)](https://doi.org/10.5281/zenodo.18895983)

MMNet is a variational autoencoder framework for extracting low-dimensional
motor-unit modes from smoothed motor-unit discharge rates. It combines a
convolutional encoder, recurrent temporal modeling, individualized mode
selection, and gradient-based attribution.

## Repository contents

- `scripts/mmnet.py`: MMNet-VAE training, deterministic reconstruction, and mode extraction
- `scripts/temporal_cv.py`: five-fold blocked temporal cross-validation
- `scripts/mode_selection.py`: individualized elbow selection by majority vote
- `scripts/integrated_gradients.py`: motor-unit and muscle attribution
- `scripts/force_coupling.py`: force–mode mutual-information analysis
- `notebooks/mmnet_demo.ipynb`: minimal training demonstration

The repository contains analysis code, not participant data. Do not commit raw
EMG, motor-unit, force, or participant-identifying files.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Input data

The training scripts expect a MATLAB file containing a two-dimensional matrix
named `concatenated_data` by default. Its shape must be `[motor units, samples]`,
with motor units from each muscle stored contiguously. The signals should be
continuous, smoothed discharge rates sampled at 2000 Hz.

## Train MMNet-VAE

The muscle counts enable the group-balanced reconstruction loss used in the
study. The best checkpoint is selected by validation reconstruction loss.
Reported reconstructions and modes use the posterior mean rather than a random
latent sample.

```bash
python scripts/mmnet.py \
  --mat_path All_MUs_S01s35bef.mat \
  --latent_dim 4 \
  --muscle_counts 10 16 11 \
  --out_dir outputs_S01 \
  --save_full_recon \
  --save_full_latent \
  --save_config
```

## Individualized mode selection

Train dimensions 1–10 and create a CSV with columns `latent_dim` and
`explained_variance`. Three elbow heuristics vote on the selected dimension; if
all disagree, Knee Locator is used as specified in the manuscript.

```bash
python scripts/mode_selection.py reconstruction_curve.csv \
  --output mode_selection.json
```

## Blocked temporal cross-validation

The cross-validation implementation uses ten contiguous blocks and five folds.
Each fold contains 70% training, 10% validation, and 20% test data; every block
serves as test data once. By default, whole-record standardization reproduces
the supplied analysis notebook. Use `--normalization training` for a stricter
training-only sensitivity analysis. In either case, 250 ms is removed from both
sides of partition boundaries, and all out-of-fold predictions are pooled
before explained variance is calculated.

```bash
python scripts/temporal_cv.py \
  --mat_path All_MUs_S01s35bef.mat \
  --latent_dim 4 \
  --muscle_counts 10 16 11 \
  --out_dir temporal_cv_S01
```

Use `--start_s` and `--end_s` when cross-validation is restricted to a
preselected stationary interval. These values are interpreted relative to the
signal after the standard 0.4-s edge trim.

## Integrated Gradients attribution

```bash
python scripts/integrated_gradients.py \
  --mat_path All_MUs_S01s35bef.mat \
  --weights outputs_S01/best_weights_latent4.weights.h5 \
  --config outputs_S01/config.json \
  --muscle_counts 10 16 11 \
  --out_dir attribution_S01
```

The script uses a zero baseline, 16 interpolation steps, absolute attribution,
and the median across overlapping windows. It writes motor-unit attribution and
normalized muscle-contribution tables.

## Force–mode coupling

```bash
python scripts/force_coupling.py \
  --modes outputs_S01/latent_full.npy \
  --force_mat force_S01.mat \
  --force_key force_filt \
  --output force_coupling_S01.json
```

The default analysis low-pass filters modes and force at 2.5 Hz, downsamples to
50 Hz, estimates mutual information with seven nearest neighbors across lags
from −150 to +150 ms, and evaluates the peak with 1000 circular-shift surrogates.

## Reproducibility notes

- The random seed is 42 by default.
- Full-signal fitting and blocked temporal cross-validation answer different
  questions and should be reported separately.
- Fold-specific explained-variance values are diagnostics. The temporal-CV
  result is the value calculated from pooled out-of-fold predictions.
- Generated data, weights, and analysis outputs are excluded by `.gitignore`.

## Citation

To cite MMNet v1.1.0, use:

> Kamankesh, A. (2026). *Motor-unit Mode Network (MMNet)* (v1.1.0) [Software].
> Zenodo. https://doi.org/10.5281/zenodo.18895983

```bibtex
@software{kamankesh_mmnet_2026,
  author  = {Kamankesh, Alireza},
  title   = {Motor-unit Mode Network (MMNet)},
  year    = {2026},
  version = {v1.1.0},
  doi     = {10.5281/zenodo.18895983},
  url     = {https://github.com/AKamankesh96/Motor-unit-Mode-Network-MMNet}
}
```

See `CITATION.cff` for machine-readable citation metadata. The DOI above
resolves to the latest archived version of MMNet.

## License

This project is released under the MIT License.
