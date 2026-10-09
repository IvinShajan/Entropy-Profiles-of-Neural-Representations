# Entropy Profiles of Neural Representations

A study of the **local order (rigidity)** of neural-network activation
distributions. Each hidden layer's activations are treated as a finite point
configuration, and its *entropy profile* — the Shannon entropy of sliding-window
counting statistics as a function of window length `L` — is compared against
three reference ensembles from random-matrix theory:

| Reference | Gaps | Behaviour | Rigidity `R` |
|-----------|------|-----------|--------------|
| **Poisson** | `~ Exp(1)` | generic / random | `~ 0` |
| **GUE** | Wigner surmise | intermediate (chaotic spectra) | `~ 0.6` |
| **Lattice** | constant | perfectly ordered / rigid | `~ 1` |

The deviation of an empirical profile from the Poisson and lattice references
defines a **Rigidity Index** `R ∈ [0, 1]` (`0` = generic, `~0.6` = GUE-like,
`1` = lattice-like). The project then asks how `R` relates to generalization,
calibration, adversarial robustness, and training dynamics — including an
honest null result and a causal intervention.

## Scientific idea

For a point set `x`, unfold it to unit mean spacing, then for a window of length
`L` slide a box across the configuration and record the number of points inside.
The distribution of those counts has entropy `H(L)`. The three references have
distinct fingerprints:

- **Poisson:** `H(L) ≈ ½ ln(2πeL)`, slope `dH/dlnL ≈ ½`, number variance
  `Σ²(L) ≈ L`.
- **GUE:** number variance `Σ²(L) ≈ (2/π²) ln L` (spectral rigidity),
  `R ≈ 0.6`.
- **Lattice:** `H(L) → 0`, slope `≈ 0`, `Σ²(L) ≈ 0`.

So a *flat* entropy profile with low slope indicates rigid, ordered structure,
while a steep profile indicates Poisson-like generic randomness.

The multi-dimensional extension replaces the 1D sliding window with a
**k-NN ball count**: for each `k`, the median k-th-nearest-neighbour distance
fixes a radius `r_k`, and the entropy of the ball-count distribution is compared
against an i.i.d. Gaussian Poisson reference using the *identical* radius rule.

## Repository layout

```
.
├── entropy_profiles.py     # Core math: references, profiles, rigidity indices
├── train_networks.py       # Training sweep (MNIST/Fashion/CIFAR/SVHN)
├── analyze.py              # Per-layer representation extraction + statistics
├── metrics.py              # Accuracy, ECE, FGSM, PGD metrics
├── correlations.py         # Rigidity vs behaviour correlation analysis
├── run_controls.py         # Untrained-network + synthetic controls (1D)
├── run_controls_multid.py  # Multi-D kNN controls
├── run_dynamics.py         # Rigidity evolution over training epochs
├── run_causal.py           # Causal rigidity-vs-spread penalty experiment
├── validate_math.py        # Sanity checks of the core math
└── figures.py              # All paper figures (PNG + SVG)
```

> **Note on paths.** Every script defines
> `BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`, so it
> expects to live in a `code/` subdirectory and writes outputs to sibling
> directories:
>
> ```
> project_root/
> ├── code/        <- the .py files here
> ├── data/        <- downloaded datasets (auto-created)
> ├── results/     <- nets/, probes/, analysis/, profiles/, causal/, *.json
> └── figures/     <- generated figures
> ```
>
> The files currently sit at the repository root. Either move them into a
> `code/` folder, or adjust `BASE` (and the `sys.path.insert(..., "code")`
> calls) to match your layout before running.

## Installation

Requires Python 3.9+ (the code uses `from __future__ import annotations` and
modern type hints).

```bash
pip install numpy scipy pandas torch torchvision matplotlib
```

A CPU is sufficient. Datasets are downloaded automatically on first run into
`data/`.

## Usage

Run the stages in order. The pipeline is resumable: trained models and analysis
JSONs are skipped if they already exist.

### 1. Verify the math (fast, no training)

```bash
python3 validate_math.py
```

Checks that synthetic Poisson / GUE / lattice configurations reproduce their
known fingerprints (KS tests, `H(L)`, number variance, `R`, `Δ₃`, unfolding).

### 2. Train the network population

```bash
python3 train_networks.py                  # full sweep (parallel workers)
python3 train_networks.py --only 3         # single config by index
python3 train_networks.py --range 0-10     # a slice of configs
python3 train_networks.py --parallel 4     # workers (default 4)
```

Trains MLPs, small CNNs, ResNet-9 and WideResNet-style models across
MNIST, Fashion-MNIST, grayscale CIFAR-10, CIFAR-10/100 and SVHN, sweeping
dropout, weight decay, learning rate, width, initialization scale and seeds.
Saves `model_*.pt`, `meta_*.json`, `log_*.json` under `results/nets/`.

### 3. Extract representations and compute statistics

```bash
python3 analyze.py                 # every trained network
python3 analyze.py --label mnist-mlp2-base
python3 analyze.py --force         # recompute existing analyses
```

For each semantic layer, captures activations on a balanced probe set and
computes:

- `R_pc1` — rigidity of the PC1 projection (1D spacing view)
- `R_gram` — rigidity of the Gram-matrix eigenvalue spectrum
- `R_vec` — rigidity of single-input coordinate vectors
- `R_knn` / `R_knn_generic` — multi-dimensional whitened k-NN indices
- `pairwise_H` — pairwise-distance entropy
- gap statistics, KS tests, `Δ₃`, slope, and baselines (`eff_rank`, CKA chain)

Writes `results/analysis/<label>.json` and `results/profiles/<label>.npz`.

### 4. Correlation analysis

```bash
python3 correlations.py
```

Builds `results/networks_table.csv` and `results/correlations.json`, computing
Spearman/Pearson correlations of each rigidity measure against generalization
gap, ECE, FGSM and PGD robustness, with bootstrap CIs, partial correlations
(controlling for accuracy, dataset, effective rank, CKA) and permutation nulls.

### 5. Controls

```bash
python3 run_controls.py           # untrained nets + Gaussian/mixture controls
python3 run_controls_multid.py    # multi-D kNN controls (i.i.d., clustered, lattice)
```

Untrained (random-init) networks should score `R ~ 0` (generic), confirming
that the measure responds to training and to genuine order rather than
architecture or dimensionality.

### 6. Training dynamics and causality

```bash
python3 run_dynamics.py           # rigidity vs epoch (saves every-2 checkpoints)
python3 run_causal.py             # penalty arms: control/penultimate/all-hidden/spread
python3 run_causal.py --arch cnn2 --lambdas 0,0.01,0.1,0.5,1.0
```

`run_causal.py` trains small MNIST models in an overfitting regime under a
differentiable penalty that pushes hidden-layer Gram spectra toward regular
(lattice/GUE-like) spacing. A matched **spread** control penalty (concentrating
the spectrum without ordering the gaps) isolates "spacing order" from "spectral
scale" as the causal ingredient.

### 7. Figures

```bash
python3 figures.py                # all figures
python3 figures.py fig1 fig9      # selected figures
```

Writes PNG and SVG to `figures/`. Figures include reference fingerprints,
untrained-vs-trained boxes, representative entropy profiles, rigidity-vs-behaviour
scatters, training dynamics, predictive-power comparisons, the multi-dimensional
k-NN analysis, and the causal dose-response.

## Key results (as documented in the code)

- Individual-layer rigidity does **not** cleanly separate trained from untrained
  networks (`fig2` — an explicit null result).
- The multi-dimensional k-NN rigidity `R_knn` is **strongly accuracy-aligned**,
  but its correlation with the generalization gap **does not survive
  partialling out accuracy** (`fig9`).
- The causal experiment tests whether *imposing* spacing regularity reduces
  overfitting, versus merely accompanying it (`fig10`).

## Outputs

| Path | Contents |
|------|----------|
| `results/nets/` | model weights, metadata, training logs, checkpoints |
| `results/probes/` | cached balanced probe sets |
| `results/analysis/*.json` | per-network, per-layer statistics + metrics |
| `results/profiles/*.npz` | empirical and reference entropy curves |
| `results/controls.json` | 1D control results |
| `results/controls_multid.json` | multi-D control results |
| `results/causal/*.json` | causal penalty sweep results |
| `results/dynamics_*.json` | per-epoch dynamics traces |
| `results/correlations.json`, `networks_table.csv` | correlation analysis |
| `figures/` | PNG + SVG figures |

## References

The reference ensembles follow standard random-matrix theory:

- Mehta, *Random Matrices* — Wigner surmise and GUE spacing statistics.
- Dyson & Mehta (1963) — `Δ₃` spectral rigidity.
- Wigner (1958) — semicircle law used for unfolding GUE eigenvalues.
