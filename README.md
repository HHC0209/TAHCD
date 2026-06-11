# TAHCD: Test-time Adaptive Hierarchical Co-enhanced Denoising Network

PyTorch reproduction of the paper:

> **"Test-time Adaptive Hierarchical Co-enhanced Denoising Network for Reliable Multimodal Classification"**  
> Shu Shen, C. L. Philip Chen, Tong Zhang  
> South China University of Technology / Pazhou Laboratory

---

## Overview

TAHCD addresses two key limitations of existing multimodal learning methods:

1. **Insufficient robustness to heterogeneous noise** — inability to jointly handle modality-specific and cross-modality noise.
2. **Limited generalization** — poor adaptability to previously unseen noise at test time.

### Three Core Components

| Component | Abbreviation | Level    | Role |
|-----------|-------------|----------|------|
| Adaptive Stable Subspace Alignment | ASSA | Global   | Remove modality-specific & cross-modality noise across all samples |
| Sample-Adaptive Confidence Alignment | SACA | Instance | Per-sample noise removal via confidence-aware slack alignment |
| Test-Time Cooperative Enhancement | TTCE | Inference | Label-free iterative adaptation to unseen noise |

---

## Project Structure

```
TAHCD/
├── models/
│   ├── __init__.py
│   └── tahcd.py          # Full model: ASSA, SACA, TTCE, FusionClassifier
├── data/
│   ├── __init__.py
│   └── datasets.py       # BRCA, ROSMAP, CUB, FOOD101 datasets + noise injection
├── configs/
│   ├── __init__.py
│   └── config.py         # Dataclass configs for all datasets & experiments
├── utils/
│   ├── __init__.py
│   └── trainer.py        # TAHCDTrainer, decoder pre-training, metrics
├── scripts/
│   └── test_run.py       # Smoke tests (no GPU needed)
├── train.py              # Main training entry point
├── evaluate.py           # Evaluation, ablation, TTCE iteration analysis
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Smoke test (synthetic data, CPU)

```bash
python scripts/test_run.py
```

Expected output: `All tests passed ✓`

### 3. Train on synthetic data

```bash
# BRCA (3-modality multi-omics, 5-class)
python train.py --dataset brca --synthetic --epochs 50

# CUB (2-modality image+text, 200-class)
python train.py --dataset cub --synthetic --epochs 50

# ROSMAP (binary, multi-omics)
python train.py --dataset rosmap --synthetic --epochs 50
```

### 4. Train on real data

Prepare your data directory with the following layout:

**Multi-omics (BRCA / ROSMAP):**
```
data_dir/
    mrna.npy       (N, d_mrna)
    dna.npy        (N, d_dna)
    mirna.npy      (N, d_mirna)
    labels.npy     (N,)
    train_indices.npy
    val_indices.npy
    test_indices.npy
```

**Image+Text (CUB / FOOD101):**
```
data_dir/
    image_features.npy   (N, d_img)   # e.g. ResNet-101 features
    text_features.npy    (N, d_txt)   # e.g. BERT features
    labels.npy           (N,)
    train_indices.npy
    val_indices.npy
    test_indices.npy
```

Then run:
```bash
python train.py --dataset cub --data_dir /path/to/cub --experiment robustness
```

---

## Experimental Protocols

### Table 1 – Robustness (noise on both train & test)
```bash
python train.py --dataset brca --synthetic --experiment robustness \
    --train_eps 5 --train_eta 0.10 --test_eps 5 --test_eta 0.10
```

### Table 3 – Generalization (clean train, noisy test)
```bash
python train.py --dataset brca --synthetic --experiment generalization \
    --test_eps 5 --test_eta 0.10 --ttce_iters 30
```

### Table 2 – Diverse noise types
```bash
# Poisson + missing (BRCA/ROSMAP):
python train.py --dataset brca --synthetic --experiment diverse_noise

# Salt-and-pepper + missing (CUB/FOOD101):
python train.py --dataset cub --synthetic --experiment diverse_noise
```

---

## Evaluation

```bash
# Standard evaluation
python evaluate.py --dataset brca --synthetic --ttce_iters 30

# With a trained checkpoint
python evaluate.py --dataset brca --checkpoint checkpoints/brca/best_model.pt

# Ablation study (Table 5)
python evaluate.py --dataset brca --synthetic --ablation

# TTCE iteration analysis (Figure 8)
python evaluate.py --dataset brca --synthetic --iter_analysis

# Noise robustness sweep (Figure 3)
python evaluate.py --dataset cub --synthetic --noise_sweep
```

---

## Key Design Choices

### ASSA (Section 3.2)
- Computes per-modality feature covariance and performs eigendecomposition.
- A learnable mask **w** (from singular values via `phi_lambda`) selects informative principal axes.
- **Inter-class orthogonality** (Eq. 6): encourages discriminative subspaces.
- **Subspace projection alignment** (Eq. 7): aligns modality projections in their respective stable subspaces to avoid erroneous cross-modal alignment.

### SACA (Section 3.3)
- Estimates Gaussian priors from ASSA's globally denoised features.
- **M modality-specific experts** + **M cross-modality experts** apply per-sample masks.
- **Confidence-aware asymmetric slack alignment** (Eq. 18-19): reweights gradients so low-confidence modalities align toward high-confidence ones.
- **Slack alignment** (Eq. 17): constrains cross-modality discrepancies within a data-driven range, preserving complementary information.

### TTCE (Section 3.4, Algorithm 1)
- At test time, iterates cooperative enhancement between ASSA and SACA:
  1. **SACA→ASSA**: uses instance-level noise (`n_s`, `n_c`) in reconstruction loss to refine global features `h↑` (Eq. 21).
  2. **ASSA→SACA**: updated `h↑` improves distribution priors; enhanced priors refine instance-level features `h_hat_s↑`, `h_hat_c↑` (Eqs. 22-25).
- Label-free: no ground-truth labels required at test time.
- Hyperparameter `α` controls prior update rate (best in [0.4, 0.6] per paper).

---

## Hyperparameter Reference

| Parameter | Paper Value | Description |
|-----------|------------|-------------|
| `feature_dim` | 256 | Latent dimension |
| `hidden_dim` | 512 | Hidden layer size |
| `alpha` | 0.5 | TTCE prior update rate |
| `ttce_iters` | 30 | TTCE enhancement iterations |
| `ttce_eta` | 0.01 | TTCE gradient step size |
| `eps` (Gaussian) | 5.0 | Noise severity |
| `eta` (alignment) | 10% | Proportion of corrupted samples |

---

## Citation

```bibtex
@article{shen2025tahcd,
  title={Test-time Adaptive Hierarchical Co-enhanced Denoising Network for Reliable Multimodal Classification},
  author={Shen, Shu and Chen, C. L. Philip and Zhang, Tong},
  journal={Pattern Recognition},
  year={2025}
}
```
