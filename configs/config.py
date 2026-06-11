"""
Experiment configurations for TAHCD.

Each config dict matches the experimental setup described in Section 4 of the paper:
  - Gaussian noise ε=5 + modality unalignment η=10%  (Tables 1 & 3)
  - Poisson / salt-and-pepper + modality missing      (Table 2)
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Default Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TAHCDConfig:
    # ── Dataset ───────────────────────────────────────────────────────────
    dataset: str = "brca"          # brca | rosmap | cub | food101
    data_dir: str = "data/raw"
    synthetic: bool = True         # use synthetic data when real data absent

    # ── Model ─────────────────────────────────────────────────────────────
    feature_dim: int = 256
    hidden_dim: int = 512
    alpha: float = 0.5             # TTCE prior update rate

    # ── Training ──────────────────────────────────────────────────────────
    pretrain_epochs: int = 20
    pretrain_lr: float = 1e-3
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    grad_clip: float = 5.0
    num_workers: int = 0           # set >0 for multi-GPU

    # ── Noise (train) ─────────────────────────────────────────────────────
    train_noise_type: str = "gaussian"    # gaussian | poisson | salt_pepper | none
    train_eps: float = 5.0
    train_cross_noise_type: str = "unalignment"   # unalignment | missing | none
    train_eta: float = 0.10

    # ── Noise (test) ──────────────────────────────────────────────────────
    test_noise_type: str = "gaussian"
    test_eps: float = 5.0
    test_cross_noise_type: str = "unalignment"
    test_eta: float = 0.10

    # ── TTCE at inference ────────────────────────────────────────────────
    ttce_iters: int = 30
    ttce_eta: float = 0.01
    val_ttce_iters: int = 0        # 0 = skip TTCE during validation for speed

    # ── Misc ──────────────────────────────────────────────────────────────
    seed: int = 42
    save_dir: str = "checkpoints"
    log_level: str = "INFO"
    device: str = "auto"           # auto | cpu | cuda | mps

    def to_dict(self):
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-specific configs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BRCAConfig(TAHCDConfig):
    dataset: str = "brca"
    # input dims: mRNA ~1000, DNA ~1000, miRNA ~503 (approximate)
    input_dims: List[int] = field(default_factory=lambda: [1000, 1000, 503])
    num_classes: int = 5
    binary: bool = False
    save_dir: str = "checkpoints/brca"


@dataclass
class ROSMAPConfig(TAHCDConfig):
    dataset: str = "rosmap"
    input_dims: List[int] = field(default_factory=lambda: [200, 200, 200])
    num_classes: int = 2
    binary: bool = True
    save_dir: str = "checkpoints/rosmap"
    # Paper uses smaller dataset → smaller batch
    batch_size: int = 32
    synthetic_n: int = 351   # actual ROSMAP size


@dataclass
class CUBConfig(TAHCDConfig):
    dataset: str = "cub"
    # ResNet-101 image features + BERT text features
    input_dims: List[int] = field(default_factory=lambda: [2048, 768])
    num_classes: int = 200
    binary: bool = False
    save_dir: str = "checkpoints/cub"
    batch_size: int = 128
    epochs: int = 150


@dataclass
class FOOD101Config(TAHCDConfig):
    dataset: str = "food101"
    input_dims: List[int] = field(default_factory=lambda: [2048, 768])
    num_classes: int = 101
    binary: bool = False
    save_dir: str = "checkpoints/food101"
    batch_size: int = 256
    epochs: int = 150


# ─────────────────────────────────────────────────────────────────────────────
# Noise sweep configs (Table 1 vs Table 3 vs Table 2)
# ─────────────────────────────────────────────────────────────────────────────

def robustness_config(base_cfg: TAHCDConfig) -> TAHCDConfig:
    """Table 1: noise on both train and test."""
    base_cfg.train_noise_type = "gaussian"
    base_cfg.train_eps = 5.0
    base_cfg.train_cross_noise_type = "unalignment"
    base_cfg.train_eta = 0.10
    base_cfg.test_noise_type = "gaussian"
    base_cfg.test_eps = 5.0
    base_cfg.test_cross_noise_type = "unalignment"
    base_cfg.test_eta = 0.10
    return base_cfg


def generalization_config(base_cfg: TAHCDConfig) -> TAHCDConfig:
    """Table 3: train on clean, test on noise."""
    base_cfg.train_noise_type = "none"
    base_cfg.train_eps = 0.0
    base_cfg.train_cross_noise_type = "none"
    base_cfg.train_eta = 0.0
    base_cfg.test_noise_type = "gaussian"
    base_cfg.test_eps = 5.0
    base_cfg.test_cross_noise_type = "unalignment"
    base_cfg.test_eta = 0.10
    base_cfg.ttce_iters = 30
    return base_cfg


def diverse_noise_config(base_cfg: TAHCDConfig, dataset: str) -> TAHCDConfig:
    """Table 2: Poisson/salt-pepper + modality missing."""
    if dataset in ("brca", "rosmap"):
        base_cfg.train_noise_type = "poisson"
        base_cfg.test_noise_type = "poisson"
    else:
        base_cfg.train_noise_type = "salt_pepper"
        base_cfg.test_noise_type = "salt_pepper"
    base_cfg.train_eps = 5.0
    base_cfg.test_eps = 5.0
    base_cfg.train_cross_noise_type = "missing"
    base_cfg.test_cross_noise_type = "missing"
    base_cfg.train_eta = 0.10
    base_cfg.test_eta = 0.10
    return base_cfg


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

CONFIG_REGISTRY = {
    "brca": BRCAConfig,
    "rosmap": ROSMAPConfig,
    "cub": CUBConfig,
    "food101": FOOD101Config,
}


def get_config(dataset: str, **overrides) -> TAHCDConfig:
    cls = CONFIG_REGISTRY[dataset.lower()]
    cfg = cls()
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg
