#!/usr/bin/env python3
"""
Main training script for TAHCD.

Usage examples:
  # Synthetic data (quick test):
  python train.py --dataset brca --synthetic

  # Real data – robustness experiment (Table 1):
  python train.py --dataset cub --data_dir /path/to/cub --experiment robustness

  # Generalization experiment (Table 3):
  python train.py --dataset brca --data_dir /path/to/brca --experiment generalization

  # Custom noise override:
  python train.py --dataset rosmap --synthetic --train_eps 3 --test_eps 5 --ttce_iters 20
"""

import argparse
import logging
import os
import random
import sys

import numpy as np
import torch

# allow running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import get_config, robustness_config, generalization_config, diverse_noise_config
from data.datasets import build_dataset, build_dataloader
from models.tahcd import TAHCD
from utils.trainer import TAHCDTrainer


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_device(cfg_device: str) -> torch.device:
    if cfg_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(cfg_device)


def setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train TAHCD")
    p.add_argument("--dataset", default="brca", choices=["brca", "rosmap", "cub", "food101"])
    p.add_argument("--data_dir", default="data/raw")
    p.add_argument("--synthetic", action="store_true",
                   help="Use synthetic random data (no real dataset needed)")
    p.add_argument("--experiment", default="robustness",
                   choices=["robustness", "generalization", "diverse_noise", "custom"],
                   help="Which experimental protocol to follow")

    # Model
    p.add_argument("--feature_dim", type=int, default=None)
    p.add_argument("--hidden_dim", type=int, default=None)
    p.add_argument("--alpha", type=float, default=None, help="TTCE prior update rate")

    # Training
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--pretrain_epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)

    # Noise
    p.add_argument("--train_noise_type", default=None,
                   choices=["gaussian", "poisson", "salt_pepper", "none"])
    p.add_argument("--train_eps", type=float, default=None)
    p.add_argument("--train_cross_noise_type", default=None,
                   choices=["unalignment", "missing", "none"])
    p.add_argument("--train_eta", type=float, default=None)
    p.add_argument("--test_noise_type", default=None,
                   choices=["gaussian", "poisson", "salt_pepper", "none"])
    p.add_argument("--test_eps", type=float, default=None)
    p.add_argument("--test_cross_noise_type", default=None,
                   choices=["unalignment", "missing", "none"])
    p.add_argument("--test_eta", type=float, default=None)

    # TTCE
    p.add_argument("--ttce_iters", type=int, default=None)
    p.add_argument("--ttce_eta", type=float, default=None)
    p.add_argument("--val_ttce_iters", type=int, default=0)

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--save_dir", default=None)
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    setup_logging(args.log_level)
    log = logging.getLogger("train")

    # ── build config ──────────────────────────────────────────────────────
    overrides = {k: v for k, v in vars(args).items()
                 if v is not None and k not in ("dataset", "experiment", "synthetic")}

    cfg = get_config(args.dataset, **overrides)
    cfg.synthetic = args.synthetic
    cfg.data_dir = args.data_dir

    # apply experiment protocol
    if args.experiment == "robustness":
        cfg = robustness_config(cfg)
    elif args.experiment == "generalization":
        cfg = generalization_config(cfg)
    elif args.experiment == "diverse_noise":
        cfg = diverse_noise_config(cfg, args.dataset)
    # "custom" → use whatever was set via CLI overrides

    # final CLI overrides (they win over experiment presets)
    for attr in ["train_noise_type", "train_eps", "train_cross_noise_type", "train_eta",
                 "test_noise_type", "test_eps", "test_cross_noise_type", "test_eta",
                 "ttce_iters", "ttce_eta", "val_ttce_iters", "epochs", "batch_size", "lr",
                 "feature_dim", "hidden_dim", "alpha", "seed", "device"]:
        val = getattr(args, attr, None)
        if val is not None:
            setattr(cfg, attr, val)

    if args.save_dir:
        cfg.save_dir = args.save_dir

    log.info(f"Config: {cfg}")
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    log.info(f"Device: {device}")

    # ── datasets ──────────────────────────────────────────────────────────
    common_train = dict(
        noise_type=cfg.train_noise_type,
        eps=cfg.train_eps,
        cross_noise_type=cfg.train_cross_noise_type,
        eta=cfg.train_eta,
        seed=cfg.seed,
    )
    common_test = dict(
        noise_type=cfg.test_noise_type,
        eps=cfg.test_eps,
        cross_noise_type=cfg.test_cross_noise_type,
        eta=cfg.test_eta,
        seed=cfg.seed + 1,
    )

    train_ds = build_dataset(cfg.dataset, cfg.data_dir, split="train",
                             synthetic=cfg.synthetic, **common_train)
    val_ds   = build_dataset(cfg.dataset, cfg.data_dir, split="val",
                             synthetic=cfg.synthetic, **common_test)
    test_ds  = build_dataset(cfg.dataset, cfg.data_dir, split="test",
                             synthetic=cfg.synthetic, **common_test)

    log.info(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    train_loader = build_dataloader(train_ds, cfg.batch_size, shuffle=True,
                                    num_workers=cfg.num_workers)
    val_loader   = build_dataloader(val_ds,   cfg.batch_size, shuffle=False,
                                    num_workers=cfg.num_workers)
    test_loader  = build_dataloader(test_ds,  cfg.batch_size, shuffle=False,
                                    num_workers=cfg.num_workers)

    # ── model ────────────────────────────────────────────────────────────
    model = TAHCD(
        input_dims=cfg.input_dims,
        num_classes=cfg.num_classes,
        feature_dim=cfg.feature_dim,
        hidden_dim=cfg.hidden_dim,
        alpha=cfg.alpha,
        ttce_iters=cfg.ttce_iters,
        ttce_eta=cfg.ttce_eta,
    )
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"TAHCD: {n_params:.2f}M parameters")

    # ── trainer ──────────────────────────────────────────────────────────
    trainer = TAHCDTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        cfg=cfg.to_dict(),
        device=device,
        binary=cfg.binary,
        save_dir=cfg.save_dir,
    )

    # 1. Pre-train decoders
    trainer.pretrain()

    # 2. Main training
    history = trainer.train()

    # 3. Final test evaluation with TTCE
    test_metrics = trainer.test(ttce_iters=cfg.ttce_iters)
    log.info("=== Final Test Metrics ===")
    for k, v in test_metrics.items():
        log.info(f"  {k}: {v:.2f}%")


if __name__ == "__main__":
    main()
