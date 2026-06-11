#!/usr/bin/env python3
"""
Evaluation script for TAHCD.

Supports:
  - Standard evaluation with TTCE
  - Ablation studies (disable ASSA / SACA / TTCE)
  - TTCE iteration analysis
  - Noise robustness sweep

Usage:
  python evaluate.py --dataset brca --synthetic --checkpoint checkpoints/brca/best_model.pt
  python evaluate.py --dataset cub  --synthetic --ablation
  python evaluate.py --dataset brca --synthetic --iter_analysis
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import get_config
from data.datasets import build_dataset, build_dataloader
from models.tahcd import TAHCD
from utils.trainer import TAHCDTrainer, compute_metrics


logger = logging.getLogger("evaluate")


# ─────────────────────────────────────────────────────────────────────────────
# Ablation wrapper
# ─────────────────────────────────────────────────────────────────────────────

class AblationTAHCD(TAHCD):
    """
    Wraps TAHCD and selectively disables components for ablation studies.
    Table 5 in the paper.
    """

    def __init__(self, *args, use_assa=True, use_saca=True, use_ttce=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_assa = use_assa
        self.use_saca = use_saca
        self.use_ttce = use_ttce

    @torch.no_grad()
    def forward_inference(self, x_list, num_iters=None, eta=None):
        iters = (num_iters or self.ttce_iters) if self.use_ttce else 0
        step = eta or self.ttce_eta

        # Encode
        z_list = [enc(x) for enc, x in zip(self.encoders, x_list)]

        if self.use_assa:
            h_list, _, _, _ = self.assa(z_list, labels=None)
        else:
            # skip ASSA → use raw latent features
            h_list = z_list

        if self.use_saca:
            mus, covs, cross_mus, cross_covs = self.saca.compute_priors(h_list)
            h_hat_s_list, h_hat_c_list, n_s_list, n_c_list, _ = self.saca(
                h_list, mus, covs, cross_mus, cross_covs
            )
        else:
            # skip SACA → pass h as both experts
            h_hat_s_list = h_list
            h_hat_c_list = h_list
            n_s_list = [torch.zeros_like(h) for h in h_list]
            n_c_list = [torch.zeros_like(h) for h in h_list]
            mus, covs, cross_mus, cross_covs = self.saca.compute_priors(h_list)

        if iters > 0 and self.use_ttce:
            with torch.enable_grad():
                h_up, hs_up, hc_up, mus_up, covs_up, _, _ = self.ttce.enhance(
                    h_list, n_s_list, n_c_list,
                    h_hat_s_list, h_hat_c_list,
                    x_list, mus, covs, cross_mus, cross_covs,
                    eta=step, num_iters=iters,
                )
        else:
            hs_up = h_hat_s_list
            hc_up = h_hat_c_list
            mus_up = mus
            covs_up = covs

        confs = self.saca.compute_confidence(
            [h.detach() for h in hs_up],
            [m.detach() if isinstance(m, torch.Tensor) else m for m in mus_up],
            [c.detach() if isinstance(c, torch.Tensor) else c for c in covs_up],
        )
        conf_stack = torch.stack(confs, dim=1)
        conf_norm = conf_stack / (conf_stack.sum(dim=1, keepdim=True) + 1e-8)
        conf_s = [conf_norm[:, m] for m in range(self.M)]

        hs_out = [h.detach() for h in hs_up]
        hc_out = [h.detach() for h in hc_up]
        return self.fusion_cls(hs_out, hc_out, conf_s, conf_s)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(model, loader, device, ttce_iters=0, binary=False):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for x_list, labels in loader:
        x_list = [x.to(device) for x in x_list]
        logits = model.forward_inference(x_list, num_iters=ttce_iters)
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    return compute_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_probs), binary=binary
    )


def build_model_from_cfg(cfg, use_assa=True, use_saca=True, use_ttce=True):
    return AblationTAHCD(
        input_dims=cfg.input_dims,
        num_classes=cfg.num_classes,
        feature_dim=cfg.feature_dim,
        hidden_dim=cfg.hidden_dim,
        alpha=cfg.alpha,
        ttce_iters=cfg.ttce_iters,
        ttce_eta=cfg.ttce_eta,
        use_assa=use_assa,
        use_saca=use_saca,
        use_ttce=use_ttce,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Ablation study (Table 5)
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(args, cfg, test_loader, device):
    """Runs all four ablation variants described in Table 5."""
    ablation_configs = [
        ("None (baseline)",     False, False, False),
        ("ASSA only",           True,  False, False),
        ("ASSA + SACA",         True,  True,  False),
        ("ASSA + SACA + TTCE",  True,  True,  True),
    ]

    print("\n" + "=" * 60)
    print("ABLATION STUDY (Table 5 reproduction)")
    print("=" * 60)
    print(f"{'Variant':<25} {'ACC':>6} {'Wt-F1':>7} {'Macro-F1':>9}")
    print("-" * 60)

    for name, use_assa, use_saca, use_ttce in ablation_configs:
        model = build_model_from_cfg(
            cfg, use_assa=use_assa, use_saca=use_saca, use_ttce=use_ttce
        ).to(device)

        # If checkpoint available, load it; else evaluate untrained (for structure check)
        if args.checkpoint and os.path.exists(args.checkpoint):
            ckpt = torch.load(args.checkpoint, map_location=device)
            try:
                model.load_state_dict(ckpt["model_state_dict"], strict=False)
            except Exception:
                pass

        iters = cfg.ttce_iters if use_ttce else 0
        m = evaluate_model(model, test_loader, device, ttce_iters=iters,
                            binary=cfg.binary)
        print(f"{name:<25} {m['acc']:>6.1f} {m['weighted_f1']:>7.1f} {m['macro_f1']:>9.1f}")

    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# TTCE iteration analysis (Figure 8)
# ─────────────────────────────────────────────────────────────────────────────

def run_iter_analysis(args, cfg, test_loader, device):
    """Replicates Figure 8: accuracy vs TTCE iterations."""
    model = build_model_from_cfg(cfg).to(device)
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    iter_values = [0, 10, 20, 30, 40, 50, 60]

    print("\n" + "=" * 45)
    print("TTCE Iteration Analysis (Figure 8)")
    print("=" * 45)
    print(f"{'Iterations':>12} {'ACC':>8}")
    print("-" * 22)

    for iters in iter_values:
        m = evaluate_model(model, test_loader, device,
                           ttce_iters=iters, binary=cfg.binary)
        print(f"{iters:>12}   {m['acc']:>7.1f}%")

    print("=" * 45)


# ─────────────────────────────────────────────────────────────────────────────
# Noise sweep (Figure 3)
# ─────────────────────────────────────────────────────────────────────────────

def run_noise_sweep(args, cfg, device):
    """Replicates Figure 3: accuracy vs noise intensity ε."""
    model = build_model_from_cfg(cfg).to(device)
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    eps_values = [0, 10, 20, 40, 60, 80, 100]

    print("\n" + "=" * 45)
    print("Noise Robustness Sweep (Figure 3)")
    print("=" * 45)
    print(f"{'ε':>5} {'ACC':>8}")
    print("-" * 15)

    for eps in eps_values:
        ds = build_dataset(
            cfg.dataset, cfg.data_dir, split="test",
            synthetic=cfg.synthetic,
            noise_type="gaussian", eps=float(eps),
            cross_noise_type="none", eta=0.0, seed=99,
        )
        loader = build_dataloader(ds, cfg.batch_size, shuffle=False,
                                  num_workers=cfg.num_workers)
        m = evaluate_model(model, loader, device,
                           ttce_iters=cfg.ttce_iters, binary=cfg.binary)
        print(f"{eps:>5}   {m['acc']:>7.1f}%")

    print("=" * 45)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate TAHCD")
    p.add_argument("--dataset", default="brca", choices=["brca", "rosmap", "cub", "food101"])
    p.add_argument("--data_dir", default="data/raw")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--checkpoint", default=None, help="Path to .pt checkpoint file")
    p.add_argument("--ttce_iters", type=int, default=30)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch_size", type=int, default=None)
    # Modes
    p.add_argument("--ablation", action="store_true", help="Run ablation study")
    p.add_argument("--iter_analysis", action="store_true", help="TTCE iteration analysis")
    p.add_argument("--noise_sweep", action="store_true", help="Noise robustness sweep")
    return p.parse_args()


def get_device(d):
    if d == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(d)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    device = get_device(args.device)

    cfg = get_config(args.dataset)
    cfg.synthetic = args.synthetic
    cfg.data_dir = args.data_dir
    cfg.ttce_iters = args.ttce_iters
    if args.batch_size:
        cfg.batch_size = args.batch_size

    # test dataset with Gaussian ε=5, η=10%
    test_ds = build_dataset(
        cfg.dataset, cfg.data_dir, split="test",
        synthetic=cfg.synthetic,
        noise_type="gaussian", eps=5.0,
        cross_noise_type="unalignment", eta=0.10,
        seed=99,
    )
    test_loader = build_dataloader(test_ds, cfg.batch_size, shuffle=False,
                                   num_workers=cfg.num_workers)

    if args.ablation:
        run_ablation(args, cfg, test_loader, device)
    elif args.iter_analysis:
        run_iter_analysis(args, cfg, test_loader, device)
    elif args.noise_sweep:
        run_noise_sweep(args, cfg, device)
    else:
        # Standard evaluation
        model = build_model_from_cfg(cfg).to(device)
        if args.checkpoint and os.path.exists(args.checkpoint):
            ckpt = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            logger.info(f"Loaded checkpoint from {args.checkpoint}")

        m = evaluate_model(model, test_loader, device,
                           ttce_iters=args.ttce_iters, binary=cfg.binary)
        print("\n=== Evaluation Results ===")
        for k, v in m.items():
            print(f"  {k:15s}: {v:.2f}%")


if __name__ == "__main__":
    main()
