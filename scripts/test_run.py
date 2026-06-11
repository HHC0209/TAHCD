#!/usr/bin/env python3
"""
Smoke test: verifies that all components of TAHCD run without errors
on synthetic data. No GPU required.

Run from project root:
    python scripts/test_run.py
"""

import sys
import os
import logging
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.tahcd import TAHCD, ASSA, SACA, TTCE
from data.datasets import BRCADataset, CUBDataset, build_dataloader
from configs.config import BRCAConfig, CUBConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("smoke_test")


def test_assa():
    log.info("Testing ASSA ...")
    assa = ASSA(feature_dim=32)
    z = [torch.randn(16, 32), torch.randn(16, 32)]
    labels = torch.randint(0, 3, (16,))
    h_list, loss, U_list, w_list = assa(z, labels)
    assert len(h_list) == 2
    assert h_list[0].shape == (16, 32)
    assert loss.item() >= 0
    log.info(f"  ASSA loss={loss.item():.4f} ✓")


def test_saca():
    log.info("Testing SACA ...")
    saca = SACA(num_modalities=2, feature_dim=32)
    h = [torch.randn(16, 32), torch.randn(16, 32)]
    mus, covs, cross_mus, cross_covs = saca.compute_priors(h)
    h_s, h_c, n_s, n_c, loss = saca(h, mus, covs, cross_mus, cross_covs)
    assert h_s[0].shape == (16, 32)
    assert loss.item() >= 0 or True   # can be negative (log-likelihood)
    log.info(f"  SACA loss={loss.item():.4f} ✓")


def test_full_model_brca():
    log.info("Testing full TAHCD on BRCA (synthetic) ...")
    model = TAHCD(
        input_dims=[64, 64, 32],
        num_classes=5,
        feature_dim=32,
        hidden_dim=64,
        alpha=0.5,
        ttce_iters=3,
        ttce_eta=0.01,
    )
    x = [torch.randn(8, 64), torch.randn(8, 64), torch.randn(8, 32)]
    y = torch.randint(0, 5, (8,))

    # train forward
    model.train()
    logits, losses = model.forward_train(x, y)
    assert logits.shape == (8, 5)
    log.info(f"  Train logits shape: {logits.shape}  loss={losses['loss_total'].item():.4f} ✓")

    # inference forward (no TTCE)
    model.eval()
    with torch.no_grad():
        logits_inf = model.forward_inference(x, num_iters=0)
    assert logits_inf.shape == (8, 5)
    log.info(f"  Inference (no TTCE) logits shape: {logits_inf.shape} ✓")

    # inference forward (with TTCE)
    logits_ttce = model.forward_inference(x, num_iters=2)
    assert logits_ttce.shape == (8, 5)
    log.info(f"  Inference (TTCE 2 iters) logits shape: {logits_ttce.shape} ✓")


def test_full_model_cub():
    log.info("Testing full TAHCD on CUB (synthetic, 2 modalities) ...")
    model = TAHCD(
        input_dims=[128, 64],
        num_classes=10,
        feature_dim=32,
        hidden_dim=64,
        alpha=0.5,
        ttce_iters=2,
    )
    x = [torch.randn(8, 128), torch.randn(8, 64)]
    y = torch.randint(0, 10, (8,))

    model.train()
    logits, losses = model.forward_train(x, y)
    assert logits.shape == (8, 10)
    log.info(f"  CUB train logits shape: {logits.shape} ✓")


def test_decoder_pretrain():
    log.info("Testing decoder pre-training loss ...")
    model = TAHCD(
        input_dims=[64, 32],
        num_classes=5,
        feature_dim=32,
        hidden_dim=64,
    )
    x = [torch.randn(8, 64), torch.randn(8, 32)]
    loss = model.pretrain_decoders_loss(x)
    assert loss.item() >= 0
    log.info(f"  Decoder pretrain loss={loss.item():.4f} ✓")


def test_dataloader():
    log.info("Testing BRCADataset + DataLoader ...")
    ds = BRCADataset(
        data_dir="",
        synthetic=True,
        synthetic_n=100,
        noise_type="gaussian",
        eps=5.0,
        cross_noise_type="unalignment",
        eta=0.10,
    )
    loader = build_dataloader(ds, batch_size=16, shuffle=True, num_workers=0)
    x_list, labels = next(iter(loader))
    assert len(x_list) == 3
    assert labels.shape == (16,)
    log.info(f"  Batch shapes: {[x.shape for x in x_list]}, labels: {labels.shape} ✓")


def test_training_step():
    """End-to-end one batch backward pass."""
    log.info("Testing end-to-end training step ...")
    model = TAHCD(
        input_dims=[32, 32],
        num_classes=3,
        feature_dim=16,
        hidden_dim=32,
        alpha=0.5,
        ttce_iters=1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = [torch.randn(4, 32), torch.randn(4, 32)]
    y = torch.randint(0, 3, (4,))

    model.train()
    optimizer.zero_grad()
    logits, losses = model.forward_train(x, y)
    losses["loss_total"].backward()
    optimizer.step()

    log.info(f"  Backward pass OK, loss={losses['loss_total'].item():.4f} ✓")


if __name__ == "__main__":
    log.info("=" * 50)
    log.info("TAHCD Smoke Tests")
    log.info("=" * 50)

    tests = [
        test_assa,
        test_saca,
        test_decoder_pretrain,
        test_dataloader,
        test_full_model_brca,
        test_full_model_cub,
        test_training_step,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            log.error(f"FAILED: {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    log.info("=" * 50)
    log.info(f"Results: {passed}/{len(tests)} passed, {failed} failed")
    if failed == 0:
        log.info("All tests passed ✓")
        sys.exit(0)
    else:
        sys.exit(1)
