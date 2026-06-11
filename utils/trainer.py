"""
Training and evaluation utilities for TAHCD.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import numpy as np


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    binary: bool = False,
) -> Dict[str, float]:
    metrics = {}
    metrics["acc"] = accuracy_score(y_true, y_pred) * 100.0
    metrics["weighted_f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100.0
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0

    if binary and y_prob is not None:
        try:
            if y_prob.ndim == 2:
                auc_prob = y_prob[:, 1]
            else:
                auc_prob = y_prob
            metrics["auc"] = roc_auc_score(y_true, auc_prob) * 100.0
        except Exception:
            metrics["auc"] = 0.0

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Decoder Pre-training
# ─────────────────────────────────────────────────────────────────────────────

def pretrain_decoders(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 20,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> None:
    """
    Pre-trains encoder+decoder pairs with reconstruction loss.
    Decoders are then fixed during main TAHCD training.
    """
    logger.info("Pre-training decoders ...")
    # Only optimize encoder & decoder parameters
    enc_dec_params = list(model.encoders.parameters()) + list(model.decoders.parameters())
    optimizer = optim.Adam(enc_dec_params, lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        n_batches = 0
        for x_list, _ in train_loader:
            x_list = [x.to(device) for x in x_list]
            optimizer.zero_grad()
            loss = model.pretrain_decoders_loss(x_list)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc_dec_params, max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()
        avg = total_loss / max(n_batches, 1)
        logger.info(f"  [Pretrain Decoder] Epoch {epoch+1}/{num_epochs}  loss={avg:.4f}")

    # Freeze decoders
    for p in model.decoders.parameters():
        p.requires_grad_(False)
    logger.info("Decoders frozen.")


# ─────────────────────────────────────────────────────────────────────────────
# Main Trainer
# ─────────────────────────────────────────────────────────────────────────────

class TAHCDTrainer:
    """
    Handles full training lifecycle for TAHCD:
      1. Pre-train decoders (reconstruction).
      2. Train full TAHCD (Ltot = Lassa + Lsaca + Lre + Lcls).
      3. Evaluate with TTCE at test time.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        cfg: Dict,
        device: torch.device,
        binary: bool = False,
        save_dir: str = "checkpoints",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.cfg = cfg
        self.device = device
        self.binary = binary
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Optimizer (exclude frozen decoder params)
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(
            trainable,
            lr=cfg.get("lr", 1e-3),
            weight_decay=cfg.get("weight_decay", 1e-4),
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.get("epochs", 100),
        )

        self.best_val_acc = 0.0
        self.best_epoch = 0

    # ── pre-training ──────────────────────────────────────────────────────

    def pretrain(self):
        pretrain_decoders(
            self.model,
            self.train_loader,
            num_epochs=self.cfg.get("pretrain_epochs", 20),
            lr=self.cfg.get("pretrain_lr", 1e-3),
            device=self.device,
        )
        # re-build optimizer after freezing decoders
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(
            trainable,
            lr=self.cfg.get("lr", 1e-3),
            weight_decay=self.cfg.get("weight_decay", 1e-4),
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.cfg.get("epochs", 100),
        )

    # ── one epoch of training ─────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_losses = {k: 0.0 for k in
                        ["loss_total", "loss_assa", "loss_saca", "loss_re", "loss_cls"]}
        n_batches = 0
        all_preds, all_labels = [], []

        for x_list, labels in self.train_loader:
            x_list = [x.to(self.device) for x in x_list]
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits, losses = self.model.forward_train(x_list, labels)
            loss = losses["loss_total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=self.cfg.get("grad_clip", 5.0),
            )
            self.optimizer.step()

            for k in total_losses:
                total_losses[k] += losses[k].item()
            n_batches += 1

            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        self.scheduler.step()

        avg_losses = {k: v / max(n_batches, 1) for k, v in total_losses.items()}
        acc = accuracy_score(all_labels, all_preds) * 100.0
        avg_losses["train_acc"] = acc
        return avg_losses

    # ── evaluation ────────────────────────────────────────────────────────

    @torch.no_grad()
    def _evaluate(
        self, loader: DataLoader, ttce_iters: int = 0
    ) -> Dict[str, float]:
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []

        for x_list, labels in loader:
            x_list = [x.to(self.device) for x in x_list]
            logits = self.model.forward_inference(x_list, num_iters=ttce_iters)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        return compute_metrics(y_true, y_pred, y_prob, binary=self.binary)

    # ── full training loop ────────────────────────────────────────────────

    def train(self) -> Dict:
        epochs = self.cfg.get("epochs", 100)
        val_ttce = self.cfg.get("val_ttce_iters", 0)   # fast val: no TTCE
        logger.info(f"Starting TAHCD training for {epochs} epochs ...")

        history = {"train": [], "val": []}

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self._train_epoch(epoch)
            val_metrics = self._evaluate(self.val_loader, ttce_iters=val_ttce)
            elapsed = time.time() - t0

            history["train"].append(train_metrics)
            history["val"].append(val_metrics)

            logger.info(
                f"Epoch {epoch:3d}/{epochs}  "
                f"loss={train_metrics['loss_total']:.4f}  "
                f"train_acc={train_metrics['train_acc']:.1f}%  "
                f"val_acc={val_metrics['acc']:.1f}%  "
                f"val_wF1={val_metrics['weighted_f1']:.1f}%  "
                f"({elapsed:.1f}s)"
            )

            # save best
            if val_metrics["acc"] > self.best_val_acc:
                self.best_val_acc = val_metrics["acc"]
                self.best_epoch = epoch
                ckpt_path = os.path.join(self.save_dir, "best_model.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "val_metrics": val_metrics,
                    },
                    ckpt_path,
                )
                logger.info(f"  ✓ New best: {self.best_val_acc:.1f}% → saved to {ckpt_path}")

        logger.info(f"Training complete. Best val acc={self.best_val_acc:.1f}% at epoch {self.best_epoch}.")
        return history

    # ── test with TTCE ────────────────────────────────────────────────────

    def test(self, ttce_iters: Optional[int] = None) -> Dict[str, float]:
        """Load best checkpoint and evaluate on test set with TTCE."""
        ckpt_path = os.path.join(self.save_dir, "best_model.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            logger.info(f"Loaded best model from epoch {ckpt['epoch']}.")

        iters = ttce_iters if ttce_iters is not None else self.cfg.get("ttce_iters", 30)
        logger.info(f"Evaluating on test set with TTCE iters={iters} ...")
        test_metrics = self._evaluate(self.test_loader, ttce_iters=iters)

        logger.info("Test Results:")
        for k, v in test_metrics.items():
            logger.info(f"  {k:15s}: {v:.2f}%")

        return test_metrics
