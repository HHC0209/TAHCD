"""
Dataset loaders for TAHCD experiments.

Supports:
  - BRCA      : multi-omics (mRNA, DNA, miRNA) – 5-class
  - ROSMAP    : multi-omics (mRNA, DNA, miRNA) – binary
  - CUB-200   : image + text  – 200-class
  - UPMC FOOD101 : image + text – 101-class

Noise injection follows the paper:
  - Modality-specific Gaussian noise: x̃ = x + σ·ε,  ε~N(0,I)
  - Cross-modality noise:
      * modality unalignment (randomly shuffled features for η% of samples)
      * modality missing     (zeros out one modality for η% of samples)
"""

import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Noise helpers
# ─────────────────────────────────────────────────────────────────────────────

def add_gaussian_noise(
    x: torch.Tensor, eps: float, seed: Optional[int] = None
) -> torch.Tensor:
    """x̃_i = x_i + σ·ε_i,  ε_i ~ N(0, I).  σ = eps."""
    if eps == 0:
        return x
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
    noise = torch.randn_like(x, generator=rng) * eps
    return x + noise


def add_poisson_noise(x: torch.Tensor, eps: float) -> torch.Tensor:
    if eps == 0:
        return x
    lam = (x.abs() * eps).clamp(min=1e-6)
    noise = torch.poisson(lam) - lam
    return x + noise


def add_salt_pepper_noise(x: torch.Tensor, eps: float) -> torch.Tensor:
    """eps fraction of pixels randomly set to ±max."""
    if eps == 0:
        return x
    mask = torch.rand_like(x)
    noisy = x.clone()
    noisy[mask < eps / 2] = x.min()
    noisy[(mask >= eps / 2) & (mask < eps)] = x.max()
    return noisy


def add_modality_unalignment(
    data_list: List[torch.Tensor], eta: float, rng: Optional[np.random.Generator] = None
) -> List[torch.Tensor]:
    """Randomly shuffle features of η% of samples for each modality independently."""
    if eta == 0:
        return data_list
    if rng is None:
        rng = np.random.default_rng()
    N = data_list[0].size(0)
    k = int(N * eta)
    out = []
    for x in data_list:
        x_noisy = x.clone()
        idx = rng.choice(N, size=k, replace=False)
        perm = rng.permutation(idx)
        x_noisy[idx] = x[perm]
        out.append(x_noisy)
    return out


def add_modality_missing(
    data_list: List[torch.Tensor], eta: float, rng: Optional[np.random.Generator] = None
) -> List[torch.Tensor]:
    """Zero out one modality for η% of samples."""
    if eta == 0:
        return data_list
    if rng is None:
        rng = np.random.default_rng()
    N = data_list[0].size(0)
    k = int(N * eta)
    out = [x.clone() for x in data_list]
    for m in range(len(data_list)):
        idx = rng.choice(N, size=k, replace=False)
        out[m][idx] = 0.0
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────────────

class MultimodalDataset(Dataset):
    """Base class. Subclasses provide (data_list, labels) in __init__."""

    def __init__(
        self,
        data_list: List[torch.Tensor],  # M tensors of shape (N, d_m)
        labels: torch.Tensor,           # (N,) long
        noise_type: str = "none",       # gaussian | poisson | salt_pepper
        eps: float = 0.0,
        cross_noise_type: str = "none", # none | unalignment | missing
        eta: float = 0.0,
        seed: int = 42,
    ):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.data_list = self._apply_noise(
            data_list, noise_type, eps, cross_noise_type, eta
        )
        self.labels = labels

    def _apply_noise(self, data_list, noise_type, eps, cross_noise_type, eta):
        noisy = []
        for x in data_list:
            if noise_type == "gaussian":
                x = add_gaussian_noise(x, eps)
            elif noise_type == "poisson":
                x = add_poisson_noise(x, eps)
            elif noise_type == "salt_pepper":
                x = add_salt_pepper_noise(x, eps)
            noisy.append(x)

        if cross_noise_type == "unalignment":
            noisy = add_modality_unalignment(noisy, eta, self.rng)
        elif cross_noise_type == "missing":
            noisy = add_modality_missing(noisy, eta, self.rng)

        return noisy

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        xs = [d[idx] for d in self.data_list]
        return xs, self.labels[idx]


# ─────────────────────────────────────────────────────────────────────────────
# BRCA Dataset
# ─────────────────────────────────────────────────────────────────────────────

class BRCADataset(MultimodalDataset):
    """
    TCGA-BRCA multi-omics dataset.
    Modalities: mRNA, DNA methylation, miRNA (all pre-extracted features).

    Expected directory layout:
        data_dir/
            mrna.npy      (N, d_mrna)
            dna.npy       (N, d_dna)
            mirna.npy     (N, d_mirna)
            labels.npy    (N,)   int in [0,4]

    If synthetic=True, generates random data for quick testing.
    """

    MODALITY_NAMES = ["mrna", "dna", "mirna"]
    NUM_CLASSES = 5

    def __init__(
        self,
        data_dir: str,
        split: str = "train",       # train | val | test
        synthetic: bool = False,
        synthetic_n: int = 500,
        **noise_kwargs,
    ):
        if synthetic:
            data_list, labels = self._make_synthetic(synthetic_n)
        else:
            data_list, labels = self._load(data_dir, split)
        super().__init__(data_list, labels, **noise_kwargs)

    @staticmethod
    def _make_synthetic(n: int):
        d = [1000, 1000, 503]   # approximate real dimensions
        data_list = [torch.randn(n, di) for di in d]
        labels = torch.randint(0, 5, (n,))
        return data_list, labels

    @staticmethod
    def _load(data_dir, split):
        split_file = os.path.join(data_dir, f"{split}_indices.npy")
        mrna  = torch.from_numpy(np.load(os.path.join(data_dir, "mrna.npy"))).float()
        dna   = torch.from_numpy(np.load(os.path.join(data_dir, "dna.npy"))).float()
        mirna = torch.from_numpy(np.load(os.path.join(data_dir, "mirna.npy"))).float()
        labs  = torch.from_numpy(np.load(os.path.join(data_dir, "labels.npy"))).long()

        if os.path.exists(split_file):
            idx = np.load(split_file)
            mrna, dna, mirna, labs = mrna[idx], dna[idx], mirna[idx], labs[idx]

        return [mrna, dna, mirna], labs


# ─────────────────────────────────────────────────────────────────────────────
# ROSMAP Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ROSMAPDataset(MultimodalDataset):
    """
    ROSMAP Alzheimer's multi-omics dataset.
    Modalities: mRNA, DNA, miRNA. Binary classification.

    Expected layout: same as BRCA but labels in {0, 1}.
    """

    MODALITY_NAMES = ["mrna", "dna", "mirna"]
    NUM_CLASSES = 2

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        synthetic: bool = False,
        synthetic_n: int = 200,
        **noise_kwargs,
    ):
        if synthetic:
            data_list, labels = self._make_synthetic(synthetic_n)
        else:
            data_list, labels = BRCADataset._load(data_dir, split)
            labels = labels.clamp(0, 1)
        super().__init__(data_list, labels, **noise_kwargs)

    @staticmethod
    def _make_synthetic(n: int):
        d = [200, 200, 200]
        data_list = [torch.randn(n, di) for di in d]
        labels = torch.randint(0, 2, (n,))
        return data_list, labels


# ─────────────────────────────────────────────────────────────────────────────
# CUB-200 Dataset
# ─────────────────────────────────────────────────────────────────────────────

class CUBDataset(MultimodalDataset):
    """
    Caltech-UCSD Birds 200 image+text dataset.
    Expects pre-extracted image/text features (e.g., from ResNet / BERT).

    Expected layout:
        data_dir/
            image_features.npy   (N, d_img)
            text_features.npy    (N, d_txt)
            labels.npy           (N,) int in [0,199]
    """

    MODALITY_NAMES = ["image", "text"]
    NUM_CLASSES = 200

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        synthetic: bool = False,
        synthetic_n: int = 1000,
        **noise_kwargs,
    ):
        if synthetic:
            data_list, labels = self._make_synthetic(synthetic_n)
        else:
            data_list, labels = self._load(data_dir, split)
        super().__init__(data_list, labels, **noise_kwargs)

    @staticmethod
    def _make_synthetic(n: int):
        data_list = [torch.randn(n, 2048), torch.randn(n, 768)]
        labels = torch.randint(0, 200, (n,))
        return data_list, labels

    @staticmethod
    def _load(data_dir, split):
        split_file = os.path.join(data_dir, f"{split}_indices.npy")
        img  = torch.from_numpy(np.load(os.path.join(data_dir, "image_features.npy"))).float()
        txt  = torch.from_numpy(np.load(os.path.join(data_dir, "text_features.npy"))).float()
        labs = torch.from_numpy(np.load(os.path.join(data_dir, "labels.npy"))).long()

        if os.path.exists(split_file):
            idx = np.load(split_file)
            img, txt, labs = img[idx], txt[idx], labs[idx]

        return [img, txt], labs


# ─────────────────────────────────────────────────────────────────────────────
# UPMC FOOD101 Dataset
# ─────────────────────────────────────────────────────────────────────────────

class FOOD101Dataset(MultimodalDataset):
    """
    UPMC FOOD101 image+text dataset (101 food categories).

    Expected layout: same as CUB but 101 classes.
    """

    MODALITY_NAMES = ["image", "text"]
    NUM_CLASSES = 101

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        synthetic: bool = False,
        synthetic_n: int = 1000,
        **noise_kwargs,
    ):
        if synthetic:
            data_list, labels = self._make_synthetic(synthetic_n)
        else:
            data_list, labels = CUBDataset._load(data_dir, split)
            labels = labels.clamp(0, 100)
        super().__init__(data_list, labels, **noise_kwargs)

    @staticmethod
    def _make_synthetic(n: int):
        data_list = [torch.randn(n, 2048), torch.randn(n, 768)]
        labels = torch.randint(0, 101, (n,))
        return data_list, labels


# ─────────────────────────────────────────────────────────────────────────────
# Collate function
# ─────────────────────────────────────────────────────────────────────────────

def multimodal_collate(batch):
    """Custom collate that handles variable numbers of modalities."""
    xs_list, labels = zip(*batch)
    M = len(xs_list[0])
    x_batched = [torch.stack([xs[m] for xs in xs_list], dim=0) for m in range(M)]
    labels_batched = torch.stack(labels, dim=0)
    return x_batched, labels_batched


# ─────────────────────────────────────────────────────────────────────────────
# Factory helpers
# ─────────────────────────────────────────────────────────────────────────────

DATASET_REGISTRY = {
    "brca": BRCADataset,
    "rosmap": ROSMAPDataset,
    "cub": CUBDataset,
    "food101": FOOD101Dataset,
}


def build_dataset(
    name: str,
    data_dir: str,
    split: str,
    synthetic: bool = True,
    **noise_kwargs,
) -> MultimodalDataset:
    cls = DATASET_REGISTRY[name.lower()]
    return cls(data_dir=data_dir, split=split, synthetic=synthetic, **noise_kwargs)


def build_dataloader(
    dataset: MultimodalDataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=multimodal_collate,
        pin_memory=True,
    )
