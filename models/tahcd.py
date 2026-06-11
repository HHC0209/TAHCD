"""
TAHCD: Test-time Adaptive Hierarchical Co-enhanced Denoising Network
Paper: "Test-time Adaptive Hierarchical Co-enhanced Denoising Network for 
        Reliable Multimodal Classification"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict


# ─────────────────────────────────────────────────────────────────────────────
# Basic Building Blocks
# ─────────────────────────────────────────────────────────────────────────────

class ModalityEncoder(nn.Module):
    """Encodes raw modality features into a latent representation (dim=256)."""

    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MaskEncoder(nn.Module):
    """Learnable encoder phi that maps a vector to a mask (sigmoid output)."""

    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class ModalityDecoder(nn.Module):
    """Decodes a latent representation back to the original feature space."""

    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# ASSA: Adaptive Stable Subspace Alignment (Global-level)
# ─────────────────────────────────────────────────────────────────────────────

class ASSA(nn.Module):
    """
    Adaptive Stable Subspace Alignment.

    Steps:
      1. Compute covariance of latent features z^m.
      2. SVD → eigenvectors U, singular values λ.
      3. Learn mask w from λ to select informative axes.
      4. Project z through masked subspace → globally denoised h.
      5. Enforce inter-class orthogonality (Lo) and subspace projection
         alignment (La) constraints.
    """

    def __init__(self, feature_dim: int = 256, lambda_encoder_hidden: int = 512):
        super().__init__()
        self.feature_dim = feature_dim
        # phi_lambda: maps singular-value vector -> mask weights
        self.lambda_encoder = MaskEncoder(feature_dim, lambda_encoder_hidden, feature_dim)

    # ── subspace construction ──────────────────────────────────────────────

    def compute_covariance(self, z: torch.Tensor) -> torch.Tensor:
        """Eq.(1): Σ^m_z = E[(z - µ)(z - µ)^T]"""
        mu = z.mean(dim=0, keepdim=True)          # (1, d)
        diff = z - mu                              # (N, d)
        cov = (diff.T @ diff) / (z.size(0) - 1)   # (d, d)
        return cov

    def svd_decompose(self, cov: torch.Tensor):
        """Eq.(3): Σ = U Λ U^T  (symmetric PSD)"""
        # torch.linalg.eigh is numerically stable for symmetric matrices
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        # sort descending
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]        # (d, d)
        return eigenvectors, eigenvalues

    def forward(
        self,
        z_list: List[torch.Tensor],                # [(N, d), ...]  one per modality
        labels: Optional[torch.Tensor] = None,     # (N,)  needed for Lo
    ) -> Tuple[List[torch.Tensor], torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Returns:
            h_list   : globally denoised features [(N, d), ...]
            loss_assa: scalar loss (Lo + La)
            U_list   : list of eigenvector matrices
            w_list   : list of mask vectors
        """
        h_list, U_list, w_list = [], [], []

        for z in z_list:
            cov = self.compute_covariance(z)           # (d, d)
            U, lam = self.svd_decompose(cov)           # (d,d), (d,)
            w = self.lambda_encoder(lam.unsqueeze(0)).squeeze(0)  # (d,)
            # Eq.(5): h = z U diag(w) U^T
            z_proj = z @ U                             # (N, d)
            h = (z_proj * w.unsqueeze(0)) @ U.T        # (N, d)
            h_list.append(h)
            U_list.append(U)
            w_list.append(w)

        loss_assa = self._compute_loss(h_list, z_list, U_list, w_list, labels)
        return h_list, loss_assa, U_list, w_list

    # ── constraints ───────────────────────────────────────────────────────

    def _compute_loss(
        self,
        h_list: List[torch.Tensor],
        z_list: List[torch.Tensor],
        U_list: List[torch.Tensor],
        w_list: List[torch.Tensor],
        labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=h_list[0].device)
        if labels is not None:
            loss = loss + self._inter_class_orthogonality(h_list, labels)
        loss = loss + self._subspace_projection_alignment(z_list, U_list, w_list)
        return loss

    def _inter_class_orthogonality(
        self, h_list: List[torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:
        """Eq.(6): Lo – removes shared spurious patterns across classes."""
        classes = labels.unique()
        C = classes.numel()
        if C < 2:
            return torch.tensor(0.0, device=h_list[0].device)

        total = torch.tensor(0.0, device=h_list[0].device)
        M = len(h_list)
        for h in h_list:
            # compute per-class mean (normalised)
            class_means = []
            for c in classes:
                mask = (labels == c)
                if mask.sum() == 0:
                    continue
                mu_c = h[mask].mean(dim=0)
                mu_c = F.normalize(mu_c, dim=0)
                class_means.append(mu_c)
            if len(class_means) < 2:
                continue
            # stack: (num_classes, d)
            mu_mat = torch.stack(class_means, dim=0)   # (C', d)
            gram = mu_mat @ mu_mat.T                   # (C', C')
            C_prime = gram.size(0)
            I = torch.eye(C_prime, device=h.device)
            # off-diagonal terms → orthogonality
            off_diag = gram - I
            total = total + off_diag.pow(2).sum() / (C_prime * (C_prime - 1) + 1e-8)

        return total / M

    def _subspace_projection_alignment(
        self,
        z_list: List[torch.Tensor],
        U_list: List[torch.Tensor],
        w_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """Eq.(7): La – aligns projections of different modalities."""
        M = len(z_list)
        if M < 2:
            return torch.tensor(0.0, device=z_list[0].device)

        # pre-compute projections
        projs = []
        for z, U, w in zip(z_list, U_list, w_list):
            p = (z @ U) * w.unsqueeze(0)              # (N, d)
            projs.append(p)

        total = torch.tensor(0.0, device=z_list[0].device)
        count = 0
        for i in range(M):
            for j in range(M):
                if i == j:
                    continue
                diff = projs[i] - projs[j]
                total = total + diff.pow(2).sum()
                count += 1
        return total / (count + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# SACA: Sample-Adaptive Confidence Alignment (Instance-level)
# ─────────────────────────────────────────────────────────────────────────────

class NoiseExpert(nn.Module):
    """A single modality-specific or cross-modality noise expert (Eqs. 11-16)."""

    def __init__(self, feature_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.phi = MaskEncoder(feature_dim, hidden_dim, feature_dim)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            w   : mask  (N, d)
            h_hat: denoised feature (N, d)
            n   : noise component   (N, d)
        """
        w = self.phi(h)          # (N, d)
        h_hat = h * w
        n = h * (1.0 - w)
        return w, h_hat, n


class SACA(nn.Module):
    """
    Sample-Adaptive Confidence Alignment.

    Uses globally denoised features from ASSA as prior estimates,
    then runs per-sample noise experts with confidence-weighted slack alignment.
    """

    def __init__(self, num_modalities: int, feature_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.M = num_modalities
        self.d = feature_dim

        # M modality-specific experts + M cross-modality experts
        self.experts_s = nn.ModuleList(
            [NoiseExpert(feature_dim, hidden_dim) for _ in range(num_modalities)]
        )
        self.experts_c = nn.ModuleList(
            [NoiseExpert(feature_dim, hidden_dim) for _ in range(num_modalities)]
        )

    # ── prior estimation ──────────────────────────────────────────────────

    @staticmethod
    def estimate_gaussian(h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """MLE Gaussian parameters from h (N, d)."""
        mu = h.mean(dim=0)                                      # (d,)
        diff = h - mu.unsqueeze(0)
        cov = (diff.T @ diff) / (h.size(0) - 1 + 1e-8)        # (d, d)
        return mu, cov

    def compute_priors(
        self, h_list: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor],
               Dict[Tuple[int,int], torch.Tensor], Dict[Tuple[int,int], torch.Tensor]]:
        """
        Returns modality-wise (mu, cov) and cross-modality discrepancy (mu, cov).
        Eqs. (9)(10).
        """
        mus, covs = [], []
        for h in h_list:
            mu, cov = self.estimate_gaussian(h)
            mus.append(mu)
            covs.append(cov)

        cross_mus, cross_covs = {}, {}
        for m in range(self.M):
            for mp in range(self.M):
                if m == mp:
                    continue
                cross_mus[(m, mp)] = mus[m] - mus[mp]
                cross_covs[(m, mp)] = covs[m] + covs[mp]

        return mus, covs, cross_mus, cross_covs

    # ── confidence ────────────────────────────────────────────────────────

    @staticmethod
    def gaussian_log_likelihood(
        x: torch.Tensor, mu: torch.Tensor, cov: torch.Tensor
    ) -> torch.Tensor:
        """Log p(x) under N(mu, cov); returns (N,) scalar per sample."""
        d = mu.size(0)
        diff = x - mu.unsqueeze(0)                            # (N, d)
        # add small diagonal for numerical stability
        cov_reg = cov + 1e-6 * torch.eye(d, device=cov.device)
        try:
            L = torch.linalg.cholesky(cov_reg)
            # log|Σ| = 2 * sum(log diag(L))
            log_det = 2.0 * L.diagonal().log().sum()
            # (x-µ)^T Σ^{-1} (x-µ)  via solve
            v = torch.linalg.solve_triangular(L, diff.T, upper=False)  # (d, N)
            maha = (v * v).sum(dim=0)                                   # (N,)
        except Exception:
            log_det = torch.log(torch.det(cov_reg) + 1e-12)
            cov_inv = torch.linalg.pinv(cov_reg)
            maha = (diff @ cov_inv * diff).sum(dim=-1)

        log_p = -0.5 * (d * 1.8379 + log_det + maha)          # -0.5*(d*log2π + …)
        return log_p                                            # (N,)

    def compute_confidence(
        self,
        h_hat_s_list: List[torch.Tensor],
        mus: List[torch.Tensor],
        covs: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """c^m_i = p^m(h_hat^m_{s,i})  →  (N,)"""
        confs = []
        for m in range(self.M):
            log_p = self.gaussian_log_likelihood(h_hat_s_list[m], mus[m], covs[m])
            confs.append(log_p.exp().clamp(min=1e-8))          # (N,)
        return confs

    # ── asymmetric confidence-weighted update ────────────────────────────

    @staticmethod
    def compute_u(confs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Eq.(19): u^m_i negatively correlated with confidence."""
        # stack: (M, N)
        stacked = torch.stack(confs, dim=0)
        scores = torch.exp(1.0 - torch.tanh(stacked))         # (M, N)
        weights = scores / (scores.sum(dim=0, keepdim=True) + 1e-8)
        return [weights[m] for m in range(weights.size(0))]   # list of (N,)

    # ── nll losses ────────────────────────────────────────────────────────

    def loss_s_nll(
        self,
        h_hat_s_list: List[torch.Tensor],
        mus: List[torch.Tensor],
        covs: List[torch.Tensor],
    ) -> torch.Tensor:
        """Eq.(17) first term: -Σ_m log p^m_h(h_hat^m_s)"""
        total = torch.tensor(0.0, device=h_hat_s_list[0].device)
        for m in range(self.M):
            log_p = self.gaussian_log_likelihood(h_hat_s_list[m], mus[m], covs[m])
            total = total - log_p.mean()
        return total

    def loss_c_nll(
        self,
        h_hat_c_list: List[torch.Tensor],
        cross_mus: Dict,
        cross_covs: Dict,
    ) -> torch.Tensor:
        """Eq.(17) second term: slack alignment via cross-modality discrepancy."""
        total = torch.tensor(0.0, device=h_hat_c_list[0].device)
        count = 0
        for m in range(self.M):
            for mp in range(self.M):
                if m == mp:
                    continue
                diff = h_hat_c_list[m] - h_hat_c_list[mp]     # (N, d)
                mu_cross = cross_mus[(m, mp)]
                cov_cross = cross_covs[(m, mp)]
                log_p = self.gaussian_log_likelihood(diff, mu_cross, cov_cross)
                total = total - log_p.mean()
                count += 1
        return total / (count + 1e-8)

    # ── forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        h_list: List[torch.Tensor],                # ASSA outputs
        mus: List[torch.Tensor],
        covs: List[torch.Tensor],
        cross_mus: Dict,
        cross_covs: Dict,
        u_weights: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[List, List, List, List, torch.Tensor]:
        """
        Returns:
            h_hat_s_list, h_hat_c_list : denoised features
            n_s_list, n_c_list         : noise components
            loss_saca                  : scalar
        """
        h_hat_s_list, h_hat_c_list = [], []
        n_s_list, n_c_list = [], []

        for m in range(self.M):
            _, h_hat_s, n_s = self.experts_s[m](h_list[m])
            _, h_hat_c, n_c = self.experts_c[m](h_list[m])
            h_hat_s_list.append(h_hat_s)
            h_hat_c_list.append(h_hat_c)
            n_s_list.append(n_s)
            n_c_list.append(n_c)

        # --- loss ---
        ls = self.loss_s_nll(h_hat_s_list, mus, covs)
        lc = self.loss_c_nll(h_hat_c_list, cross_mus, cross_covs)
        loss_saca = ls + lc

        return h_hat_s_list, h_hat_c_list, n_s_list, n_c_list, loss_saca


# ─────────────────────────────────────────────────────────────────────────────
# TTCE: Test-Time Cooperative Enhancement
# ─────────────────────────────────────────────────────────────────────────────

class TTCE(nn.Module):
    """
    Test-Time Cooperative Enhancement.

    At inference, iteratively:
      1. SACA→ASSA: incorporate instance-level noise into reconstruction loss
         to refine globally denoised h (Eq. 21).
      2. ASSA→SACA: update priors with enhanced h↑, then refine instance-level
         denoised features (Eqs. 22-25).
    """

    def __init__(self, decoders: nn.ModuleList, alpha: float = 0.5):
        super().__init__()
        self.decoders = decoders    # Ψ^m
        self.alpha = alpha

    def enhance(
        self,
        # ASSA globals
        h_list: List[torch.Tensor],
        # SACA instance noise
        n_s_list: List[torch.Tensor],
        n_c_list: List[torch.Tensor],
        # SACA instance features
        h_hat_s_list: List[torch.Tensor],
        h_hat_c_list: List[torch.Tensor],
        # raw input (for reconstruction target)
        x_list: List[torch.Tensor],
        # current priors
        mus: List[torch.Tensor],
        covs: List[torch.Tensor],
        cross_mus: Dict,
        cross_covs: Dict,
        # hyper-params
        eta: float = 0.01,
        num_iters: int = 30,
    ) -> Tuple[List, List, List, List]:
        """
        Returns enhanced h↑ list, h_hat_s↑ list, h_hat_c↑ list,
        and updated (mus↑, covs↑, cross_mus↑, cross_covs↑).
        """
        M = len(h_list)

        # working copies (detached from graph for iterative update)
        h_up = [h.detach().clone().requires_grad_(True) for h in h_list]
        hs_up = [hs.detach().clone() for hs in h_hat_s_list]
        hc_up = [hc.detach().clone() for hc in h_hat_c_list]
        mus_up = [mu.detach().clone() for mu in mus]
        covs_up = [c.detach().clone() for c in covs]
        cross_mus_up = {k: v.detach().clone() for k, v in cross_mus.items()}
        cross_covs_up = {k: v.detach().clone() for k, v in cross_covs.items()}

        for _ in range(num_iters):
            # ── Step 1: SACA→ASSA (reconstruction loss to enhance h) ──
            # Eq.(20): L_re = mean_m ||Ψ^m(h^m + n^m_s + n^m_c) - x^m||^2_F
            # We do gradient step on h_up
            loss_re = torch.tensor(0.0, device=h_list[0].device)
            for m in range(M):
                # stop grad through noise; only h_up is updated
                recon_input = h_up[m] + n_s_list[m].detach() + n_c_list[m].detach()
                recon = self.decoders[m](recon_input)
                loss_re = loss_re + F.mse_loss(recon, x_list[m])
            loss_re = loss_re / M

            # gradient w.r.t. h_up
            grads = torch.autograd.grad(
                loss_re, h_up, create_graph=False, allow_unused=True
            )
            with torch.no_grad():
                for m in range(M):
                    if grads[m] is not None:
                        # Eq.(21): h↑ ← h↑ - eta * grad
                        h_up[m] = h_up[m] - eta * grads[m]
                        h_up[m] = h_up[m].detach().requires_grad_(True)

            # ── Step 2: ASSA→SACA (update priors with enhanced h) ──
            # Eqs.(22)(23): µ↑ = (1-α)µ + α·∆µ↑, same for Σ
            with torch.no_grad():
                for m in range(M):
                    delta_mu = h_up[m].mean(dim=0)
                    diff = h_up[m] - delta_mu.unsqueeze(0)
                    delta_cov = (diff.T @ diff) / (h_up[m].size(0) - 1 + 1e-8)

                    mus_up[m] = (1 - self.alpha) * mus_up[m] + self.alpha * delta_mu
                    covs_up[m] = (1 - self.alpha) * covs_up[m] + self.alpha * delta_cov

                # update cross-modality discrepancy distributions
                for m in range(M):
                    for mp in range(M):
                        if m == mp:
                            continue
                        cross_mus_up[(m, mp)] = mus_up[m] - mus_up[mp]
                        cross_covs_up[(m, mp)] = covs_up[m] + covs_up[mp]

                # Eqs.(24)(25): update h_hat_s↑ and h_hat_c↑
                for m in range(M):
                    cov_inv = torch.linalg.pinv(
                        covs_up[m] + 1e-6 * torch.eye(covs_up[m].size(0), device=covs_up[m].device)
                    )
                    # h_hat_s↑ = h_hat_s↑ - eta * Σ↑^{-1} (h_hat_s↑ - µ↑)
                    hs_up[m] = hs_up[m] - eta * (hs_up[m] - mus_up[m].unsqueeze(0)) @ cov_inv.T

                    cross_sum = torch.zeros_like(hc_up[m])
                    for mp in range(M):
                        if m == mp:
                            continue
                        c_inv = torch.linalg.pinv(
                            cross_covs_up[(m, mp)] + 1e-6 * torch.eye(
                                cross_covs_up[(m, mp)].size(0), device=cross_covs_up[(m, mp)].device
                            )
                        )
                        mu_cross = cross_mus_up[(m, mp)]
                        # (h_hat_c^m - h_hat_c^{m'} - µ^{m-m'})
                        residual = hc_up[m] - hc_up[mp] - mu_cross.unsqueeze(0)
                        cross_sum = cross_sum + residual @ c_inv.T
                    hc_up[m] = hc_up[m] - eta * cross_sum

        return h_up, hs_up, hc_up, mus_up, covs_up, cross_mus_up, cross_covs_up


# ─────────────────────────────────────────────────────────────────────────────
# Fusion & Classifier
# ─────────────────────────────────────────────────────────────────────────────

class FusionClassifier(nn.Module):
    """Confidence-weighted fusion then linear classification (Eqs. 26)."""

    def __init__(self, feature_dim: int, num_classes: int, hidden_dim: int = 512):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        h_hat_s_list: List[torch.Tensor],         # (N, d) per modality
        h_hat_c_list: List[torch.Tensor],
        conf_s: List[torch.Tensor],                # (N,) per modality
        conf_c: List[torch.Tensor],
    ) -> torch.Tensor:
        """Eq.(26): f^mm = Σ_m (conf_s^m * h_hat_s^m + conf_c^m * h_hat_c^m)"""
        fused = torch.zeros_like(h_hat_s_list[0])
        for m, (hs, hc, cs, cc) in enumerate(zip(h_hat_s_list, h_hat_c_list, conf_s, conf_c)):
            fused = fused + cs.unsqueeze(1) * hs + cc.unsqueeze(1) * hc
        return self.classifier(fused)


# ─────────────────────────────────────────────────────────────────────────────
# Full TAHCD Model
# ─────────────────────────────────────────────────────────────────────────────

class TAHCD(nn.Module):
    """
    Full TAHCD model combining ASSA, SACA, TTCE and a fusion classifier.

    Args:
        input_dims   : list of input feature dimensions per modality
        num_classes  : number of output classes
        feature_dim  : shared latent dimension (256 per paper)
        hidden_dim   : hidden layer size (512 per paper)
        alpha        : TTCE prior update rate  (Eq. 22-23)
        ttce_iters   : number of TTCE enhancement iterations at test time
        ttce_eta     : step size for TTCE gradient update
    """

    def __init__(
        self,
        input_dims: List[int],
        num_classes: int,
        feature_dim: int = 256,
        hidden_dim: int = 512,
        alpha: float = 0.5,
        ttce_iters: int = 30,
        ttce_eta: float = 0.01,
    ):
        super().__init__()
        self.M = len(input_dims)
        self.feature_dim = feature_dim
        self.ttce_iters = ttce_iters
        self.ttce_eta = ttce_eta

        # ── Modality encoders ──────────────────────────────────────────────
        self.encoders = nn.ModuleList(
            [ModalityEncoder(d, hidden_dim, feature_dim) for d in input_dims]
        )

        # ── Decoders (pre-trained separately, fixed during TAHCD training) ──
        self.decoders = nn.ModuleList(
            [ModalityDecoder(feature_dim, hidden_dim, d) for d in input_dims]
        )

        # ── ASSA (global) ──────────────────────────────────────────────────
        self.assa = ASSA(feature_dim, hidden_dim)

        # ── SACA (instance) ────────────────────────────────────────────────
        self.saca = SACA(self.M, feature_dim, hidden_dim)

        # ── TTCE ───────────────────────────────────────────────────────────
        self.ttce = TTCE(self.decoders, alpha)

        # ── Fusion + Classifier ────────────────────────────────────────────
        self.fusion_cls = FusionClassifier(feature_dim, num_classes, hidden_dim)

    # ── pre-train decoders ─────────────────────────────────────────────────

    def pretrain_decoders_loss(self, x_list: List[torch.Tensor]) -> torch.Tensor:
        """Reconstruction loss for pre-training decoders."""
        total = torch.tensor(0.0, device=x_list[0].device)
        for m, (encoder, decoder) in enumerate(zip(self.encoders, self.decoders)):
            z = encoder(x_list[m])
            recon = decoder(z)
            total = total + F.mse_loss(recon, x_list[m])
        return total / self.M

    # ── training forward ───────────────────────────────────────────────────

    def forward_train(
        self,
        x_list: List[torch.Tensor],
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Full training pass.
        Returns:
            logits     : (N, C)
            loss_dict  : individual losses for logging
        """
        # 1. Encode
        z_list = [enc(x) for enc, x in zip(self.encoders, x_list)]

        # 2. ASSA – global denoising
        h_list, loss_assa, U_list, w_list = self.assa(z_list, labels)

        # 3. SACA – priors
        mus, covs, cross_mus, cross_covs = self.saca.compute_priors(h_list)

        # 4. SACA – instance denoising (confidence weights from previous iter)
        h_hat_s_list, h_hat_c_list, n_s_list, n_c_list, loss_saca = self.saca(
            h_list, mus, covs, cross_mus, cross_covs
        )

        # 5. TTCE reconstruction loss (Eq. 20)
        loss_re = torch.tensor(0.0, device=x_list[0].device)
        for m in range(self.M):
            recon_in = h_list[m] + n_s_list[m] + n_c_list[m]
            recon = self.decoders[m](recon_in)
            loss_re = loss_re + F.mse_loss(recon, x_list[m])
        loss_re = loss_re / self.M

        # 6. Confidence-weighted fusion
        confs = self.saca.compute_confidence(h_hat_s_list, mus, covs)
        # normalise confidences across modalities per sample
        conf_stack = torch.stack(confs, dim=1)            # (N, M)
        conf_norm = conf_stack / (conf_stack.sum(dim=1, keepdim=True) + 1e-8)
        conf_s = [conf_norm[:, m] for m in range(self.M)]
        conf_c = conf_s   # same normalised confidence for both experts

        logits = self.fusion_cls(h_hat_s_list, h_hat_c_list, conf_s, conf_c)

        loss_cls = F.cross_entropy(logits, labels)
        loss_total = loss_assa + loss_saca + loss_re + loss_cls

        losses = {
            "loss_total": loss_total,
            "loss_assa": loss_assa,
            "loss_saca": loss_saca,
            "loss_re": loss_re,
            "loss_cls": loss_cls,
        }
        return logits, losses

    # ── inference forward (with TTCE) ─────────────────────────────────────

    @torch.no_grad()
    def forward_inference(
        self,
        x_list: List[torch.Tensor],
        num_iters: Optional[int] = None,
        eta: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Inference with TTCE cooperative enhancement.
        (Algorithm 1 in the paper)
        """
        iters = num_iters if num_iters is not None else self.ttce_iters
        step = eta if eta is not None else self.ttce_eta

        # Step 1: encode
        z_list = [enc(x) for enc, x in zip(self.encoders, x_list)]

        # Step 2: ASSA
        h_list, _, _, _ = self.assa(z_list, labels=None)

        # Step 3: priors
        mus, covs, cross_mus, cross_covs = self.saca.compute_priors(h_list)

        # Step 4: SACA
        h_hat_s_list, h_hat_c_list, n_s_list, n_c_list, _ = self.saca(
            h_list, mus, covs, cross_mus, cross_covs
        )

        if iters > 0:
            # need grads for TTCE h_up update (re-enable for specific tensors)
            with torch.enable_grad():
                h_up, hs_up, hc_up, mus_up, covs_up, cross_mus_up, cross_covs_up = \
                    self.ttce.enhance(
                        h_list, n_s_list, n_c_list,
                        h_hat_s_list, h_hat_c_list,
                        x_list, mus, covs, cross_mus, cross_covs,
                        eta=step, num_iters=iters,
                    )
        else:
            h_up = h_list
            hs_up = h_hat_s_list
            hc_up = h_hat_c_list
            mus_up = mus
            covs_up = covs

        # Step 5: confidence-weighted fusion
        confs = self.saca.compute_confidence(
            [h.detach() for h in hs_up],
            [m.detach() if isinstance(m, torch.Tensor) else m for m in mus_up],
            [c.detach() if isinstance(c, torch.Tensor) else c for c in covs_up],
        )
        conf_stack = torch.stack(confs, dim=1)
        conf_norm = conf_stack / (conf_stack.sum(dim=1, keepdim=True) + 1e-8)
        conf_s = [conf_norm[:, m] for m in range(self.M)]

        hs_out = [h.detach() if isinstance(h, torch.Tensor) else h for h in hs_up]
        hc_out = [h.detach() if isinstance(h, torch.Tensor) else h for h in hc_up]

        logits = self.fusion_cls(hs_out, hc_out, conf_s, conf_s)
        return logits

    def forward(
        self,
        x_list: List[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        inference: bool = False,
    ):
        if inference or not self.training:
            return self.forward_inference(x_list)
        return self.forward_train(x_list, labels)
