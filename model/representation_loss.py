import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy.ndimage import binary_dilation

import time


# ============================================================================

# ============================================================================

class InstanceToInstanceContrastive(nn.Module):
    """Contrast local instances using structural patch similarity."""

    def __init__(
            self,
            patch_size: int = 5,
            temperature: float = 0.1,
            top_k: int = 5,
            min_dice: float = 0.3,
            use_original_resolution: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.temperature = temperature
        self.top_k = top_k
        self.min_dice = min_dice
        
        self.padding = patch_size // 2 
        self.use_original_resolution = use_original_resolution

    def extract_patches_parallel(
            self,
            volume: torch.Tensor,
            batch_indices: torch.Tensor,
            coords: torch.Tensor,
            patch_size: int = None, 
    ) -> torch.Tensor:
        """Extract centered patches with vectorized indexing."""
        B, D, H, W = volume.shape
        
        
        P = patch_size if patch_size is not None else self.patch_size
        pad = P // 2
        
        N = coords.shape[0]

        
        volume_padded = F.pad(
            volume.unsqueeze(1),
            (pad, pad, pad, pad, pad, pad),
            mode='constant', value=0
        ).squeeze(1)

        
        mesh = torch.meshgrid(
            torch.arange(-pad, pad + 1, device=volume.device),
            torch.arange(-pad, pad + 1, device=volume.device),
            torch.arange(-pad, pad + 1, device=volume.device),
            indexing='ij'
        )
        offsets = torch.stack(mesh, dim=-1).reshape(1, -1, 3)
        
        
        sample_coords = coords.unsqueeze(1) + offsets + pad

        
        z = sample_coords[..., 0].clamp(0, D + 2 * pad - 1)
        y = sample_coords[..., 1].clamp(0, H + 2 * pad - 1)
        x = sample_coords[..., 2].clamp(0, W + 2 * pad - 1)

        
        b_idx = batch_indices.unsqueeze(1).expand(-1, P * P * P)
        patches_flat = volume_padded[b_idx, z, y, x]

        return patches_flat.view(N, P, P, P)

    def forward(
            self,
            features: torch.Tensor,
            skeleton_dilated: torch.Tensor,
            skeleton: torch.Tensor,
            num_samples: int = 256,
    ) -> torch.Tensor:
        B, C, D_feat, H_feat, W_feat = features.shape
        device = features.device

        
        if skeleton_dilated.dim() == 5:
            skeleton_dilated = skeleton_dilated.squeeze(1)
        
        
        stride_d = skeleton_dilated.shape[1] // D_feat
        stride_h = skeleton_dilated.shape[2] // H_feat
        stride_w = skeleton_dilated.shape[3] // W_feat
        
        
        
        if (stride_d > 1 or stride_h > 1 or stride_w > 1):
            
            
            skeleton_dilated_ds = F.interpolate(
                skeleton_dilated.unsqueeze(1).float(),
                size=(D_feat, H_feat, W_feat),
                mode='trilinear' 
            ).squeeze(1)
        else:
            skeleton_dilated_ds = skeleton_dilated.float()

        
        max_val = skeleton_dilated_ds.view(B, -1).max(dim=1, keepdim=True)[0]
        skeleton_dilated_ds = skeleton_dilated_ds / (max_val.view(B, 1, 1, 1) + 1e-6)

        
        if skeleton.dim() == 5:
            skeleton = skeleton.squeeze(1)
            
        
        if skeleton.shape[1:] != (D_feat, H_feat, W_feat):
            skeleton_low = F.max_pool3d(skeleton.float().unsqueeze(1), kernel_size=stride_d, stride=stride_d).squeeze(1)
        else:
            skeleton_low = skeleton
            
        mask = skeleton_low > 0.5
        coords_all = torch.nonzero(mask)

        if coords_all.shape[0] < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        
        total_samples = min(coords_all.shape[0], B * num_samples)
        perm = torch.randperm(coords_all.shape[0], device=device)[:total_samples]
        sampled_coords = coords_all[perm]

        b_idx = sampled_coords[:, 0]
        z_idx, y_idx, x_idx = sampled_coords[:, 1], sampled_coords[:, 2], sampled_coords[:, 3]
        
        
        sampled_features = features[b_idx, :, z_idx, y_idx, x_idx]
        sampled_features = F.normalize(sampled_features, dim=1)
        N = sampled_features.shape[0]

        
        if not self.use_original_resolution:
            
            patches = self.extract_patches_parallel(
                skeleton_dilated_ds, 
                b_idx, 
                sampled_coords[:, 1:],
                patch_size=self.patch_size 
            )
        else:
            
            coords_orig = sampled_coords[:, 1:] * stride_d 
            
            large_patch_size = self.patch_size * 2 - 1
            
            patches = self.extract_patches_parallel(
                skeleton_dilated,  
                b_idx, 
                coords_orig,       
                patch_size=large_patch_size 
            )

        
        patches_flat = patches.reshape(N, -1).float()
        
        
        p_max, _ = patches_flat.max(dim=1, keepdim=True)
        patches_flat = patches_flat / (p_max + 1e-6)

        intersection = torch.mm(patches_flat, patches_flat.T)
        sums = patches_flat.sum(dim=1)
        union = sums.unsqueeze(1) + sums.unsqueeze(0)
        dice_matrix = (2 * intersection) / (union + 1e-6)

        
        mask_diag = torch.eye(N, device=device, dtype=torch.bool)
        dice_matrix = dice_matrix.masked_fill(mask_diag, -1)
        valid_dice_mask = dice_matrix >= 0.0
        mean_dice = (dice_matrix * valid_dice_mask).sum(dim=1) / (valid_dice_mask.sum(dim=1).clamp_min(1))

        # ==========================
        
        # ==========================
        
        pos_abs_thr = 0.65   
        pos_gap_thr = 0.30   
        max_pos = 10         

        
        
        
        pos_mask = (dice_matrix >= pos_abs_thr) & (dice_matrix >= (mean_dice.unsqueeze(1) + pos_gap_thr))

        
        

        
        
        if max_pos is not None and max_pos > 0:
            
            masked_scores = dice_matrix.masked_fill(~pos_mask, float("-inf"))
            
            
            top_scores, top_idx = torch.topk(masked_scores, k=min(max_pos, N-1), dim=1)
            
            pos_mask_cap = torch.zeros_like(pos_mask)
            
            keep = top_scores > float("-1e8")
            row_ids = torch.arange(N, device=device).unsqueeze(1).expand_as(top_idx)
            pos_mask_cap[row_ids[keep], top_idx[keep]] = True
            pos_mask = pos_mask_cap

        
        num_pos = pos_mask.sum(dim=1)
        anchor_mask = num_pos > 0
        if anchor_mask.sum() < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # ==========================
        
        # ==========================
        logits = torch.mm(sampled_features, sampled_features.T) / self.temperature  # [N, N]
        logits = logits.masked_fill(mask_diag, -1e9)  

        
        
        t_w = 0.3
        
        pos_scores = (dice_matrix / t_w).masked_fill(~pos_mask, float("-inf"))
        
        row_has_pos = torch.isfinite(pos_scores).any(dim=1)  # True if at least one finite value
        anchor_mask = anchor_mask & row_has_pos
        if anchor_mask.sum() < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        
        weights = torch.zeros_like(pos_scores)
        weights[anchor_mask] = torch.softmax(pos_scores[anchor_mask], dim=1)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

        
        log_softmax = logits - torch.logsumexp(logits, dim=1, keepdim=True)  # [N, N]
        loss_per_sample = -(weights * log_softmax).sum(dim=1)  # [N]

        loss = loss_per_sample[anchor_mask].mean()

        
        log_probs = torch.log(weights + 1e-9)
        entropy_per_sample = - (weights * log_probs).sum(dim=1)
        entropy = entropy_per_sample.mean()

        return loss



class InstanceToPrototypeContrastive(nn.Module):
    """Align sampled instances with their assigned prototypes."""

    def __init__(self, temperature: float = 0.1, bg_ratio: float = 0.3):
        super().__init__()
        self.temperature = temperature
        self.bg_ratio = bg_ratio

    def forward(
        self,
        features: torch.Tensor,  # [B, C, D, H, W]
        prototypes: torch.Tensor,  # [K, C]
        Q: torch.Tensor,  # [B, K, D, H, W]
        skeleton: torch.Tensor,  # [B, D, H, W]
    ) -> torch.Tensor:
        
        device = features.device
        B, C, D, H, W = features.shape
        
        
        # [B, C, D, H, W] -> [B, D, H, W, C]
        features_perm = features.permute(0, 2, 3, 4, 1)
        features_norm = F.normalize(features_perm, dim=-1)
        prototypes_norm = F.normalize(prototypes, dim=1)
        
        # [B, K, D, H, W] -> [B, D, H, W, K]
        Q_perm = Q.permute(0, 2, 3, 4, 1)

        
        fg_mask = skeleton > 0.5  # [B, D, H, W]
        
        
        num_fg_total = fg_mask.sum().item()
        
        if num_fg_total == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        
        
        
        
        bg_candidates = ~fg_mask # [B, D, H, W]
        num_bg_total = bg_candidates.sum().item()
        
        
        num_bg_sample = int(num_fg_total * self.bg_ratio)
        num_bg_sample = min(num_bg_sample, num_bg_total) 
        
        if num_bg_sample > 0:
            
            rand_scores = torch.rand_like(skeleton)
            
            rand_scores = rand_scores.masked_fill(fg_mask, -1.0)
            _, topk_indices = torch.topk(rand_scores.view(-1), k=num_bg_sample)
            
            bg_mask_flat = torch.zeros(B*D*H*W, dtype=torch.bool, device=device)
            bg_mask_flat[topk_indices] = True
            bg_mask = bg_mask_flat.view(B, D, H, W)
        else:
            bg_mask = torch.zeros_like(fg_mask)

        
        combined_mask = fg_mask | bg_mask  # [B, D, H, W]
        
        
        # feat_samples: [N_total, C]
        feat_samples = features_norm[combined_mask]
        # Q_samples: [N_total, K]
        # Q is a self-generated assignment target.  Letting the CE gradient
        # flow through that target rewards making the current winner even more
        # confident and creates a positive feedback loop toward one prototype.
        # Treat assignments as pseudo-labels; features/prototypes still receive
        # gradients through ``sim``.
        Q_samples = Q_perm[combined_mask].detach()
        # A target identical to the model's own softmax gives almost zero
        # within-group gradient.  Sharpen it so IP contrastive learning really
        # pulls a sample toward its preferred prototypes, while the marginal
        # assignment regularizer below keeps usage globally balanced.
        Q_samples = Q_samples.pow(2)
        Q_samples = Q_samples / Q_samples.sum(dim=1, keepdim=True).clamp_min(1e-8)
        
        if feat_samples.shape[0] == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        
        # [N, C] @ [C, K] -> [N, K]
        sim = torch.mm(feat_samples, prototypes_norm.T) / self.temperature
        
        
        log_softmax = F.log_softmax(sim, dim=1)
        loss = -(Q_samples * log_softmax).sum(dim=1).mean()

        
        # log_softmax = sim - torch.logsumexp(sim, dim=1, keepdim=True) # (N,K)
        # loss = -(Q_samples * log_softmax).sum(dim=1).mean()

        
        log_probs = torch.log(Q_samples + 1e-9)
        entropy_per_sample = - (Q_samples * log_probs).sum(dim=1)
        entropy = entropy_per_sample.mean()

        
        pred = torch.mm(features_norm[fg_mask], prototypes_norm.T).argmax(dim=1)
        label = Q_perm[fg_mask].argmax(dim=1)
        fg_acc = (pred == label).float().mean()

        return loss

    def sample_back_mask(self, skeleton_mask: torch.Tensor, num_samples: int) -> torch.Tensor:
            """Sample a fixed number of background positions."""
            device = skeleton_mask.device
            D, H, W = skeleton_mask.shape

            
            bg_indices = torch.nonzero(skeleton_mask < 0.5)  # [N_bg, 3]
            num_bg = bg_indices.shape[0]

            if num_bg == 0:
                return torch.zeros_like(skeleton_mask, dtype=torch.bool)

            
            if num_bg <= num_samples:
                selected_indices = bg_indices
            else:
                
                perm = torch.randperm(num_bg, device=device)[:num_samples]
                selected_indices = bg_indices[perm]

            
            bg_sample_mask = torch.zeros_like(skeleton_mask, dtype=torch.bool)
            
            bg_sample_mask[selected_indices[:, 0], selected_indices[:, 1], selected_indices[:, 2]] = True

            return bg_sample_mask


class PrototypeDiversityLoss(nn.Module):
    """Penalize excessive similarity between distinct prototypes."""

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, prototypes: torch.Tensor) -> torch.Tensor:
        """Compute diversity loss for the prototype set."""
        if prototypes.dim() > 2:
            prototypes = prototypes.squeeze(-1).squeeze(-1).squeeze(-1)

        K, C = prototypes.shape

        
        proto_norm = F.normalize(prototypes, dim=1)

        
        sim_matrix = torch.mm(proto_norm, proto_norm.T)

        
        mask = ~torch.eye(K, dtype=torch.bool, device=prototypes.device)
        off_diag = sim_matrix[mask]

        
        loss = F.relu(off_diag - self.margin).mean()

        return loss
    

class InstanceToPrototypeEntropy(nn.Module):
    """Encourage balanced use of foreground prototypes."""

    def __init__(self, num_foreground: int):
        super().__init__()
        self.num_foreground = num_foreground

    def forward(self, fg_soft_sim_prob: torch.Tensor) -> torch.Tensor:
        """Compute foreground assignment entropy loss."""

        p = fg_soft_sim_prob[:, :self.num_foreground]
        p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-8)
        p_bar = p.mean(dim=0)
        epsilon = 1e-8
        # Mutual-information style regularization: the marginal KL prevents
        # global collapse, while a smaller conditional-entropy term prevents
        # the trivial solution where every point is uniformly assigned to all
        # prototypes.
        marginal_kl = (
            p_bar * torch.log(p_bar * self.num_foreground + epsilon)
        ).sum()
        conditional_entropy = -(
            p * torch.log(p + epsilon)
        ).sum(dim=1).mean()
        loss = marginal_kl + conditional_entropy

        return loss
