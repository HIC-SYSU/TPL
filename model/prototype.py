"""Latent and spatial prototype assignment components."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class TopologicalPrototype(nn.Module):
    """Maintain latent and spatial prototypes and compute dense assignments."""

    def __init__(
            self,
            num_prototypes: int = 64,
            num_foreground: int = 60,  
            feature_dim: int = 128,
            prototype_size: int = 5,  
            alpha: float = 1.0,  
            beta: float = 0.0,  
            temperature: float = 0.2,
            ema_decay: float = 0.99,
    ):
        super().__init__()

        self.num_prototypes = num_prototypes
        self.num_foreground = num_foreground
        self.num_background = num_prototypes - num_foreground
        self.feature_dim = feature_dim
        self.prototype_size = prototype_size
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.ema_decay = ema_decay

        
        
        self.implicit_prototypes = nn.Parameter(
            torch.randn(num_prototypes, feature_dim)
        )
        nn.init.xavier_uniform_(self.implicit_prototypes)

        
        
        self.register_buffer(
            'explicit_prototypes',
            torch.zeros(num_prototypes, 1, prototype_size, prototype_size, prototype_size)
        )
        self._init_explicit_prototypes()

        
        
        self.register_buffer(
            'hit_counts',
            torch.zeros(num_prototypes)
        )
        self.register_buffer(
            'total_counts',
            torch.zeros(1)
        )

        
        self.register_buffer(
            'similarity_thresholds',
            torch.ones(num_prototypes) * 0.5
        )

    def _init_explicit_prototypes(self):
        """Initialize spatial prototype templates."""
        S = self.prototype_size
        center = S // 2

        with torch.no_grad():
            for k in range(self.num_foreground):
                template = torch.zeros(1, S, S, S)

                
                topo_type = k % 4  # 0: terminal, 1-2: tubular, 3: bifurcation

                if topo_type == 0:
                    
                    template[0, center, center, center:] = 1.0

                elif topo_type == 1:
                    
                    template[0, center, center, :] = 1.0

                elif topo_type == 2:
                    
                    template[0, center, center, center:] = 1.0
                    template[0, center:, center, center] = 1.0

                else:
                    
                    template[0, center, center, :center + 1] = 1.0
                    template[0, :center + 1, center, center] = 1.0
                    template[0, center:, center, center] = 1.0

                self.explicit_prototypes[k] = template

            
            for k in range(self.num_foreground, self.num_prototypes):
                self.explicit_prototypes[k] = torch.zeros(1, S, S, S)

    def forward(
            self,
            features: torch.Tensor,  # [B, C, D, H, W]
            skeleton_dilated: Optional[torch.Tensor] = None,
            skeleton: Optional[torch.Tensor] = None,  # [B, 1, D', H', W']
            topk: int = 3,
            return_assignment: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Compute prototype similarities and dense assignments."""
        B, C, D, H, W = features.shape
        device = features.device

        outputs = {}

        
        imp_sim = self.compute_implicit_similarity(features)
        outputs['implicit_similarity'] = imp_sim

        
        foreground_mask = None
        exp_sim = None

        if skeleton is not None:
            
            
            foreground_mask = skeleton > 0.5

            if self.beta != 0 and skeleton_dilated is not None:
                _, _, D_feat, H_feat, W_feat = features.shape
                if skeleton_dilated.dim() == 4:
                    skeleton_dilated = skeleton_dilated.unsqueeze(1)

                
                if skeleton_dilated.shape[-3:] != (D_feat, H_feat, W_feat):
                    skeleton_dilated_ds = F.interpolate(
                        skeleton_dilated.float(),
                        size=(D_feat, H_feat, W_feat),
                        mode='trilinear',
                        align_corners=False,
                    )
                else:
                    skeleton_dilated_ds = skeleton_dilated.float()

                exp_sim = self.compute_explicit_similarity(skeleton_dilated_ds)
                outputs['explicit_similarity'] = exp_sim

        
        if return_assignment:
            
            assignment_result = self.compute_assignment(
                imp_sim, exp_sim,
                foreground_mask=foreground_mask,
                confidence_threshold=1.5,
                topk=topk,
            )
            outputs.update(assignment_result)

        return outputs

    def compute_implicit_similarity(
            self,
            features: torch.Tensor,  # [B, C, D, H, W]
    ) -> torch.Tensor:
        """Compute cosine similarity to latent prototypes."""
        B, C, D, H, W = features.shape
        K = self.num_prototypes

        
        features_norm = F.normalize(features, dim=1)  # [B, C, D, H, W]

        
        prototypes_norm = F.normalize(self.implicit_prototypes, dim=1)  # [K, C]

        
        # features: [B, C, D, H, W] -> [B, D, H, W, C]
        # prototypes: [K, C]
        # result: [B, D, H, W, K] -> [B, K, D, H, W]

        features_perm = features_norm.permute(0, 2, 3, 4, 1)  # [B, D, H, W, C]
        similarity = torch.einsum('bdhwc,kc->bdhwk', features_perm, prototypes_norm)
        similarity = similarity.permute(0, 4, 1, 2, 3)  # [B, K, D, H, W]

        return similarity

    def compute_explicit_similarity(
            self,
            skeleton_dilated: torch.Tensor,  
    ) -> torch.Tensor:
        """Compute local Dice similarity to spatial prototypes."""
        return self.dice_conv(skeleton_dilated, self.explicit_prototypes)

    @torch.no_grad()
    def dice_conv(
            self,
            label: torch.Tensor,  
            kernel: torch.Tensor,  
            eps: float = 1e-6,
    ) -> torch.Tensor:
        """Compute local Dice scores with convolution."""
        with torch.amp.autocast('cuda', enabled=False):
            label = label.float()
            kernel = kernel.float()

            
            # label_binary = (label > 0.5).float()
            # kernel_binary = (kernel > 0.5).float()
            label_binary = label.float()
            kernel_binary = kernel.float()

            K, _, S, S, S = kernel.shape
            padding = S // 2

            
            
            sum_P = kernel_binary.sum(dim=[1, 2, 3, 4])  # [K]

            
            
            intersection = F.conv3d(
                label_binary,
                kernel_binary,
                padding=padding
            )  # [B, K, D, H, W]

            
            
            ones_kernel = torch.ones(1, 1, S, S, S, device=label.device)
            sum_Y = F.conv3d(
                label_binary,
                ones_kernel,
                padding=padding
            )  # [B, 1, D, H, W]

            
            # Dice = 2 * intersection / (sum_Y + sum_P)
            
            sum_P_broadcast = sum_P.view(1, K, 1, 1, 1)

            dice = (2 * intersection) / (sum_Y + sum_P_broadcast + eps)

            # Clamp to [0, 1]
            dice = dice.clamp(0, 1)

        return dice

    def compute_assignment(
            self,
            imp_sim: torch.Tensor,  # [B, K, D, H, W]
            exp_sim: Optional[torch.Tensor] = None,  # [B, K, D, H, W]
            foreground_mask: Optional[torch.Tensor] = None, # [B, D, H, W] bool
            confidence_threshold: float = 2.0,
            topk: int = 3,
    ) -> Dict[str, torch.Tensor]:   
        """Compute constrained dense prototype assignments."""
        B, K, D, H, W = imp_sim.shape
        device = imp_sim.device
        
        
        if exp_sim is not None:
            combined_sim = self.alpha * imp_sim + self.beta * exp_sim
        else:
            combined_sim = imp_sim
        
        
        sim_flat = combined_sim.permute(0, 2, 3, 4, 1).reshape(-1, K)  # [N, K]
        N = sim_flat.shape[0]

        fg_soft_sim_prob = None
        # =======================================================
        if foreground_mask is not None:
            mask_flat = foreground_mask.reshape(-1).bool() # [N]
            
            
            neg_inf = -1e9
            
            
            # sim_flat[mask_flat, self.num_foreground:] = neg_inf
            
            
            
            proto_indices = torch.arange(K, device=device).unsqueeze(0)
            
            
            
            
            
            
            
            is_fg_pixel = mask_flat.unsqueeze(1)    # [N, 1]
            is_bg_proto = proto_indices >= self.num_foreground # [1, K]
            is_fg_proto = proto_indices < self.num_foreground  # [1, K]
            
            
            
            mask_fg_to_bg = is_fg_pixel & is_bg_proto
            sim_flat = sim_flat.masked_fill(mask_fg_to_bg, neg_inf)
            
            
            mask_bg_to_fg = (~is_fg_pixel) & is_fg_proto
            sim_flat = sim_flat.masked_fill(mask_bg_to_fg, neg_inf)

        # =======================================================

        # The differentiable assignment must be computed after applying the
        # foreground/background constraints.  Otherwise the hard forward pass
        # is masked while the STE backward pass (and the utilization entropy
        # loss) still distributes probability mass over invalid prototypes.
        Q_soft_flat = F.softmax(sim_flat / self.temperature, dim=1)
        if foreground_mask is not None:
            fg_soft_sim_prob = Q_soft_flat[mask_flat]

        
        top2_sim, _ = torch.topk(sim_flat, k=2, dim=1)
        max_sim = top2_sim[:, 0]
        second_max_sim = top2_sim[:, 1]
        confidence_ratio = max_sim / (second_max_sim + 1e-6)
        
        
        high_conf_mask = confidence_ratio > confidence_threshold
        
        
        Q_hard_flat = torch.zeros_like(sim_flat)
        
        
        if high_conf_mask.any():
            high_conf_sim = sim_flat[high_conf_mask]
            high_conf_assign = torch.argmax(high_conf_sim, dim=1)
            
            one_hot = torch.zeros_like(high_conf_sim)
            one_hot.scatter_(1, high_conf_assign.unsqueeze(1), 1.0)
            Q_hard_flat[high_conf_mask] = one_hot
        
        
        low_conf_mask = ~high_conf_mask
        if low_conf_mask.any():
            low_conf_sim = sim_flat[low_conf_mask]

            actual_k = min(topk, K)
            topk_val, topk_idx = torch.topk(low_conf_sim, k=actual_k, dim=1)
            
            scaled = topk_val / self.temperature
            scaled = scaled - scaled.max(dim=1, keepdim=True)[0]
            prob_topk = F.softmax(scaled, dim=1)
            
            sparse_assign = torch.zeros_like(low_conf_sim, dtype=prob_topk.dtype, device=prob_topk.device)
            sparse_assign.scatter_(1, topk_idx, prob_topk)
            sparse_assign = sparse_assign.to(dtype=Q_hard_flat.dtype, device=Q_hard_flat.device)
            Q_hard_flat[low_conf_mask] = sparse_assign
        
        # Use the differentiable constrained assignment in the actual graph.
        # The previous STE exposed a top-k/hard Q in the forward pass while the
        # utilization regularizer optimized a different dense soft Q.  It was
        # therefore possible to have high soft entropy but only 1--3 prototypes
        # participating in the graph.  Keep the sparse assignment solely as a
        # diagnostic argmax/confidence view.
        Q_final_flat = Q_soft_flat
        
        
        Q = Q_final_flat.reshape(B, D, H, W, K).permute(0, 4, 1, 2, 3)
        hard_assignment = Q_hard_flat.argmax(dim=1).reshape(B, D, H, W)
        confidence_mask = high_conf_mask.reshape(B, D, H, W)
        
        if fg_soft_sim_prob is not None:
            return {
                'Q': Q,
                'hard_assignment': hard_assignment,
                'confidence_mask': confidence_mask,
                'confidence_ratio': confidence_ratio.reshape(B, D, H, W),
                'fg_soft_sim_prob': fg_soft_sim_prob
            }
        else:
            return {
                'Q': Q,
                'hard_assignment': hard_assignment,
                'confidence_mask': confidence_mask,
                'confidence_ratio': confidence_ratio.reshape(B, D, H, W),
            }


    @torch.no_grad()
    def update_explicit_prototypes(
            self,
            skeleton: torch.Tensor,  
            Q: torch.Tensor,  # [B, K, D, H, W]
            hard_assignment: torch.Tensor,  # [B, D, H, W]
    ):
        """Update spatial prototypes with assignment-weighted patches."""
        B, _, D, H, W = skeleton.shape
        S = self.prototype_size
        padding = S // 2

        for k in range(self.num_foreground):
            
            mask_k = (hard_assignment == k).float().unsqueeze(1)  # [B, 1, D, H, W]

            
            weights = Q[:, k:k + 1] * mask_k  # [B, 1, D, H, W]
            total_weight = weights.sum() + 1e-6

            if total_weight < 1.0:
                continue  

            
            positions = torch.nonzero(mask_k.squeeze(1) > 0.5)  # [N, 4] (b, d, h, w)

            if len(positions) == 0:
                continue

            
            num_samples = min(len(positions), 100)
            indices = torch.randperm(len(positions))[:num_samples]
            sampled_positions = positions[indices]

            
            patches = []
            for pos in sampled_positions:
                b, d, h, w = pos.tolist()
                patch = self._extract_patch(skeleton[b, 0], d, h, w, S)
                if patch is not None:
                    patches.append(patch)

            if len(patches) > 0:
                mean_patch = torch.stack(patches).mean(dim=0)

                
                self.explicit_prototypes[k] = (
                        self.ema_decay * self.explicit_prototypes[k] +
                        (1 - self.ema_decay) * mean_patch.unsqueeze(0)
                )

    def _extract_patch(
            self,
            volume: torch.Tensor,  # [D, H, W]
            d: int, h: int, w: int,
            size: int,
    ) -> Optional[torch.Tensor]:
        """Extract a centered spatial patch with boundary padding."""
        D, H, W = volume.shape
        half = size // 2

        
        d_start, d_end = d - half, d + half + 1
        h_start, h_end = h - half, h + half + 1
        w_start, w_end = w - half, w + half + 1

        if d_start < 0 or d_end > D or h_start < 0 or h_end > H or w_start < 0 or w_end > W:
            
            patch = torch.zeros(1, size, size, size, device=volume.device)

            
            src_d = slice(max(0, d_start), min(D, d_end))
            src_h = slice(max(0, h_start), min(H, h_end))
            src_w = slice(max(0, w_start), min(W, w_end))

            dst_d = slice(max(0, -d_start), size - max(0, d_end - D))
            dst_h = slice(max(0, -h_start), size - max(0, h_end - H))
            dst_w = slice(max(0, -w_start), size - max(0, w_end - W))

            patch[0, dst_d, dst_h, dst_w] = volume[src_d, src_h, src_w]
            return patch
        else:
            return volume[d_start:d_end, h_start:h_end, w_start:w_end].unsqueeze(0)

    @torch.no_grad()
    def update_statistics(
            self,
            hard_assignment: torch.Tensor,  # [B, D, H, W]
            foreground_mask: torch.Tensor,  # [B, D, H, W] bool
    ):
        """Update prototype assignment statistics."""
        
        valid_assignments = hard_assignment[foreground_mask]

        for k in range(self.num_prototypes):
            self.hit_counts[k] += (valid_assignments == k).sum()

        self.total_counts += foreground_mask.sum()

    def get_prototype_utilization(self) -> torch.Tensor:
        """Return normalized prototype utilization."""
        if self.total_counts > 0:
            return self.hit_counts / self.total_counts
        else:
            return torch.zeros_like(self.hit_counts)
