"""Graph construction, topology prediction, and point supervision."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class GraphConstructor(nn.Module):
    """Construct dense 26-neighbor graphs from assignments and port probabilities."""

    def __init__(
        self,
        num_prototypes: int = 16,
        connectivity: int = 26,
    ):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.connectivity = connectivity

        
        offsets = self._generate_offsets(connectivity)
        self.register_buffer('neighbor_offsets', offsets)
        self.num_neighbors = len(offsets)

        
        kernel = self._create_neighbor_kernel(offsets)
        self.register_buffer('neighbor_kernel', kernel)

        
        reverse_idx = self._compute_reverse_indices(offsets)
        self.register_buffer('reverse_indices', reverse_idx)

    def _generate_offsets(self, connectivity: int) -> torch.Tensor:
        offsets = []
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    dist = abs(dz) + abs(dy) + abs(dx)
                    if connectivity == 6 and dist > 1:
                        continue
                    if connectivity == 18 and dist > 2:
                        continue
                    offsets.append([dz, dy, dx])
        return torch.tensor(offsets, dtype=torch.long)

    def _create_neighbor_kernel(self, offsets: torch.Tensor) -> torch.Tensor:
        N = len(offsets)
        kernel = torch.zeros(N, 1, 3, 3, 3)
        for i, off in enumerate(offsets):
            z, y, x = off[0] + 1, off[1] + 1, off[2] + 1
            kernel[i, 0, z, y, x] = 1.0
        return kernel

    def _compute_reverse_indices(self, offsets: torch.Tensor) -> torch.Tensor:
        """Return the reverse direction index for every offset."""
        N = len(offsets)
        reverse_idx = torch.zeros(N, dtype=torch.long)

        for i, off in enumerate(offsets):
            rev_off = -off
            for j, other in enumerate(offsets):
                if torch.equal(other, rev_off):
                    reverse_idx[i] = j
                    break

        return reverse_idx

    def get_neighbor_Q(self, Q: torch.Tensor) -> torch.Tensor:
        """Gather neighboring prototype assignments in every direction."""
        B, K, D, H, W = Q.shape
        N = self.num_neighbors

        Q_flat = Q.reshape(B * K, 1, D, H, W)
        Q_neighbors_flat = F.conv3d(Q_flat, self.neighbor_kernel, padding=1)
        Q_neighbors = Q_neighbors_flat.reshape(B, K, N, D, H, W)

        return Q_neighbors

    @torch.no_grad()
    def build_dense_graph(
        self,
        Q: torch.Tensor,  
        port_probs: torch.Tensor,  
    ) -> torch.Tensor:
        """Build a detached dense graph for shortest-path search."""
        B, K, D, H, W = Q.shape
        N = self.num_neighbors
        device = Q.device

        # port_probs: [K, 26]
        
        port_reverse = port_probs[:, self.reverse_indices]  # [K, 26]

        
        # Q: [B, K, D, H, W], port_probs: [K, 26]
        
        w_forward = torch.einsum('bkdhw,kn->bndhw', Q, port_probs)

        
        Q_neighbors = self.get_neighbor_Q(Q)

        
        # Q_neighbors: [B, K, 26, D, H, W], port_reverse: [K, 26]
        
        w_backward = torch.einsum('bkndhw,kn->bndhw', Q_neighbors, port_reverse)

        
        # edge_weights = w_forward * w_backward  # [B, 26, D, H, W]
        # edge_weights = (w_forward + w_backward) / 2  # [B, 26, D, H, W]
        edge_weights = torch.sqrt(w_forward * w_backward + 1e-6)  # [B, 26, D, H, W]

        
        edge_weights = edge_weights.permute(0, 2, 3, 4, 1).contiguous()

        return edge_weights

    def build_graph_with_grad(
        self,
        Q: torch.Tensor,  
        port_probs: torch.Tensor,  
    ) -> torch.Tensor:
        """Build differentiable dense edge weights."""
        B, K, D, H, W = Q.shape
        device = Q.device

        # port_reverse: [K, 26]
        port_reverse = port_probs[:, self.reverse_indices]

        # w_forward: [B, 26, D, H, W]
        w_forward = torch.einsum('bkdhw,kn->bndhw', Q, port_probs)

        # Q_neighbors: [B, K, 26, D, H, W]
        Q_neighbors = self.get_neighbor_Q(Q)

        # w_backward: [B, 26, D, H, W]
        w_backward = torch.einsum('bkndhw,kn->bndhw', Q_neighbors, port_reverse)

        
        # edge_weights = w_forward * w_backward
        # edge_weights = (w_forward + w_backward) / 2
        edge_weights = torch.sqrt(w_forward * w_backward + 1e-6)
        edge_weights = edge_weights.permute(0, 2, 3, 4, 1).contiguous()

        return edge_weights

    def compute_edge_weight_at(
        self,
        Q: torch.Tensor,  
        port_probs: torch.Tensor,  
        coords: torch.Tensor,  # [M, 4] (batch_idx, z, y, x)
        neighbor_indices: torch.Tensor,  
    ) -> torch.Tensor:
        """Compute directional edge weights at selected coordinates."""
        M = coords.shape[0]
        B, K, D, H, W = Q.shape
        device = Q.device

        
        batch_idx = coords[:, 0]
        z = coords[:, 1]
        y = coords[:, 2]
        x = coords[:, 3]

        Q_source = Q[batch_idx, :, z, y, x]  # [M, K]

        
        offsets = self.neighbor_offsets[neighbor_indices]  # [M, 3]
        z_n = (z + offsets[:, 0]).clamp(0, D - 1)
        y_n = (y + offsets[:, 1]).clamp(0, H - 1)
        x_n = (x + offsets[:, 2]).clamp(0, W - 1)

        
        Q_neighbor = Q[batch_idx, :, z_n, y_n, x_n]

        
        # port_probs: [K, 26], neighbor_indices: [M]
        port_forward = port_probs[:, neighbor_indices].T  # [M, K]

        
        reverse_neighbor_indices = self.reverse_indices[neighbor_indices]  # [M]
        port_backward = port_probs[:, reverse_neighbor_indices].T  # [M, K]

        # w_forward = Σ_k Q_k(u) * port_k(δ)
        w_forward = (Q_source * port_forward).sum(dim=1)  # [M]

        # w_backward = Σ_l Q_l(v) * port_l(-δ)
        w_backward = (Q_neighbor * port_backward).sum(dim=1)  # [M]

        
        # edge_weights = w_forward * w_backward  # [M]
        # edge_weights = (w_forward + w_backward) / 2  # [M]
        edge_weights = torch.sqrt(w_forward * w_backward + 1e-6)

        return edge_weights


# ============================================================================

# ============================================================================


class TopologicalPredictor(nn.Module):
    """Predict degree classes and directional port probabilities from features."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        num_directions: int = 26,
        peakness_alpha: float = 2.0,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_directions = num_directions
        self.peakness_alpha = peakness_alpha

        
        self.shared_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        
        self.degree_head = nn.Linear(hidden_dim, 4)

        
        self.port_head = nn.Linear(hidden_dim, num_directions)

        
        self.register_buffer('angular_neighbors', self._build_angular_neighbors())

    def _build_angular_neighbors(self) -> torch.Tensor:
        """Build angular neighborhoods for directional smoothing."""
        directions = []
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    directions.append([dz, dy, dx])

        directions = np.array(directions, dtype=np.float32)
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions_norm = directions / norms

        cos_sim = np.dot(directions_norm, directions_norm.T)
        threshold = 0.5

        max_neighbors = 12
        angular_neighbors = np.full((26, max_neighbors), -1, dtype=np.int64)

        for i in range(26):
            neighbors = []
            for j in range(26):
                if i != j and cos_sim[i, j] > threshold:
                    neighbors.append(j)
            for k, n in enumerate(neighbors[:max_neighbors]):
                angular_neighbors[i, k] = n

        return torch.from_numpy(angular_neighbors)

    def compute_peakness(self, activations: torch.Tensor) -> torch.Tensor:
        """Measure concentration of directional port probabilities."""
        K, N = activations.shape
        device = activations.device

        neighbors = self.angular_neighbors  # [26, max_neighbors]

        
        neighbor_weights = torch.zeros(N, N, device=device)
        for i in range(N):
            valid_mask = neighbors[i] >= 0
            valid_neighbors = neighbors[i][valid_mask]
            num_valid = len(valid_neighbors)
            if num_valid > 0:
                neighbor_weights[i, valid_neighbors] = 1.0 / num_valid

        
        neighbor_mean = torch.matmul(activations, neighbor_weights.T)  # [K, 26]

        
        peakness = activations - neighbor_mean

        return peakness

    def forward(self, prototypes: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict topology logits and calibrated port probabilities."""
        
        shared = self.shared_mlp(prototypes)  # [K, hidden_dim]

        
        degree_logits = self.degree_head(shared)  # [K, 4]

        
        port_raw = self.port_head(shared)  # [K, 26]

        # Soft NMS: peakness score
        peakness = self.compute_peakness(port_raw)

        
        port_probs = torch.sigmoid(port_raw) * torch.sigmoid(self.peakness_alpha * peakness)

        # The graph is allowed a port-probability budget equal to the expected
        # degree.  This removes the shortcut in which a geodesic loss lowers
        # many edge costs by broadening all 26 ports simultaneously.  Degree
        # is detached here on purpose: it is learned by its explicit degree
        # supervision, while path gradients only redistribute the port budget.
        degree_values = torch.arange(
            degree_logits.shape[-1],
            device=degree_logits.device,
            dtype=degree_logits.dtype,
        )
        expected_degree = (
            F.softmax(degree_logits, dim=-1) * degree_values
        ).sum(dim=-1)
        raw_port_mass = port_probs.sum(dim=-1)
        graph_port_scale = (
            expected_degree.detach()
            / raw_port_mass.clamp_min(1e-6)
        ).clamp(max=1.0)
        graph_port_probs = port_probs * graph_port_scale.unsqueeze(-1)

        return {
            'degree_logits': degree_logits,  # [K, 4]
            'expected_degree': expected_degree,
            'port_probs': port_probs,        # supervised, unnormalized [K, 26]
            'graph_port_probs': graph_port_probs,  # graph-only [K, 26]
            'graph_port_scale': graph_port_scale,
            'port_logits': port_raw,          # [K, 26]
            'peakness': peakness,            # [K, 26]
        }


# ============================================================================

# ============================================================================


class TopologySupervisionLoss(nn.Module):
    def __init__(
        self,
        num_prototypes: int = 64,
        num_foreground: int = 60,
        lambda_degree: float = 1.0,
        lambda_port: float = 1.0,
        lambda_port_rank: float = 0.5,
        lambda_consistency: float = 0.1,
        lambda_sparsity: float = 0.5,
    ):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.num_foreground = num_foreground 
        self.lambda_degree = lambda_degree
        self.lambda_port = lambda_port
        self.lambda_port_rank = lambda_port_rank
        self.lambda_consistency = lambda_consistency
        self.lambda_sparsity = lambda_sparsity
        class_weights = torch.tensor([1.0, 5.0, 1.0, 10.0]) 
        self.register_buffer('class_weights', class_weights)
        self.register_buffer(
            'last_sampled_degree_histogram',
            torch.zeros(4, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            'last_topology_contrast_metrics',
            torch.zeros(4, dtype=torch.float32),
            persistent=False,
        )

    def forward(
    self,
    features: torch.Tensor,           # [B, C, D', H', W']
    mlp: torch.nn.Module,
    Q: torch.Tensor,
    prototype_topo_pred,
    num_fg_prototype,
    degree_gt: torch.Tensor,          # [B, D, H, W]
    neighbor_mask_gt: torch.Tensor,   # [B, D, H, W, 26]
    skeleton: torch.Tensor,           
    num_samples: int = 256,
    search_radius: int = 2,
    ) -> Dict[str, torch.Tensor]:
        
        B, C, D_feat, H_feat, W_feat = features.shape
        D_orig, H_orig, W_orig = degree_gt.shape[1:]
        device = features.device

        
        stride_d = D_orig // D_feat
        stride_h = H_orig // H_feat
        stride_w = W_orig // W_feat

        if skeleton.dim() == 5:
            skeleton = skeleton.squeeze(1)

        
        fg_mask = skeleton > 0.5
        fg_coords = torch.nonzero(fg_mask)
        self.last_sampled_degree_histogram.zero_()
        self.last_topology_contrast_metrics.zero_()

        if fg_coords.shape[0] < 2:
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return {
                'loss_degree_fg': zero_loss, 'loss_port_fg': zero_loss,
                'loss_consistency_fg': zero_loss, 'loss_sparsity_fg': zero_loss,
                'loss_degree_bg': zero_loss, 'loss_port_bg': zero_loss,
                'loss_consistency_bg': zero_loss, 'loss_sparsity_bg': zero_loss,
                'loss_degree': zero_loss, 'loss_port': zero_loss,
                'loss_consistency': zero_loss, 'loss_sparsity': zero_loss,
                'loss_topology_feature': zero_loss,
            }
        
        # Map feature-grid skeleton points to original-space topology labels.
        b_all, z_all, y_all, x_all = fg_coords.T
        skeleton_gt = (degree_gt > 0).float()
        z_orig, y_orig, x_orig, valid_mask = self.find_nearest_skeleton_point(
            z_all * stride_d + stride_d // 2,
            y_all * stride_h + stride_h // 2,
            x_all * stride_w + stride_w // 2,
            b_all,
            skeleton_gt,
            search_radius=search_radius,
        )

        
        if valid_mask.sum() < 2:
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return {
                'loss_degree_fg': zero_loss, 'loss_port_fg': zero_loss,
                'loss_consistency_fg': zero_loss, 'loss_sparsity_fg': zero_loss,
                'loss_degree_bg': zero_loss, 'loss_port_bg': zero_loss,
                'loss_consistency_bg': zero_loss, 'loss_sparsity_bg': zero_loss,
                'loss_degree': zero_loss, 'loss_port': zero_loss,
                'loss_consistency': zero_loss, 'loss_sparsity': zero_loss,
                'loss_topology_feature': zero_loss,
            }

        valid_coords = fg_coords[valid_mask]
        b_valid = b_all[valid_mask]
        z_orig = z_orig[valid_mask]
        y_orig = y_orig[valid_mask]
        x_orig = x_orig[valid_mask]
        degree_valid = degree_gt[
            b_valid, z_orig, y_orig, x_orig
        ].long().clamp(max=3)
        # Uniform foreground sampling without degree quotas or replacement.
        sample_count = min(degree_valid.numel(), B * num_samples)
        selected = torch.randperm(
            degree_valid.numel(), device=device
        )[:sample_count]
        fg_sampled = valid_coords[selected]
        b_fg = b_valid[selected]
        z_orig = z_orig[selected]
        y_orig = y_orig[selected]
        x_orig = x_orig[selected]
        target_degree_fg = degree_valid[selected]
        self.last_sampled_degree_histogram.copy_(
            torch.bincount(target_degree_fg, minlength=4)[:4]
        )

        z_fg, y_fg, x_fg = fg_sampled[:, 1:].T
        feat_fg = F.normalize(
            features[b_fg, :, z_fg, y_fg, x_fg], dim=1
        )
        Q_fg = Q[b_fg, :, z_fg, y_fg, x_fg]
        N_fg = feat_fg.shape[0]

        
        target_port_fg = neighbor_mask_gt[b_fg, z_orig, y_orig, x_orig].float()

        loss_topology_feature = self.topology_feature_contrastive_loss(
            feat_fg, target_port_fg, target_degree_fg, selected
        )

        
        pred_fg = mlp(feat_fg)
        degree_logits_fg = pred_fg['degree_logits']

        
        target_degree_fg = target_degree_fg.clamp(max=degree_logits_fg.shape[1] - 1)
        # Class weights affect the loss only; sampling remains uniform.
        weights_fg = self.class_weights[target_degree_fg].sqrt()

        
        with torch.amp.autocast('cuda', enabled=False):
            # Degree Loss
            loss_degree_fg = F.cross_entropy(
                degree_logits_fg.float(), 
                target_degree_fg,
                weight=self.class_weights
            )
            
            # Balanced focal port loss.  Normalize positive and negative
            # directions separately so the roughly 24 negatives cannot be
            # minimized simply by driving every direction toward zero.
            # Supervise the probability that is actually consumed by the
            # graph.  Supervising only sigmoid(port_logits) leaves the
            # multiplicative soft-NMS peakness term unconstrained.
            p = pred_fg['port_probs'].float().clamp(1e-6, 1.0 - 1e-6)
            loss_port_per_sample = F.binary_cross_entropy(
                p,
                target_port_fg,
                reduction='none'
            )
            positive = target_port_fg
            negative = 1.0 - target_port_fg
            positive_loss = (
                (1.0 - p).pow(2) * loss_port_per_sample * positive
            ).sum(dim=1) / positive.sum(dim=1).clamp_min(1.0)
            negative_loss = (
                p.pow(2) * loss_port_per_sample * negative
            ).sum(dim=1) / negative.sum(dim=1).clamp_min(1.0)
            # Directly optimize the ranking used by degree-conditioned top-k
            # graph ports: every positive direction should outrank negatives.
            effective_logits = torch.logit(p)
            pairwise_mask = positive.unsqueeze(2) * negative.unsqueeze(1)
            pairwise_rank = F.softplus(
                effective_logits.unsqueeze(1)
                - effective_logits.unsqueeze(2)
                + 0.2
            )
            rank_loss = (
                pairwise_rank * pairwise_mask
            ).sum(dim=(1, 2)) / pairwise_mask.sum(dim=(1, 2)).clamp_min(1.0)
            loss_port_fg = (
                (
                    positive_loss
                    + negative_loss
                    + self.lambda_port_rank * rank_loss
                ) * weights_fg
            ).sum() / weights_fg.sum().clamp_min(1e-6)

        # Consistency Loss
        pred_degree_prob_fg = F.softmax(degree_logits_fg, dim=1)
        vals = torch.tensor([0., 1., 2., 3.], device=device)
        expected_degree_fg = (pred_degree_prob_fg * vals).sum(dim=1)
        port_probs_fg = pred_fg['port_probs']
        port_sum_fg = port_probs_fg.sum(dim=1)
        loss_consistency_fg = F.smooth_l1_loss(
            port_sum_fg, expected_degree_fg
        )

        # Sparsity Loss
        # loss_sparsity_fg = port_probs_fg.mean()
        loss_sparsity_fg = self.threshold_sparsity_loss(port_probs_fg, 0.1)

        
        with torch.no_grad():
            proto_evidence = Q_fg.sum(dim=0) 
            proto_target_degree = torch.matmul(Q_fg.T, target_degree_fg.float()) / (proto_evidence + 1e-6)
            proto_target_port = torch.matmul(Q_fg.T, target_port_fg) / (proto_evidence.unsqueeze(1) + 1e-6)

        proto_port_probs = prototype_topo_pred['port_probs']
        proto_degree_logits = prototype_topo_pred['degree_logits']

        num_prototype = proto_port_probs.shape[0]

        # 
        proto_degree_prob = F.softmax(proto_degree_logits, dim=1)
        vals = torch.tensor([0., 1., 2., 3.], device=proto_degree_logits.device)
        expected_degree = (proto_degree_prob * vals).sum(dim=1)

        
        target_degree_bg = torch.zeros(num_prototype - num_fg_prototype, dtype=torch.long, device=device)
        target_port_bg = torch.zeros(num_prototype - num_fg_prototype, 26, device=device)
        with torch.amp.autocast('cuda', enabled=False):
            # Degree Loss
            loss_degree_bg_p = F.cross_entropy(
                proto_degree_logits[num_fg_prototype:].float(),
                target_degree_bg,
                weight=self.class_weights
            )
            
            # Port Loss
            loss_port_bg_p = F.binary_cross_entropy(
                proto_port_probs[num_fg_prototype:].float().clamp(
                    1e-6, 1.0 - 1e-6
                ),
                target_port_bg,
                reduction='mean'
            )

        
        # loss_proto_degree_sup = (expected_degree - proto_target_degree[:num_fg_prototype,]).pow(2)
        
        
        # if valid_proto_mask.any():
        #     loss_degree_p = (loss_proto_degree_sup * proto_evidence).sum() / (proto_evidence.sum() + 1e-6)
        # else:
        #     loss_degree_p = torch.tensor(0.0, device=device, requires_grad=True)

        
        # loss_proto_port_sup = F.binary_cross_entropy(
        #     port_probs.clamp(1e-6, 1-1e-6), 
        #     proto_target_port.clamp(0, 1),
        #     reduction='none'
        # ).sum(dim=1) # [K]
        
        # if valid_proto_mask.any():
        #     loss_proto_port = (loss_proto_port_sup * proto_evidence).sum() / (proto_evidence.sum() + 1e-6)
        # else:
        #     loss_proto_port = torch.tensor(0.0, device=device, requires_grad=True)

        
        port_sum = proto_port_probs.sum(dim=1)
        loss_consistency_p = 3 * F.smooth_l1_loss(port_sum, expected_degree)
        # loss_sparsity_p = proto_port_probs.mean()
        loss_sparsity_p = self.threshold_sparsity_loss(proto_port_probs, 0.1)

        
        bg_mask = skeleton < 0.5
        bg_coords = torch.nonzero(bg_mask)

        if bg_coords.shape[0] < 2:
            loss_degree_bg = torch.tensor(0.0, device=device, requires_grad=True)
            loss_port_bg = torch.tensor(0.0, device=device, requires_grad=True)
            loss_consistency_bg = torch.tensor(0.0, device=device, requires_grad=True)
            loss_sparsity_bg = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            
            num_bg = min(bg_coords.shape[0], N_fg//4)
            perm = torch.randperm(bg_coords.shape[0], device=device)[:num_bg]
            bg_sampled = bg_coords[perm]

            b_bg = bg_sampled[:, 0]
            z_bg = bg_sampled[:, 1]
            y_bg = bg_sampled[:, 2]
            x_bg = bg_sampled[:, 3]

            
            feat_bg = features[b_bg, :, z_bg, y_bg, x_bg]
            feat_bg = F.normalize(feat_bg, dim=1)
            N_bg = feat_bg.shape[0]

            
            target_degree_bg = torch.zeros(N_bg, dtype=torch.long, device=device)
            target_port_bg = torch.zeros(N_bg, 26, device=device)

            
            pred_bg = mlp(feat_bg)
            degree_logits_bg = pred_bg['degree_logits']

            
            with torch.amp.autocast('cuda', enabled=False):
                # Degree Loss
                loss_degree_bg = F.cross_entropy(
                    degree_logits_bg.float(),
                    target_degree_bg,
                    weight=self.class_weights
                )
                
                # Port Loss
                loss_port_bg = F.binary_cross_entropy(
                    pred_bg['port_probs'].float().clamp(1e-6, 1.0 - 1e-6),
                    target_port_bg,
                    reduction='mean'
                )

            
            port_probs_bg = pred_bg['port_probs']
            port_sum_bg = port_probs_bg.sum(dim=1)
            expected_degree_bg = torch.zeros(N_bg, device=device)
            loss_consistency_bg = F.smooth_l1_loss(
                port_sum_bg, expected_degree_bg
            )

            # Sparsity Loss
            loss_sparsity_bg = port_probs_bg.mean()

        
        loss_degree = loss_degree_fg + loss_degree_bg + loss_degree_bg_p
        loss_port = loss_port_fg + loss_port_bg + loss_port_bg_p
        loss_consistency = loss_consistency_fg + loss_consistency_bg + loss_consistency_p
        loss_sparsity = loss_sparsity_fg + loss_sparsity_bg + loss_sparsity_p

        return {
            
            'loss_degree_fg': self.lambda_degree * loss_degree_fg,
            'loss_port_fg': self.lambda_port * loss_port_fg,
            'loss_consistency_fg': self.lambda_consistency * loss_consistency_fg,
            'loss_sparsity_fg': self.lambda_sparsity * loss_sparsity_fg,
            
            
            'loss_degree_bg': self.lambda_degree * loss_degree_bg,
            'loss_port_bg': self.lambda_port * loss_port_bg,
            'loss_consistency_bg': self.lambda_consistency * loss_consistency_bg,
            'loss_sparsity_bg': self.lambda_sparsity * loss_sparsity_bg,
            
            
            'loss_degree': self.lambda_degree * loss_degree,
            'loss_port': self.lambda_port * loss_port,
            'loss_consistency': self.lambda_consistency * loss_consistency,
            'loss_sparsity': self.lambda_sparsity * loss_sparsity,
            'loss_topology_feature': loss_topology_feature,
        }

    def topology_feature_contrastive_loss(
        self,
        features: torch.Tensor,
        target_ports: torch.Tensor,
        target_degrees: torch.Tensor,
        source_indices: torch.Tensor,
        negative_margin: float = 0.2,
        max_hard_negatives: int = 1024,
    ) -> torch.Tensor:
        """Contrast topology features using only current-patch pairs."""
        del target_degrees
        with torch.amp.autocast('cuda', enabled=False):
            feature_vectors = F.normalize(features.float(), dim=1)
            ports = target_ports.float()
            feature_similarity = feature_vectors @ feature_vectors.T
            port_intersection = ports @ ports.T
            port_difference = (
                ports.unsqueeze(1) - ports.unsqueeze(0)
            ).abs().sum(dim=-1)
            num_points = feature_vectors.shape[0]
            upper = torch.triu(
                torch.ones(
                    num_points, num_points, dtype=torch.bool,
                    device=features.device,
                ),
                diagonal=1,
            )
            distinct = source_indices[:, None] != source_indices[None, :]
            valid_pairs = upper & distinct
            positive_mask = valid_pairs & (port_difference == 0)
            negative_mask = valid_pairs & (port_intersection == 0)
            positive_similarity = feature_similarity[positive_mask]
            negative_similarity = feature_similarity[negative_mask]
            zero = feature_similarity.sum() * 0.0
            positive_loss = (
                (1.0 - positive_similarity).mean()
                if positive_similarity.numel() > 0 else zero
            )
            if negative_similarity.numel() > 0:
                hard_count = min(max_hard_negatives, negative_similarity.numel())
                hard_negatives = torch.topk(
                    negative_similarity, k=hard_count
                ).values
                negative_loss = F.relu(
                    hard_negatives - negative_margin
                ).pow(2).mean()
            else:
                negative_loss = zero
            self.last_topology_contrast_metrics.copy_(torch.tensor([
                float(positive_similarity.numel()),
                float(negative_similarity.numel()),
                float(positive_similarity.detach().mean())
                if positive_similarity.numel() > 0 else 0.0,
                float(negative_similarity.detach().mean())
                if negative_similarity.numel() > 0 else 0.0,
            ], device=self.last_topology_contrast_metrics.device))
            return positive_loss + negative_loss

    def threshold_sparsity_loss(self, probs, threshold=0.1):
        del threshold
        # Encourage confident (near 0 or 1) ports.  The previous expression
        # relu(0.1 - p) did the opposite for absent directions by penalizing
        # probabilities below 0.1.
        return (probs * (1.0 - probs)).mean()

    def find_nearest_skeleton_point(
        self,
        z_orig: torch.Tensor,
        y_orig: torch.Tensor,
        x_orig: torch.Tensor,
        b_idx: torch.Tensor,
        skeleton_gt: torch.Tensor,
        search_radius: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Find the nearest skeleton point for each coordinate."""
        device = z_orig.device
        N = z_orig.shape[0]
        B, D, H, W = skeleton_gt.shape

        
        offsets = []
        for dz in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                for dx in range(-search_radius, search_radius + 1):
                    offsets.append([dz, dy, dx])
        offsets = torch.tensor(offsets, device=device, dtype=torch.long)
        dists = (offsets.float() ** 2).sum(dim=1).sqrt()
        sorted_idx = torch.argsort(dists)
        offsets = offsets[sorted_idx]  
        K = offsets.shape[0]

        
        z_search = z_orig.unsqueeze(1) + offsets[:, 0].unsqueeze(0)  # [N, K]
        y_search = y_orig.unsqueeze(1) + offsets[:, 1].unsqueeze(0)
        x_search = x_orig.unsqueeze(1) + offsets[:, 2].unsqueeze(0)

        
        z_search = z_search.clamp(0, D - 1)
        y_search = y_search.clamp(0, H - 1)
        x_search = x_search.clamp(0, W - 1)

        
        b_expand = b_idx.unsqueeze(1).expand(-1, K)

        
        skel_vals = skeleton_gt[b_expand, z_search, y_search, x_search]

        
        is_valid = skel_vals > 0.5  # [N, K]
        
        
        
        first_valid_idx = is_valid.float().argmax(dim=1)  # [N]
        has_valid = is_valid.any(dim=1)  # [N]

        
        batch_indices = torch.arange(N, device=device)
        z_new = z_search[batch_indices, first_valid_idx]
        y_new = y_search[batch_indices, first_valid_idx]
        x_new = x_search[batch_indices, first_valid_idx]

        return z_new, y_new, x_new, has_valid
