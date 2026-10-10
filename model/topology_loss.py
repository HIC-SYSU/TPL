"""Topology-aware path and edge-cost objectives."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from .graph import GraphConstructor
# from .path_sampler import PathSampler
from .path_sampler import BottleneckPathSampler


class SkeletonGraph:
    """Vectorized sparse graph representation of a skeleton."""

    def __init__(self, skeleton: np.ndarray, connectivity: int = 26):
        self.shape = skeleton.shape
        D, H, W = self.shape
        
        self.skel_coords = np.argwhere(skeleton > 0.5)
        self.n_nodes = len(self.skel_coords)

        if self.n_nodes == 0:
            self.valid = False
            return

        self.valid = True
        self.kdtree = cKDTree(self.skel_coords)
        
        
        self.neighbor_offsets = self._generate_offsets(connectivity)
        self.offset_distances = np.linalg.norm(self.neighbor_offsets, axis=1)
        
        
        self.linear_to_skel = np.full(D * H * W, -1, dtype=np.int32)
        skel_linear = self.skel_coords[:, 0] * H * W + self.skel_coords[:, 1] * W + self.skel_coords[:, 2]
        self.linear_to_skel[skel_linear] = np.arange(self.n_nodes)
        
        self.graph = self._build_graph_vectorized()
        
        
        self._precompute = self.n_nodes <= 5000
        self.geodesic_matrix = dijkstra(self.graph, directed=False) if self._precompute else None

    def _generate_offsets(self, connectivity: int) -> np.ndarray:
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
        return np.array(offsets)

    def _build_graph_vectorized(self) -> csr_matrix:
        D, H, W = self.shape
        n_offsets = len(self.neighbor_offsets)
        
        # [N, n_offsets, 3]
        all_neighbors = self.skel_coords[:, None, :] + self.neighbor_offsets[None, :, :]
        nz, ny, nx = all_neighbors[..., 0], all_neighbors[..., 1], all_neighbors[..., 2]
        
        valid = (nz >= 0) & (nz < D) & (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
        neighbor_linear = np.where(valid, nz * H * W + ny * W + nx, 0).astype(np.int64)
        neighbor_skel = self.linear_to_skel[neighbor_linear]
        
        edge_mask = valid & (neighbor_skel >= 0)
        src, dir_idx = np.where(edge_mask)
        tgt = neighbor_skel[edge_mask]
        weights = self.offset_distances[dir_idx]
        
        return csr_matrix((weights, (src, tgt)), shape=(self.n_nodes, self.n_nodes))

    def project_batch(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project coordinates onto their nearest skeleton nodes."""
        if not self.valid or len(points) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        return self.kdtree.query(points)[::-1]  # query returns (dist, idx)

    def get_geodesic_matrix_rows(self, source_indices: np.ndarray) -> np.ndarray:
        """Return cached geodesic distance rows."""
        if not self.valid:
            return np.array([])
        
        unique_sources = np.unique(source_indices)
        if self._precompute:
            return self.geodesic_matrix[unique_sources]
        else:
            return dijkstra(self.graph, directed=False, indices=unique_sources.tolist())


class TopologyAwareLoss(nn.Module):
    """Optimize predicted graph costs against skeleton geodesics."""

    def __init__(
        self,
        num_prototypes: int = 16,
        num_sample_pairs: int = 32,
        alpha: float = 3.0,
        lambda_shortcut: float = 1.0,
        lambda_spectral: float = 0.1,
        connectivity: int = 26,
        num_candidate_pairs: int = 128,
        nms_distance: float = 5.0,
        min_edge_conductivity: float = 0.02,
        max_conductivity_grad_norm: float = 2.0,
    ):
        super().__init__()

        self.num_sample_pairs = num_sample_pairs
        self.num_candidate_pairs = num_candidate_pairs
        self.nms_distance = nms_distance
        
        self.alpha = alpha
        self.lambda_shortcut = lambda_shortcut
        self.lambda_spectral = lambda_spectral
        self.connectivity = connectivity
        self.min_edge_conductivity = min_edge_conductivity
        self.max_conductivity_grad_norm = max_conductivity_grad_norm

        self.graph_constructor = GraphConstructor(
            num_prototypes=num_prototypes,
            connectivity=connectivity,
        )
        self.path_sampler = BottleneckPathSampler(
            num_sample_pairs=num_sample_pairs,
            num_candidate_pairs=num_candidate_pairs,
            nms_distance=nms_distance,
            connectivity=connectivity,
        )

        
        self.neighbor_offsets = self._generate_offsets(connectivity)
        self.n_offsets = len(self.neighbor_offsets)
        
        
        
        self.offset_to_idx = {}
        for i, off in enumerate(self.neighbor_offsets):
            key = (int(off[0]), int(off[1]), int(off[2]))
            self.offset_to_idx[key] = i
        
        
        
        self.offset_lookup = np.full(27, -1, dtype=np.int32)
        for i, off in enumerate(self.neighbor_offsets):
            code = (int(off[0]) + 1) * 9 + (int(off[1]) + 1) * 3 + (int(off[2]) + 1)
            self.offset_lookup[code] = i

    def _generate_offsets(self, connectivity: int) -> np.ndarray:
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
        return np.array(offsets, dtype=np.int32)

    def forward(
        self,
        Q: torch.Tensor,
        port_probs: torch.Tensor,
        skeleton_gt: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, K, D, H, W = Q.shape
        device = Q.device

        
        with torch.no_grad():
            edge_weights_dense = self.graph_constructor.build_dense_graph(Q, port_probs)

        
        path_pairs = self.path_sampler(edge_weights_dense, skeleton_gt)

        if len(path_pairs) == 0:
            zeros = torch.tensor(0.0, device=device)
            return {
                'loss_backbone': zeros,
                'loss_shortcut': zeros,
                'loss_local': zeros,
                'loss_spectral': zeros,
                'loss_topology': zeros,
            }

        
        skeleton_graphs = {
            b: SkeletonGraph(skeleton_gt[b, 0].cpu().numpy() > 0.5)
            for b in range(B)
        }

        
        reference_data, optimal_data = self._process_all_paths_vectorized(
            path_pairs, skeleton_graphs
        )

        
        #    c_e = ||e|| / (E_e + eps)
        #    c_e* = ||e|| (1 + eta(e))
        
        loss_backbone = self._edge_cost_regression_loss(
            reference_data, Q, port_probs, device
        )
        loss_shortcut = self._edge_cost_regression_loss(
            optimal_data, Q, port_probs, device
        )

        loss_shortcut = self.lambda_shortcut * loss_shortcut
        loss_local = loss_backbone + loss_shortcut

        # Spectral loss
        if self.lambda_spectral > 0:
            loss_spectral = self.lambda_spectral * self._compute_spectral_loss(Q, skeleton_gt)
        else:
            loss_spectral = torch.tensor(0.0, device=device)

        return {
            'loss_backbone': loss_backbone,
            'loss_shortcut': loss_shortcut,
            'loss_local': loss_local,
            'loss_spectral': loss_spectral,
            'loss_topology': loss_local + loss_spectral,
        }

    def _edge_cost_regression_loss(
        self,
        edge_data: Dict[str, np.ndarray],
        Q: torch.Tensor,
        port_probs: torch.Tensor,
        device: torch.device,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Regress selected edge costs to geodesic targets."""
        if edge_data['count'] == 0:
            return torch.tensor(0.0, device=device)

        # Disconnected skeleton components can produce infinite geodesic
        # excess. Such edges have no finite regression target and must not enter
        # the regression (clipping infinity would invent an arbitrary target).
        deltas_np = np.asarray(edge_data['deltas'])
        valid_np = np.isfinite(deltas_np) & (deltas_np >= 0)
        if not valid_np.any():
            return (Q.sum() + port_probs.sum()) * 0.0

        coords_t = torch.as_tensor(
            np.asarray(edge_data['coords'])[valid_np],
            device=device,
            dtype=torch.long,
        )
        neighbor_t = torch.as_tensor(
            np.asarray(edge_data['neighbors'])[valid_np],
            device=device,
            dtype=torch.long,
        )
        deltas_t = torch.as_tensor(
            deltas_np[valid_np], device=device, dtype=torch.float32
        )

        conductivity = self.graph_constructor.compute_edge_weight_at(
            Q, port_probs, coords_t, neighbor_t
        )
        if conductivity.requires_grad and self.max_conductivity_grad_norm > 0:
            max_norm = self.max_conductivity_grad_norm

            def _clip_conductivity_gradient(gradient: torch.Tensor) -> torch.Tensor:
                gradient_fp32 = torch.nan_to_num(
                    gradient.float(), nan=0.0, posinf=0.0, neginf=0.0
                )
                norm = torch.linalg.vector_norm(gradient_fp32)
                scale = torch.clamp(max_norm / norm.clamp_min(1e-12), max=1.0)
                return (gradient_fp32 * scale).to(gradient.dtype)

            conductivity.register_hook(_clip_conductivity_gradient)
        neighbor_offsets = torch.as_tensor(
            self.neighbor_offsets,
            device=device,
            dtype=torch.float32,
        )
        edge_lengths = torch.linalg.vector_norm(
            neighbor_offsets[neighbor_t], dim=1
        )

        with torch.amp.autocast('cuda', enabled=False):
            conductivity_fp32 = conductivity.float()
            # Straight-through lower bound: forward costs cannot become
            # singular, while low-conductivity edges still receive a gradient
            # that pushes them back into the well-conditioned region.
            conductivity_floor = conductivity_fp32.clamp_min(
                self.min_edge_conductivity
            )
            conductivity_safe = conductivity_fp32 + (
                conductivity_floor - conductivity_fp32
            ).detach()
            edge_costs = edge_lengths / (conductivity_safe + eps)
            target_costs = edge_lengths * (1.0 + deltas_t)
            return F.mse_loss(edge_costs, target_costs)
    
    def generalized_focal_loss(self, prediction, target, gamma=2.0, alpha=1.0):
        """Compute a focal-weighted loss for continuous targets."""
        
        bce_loss = F.binary_cross_entropy(prediction, target, reduction='none')
        
        
        
        modulating_factor = (prediction - target).abs().pow(gamma)
        
        
        focal_loss = alpha * modulating_factor * bce_loss
    
        return focal_loss.mean()

    def _process_all_paths_vectorized(
        self,
        path_pairs: List[Dict],
        skeleton_graphs: Dict[int, SkeletonGraph],
    ) -> Tuple[Dict, Dict]:
        """Collect path edges and targets in batched arrays."""
        
        all_backbone = []  # List of (batch, z, y, x, neighbor_idx)
        all_opt_edges = []  # List of (batch, z, y, x, neighbor_idx, path_idx, edge_idx)
        
        path_sources = []  
        path_targets = []  
        path_batch_indices = []  

        for path_idx, pinfo in enumerate(path_pairs):
            batch_idx = pinfo['batch_idx']
            path_ref = pinfo['path_ref']
            path_opt = pinfo['path_opt']
            
            
            ref_edges = self._extract_path_edges_numpy(path_ref, batch_idx)
            all_backbone.append(ref_edges)
            
            
            opt_edges = self._extract_path_edges_numpy(path_opt, batch_idx)
            
            n_edges = len(opt_edges)
            if n_edges > 0:
                opt_with_idx = np.column_stack([
                    opt_edges,
                    np.full(n_edges, path_idx, dtype=np.int32),
                    np.arange(n_edges, dtype=np.int32)
                ])
                all_opt_edges.append(opt_with_idx)
            
            path_sources.append(pinfo['source'])
            path_targets.append(pinfo['target'])
            path_batch_indices.append(batch_idx)

        
        if all_backbone:
            backbone_arr = np.vstack([e for e in all_backbone if len(e) > 0]) if any(len(e) > 0 for e in all_backbone) else np.array([]).reshape(0, 5)
        else:
            backbone_arr = np.array([]).reshape(0, 5)

        
        if all_opt_edges:
            opt_arr = np.vstack(all_opt_edges)  # [M, 7]: batch, z, y, x, neighbor, path_idx, edge_idx
        else:
            opt_arr = np.array([]).reshape(0, 7)

        
        optimal_deltas = np.zeros(len(opt_arr))
        
        if len(opt_arr) > 0:
            optimal_deltas = self._compute_deltas_batch(
                opt_arr,
                path_sources,
                path_targets,
                path_batch_indices,
                skeleton_graphs
            )

        
        reference_data = {
            'coords': backbone_arr[:, :4].astype(np.int64) if len(backbone_arr) > 0 else np.array([]).reshape(0, 4),
            'neighbors': backbone_arr[:, 4].astype(np.int64) if len(backbone_arr) > 0 else np.array([]),
            'deltas': np.zeros(len(backbone_arr), dtype=np.float32),
            'count': len(backbone_arr),
        }
        
        optimal_data = {
            'coords': opt_arr[:, :4].astype(np.int64) if len(opt_arr) > 0 else np.array([]).reshape(0, 4),
            'neighbors': opt_arr[:, 4].astype(np.int64) if len(opt_arr) > 0 else np.array([]),
            'deltas': optimal_deltas,
            'count': len(opt_arr),
        }

        return reference_data, optimal_data

    def _extract_path_edges_numpy(
        self,
        path: List[np.ndarray],
        batch_idx: int,
    ) -> np.ndarray:
        """Convert a coordinate path into directional edges."""
        if len(path) < 2:
            return np.array([]).reshape(0, 5)

        
        path_arr = np.array([p if isinstance(p, np.ndarray) else np.array(p) for p in path])
        
        
        offsets = (path_arr[1:] - path_arr[:-1]).astype(np.int32)  # [L-1, 3]
        
        
        codes = (offsets[:, 0] + 1) * 9 + (offsets[:, 1] + 1) * 3 + (offsets[:, 2] + 1)
        
        
        neighbor_indices = self.offset_lookup[codes]  # [L-1]
        
        
        valid_mask = neighbor_indices >= 0
        valid_coords = path_arr[:-1][valid_mask]  # [N_valid, 3]
        valid_neighbors = neighbor_indices[valid_mask]  # [N_valid]
        
        n_valid = valid_mask.sum()
        if n_valid == 0:
            return np.array([]).reshape(0, 5)

        
        result = np.column_stack([
            np.full(n_valid, batch_idx, dtype=np.int32),
            valid_coords.astype(np.int32),
            valid_neighbors
        ])
        
        return result

    def _path_to_edge_set_fast(self, path: List[np.ndarray]) -> set:
        """Encode a path as a set of directed edges."""
        edges = set()
        for i in range(len(path) - 1):
            c1 = tuple(int(x) for x in path[i])
            c2 = tuple(int(x) for x in path[i + 1])
            edges.add((c1, c2))
        return edges

    def _compute_deltas_batch(
        self,
        optimal_edges: np.ndarray,  # [M, 7]
        path_sources: List[np.ndarray],
        path_targets: List[np.ndarray],
        path_batch_indices: List[int],
        skeleton_graphs: Dict[int, SkeletonGraph],
    ) -> np.ndarray:
        """Compute geodesic excess for selected path edges."""
        M = len(optimal_edges)
        if M == 0:
            return np.array([])

        deltas = np.zeros(M)
        
        
        batch_indices = optimal_edges[:, 0].astype(int)
        unique_batches = np.unique(batch_indices)
        
        for b in unique_batches:
            b = int(b)  
            skel_graph = skeleton_graphs.get(b)
            if skel_graph is None or not skel_graph.valid:
                continue
            
            
            batch_mask = batch_indices == b
            batch_edges = optimal_edges[batch_mask]
            batch_local_indices = np.where(batch_mask)[0]
            
            
            points_start = batch_edges[:, 1:4].astype(np.float64)  # [N_b, 3]
            neighbor_indices = batch_edges[:, 4].astype(int)
            points_end = points_start + self.neighbor_offsets[neighbor_indices]
            path_indices = batch_edges[:, 5].astype(int)  # [N_b]
            
            
            proj_dists_start, proj_indices_start = skel_graph.kdtree.query(points_start)
            proj_dists_end, proj_indices_end = skel_graph.kdtree.query(points_end)
            proj_indices_start = np.asarray(proj_indices_start).astype(int)
            proj_indices_end = np.asarray(proj_indices_end).astype(int)
            proj_dists_start = np.asarray(proj_dists_start).astype(float)
            proj_dists_end = np.asarray(proj_dists_end).astype(float)
            
            
            unique_paths = np.unique(path_indices)
            
            
            all_u_skel = []
            all_v_skel = []
            path_to_uv = {}
            
            for p_idx in unique_paths:
                p_idx = int(p_idx)  
                if p_idx >= len(path_sources):
                    continue
                source = path_sources[p_idx]
                target = path_targets[p_idx]
                
                
                _, u_idx = skel_graph.kdtree.query(source)
                _, v_idx = skel_graph.kdtree.query(target)
                u_idx = int(u_idx)
                v_idx = int(v_idx)
                
                all_u_skel.append(u_idx)
                all_v_skel.append(v_idx)
                path_to_uv[p_idx] = (u_idx, v_idx)
            
            if not path_to_uv:
                continue
            
            
            all_sources = list(set(all_u_skel) | set(all_v_skel))
            
            if skel_graph._precompute:
                geo_matrix = skel_graph.geodesic_matrix
            else:
                geo_matrix = dijkstra(skel_graph.graph, directed=False, indices=all_sources)
                source_to_row = {s: i for i, s in enumerate(all_sources)}
            
            
            for i, local_idx in enumerate(batch_local_indices):
                p_idx = int(path_indices[i])
                if p_idx not in path_to_uv:
                    continue
                    
                u_skel, v_skel = path_to_uv[p_idx]
                D_uv = (
                    geo_matrix[u_skel, v_skel]
                    if skel_graph._precompute
                    else geo_matrix[source_to_row[u_skel], v_skel]
                )

                endpoint_etas = []
                for x_skel, x_proj in (
                    (int(proj_indices_start[i]), float(proj_dists_start[i])),
                    (int(proj_indices_end[i]), float(proj_dists_end[i])),
                ):
                    if skel_graph._precompute:
                        D_ux = x_proj + geo_matrix[u_skel, x_skel]
                        D_xv = x_proj + geo_matrix[v_skel, x_skel]
                    else:
                        D_ux = x_proj + geo_matrix[source_to_row[u_skel], x_skel]
                        D_xv = x_proj + geo_matrix[source_to_row[v_skel], x_skel]
                    endpoint_etas.append(max(0.0, D_ux + D_xv - D_uv))

                deltas[local_idx] = 0.5 * (endpoint_etas[0] + endpoint_etas[1])

        return deltas

    def _compute_spectral_loss(
        self,
        Q: torch.Tensor,
        skeleton_gt: torch.Tensor,
    ) -> torch.Tensor:
        """Compute global spectral alignment loss."""
        B = Q.shape[0]
        device = Q.device
        
        total_loss = torch.tensor(0.0, device=device)
        num_valid = 0

        for b in range(B):
            loss_b = self._compute_spectral_loss_single(Q[b], skeleton_gt[b, 0])
            if loss_b is not None:
                total_loss = total_loss + loss_b
                num_valid += 1

        return total_loss / max(num_valid, 1)

    def _compute_spectral_loss_single(
        self,
        Q: torch.Tensor,
        skeleton: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        K, D, H, W = Q.shape
        device = Q.device

        skel_mask = skeleton > 0.5
        skel_coords = torch.nonzero(skel_mask)
        N = skel_coords.shape[0]

        if N < 10 or N > 3000:
            return None

        
        diff = skel_coords.unsqueeze(0) - skel_coords.unsqueeze(1)
        dist_sq = (diff ** 2).sum(dim=-1)
        A_gt = (dist_sq <= 3).float()
        A_gt.fill_diagonal_(0)

        
        z, y, x = skel_coords[:, 0], skel_coords[:, 1], skel_coords[:, 2]
        Q_nodes = F.normalize(Q[:, z, y, x].T, dim=1)
        A_pred = (torch.mm(Q_nodes, Q_nodes.T) + 1) / 2
        A_pred.fill_diagonal_(0)

        
        neighbor_mask = A_gt > 0.5
        loss_pos = ((A_pred[neighbor_mask] - 1.0) ** 2).mean() if neighbor_mask.sum() > 0 else torch.tensor(0.0, device=device)

        non_neighbor_mask = A_gt < 0.5
        num_neg = min(int(neighbor_mask.sum().item() * 2), 1000)
        
        if num_neg > 0 and non_neighbor_mask.sum() > 0:
            neg_indices = torch.nonzero(non_neighbor_mask.flatten()).squeeze(-1)
            if len(neg_indices) > num_neg:
                neg_indices = neg_indices[torch.randperm(len(neg_indices), device=device)[:num_neg]]
            loss_neg = (A_pred.flatten()[neg_indices] ** 2).mean()
        else:
            loss_neg = torch.tensor(0.0, device=device)

        return loss_pos + 0.1 * loss_neg
    

    def forward_path(
        self,
        path_pairs,
        Q: torch.Tensor,
        port_probs: torch.Tensor,
        skeleton_gt: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, K, D, H, W = Q.shape
        device = Q.device

        
        with torch.no_grad():
            edge_weights_dense = self.graph_constructor.build_dense_graph(Q, port_probs)

        
        skeleton_graphs = {
            b: SkeletonGraph(skeleton_gt[b, 0].cpu().numpy() > 0.5)
            for b in range(B)
        }

        
        reference_data, optimal_data = self._process_all_paths_vectorized(
            path_pairs, skeleton_graphs
        )

        
        loss_backbone = self._edge_cost_regression_loss(
            reference_data, Q, port_probs, device
        )
        loss_shortcut = self._edge_cost_regression_loss(
            optimal_data, Q, port_probs, device
        )

        loss_shortcut = self.lambda_shortcut * loss_shortcut
        loss_local = loss_backbone + loss_shortcut

        # Spectral loss
        if self.lambda_spectral > 0:
            loss_spectral = self.lambda_spectral * self._compute_spectral_loss(Q, skeleton_gt)
        else:
            loss_spectral = torch.tensor(0.0, device=device)

        return {
            'loss_backbone': loss_backbone,
            'loss_shortcut': loss_shortcut,
            'loss_local': loss_local,
            'loss_spectral': loss_spectral,
            'loss_topology': loss_local + loss_spectral,
        }
