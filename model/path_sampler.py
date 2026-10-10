"""Shortest-path sampling and adaptive bottleneck mining utilities."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


class PathSampler(nn.Module):
    """Sample reference paths on the skeleton and shortest paths on the predicted graph."""

    def __init__(
            self,
            num_sample_pairs: int = 32,
            connectivity: int = 26,  # 26-connectivity for 3D
    ):
        super().__init__()
        self.num_sample_pairs = num_sample_pairs
        self.connectivity = connectivity

        
        self.neighbor_offsets = self._get_neighbor_offsets()

    def _get_neighbor_offsets(self) -> np.ndarray:
        """Return offsets for the 26-connected neighborhood."""
        offsets = []
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    offsets.append([dz, dy, dx])
        return np.array(offsets)  # [26, 3]

    def forward(
            self,
            edge_weights_pred: torch.Tensor,
            skeleton_gt: torch.Tensor,
            num_pairs: Optional[int] = None,
        ) -> List[Dict]:
        """Sample path pairs for every item in the batch."""
        if num_pairs is None:
            num_pairs = self.num_sample_pairs

        B = edge_weights_pred.shape[0]
        all_path_pairs = []

        for b in range(B):
            skeleton_np = skeleton_gt[b, 0].cpu().numpy() > 0.5
            edge_weights_np = edge_weights_pred[b].detach().cpu().float().numpy()
            
            skel_coords = np.argwhere(skeleton_np)
            if len(skel_coords) < 2:
                continue

            D, H, W = skeleton_np.shape
            
            
            gt_graph, coord_to_idx, idx_to_coord = self._build_skeleton_graph_fully_vectorized(
                skeleton_np, skel_coords
            )
            pred_graph = self._build_prediction_graph_fully_vectorized(edge_weights_np, (D, H, W))

            
            sampled_pairs = self._sample_node_pairs(skel_coords, num_pairs)
            if not sampled_pairs:
                continue

            
            
            valid_pairs_skel = []  # (u_idx_skel, v_idx_skel, u_coord, v_coord)
            valid_pairs_full = []  # (u_idx_full, v_idx_full)
            
            for u_coord, v_coord in sampled_pairs:
                u_tuple, v_tuple = tuple(u_coord), tuple(v_coord)
                if u_tuple not in coord_to_idx or v_tuple not in coord_to_idx:
                    continue
                
                u_idx_skel = coord_to_idx[u_tuple]
                v_idx_skel = coord_to_idx[v_tuple]
                u_idx_full = u_coord[0] * H * W + u_coord[1] * W + u_coord[2]
                v_idx_full = v_coord[0] * H * W + v_coord[1] * W + v_coord[2]
                
                valid_pairs_skel.append((u_idx_skel, v_idx_skel, u_coord, v_coord))
                valid_pairs_full.append((u_idx_full, v_idx_full))

            if not valid_pairs_skel:
                continue

            
            unique_sources_skel = list(set(p[0] for p in valid_pairs_skel))
            unique_sources_full = list(set(p[0] for p in valid_pairs_full))
            
            
            dist_skel, pred_skel = dijkstra(
                gt_graph, directed=False,
                indices=unique_sources_skel,
                return_predecessors=True
            )
            source_to_row_skel = {s: i for i, s in enumerate(unique_sources_skel)}
            
            
            dist_full, pred_full = dijkstra(
                pred_graph, directed=False,
                indices=unique_sources_full,
                return_predecessors=True
            )
            source_to_row_full = {s: i for i, s in enumerate(unique_sources_full)}

            
            for i, (u_skel, v_skel, u_coord, v_coord) in enumerate(valid_pairs_skel):
                u_full, v_full = valid_pairs_full[i]
                
                
                row_skel = source_to_row_skel[u_skel]
                row_full = source_to_row_full[u_full]
                
                d_ref = dist_skel[row_skel, v_skel]
                d_opt = dist_full[row_full, v_full]
                
                if np.isinf(d_ref) or np.isinf(d_opt):
                    continue
                
                
                path_ref, edges_ref = self._backtrack_path(
                    pred_skel[row_skel], u_skel, v_skel, idx_to_coord
                )
                path_opt, edges_opt = self._backtrack_path_full(
                    pred_full[row_full], u_full, v_full, (D, H, W)
                )
                
                if path_ref is None or path_opt is None:
                    continue

                all_path_pairs.append({
                    'source': u_coord,
                    'target': v_coord,
                    'path_ref': path_ref,
                    'path_opt': path_opt,
                    'edges_ref': edges_ref,
                    'edges_opt': edges_opt,
                    'distance_ref': d_ref,
                    'distance_opt': d_opt,
                    'batch_idx': b,
                })

        return all_path_pairs

    def _backtrack_path(self, predecessors, source, target, idx_to_coord):
        """Recover a skeleton path from a predecessor array."""
        path_indices = []
        current = target
        while current != source:
            path_indices.append(current)
            current = predecessors[current]
            if current == -9999:
                return None, []
        path_indices.append(source)
        path_indices.reverse()
        
        path_coords = [np.array(idx_to_coord[i]) for i in path_indices]
        edges = [(path_indices[i], path_indices[i+1]) for i in range(len(path_indices)-1)]
        return path_coords, edges
    
    def _backtrack_path_full(self, predecessors, source, target, shape):
        """Recover a dense-grid path from a predecessor array."""
        D, H, W = shape
        
        
        path_indices = []
        current = target
        while current != source:
            path_indices.append(current)
            current = predecessors[current]
            if current == -9999:
                return None, []
        path_indices.append(source)
        path_indices.reverse()
        
        
        def idx_to_coord(idx):
            z = idx // (H * W)
            y = (idx % (H * W)) // W
            x = idx % W
            return np.array([z, y, x])
        
        path_coords = [idx_to_coord(i) for i in path_indices]
        
        
        edges = []
        for i in range(len(path_coords) - 1):
            offset = path_coords[i + 1] - path_coords[i]
            for k, ref_offset in enumerate(self.neighbor_offsets):
                if np.array_equal(offset, ref_offset):
                    edges.append({
                        'coord': path_coords[i],
                        'neighbor_idx': k,
                        'next_coord': path_coords[i + 1],
                    })
                    break
        
        return path_coords, edges
    
    def _build_skeleton_graph_fully_vectorized(
            self,
            skeleton: np.ndarray,
            skel_coords: np.ndarray,
    ) -> Tuple[csr_matrix, Dict, Dict]:
        """Build a sparse skeleton graph with vectorized indexing."""
        n_nodes = len(skel_coords)
        D, H, W = skeleton.shape
        
        
        coord_to_idx = {tuple(c): i for i, c in enumerate(skel_coords)}
        idx_to_coord = {i: tuple(c) for i, c in enumerate(skel_coords)}
        
        
        linear_to_skel_idx = np.full(D * H * W, -1, dtype=np.int32)
        skel_linear = skel_coords[:, 0] * H * W + skel_coords[:, 1] * W + skel_coords[:, 2]
        linear_to_skel_idx[skel_linear] = np.arange(n_nodes)
        
        
        all_neighbors = skel_coords[:, None, :] + self.neighbor_offsets[None, :, :]
        nz, ny, nx = all_neighbors[..., 0], all_neighbors[..., 1], all_neighbors[..., 2]
        
        
        valid_mask = (nz >= 0) & (nz < D) & (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
        
        
        neighbor_linear = np.where(valid_mask, nz * H * W + ny * W + nx, 0).astype(np.int64)
        
        
        neighbor_skel_idx = linear_to_skel_idx[neighbor_linear]  # [N, 26]
        
        
        edge_mask = valid_mask & (neighbor_skel_idx >= 0)  # [N, 26]
        
        
        src_node_idx, direction_idx = np.where(edge_mask)
        tgt_node_idx = neighbor_skel_idx[edge_mask]
        
        
        offset_distances = np.linalg.norm(self.neighbor_offsets, axis=1)
        edge_weights = offset_distances[direction_idx]
        
        graph = csr_matrix(
            (edge_weights, (src_node_idx, tgt_node_idx)), 
            shape=(n_nodes, n_nodes)
        )
        
        return graph, coord_to_idx, idx_to_coord
    
    def _build_prediction_graph_fully_vectorized(
            self,
            edge_weights: np.ndarray,  # [D, H, W, 26]
            shape: Tuple[int, int, int],
    ) -> csr_matrix:
        """Build the predicted dense-grid graph with vectorized indexing."""
        D, H, W = shape
        n_nodes = D * H * W
        eps = 1e-6
        
        
        z_flat, y_flat, x_flat = np.meshgrid(
            np.arange(D), np.arange(H), np.arange(W), indexing='ij'
        )
        all_coords = np.stack([z_flat.ravel(), y_flat.ravel(), x_flat.ravel()], axis=1)  # [N, 3]
        
        
        # all_coords[:, None, :] -> [N, 1, 3]
        # self.neighbor_offsets[None, :, :] -> [1, 26, 3]
        all_neighbors = all_coords[:, None, :] + self.neighbor_offsets[None, :, :]  # [N, 26, 3]
        
        
        nz, ny, nx = all_neighbors[..., 0], all_neighbors[..., 1], all_neighbors[..., 2]
        valid_mask = (nz >= 0) & (nz < D) & (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
        
        
        
        src_indices = np.arange(n_nodes)[:, None] * np.ones(26, dtype=np.int64)[None, :]  # [N, 26]
        
        
        
        tgt_indices = np.where(valid_mask, nz * H * W + ny * W + nx, 0).astype(np.int64)  # [N, 26]
        
        
        
        #     c_e = ||x-y|| / (E_e + eps)
        
        
        
        edge_weights_flat = edge_weights.reshape(n_nodes, 26)  # [N, 26]
        offset_distances = np.linalg.norm(self.neighbor_offsets, axis=1).astype(np.float32)
        distances = offset_distances[None, :] / (edge_weights_flat + eps)
        
        
        valid_src = src_indices[valid_mask]  # [num_valid_edges]
        valid_tgt = tgt_indices[valid_mask]  # [num_valid_edges]
        valid_dist = distances[valid_mask]   # [num_valid_edges]
        
        
        graph = csr_matrix((valid_dist, (valid_src, valid_tgt)), shape=(n_nodes, n_nodes))
        
        return graph

    def _sample_node_pairs(
            self,
            coords: np.ndarray,
            num_pairs: int,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Sample endpoint pairs from skeleton coordinates."""
        n = len(coords)
        pairs = []

        
        min_distance = 5  

        attempts = 0
        max_attempts = num_pairs * 10

        while len(pairs) < num_pairs and attempts < max_attempts:
            i, j = np.random.choice(n, 2, replace=False)
            u, v = coords[i], coords[j]

            dist = np.linalg.norm(u - v)
            if dist >= min_distance:
                pairs.append((u.copy(), v.copy()))

            attempts += 1

        return pairs

    def _find_path_on_graph(
            self,
            graph: csr_matrix,
            source: int,
            target: int,
            idx_to_coord: Dict,
    ) -> Tuple[Optional[List], float, List]:
        """Find a shortest path on a sparse skeleton graph."""
        
        dist_matrix, predecessors = dijkstra(
            graph,
            directed=False,
            indices=source,
            return_predecessors=True
        )

        distance = dist_matrix[target]

        if np.isinf(distance):
            return None, float('inf'), []

        
        path_indices = []
        current = target
        while current != source:
            path_indices.append(current)
            current = predecessors[current]
            if current == -9999:  
                return None, float('inf'), []
        path_indices.append(source)
        path_indices.reverse()

        
        path_coords = [np.array(idx_to_coord[i]) for i in path_indices]

        
        edges = []
        for i in range(len(path_indices) - 1):
            edges.append((path_indices[i], path_indices[i + 1]))

        return path_coords, distance, edges

    def _find_path_on_full_graph(
            self,
            graph: csr_matrix,
            source: int,
            target: int,
            shape: Tuple[int, int, int],
    ) -> Tuple[Optional[List], float, List]:
        """Find a shortest path on the full predicted graph."""
        D, H, W = shape

        
        dist_matrix, predecessors = dijkstra(
            graph,
            directed=False,
            indices=source,
            return_predecessors=True
        )

        distance = dist_matrix[target]

        if np.isinf(distance):
            return None, float('inf'), []

        
        path_indices = []
        current = target
        while current != source:
            path_indices.append(current)
            current = predecessors[current]
            if current == -9999:
                return None, float('inf'), []
        path_indices.append(source)
        path_indices.reverse()

        
        def idx_to_coord(idx):
            z = idx // (H * W)
            remainder = idx % (H * W)
            y = remainder // W
            x = remainder % W
            return np.array([z, y, x])

        path_coords = [idx_to_coord(i) for i in path_indices]

        
        edges = []
        for i in range(len(path_coords) - 1):
            coord = path_coords[i]
            next_coord = path_coords[i + 1]
            offset = next_coord - coord

            
            for k, ref_offset in enumerate(self.neighbor_offsets):
                if np.array_equal(offset, ref_offset):
                    edges.append({
                        'coord': coord,  # [z, y, x]
                        'neighbor_idx': k,  # 0-25
                        'next_coord': next_coord,
                    })
                    break

        return path_coords, distance, edges

    def get_path_edges_tensor(
            self,
            path_info: Dict,
            shape: Tuple[int, int, int],
            device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert sampled path edges to tensor indices."""
        edges = path_info['edges_opt']

        if len(edges) == 0:
            return torch.zeros(0, 3, dtype=torch.long, device=device), \
                torch.zeros(0, dtype=torch.long, device=device)

        coords = torch.tensor(
            [e['coord'] for e in edges],
            dtype=torch.long,
            device=device
        )
        neighbor_indices = torch.tensor(
            [e['neighbor_idx'] for e in edges],
            dtype=torch.long,
            device=device
        )

        return coords, neighbor_indices


# Adaptive bottleneck selection used by the TPL topology loss.
class BottleneckPathSampler(PathSampler):
    """
    Path sampler with adaptive topological bottleneck mining.

    Args:
        num_sample_pairs: selected path number M.
        num_candidate_pairs: candidate path number M_c.
        nms_distance: GT geodesic distance threshold d_nms.
        connectivity: graph connectivity, expected to be 26 in current TPL.
    """

    def __init__(
        self,
        num_sample_pairs: int = 32,
        num_candidate_pairs: int = 128,
        nms_distance: float = 5.0,
        connectivity: int = 26,
    ):
        super().__init__(num_sample_pairs=num_sample_pairs, connectivity=connectivity)
        self.num_candidate_pairs = num_candidate_pairs
        self.nms_distance = nms_distance

    def forward(
        self,
        edge_weights_pred: torch.Tensor,
        skeleton_gt: torch.Tensor,
        num_pairs: Optional[int] = None,
        num_candidate_pairs: Optional[int] = None,
    ) -> List[Dict]:
        if num_pairs is None:
            num_pairs = self.num_sample_pairs
        if num_candidate_pairs is None:
            num_candidate_pairs = self.num_candidate_pairs

        num_candidate_pairs = max(int(num_candidate_pairs), int(num_pairs))
        all_path_pairs = []

        B = edge_weights_pred.shape[0]
        for b in range(B):
            skeleton_np = skeleton_gt[b, 0].cpu().numpy() > 0.5
            edge_weights_np = edge_weights_pred[b].detach().cpu().float().numpy()

            skel_coords = np.argwhere(skeleton_np)
            if len(skel_coords) < 2:
                continue

            D, H, W = skeleton_np.shape
            gt_graph, coord_to_idx, idx_to_coord = self._build_skeleton_graph_fully_vectorized(
                skeleton_np, skel_coords
            )
            pred_graph = self._build_prediction_graph_fully_vectorized(
                edge_weights_np, (D, H, W)
            )

            candidate_pairs = self._sample_node_pairs(skel_coords, num_candidate_pairs)
            if not candidate_pairs:
                continue

            candidates, influence = self._build_candidate_pool(
                candidate_pairs=candidate_pairs,
                batch_idx=b,
                shape=(D, H, W),
                gt_graph=gt_graph,
                pred_graph=pred_graph,
                coord_to_idx=coord_to_idx,
                idx_to_coord=idx_to_coord,
                skel_coords=skel_coords,
            )
            if not candidates:
                continue

            selected = self._select_by_node_influence(
                candidates=candidates,
                influence=influence,
                gt_graph=gt_graph,
                num_pairs=num_pairs,
            )
            all_path_pairs.extend(selected)

        return all_path_pairs

    def _build_candidate_pool(
        self,
        candidate_pairs: List[Tuple[np.ndarray, np.ndarray]],
        batch_idx: int,
        shape: Tuple[int, int, int],
        gt_graph,
        pred_graph,
        coord_to_idx: Dict[Tuple[int, int, int], int],
        idx_to_coord: Dict[int, Tuple[int, int, int]],
        skel_coords: np.ndarray,
    ) -> Tuple[List[Dict], np.ndarray]:
        D, H, W = shape
        candidates = []
        influence = np.zeros(len(skel_coords), dtype=np.float64)
        skel_tree = cKDTree(skel_coords)

        valid_pairs_skel = []
        valid_pairs_full = []
        for u_coord, v_coord in candidate_pairs:
            u_tuple = tuple(int(x) for x in u_coord)
            v_tuple = tuple(int(x) for x in v_coord)
            if u_tuple not in coord_to_idx or v_tuple not in coord_to_idx:
                continue

            u_skel = coord_to_idx[u_tuple]
            v_skel = coord_to_idx[v_tuple]
            u_full = int(u_coord[0] * H * W + u_coord[1] * W + u_coord[2])
            v_full = int(v_coord[0] * H * W + v_coord[1] * W + v_coord[2])
            valid_pairs_skel.append((u_skel, v_skel, u_coord.copy(), v_coord.copy()))
            valid_pairs_full.append((u_full, v_full))

        if not valid_pairs_skel:
            return candidates, influence

        unique_sources_skel = sorted(set(p[0] for p in valid_pairs_skel))
        unique_sources_full = sorted(set(p[0] for p in valid_pairs_full))

        dist_skel, pred_skel = dijkstra(
            gt_graph,
            directed=False,
            indices=unique_sources_skel,
            return_predecessors=True,
        )
        dist_full, pred_full = dijkstra(
            pred_graph,
            directed=False,
            indices=unique_sources_full,
            return_predecessors=True,
        )
        source_to_row_skel = {s: i for i, s in enumerate(unique_sources_skel)}
        source_to_row_full = {s: i for i, s in enumerate(unique_sources_full)}
        distance_cache = {s: dist_skel[source_to_row_skel[s]] for s in unique_sources_skel}

        for i, (u_skel, v_skel, u_coord, v_coord) in enumerate(valid_pairs_skel):
            u_full, v_full = valid_pairs_full[i]
            row_skel = source_to_row_skel[u_skel]
            row_full = source_to_row_full[u_full]

            d_ref = dist_skel[row_skel, v_skel]
            d_opt = dist_full[row_full, v_full]
            if np.isinf(d_ref) or np.isinf(d_opt):
                continue

            path_ref, edges_ref = self._backtrack_path(
                pred_skel[row_skel], u_skel, v_skel, idx_to_coord
            )
            path_opt, edges_opt = self._backtrack_path_full(
                pred_full[row_full], u_full, v_full, (D, H, W)
            )
            if path_ref is None or path_opt is None:
                continue

            score = self._compute_path_discrepancy(
                path_opt=path_opt,
                u_skel=u_skel,
                v_skel=v_skel,
                gt_graph=gt_graph,
                skel_tree=skel_tree,
                distance_cache=distance_cache,
            )

            ref_node_indices = np.array(
                [coord_to_idx[tuple(int(x) for x in coord)] for coord in path_ref],
                dtype=np.int64,
            )
            influence[ref_node_indices] += score

            candidates.append(
                {
                    'source': u_coord,
                    'target': v_coord,
                    'path_ref': path_ref,
                    'path_opt': path_opt,
                    'edges_ref': edges_ref,
                    'edges_opt': edges_opt,
                    'distance_ref': d_ref,
                    'distance_opt': d_opt,
                    'batch_idx': batch_idx,
                    'score': float(score),
                    '_ref_node_set': set(int(x) for x in ref_node_indices),
                }
            )

        return candidates, influence

    def _compute_path_discrepancy(
        self,
        path_opt: List[np.ndarray],
        u_skel: int,
        v_skel: int,
        gt_graph,
        skel_tree: cKDTree,
        distance_cache: Dict[int, np.ndarray],
    ) -> float:
        if len(path_opt) < 2:
            return 0.0

        dist_u = self._get_distance_row(gt_graph, u_skel, distance_cache)
        dist_v = self._get_distance_row(gt_graph, v_skel, distance_cache)
        d_uv = dist_u[v_skel]
        if np.isinf(d_uv):
            return 0.0

        points = np.asarray(path_opt, dtype=np.float64)
        proj_dists, proj_indices = skel_tree.query(points)
        proj_indices = np.asarray(proj_indices, dtype=np.int64)
        proj_dists = np.asarray(proj_dists, dtype=np.float64)

        node_eta = dist_u[proj_indices] + dist_v[proj_indices] + 2.0 * proj_dists - d_uv
        node_eta = np.maximum(node_eta, 0.0)
        edge_eta = 0.5 * (node_eta[:-1] + node_eta[1:])
        finite_eta = edge_eta[np.isfinite(edge_eta)]
        if len(finite_eta) == 0:
            return 0.0
        return float(finite_eta.sum() / max(len(edge_eta), 1))

    def _select_by_node_influence(
        self,
        candidates: List[Dict],
        influence: np.ndarray,
        gt_graph,
        num_pairs: int,
    ) -> List[Dict]:
        selected = []
        selected_high_nodes = []
        remaining = list(range(len(candidates)))
        distance_cache = {}

        for node_idx in np.argsort(-influence):
            node_idx = int(node_idx)
            if influence[node_idx] <= 0 and selected:
                break

            if self._passes_nms(node_idx, selected_high_nodes, gt_graph, distance_cache):
                selected_high_nodes.append(node_idx)

                while len(selected) < num_pairs:
                    candidate_idx = self._pop_first_candidate_through_node(
                        remaining, candidates, node_idx
                    )
                    if candidate_idx is None:
                        break
                    selected.append(candidates[candidate_idx])

            if len(selected) >= num_pairs:
                break

        if len(selected) < num_pairs:
            selected.extend(candidates[i] for i in remaining[: num_pairs - len(selected)])

        for item in selected:
            item.pop('_ref_node_set', None)
        return selected[:num_pairs]

    def _passes_nms(
        self,
        node_idx: int,
        selected_high_nodes: List[int],
        gt_graph,
        distance_cache: Dict[int, np.ndarray],
    ) -> bool:
        if not selected_high_nodes:
            return True
        dist_row = self._get_distance_row(gt_graph, node_idx, distance_cache)
        min_dist = np.min(dist_row[np.asarray(selected_high_nodes, dtype=np.int64)])
        return bool(min_dist > self.nms_distance)

    @staticmethod
    def _pop_first_candidate_through_node(
        remaining: List[int],
        candidates: List[Dict],
        node_idx: int,
    ) -> Optional[int]:
        for pos, candidate_idx in enumerate(remaining):
            if node_idx in candidates[candidate_idx]['_ref_node_set']:
                return remaining.pop(pos)
        return None

    @staticmethod
    def _get_distance_row(gt_graph, node_idx: int, cache: Dict[int, np.ndarray]) -> np.ndarray:
        if node_idx not in cache:
            cache[node_idx] = dijkstra(gt_graph, directed=False, indices=node_idx)
        return cache[node_idx]
