"""Prepare skeletons, topology targets, foreground samples, and the data index."""

import os
import numpy as np
import SimpleITK as sitk
from typing import Dict, List, Tuple, Optional, Union
from scipy import ndimage
from scipy.spatial.distance import cdist
from collections import deque
import pickle
import argparse
from tqdm import tqdm

try:
    from skimage.morphology import skeletonize as skeletonize_3d
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: skimage not available, using fallback skeletonization")

class LabelTopologyProcessor:
    """Extract skeleton topology and directional ports from a binary label."""

    def __init__(
            self,
            connectivity: int = 26,
            bifurcation_merge_radius: float = 2.0,
    ):
        self.connectivity = connectivity
        self.bifurcation_merge_radius = bifurcation_merge_radius
        self.neighbor_offsets = self._generate_offsets()

    def _generate_offsets(self) -> np.ndarray:
        """Return offsets for the 26-connected neighborhood."""
        offsets = []
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    offsets.append([dz, dy, dx])
        return np.array(offsets)

    def process(
            self,
            label: np.ndarray,  # [Z, Y, X] - sitk format
            return_all: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Extract all topology targets from one binary label volume."""

        skeleton = self.extract_skeleton(label)

        degree_map_raw = self.compute_degree(skeleton)

        degree_map, bifurcation_points = self.deduplicate_bifurcations(
            skeleton, degree_map_raw
        )

        terminal_points = np.argwhere((degree_map_raw == 1)&(skeleton > 0))
        print(f"Found {len(terminal_points)} endpoints")

        topology_class = self.classify_topology(degree_map, skeleton)

        neighbor_mask, degree_map = self.compute_patch_neighbor_mask(skeleton,
                                                                     bifurcation_points=bifurcation_points,
                                                                     terminal_points=terminal_points)

        results = {
            'skeleton': skeleton.astype(np.uint8),
            'degree_map': degree_map.astype(np.uint8),
            'topology_class': topology_class.astype(np.uint8),
            'neighbor_mask': neighbor_mask,
            'bifurcation_points': bifurcation_points,
            'terminal_points': terminal_points,
        }

        if return_all:
            results['degree_map_raw'] = degree_map_raw.astype(np.uint8)

        return results

    def extract_skeleton(self, label: np.ndarray) -> np.ndarray:
        """Convert a binary label into a one-voxel-wide skeleton."""
        binary = (label > 0.5).astype(np.uint8)

        if HAS_SKIMAGE:
            skeleton = skeletonize_3d(binary)
        else:
            skeleton = self._fallback_skeleton(binary)

        return (skeleton > 0).astype(np.uint8)

    def _fallback_skeleton(self, binary: np.ndarray) -> np.ndarray:
        """Compute a morphological skeleton without scikit-image."""
        from scipy.ndimage import binary_erosion, binary_dilation

        skeleton = np.zeros_like(binary)
        temp = binary.copy()
        struct = ndimage.generate_binary_structure(3, 1)

        while temp.sum() > 0:
            eroded = binary_erosion(temp, struct)
            opened = binary_dilation(eroded, struct)
            skeleton |= (temp & ~opened)
            temp = eroded

        return skeleton

    def compute_degree(self, skeleton: np.ndarray) -> np.ndarray:
        """Count occupied neighbors at each skeleton point."""
        kernel = np.ones((3, 3, 3), dtype=np.float32)
        kernel[1, 1, 1] = 0

        neighbor_count = ndimage.convolve(
            skeleton.astype(np.float32),
            kernel,
            mode='constant',
            cval=0
        )

        degree_map = np.where(skeleton > 0, neighbor_count, 0)
        return degree_map.astype(np.int32)

    def deduplicate_bifurcations(
            self,
            skeleton: np.ndarray,
            degree_map: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Collapse nearby bifurcation candidates to one representative."""
        bifurcation_mask = (skeleton > 0) & (degree_map >= 3)
        candidates = np.argwhere(bifurcation_mask)

        if len(candidates) == 0:
            return degree_map.copy(), np.array([]).reshape(0, 3)

        clusters = self._cluster_points(candidates, self.bifurcation_merge_radius)

        degree_map_cleaned = degree_map.copy()
        kept_bifurcations = []

        for cluster_indices in clusters:
            cluster_points = candidates[cluster_indices]
            cluster_degrees = [degree_map[tuple(p)] for p in cluster_points]

            best_idx = np.argmax(cluster_degrees)
            representative = cluster_points[best_idx]
            kept_bifurcations.append(representative)

            for i, point in enumerate(cluster_points):
                if i != best_idx:
                    degree_map_cleaned[tuple(point)] = 2

        bifurcation_points = np.array(kept_bifurcations) if kept_bifurcations else np.array([]).reshape(0, 3)

        return degree_map_cleaned, bifurcation_points

    def _cluster_points(
            self,
            points: np.ndarray,
            radius: float,
    ) -> List[List[int]]:
        """Group points by radius-connected components."""
        n = len(points)
        if n == 0:
            return []

        dist_matrix = cdist(points, points)
        adjacency = dist_matrix <= radius

        visited = np.zeros(n, dtype=bool)
        clusters = []

        for start in range(n):
            if visited[start]:
                continue

            cluster = []
            queue = deque([start])
            visited[start] = True

            while queue:
                node = queue.popleft()
                cluster.append(node)

                neighbors = np.where(adjacency[node] & ~visited)[0]
                for neighbor in neighbors:
                    visited[neighbor] = True
                    queue.append(neighbor)

            clusters.append(cluster)

        return clusters

    def classify_topology(
            self,
            degree_map: np.ndarray,
            skeleton: np.ndarray,
    ) -> np.ndarray:
        """Assign endpoint, tubular, and bifurcation classes."""
        topology = np.zeros_like(degree_map, dtype=np.uint8)
        skeleton_mask = skeleton > 0

        topology[(skeleton_mask) & (degree_map == 1)] = 1
        topology[(skeleton_mask) & (degree_map == 2)] = 2
        topology[(skeleton_mask) & (degree_map >= 3)] = 3

        return topology

    def _get_patch_boundary_coords(self, radius: int) -> np.ndarray:
        coords = []
        for dz in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if abs(dz) == radius or abs(dy) == radius or abs(dx) == radius:
                        coords.append([dz, dy, dx])
        return np.array(coords, dtype=np.int32)

    def compute_patch_neighbor_mask(
        self,
        skeleton: np.ndarray,
        dilation_radius: int = 1,
        patch_radius: int = 3,
        bifurcation_points: np.ndarray = None,
        terminal_points: np.ndarray = None,
        correction_radius: int = 3,
        hard: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate directional ports and corrected degrees from local patches."""
        from scipy.ndimage import binary_dilation, generate_binary_structure, label
        from scipy.spatial import cKDTree

        Z, Y, X = skeleton.shape

        struct = generate_binary_structure(3, 3)
        dilated = skeleton.copy()
        for _ in range(dilation_radius):
            dilated = binary_dilation(dilated, struct)
        dilated = dilated.astype(np.float32)

        skel_points = np.argwhere(skeleton > 0)
        if len(skel_points) == 0:
            return np.zeros((Z, Y, X, 26), dtype=np.float32), np.zeros((Z, Y, X), dtype=np.int32)

        offsets = self.neighbor_offsets.astype(np.float32)
        offsets_norm = offsets / (np.linalg.norm(offsets, axis=1, keepdims=True) + 1e-6)

        boundary_coords = self._get_patch_boundary_coords(patch_radius)
        boundary_dirs = boundary_coords.astype(np.float32)
        boundary_dirs = boundary_dirs / (np.linalg.norm(boundary_dirs, axis=1, keepdims=True) + 1e-6)
        boundary_to_dir_sim = np.dot(boundary_dirs, offsets_norm.T)

        boundary_size = 2 * patch_radius + 1
        boundary_to_grid = {}
        for bi, (bz, by, bx) in enumerate(boundary_coords):
            grid_coord = (bz + patch_radius, by + patch_radius, bx + patch_radius)
            boundary_to_grid[bi] = grid_coord

        neighbor_mask = np.zeros((Z, Y, X, 26), dtype=np.float32)
        degree_map = np.zeros((Z, Y, X), dtype=np.int32)

        for center in tqdm(skel_points, desc="Computing patch ports (pass 1)"):
            cz, cy, cx = center

            activated = []
            for bi, (bz, by, bx) in enumerate(boundary_coords):
                az, ay, ax = cz + bz, cy + by, cx + bx
                if 0 <= az < Z and 0 <= ay < Y and 0 <= ax < X:
                    if dilated[az, ay, ax] > 0:
                        activated.append(bi)

            if len(activated) == 0:
                degree_map[cz, cy, cx] = 1
                continue

            boundary_grid = np.zeros((boundary_size, boundary_size, boundary_size), dtype=np.uint8)
            for bi in activated:
                gz, gy, gx = boundary_to_grid[bi]
                boundary_grid[gz, gy, gx] = 1

            labeled, num_exits = label(boundary_grid, structure=generate_binary_structure(3, 3))

            port_weights = np.zeros(26, dtype=np.float32)

            for exit_id in range(1, num_exits + 1):
                exit_mask = labeled == exit_id
                exit_boundary_indices = [bi for bi in activated if exit_mask[boundary_to_grid[bi]]]

                if len(exit_boundary_indices) == 0:
                    continue

                if not hard:
                    exit_sims = boundary_to_dir_sim[exit_boundary_indices].mean(axis=0)
                    threshold = exit_sims.max() * 0.5
                    exit_sims[exit_sims < threshold] = 0

                    if exit_sims.sum() > 0:
                        exit_sims = exit_sims / exit_sims.sum()

                    port_weights += exit_sims

                if hard:
                    exit_sims = boundary_to_dir_sim[exit_boundary_indices].mean(axis=0)
                    best_direction_idx = np.argmax(exit_sims)
                    port_weights[best_direction_idx] = 1.0

            neighbor_mask[cz, cy, cx] = port_weights
            degree_map[cz, cy, cx] = num_exits

        if bifurcation_points is not None or terminal_points is not None:
            neighbor_mask, degree_map = self._correct_degree(
                skeleton, neighbor_mask, degree_map,
                bifurcation_points, terminal_points,
                correction_radius, offsets_norm
            )

        self._print_degree_stats(degree_map, skeleton)

        return neighbor_mask, degree_map

    def _print_degree_stats(self, degree_map: np.ndarray, skeleton: np.ndarray):
        """Print the distribution of corrected skeleton degrees."""
        skel_mask = skeleton > 0
        degrees = degree_map[skel_mask]
        total = len(degrees)

        stats = {
            0: (degrees == 0).sum(),
            1: (degrees == 1).sum(),
            2: (degrees == 2).sum(),
            3: (degrees == 3).sum(),
            '4+': (degrees >= 4).sum(),
        }

        print(f"\nDegree statistics ({total} skeleton points):")
        print(f"  Degree 0 (isolated):     {stats[0]:5d} ({100*stats[0]/total:.1f}%)")
        print(f"  Degree 1 (endpoint):     {stats[1]:5d} ({100*stats[1]/total:.1f}%)")
        print(f"  Degree 2 (segment):      {stats[2]:5d} ({100*stats[2]/total:.1f}%)")
        print(f"  Degree 3 (bifurcation): {stats[3]:5d} ({100*stats[3]/total:.1f}%)")
        print(f"  Degree 4+ (multi-branch): {stats['4+']:5d} ({100*stats['4+']/total:.1f}%)")

    def _correct_degree(
        self,
        skeleton: np.ndarray,
        neighbor_mask: np.ndarray,
        degree_map: np.ndarray,
        bifurcation_points: np.ndarray,
        terminal_points: np.ndarray,
        correction_radius: int,
        offsets_norm: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Correct spurious endpoints and bifurcations using reference landmarks."""
        from scipy.spatial import cKDTree

        Z, Y, X = skeleton.shape
        skel_points = np.argwhere(skeleton > 0)

        degree2_mask = (skeleton > 0) & (degree_map == 2)
        degree2_points = np.argwhere(degree2_mask)

        if len(degree2_points) == 0:
            return neighbor_mask, degree_map

        degree2_tree = cKDTree(degree2_points)

        true_bif_set = set()
        true_term_set = set()

        if bifurcation_points is not None and len(bifurcation_points) > 0:
            for p in bifurcation_points:
                true_bif_set.add(tuple(p))

        if terminal_points is not None and len(terminal_points) > 0:
            for p in terminal_points:
                true_term_set.add(tuple(p))
            terminal_tree = cKDTree(terminal_points)
        else:
            terminal_tree = None

        corrected_count = {'bif_to_2': 0, 'term_to_2': 0}

        for point in skel_points:
            pz, py, px = point
            current_degree = degree_map[pz, py, px]
            point_tuple = tuple(point)

            if current_degree >= 3 and point_tuple not in true_bif_set:
                dist, idx = degree2_tree.query(point, k=1)

                if dist <= correction_radius:
                    nearest_d2 = degree2_points[idx]
                    nz, ny, nx = nearest_d2

                    current_port = neighbor_mask[pz, py, px].copy()

                    ref_port = neighbor_mask[nz, ny, nx]

                    new_port = self._select_similar_ports(
                        current_port, ref_port, offsets_norm, keep_k=2
                    )

                    neighbor_mask[pz, py, px] = new_port
                    degree_map[pz, py, px] = 2
                    corrected_count['bif_to_2'] += 1

            elif current_degree == 1 and point_tuple not in true_term_set:
                if terminal_tree is not None:
                    dist, idx = terminal_tree.query(point, k=1)
                    nearest_term = terminal_points[idx]

                    direction = nearest_term - point
                    norm = np.linalg.norm(direction)

                    if norm > 1e-6:
                        direction = direction / norm

                        sims = np.dot(offsets_norm, direction)
                        best_dir_idx = np.argmax(sims)

                        current_port = neighbor_mask[pz, py, px].copy()
                        current_main_idx = np.argmax(current_port)

                        new_port = np.zeros(26, dtype=np.float32)
                        new_port[current_main_idx] = 1.0
                        new_port[best_dir_idx] = 1.0

                        if current_main_idx != best_dir_idx:
                            neighbor_mask[pz, py, px] = new_port
                            degree_map[pz, py, px] = 2
                            corrected_count['term_to_2'] += 1

        print(f"\nDegree correction statistics:")
        print(f"  Spurious bifurcations corrected: {corrected_count['bif_to_2']}")
        print(f"  Spurious endpoints corrected: {corrected_count['term_to_2']}")

        return neighbor_mask, degree_map

    def _select_similar_ports(
            self,
            current_port: np.ndarray,
            ref_port: np.ndarray,
            offsets_norm: np.ndarray,
            keep_k: int = 2,
    ) -> np.ndarray:
        """Keep the current directions most similar to a reference port pattern."""
        new_port = np.zeros_like(current_port)

        current_active = np.where(current_port > 0.1)[0]
        ref_active = np.where(ref_port > 0.1)[0]

        if len(current_active) == 0 or len(ref_active) == 0:
            return current_port

        similarities = []
        for c_idx in current_active:
            c_dir = offsets_norm[c_idx]
            max_sim = max(np.dot(c_dir, offsets_norm[r_idx]) for r_idx in ref_active)
            similarities.append((c_idx, max_sim, current_port[c_idx]))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:keep_k]

        for idx, sim, weight in top_k:
            new_port[idx] = weight

        if new_port.sum() > 0:
            new_port = new_port / new_port.sum() * keep_k

        return new_port

def sample_foreground_locations(
        seg: np.ndarray,
        classes_or_regions: List[Union[int, List[int]]] = [1],
        seed: int = 1234,
        num_samples: int = 10000,
        min_percent_coverage: float = 0.01,
        verbose: bool = False,
) -> Dict:
    """Sample foreground coordinates for patch extraction."""
    rndst = np.random.RandomState(seed)
    class_locs = {}

    for c in classes_or_regions:
        k = c if not isinstance(c, list) else tuple(c)

        if isinstance(c, (tuple, list)):
            mask = seg == c[0]
            for cc in c[1:]:
                mask = mask | (seg == cc)
            all_locs = np.argwhere(mask)
        else:
            all_locs = np.argwhere(seg == c)

        if len(all_locs) == 0:
            class_locs[k] = np.array([]).reshape(0, 3)
            continue

        target_num_samples = min(num_samples, len(all_locs))
        target_num_samples = max(target_num_samples, int(np.ceil(len(all_locs) * min_percent_coverage)))

        selected = all_locs[rndst.choice(len(all_locs), target_num_samples, replace=False)]
        class_locs[k] = selected

        if verbose:
            print(f"  Class {c}: sampled {target_num_samples} / {len(all_locs)} locations")

    return class_locs

def find_label_path(cta_filename: str, label_root: str) -> Optional[str]:
    """Resolve a label path from a CTA filename."""
    base_name = cta_filename

    patterns = [
        (lambda x: x[:-7] if x.endswith('.nii.gz') else x[:-4], '_refine_aorta_and_arteries.nii.gz'),
        (lambda x: x[:-12] if '_0000.nii.gz' in x else x[:-7], '_Label.nii.gz'),
        (lambda x: x[:-7] if x.endswith('.nii.gz') else x[:-4], '_label.nii'),
        (lambda x: x[:-11] if x.endswith('.nii.gz') else x[:-8], '_label.nii'),
        (lambda x: x[:-7] if x.endswith('.nii.gz') else x[:-4], '_label.nii.gz'),
        (lambda x: x[:-7] if x.endswith('.nii.gz') else x[:-4], '.nii.gz'),
        (lambda x: x[:-11] if x.endswith('.nii.gz') else x[:-8], '_Label.nii.gz'),
        (lambda x: x[:-11] if x.endswith('.nii.gz') else x[:-8], '_refine_aorta_and_arteries.nii.gz'),
    ]

    for name_func, suffix in patterns:
        try:
            label_name = name_func(base_name) + suffix
            label_path = os.path.join(label_root, label_name)
            if os.path.exists(label_path):
                return label_path
        except:
            continue

    return None

def get_case_id(filename: str) -> str:
    """Return a normalized case identifier from a filename."""
    if filename.endswith('.nii.gz'):
        base = filename[:-7]
    elif filename.endswith('.nii'):
        base = filename[:-4]
    else:
        base = filename

    for suffix in ['_0000', '_Label', '_label', '_refine_aorta_and_arteries']:
        if base.endswith(suffix):
            base = base[:-len(suffix)]

    return base

def load_filepath_list(txt_path):
    file_path_list = []
    with open(txt_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()
            if not lines:
                break
            parts = lines.split()
            if len(parts) >= 6:
                file_path_list.append(parts)
            elif len(parts) >= 4:
                file_path_list.append(parts + [None, None])
            elif len(parts) == 2:
                file_path_list.append(parts + [None, None, None, None])
    return file_path_list

class TPLPreprocessor:
    """Generate topology targets and metadata for a TPL dataset."""

    def __init__(
            self,
            dataset_root: str,
            bifurcation_merge_radius: float = 2.5,
            num_foreground_samples: int = 10000,
            seed: int = 1234,
    ):
        self.dataset_root = dataset_root
        self.cta_root = os.path.join(dataset_root, 'cta')
        self.label_root = os.path.join(dataset_root, 'label')
        self.skeleton_root = os.path.join(dataset_root, 'skeleton')
        self.properties_root = os.path.join(dataset_root, 'properties')
        self.data_txt_path = os.path.join(dataset_root, 'data.txt')

        self.bifurcation_merge_radius = bifurcation_merge_radius
        self.num_foreground_samples = num_foreground_samples
        self.seed = seed

        os.makedirs(self.skeleton_root, exist_ok=True)
        os.makedirs(self.properties_root, exist_ok=True)

        self.topo_processor = LabelTopologyProcessor(
            bifurcation_merge_radius=bifurcation_merge_radius
        )

    def process_single_case(
            self,
            cta_path: str,
            label_path: str,
            case_id: str,
    ) -> Tuple[str, str, str, str]:
        """Generate and save topology targets for one case."""

        label_sitk = sitk.ReadImage(label_path)
        label = sitk.GetArrayFromImage(label_sitk)  # [Z, Y, X]

        label_binary = (label > 0.5).astype(np.uint8)

        topo_results = self.topo_processor.process(label_binary)

        skeleton_path = os.path.join(self.skeleton_root, f"{case_id}_skeleton.nii.gz")
        skeleton_sitk = sitk.GetImageFromArray(topo_results['skeleton'])
        skeleton_sitk.CopyInformation(label_sitk)
        sitk.WriteImage(skeleton_sitk, skeleton_path)

        degree_map_path = os.path.join(self.skeleton_root, f"{case_id}_degree_map.nii.gz")
        degree_sitk = sitk.GetImageFromArray(topo_results['degree_map'])
        degree_sitk.CopyInformation(label_sitk)
        sitk.WriteImage(degree_sitk, degree_map_path)

        neighbor_mask = topo_results['neighbor_mask']
        coords = np.argwhere(topo_results['skeleton'] == 1)
        values = neighbor_mask[coords[:, 0], coords[:, 1], coords[:, 2]]

        topology_path = os.path.join(self.skeleton_root, f"{case_id}_topology.npz")
        np.savez_compressed(
            topology_path,
            coords=coords,
            values=values,
            shape=label.shape,
            neighbor_mask=topo_results['neighbor_mask'],
            bifurcation_points=topo_results['bifurcation_points'],
            terminal_points=topo_results['terminal_points'],
        )

        class_locs = sample_foreground_locations(
            seg=label_binary,
            classes_or_regions=[1],
            seed=self.seed,
            num_samples=self.num_foreground_samples,
        )

        properties = {
            'class_locs': {
                'coronary_artery': class_locs[1]
            },
            'shape': label.shape,
            'spacing': label_sitk.GetSpacing(),
            'origin': label_sitk.GetOrigin(),
            'direction': label_sitk.GetDirection(),
        }

        properties_path = os.path.join(self.properties_root, f"{case_id}_properties.pkl")
        with open(properties_path, 'wb') as f:
            pickle.dump(properties, f)

        return skeleton_path, degree_map_path, topology_path, properties_path

    def process_dataset(self, verbose: bool = True) -> List[Tuple]:
        """Process all discoverable CTA and label pairs."""
        cta_files = sorted([f for f in os.listdir(self.cta_root)
                            if f.endswith('.nii.gz') or f.endswith('.nii')])

        if verbose:
            print(f"Found {len(cta_files)} CTA files in {self.cta_root}")

        data_list = []
        failed_cases = []

        iterator = tqdm(cta_files, desc="Processing") if verbose else cta_files

        for cta_filename in iterator:
            cta_path = os.path.join(self.cta_root, cta_filename)
            label_path = find_label_path(cta_filename, self.label_root)

            if label_path is None:
                if verbose:
                    print(f"\nWarning: No label found for {cta_filename}")
                failed_cases.append(cta_filename)
                continue

            case_id = get_case_id(cta_filename)

            try:
                skeleton_path, degree_map_path, topology_path, properties_path = \
                    self.process_single_case(cta_path, label_path, case_id)

                data_list.append((
                    cta_path, label_path,
                    skeleton_path, degree_map_path, topology_path,
                    properties_path
                ))

            except Exception as e:
                if verbose:
                    print(f"\nError processing {cta_filename}: {e}")
                failed_cases.append(cta_filename)
                continue

        self._write_data_txt(data_list)

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Processing complete!")
            print(f"  Successfully processed: {len(data_list)} cases")
            print(f"  Failed: {len(failed_cases)} cases")
            if failed_cases:
                print(f"  Failed cases: {failed_cases[:5]}{'...' if len(failed_cases) > 5 else ''}")
            print(f"  Data index saved to: {self.data_txt_path}")
            print(f"{'=' * 60}")

        return data_list

    def _write_data_txt(self, data_list: List[Tuple]):
        """Write the six-column dataset index."""
        with open(self.data_txt_path, 'w') as f:
            for item in data_list:
                f.write(' '.join(item) + '\n')

    @staticmethod
    def load_data_txt(data_txt_path: str) -> List[Dict[str, str]]:
        """Read a six-column dataset index into dictionaries."""
        data_list = []
        with open(data_txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    data_list.append({
                        'cta': parts[0],
                        'label': parts[1],
                        'skeleton': parts[2],
                        'degree_map': parts[3],
                        'topology': parts[4],
                        'properties': parts[5],
                    })
        return data_list

def preprocess_tpl(dataset_root: str, **kwargs):
    """Run TPL preprocessing with the supplied configuration."""
    preprocessor = TPLPreprocessor(dataset_root, **kwargs)
    return preprocessor.process_dataset()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess a dataset for TPL')
    parser.add_argument('--dataset_root', type=str,
                        required=True,
                        help='Dataset root directory')
    parser.add_argument('--bifurcation_radius', type=float, default=2.5,
                        help='Bifurcation merge radius')
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Number of foreground samples')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed')

    args = parser.parse_args()

    print(f"Preprocessing TPL dataset at: {args.dataset_root}")
    print(f"  Bifurcation merge radius: {args.bifurcation_radius}")
    print(f"  Foreground samples: {args.num_samples}")
    print(f"  Random seed: {args.seed}")
    print()

    preprocess_tpl(
        dataset_root=args.dataset_root,
        bifurcation_merge_radius=args.bifurcation_radius,
        num_foreground_samples=args.num_samples,
        seed=args.seed,
    )