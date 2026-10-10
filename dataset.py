import torch
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset
import pickle
import os

class TPLTopologyDataset(Dataset):
    """Load CTA volumes, labels, and precomputed topology targets."""

    def __init__(
        self,
        is_train,
        fold_th,
        k_fold,
        dataset_txt_path,
        intensity_lower: float = -200.0,
        intensity_upper: float = 800.0,
    ):
        self.is_train = is_train
        self.fold_th = fold_th
        self.k_fold = k_fold
        if intensity_lower >= intensity_upper:
            raise ValueError('intensity_lower must be smaller than intensity_upper')
        self.intensity_lower = float(intensity_lower)
        self.intensity_upper = float(intensity_upper)

        self.filepath_list = self.load_filepath_list(dataset_txt_path)
        self.dataset = dataset_txt_path.split("/")[-2]
        self.train_filepath_list, self.val_filepath_list = self.split_data_x_fold(
            self.filepath_list, self.fold_th, self.k_fold
        )

        self.intensity_func = self.process_intensity()
        self.label_func = self.process_label()

    def load_filepath_list(self, txt_path):
        """Read dataset records from a whitespace-delimited index file."""
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

    def process_intensity(self):
        """Clip and scale CTA intensities to [0, 1].

        The default window is [-200, 800]. For ASOCA, use [200, 2500]
        through ``intensity_lower`` and ``intensity_upper``.
        """
        lower = self.intensity_lower
        upper = self.intensity_upper

        def cut_normalize(cta_array):
            cta_array = cta_array.astype(np.float32)
            cta_array[cta_array > upper] = upper
            cta_array[cta_array < lower] = lower
            return (cta_array - lower) / (upper - lower)

        return cut_normalize

    def process_label(self):
        """Return the dataset-specific binary label conversion function."""
        if self.dataset == 'ENTIRE':
            def process_label(label_array):
                label_array[label_array == 1] = 0
                label_array[label_array != 0] = 1
                return label_array
        else:
            def process_label(label_array):
                return (label_array > 0).astype(np.float32)
        return process_label

    def load_topology_data(self, skeleton_path, degree_map_path, topology_path):
        """Load skeleton, degree, and sparse directional topology targets."""
        result = {}

        if skeleton_path and os.path.exists(skeleton_path):
            skeleton_sitk = sitk.ReadImage(skeleton_path)
            result['skeleton'] = sitk.GetArrayFromImage(skeleton_sitk)  # [Z, Y, X]
        else:
            return None

        if degree_map_path and os.path.exists(degree_map_path):
            degree_sitk = sitk.ReadImage(degree_map_path)
            result['degree_map'] = sitk.GetArrayFromImage(degree_sitk)  # [Z, Y, X]
        else:
            result['degree_map'] = np.zeros_like(result['skeleton'])

        if topology_path and os.path.exists(topology_path):
            data = np.load(topology_path)
            result['coords'] = data['coords']
            result['ports'] = data['values']
            result['shape'] = data['shape']
            result['bifurcation_points'] = data['bifurcation_points']
            result['terminal_points'] = data['terminal_points']
        else:
            shape = result['skeleton'].shape
            result['coords'] = np.array([]).reshape(0, 3)
            result['ports'] = np.array([]).reshape(0, 3)
            result['shape'] = shape
            result['bifurcation_points'] = np.array([]).reshape(0, 3)
            result['terminal_points'] = np.array([]).reshape(0, 3)

        return result

    def load_properties(self, properties_path):
        """Load foreground sampling metadata."""
        if properties_path is None or not os.path.exists(properties_path):
            return None

        with open(properties_path, 'rb') as f:
            properties = pickle.load(f)

        return properties

    def __getitem__(self, index):
        if self.is_train:
            filepath = self.train_filepath_list[index]
        else:
            filepath = self.val_filepath_list[index]

        img_path = filepath[0]
        label_path = filepath[1]
        skeleton_path = filepath[2] if len(filepath) > 2 else None
        degree_map_path = filepath[3] if len(filepath) > 3 else None
        topology_path = filepath[4] if len(filepath) > 4 else None
        properties_path = filepath[5] if len(filepath) > 5 else None

        img = sitk.ReadImage(img_path, sitk.sitkInt32)
        label = sitk.ReadImage(label_path, sitk.sitkInt8)

        img_array = sitk.GetArrayFromImage(img)  # [Z, Y, X]
        label_array = sitk.GetArrayFromImage(label)  # [Z, Y, X]

        img_array = self.intensity_func(img_array)
        label_array = self.label_func(label_array)

        topo_data = self.load_topology_data(skeleton_path, degree_map_path, topology_path)

        properties = self.load_properties(properties_path)

        if properties is not None:
            class_locations = properties['class_locs']
        else:
            class_locations = {'coronary_artery': np.argwhere(label_array == 1)}

        img_tensor = torch.from_numpy(img_array).unsqueeze(0).float()
        label_tensor = torch.from_numpy(label_array).unsqueeze(0).float()

        result = {
            "data": img_tensor,
            "seg": label_tensor,
            "properties": {
                'class_locations': class_locations,
                "img_path": img_path,
                "label_path": label_path,
            },
        }

        if topo_data is not None:
            result['skeleton'] = torch.from_numpy(topo_data['skeleton']).float()
            result['degree_map'] = torch.from_numpy(topo_data['degree_map']).long()
            result['coords'] = torch.from_numpy(topo_data['coords']).long()
            result['ports'] = torch.from_numpy(topo_data['ports'])
            result['bifurcation_points'] = topo_data['bifurcation_points']
            result['terminal_points'] = topo_data['terminal_points']
        else:
            shape = label_array.shape
            result['skeleton'] = torch.zeros(shape).float()
            result['degree_map'] = torch.zeros(shape).long()
            result['coords'] = np.array([]).reshape(0, 3)
            result['ports'] = np.array([]).reshape(0, 3)
            result['bifurcation_points'] = np.array([]).reshape(0, 3)
            result['terminal_points'] = np.array([]).reshape(0, 3)

        return result

    def __len__(self):
        if self.is_train:
            return len(self.train_filepath_list)
        else:
            return len(self.val_filepath_list)

    def get_total_data_num(self):
        return len(self.filepath_list)

    def split_data_x_fold(self, filepath, fold_th=1, k_fold=5):
        """Split records into deterministic training and validation folds."""
        np.random.seed(888)

        data_num = len(filepath)
        shuffle_indices = np.random.permutation(np.arange(data_num))
        train_indice, val_indice = self.create_indice(shuffle_indices, i=fold_th, folds_all=k_fold)

        train_path = [filepath[i] for i in train_indice]
        val_path = [filepath[i] for i in val_indice]

        return train_path, val_path

    def create_indice(self, shuffle_indices, i=1, folds_all=5):
        """Return training and validation indices for one fold."""
        a = np.linspace(0, len(shuffle_indices), folds_all + 1, dtype=int)

        if i == folds_all:
            train = shuffle_indices[:a[i - 1]]
            val = shuffle_indices[a[i - 1]:]
        elif i == 1:
            train = shuffle_indices[a[i]:]
            val = shuffle_indices[:a[i]]
        else:
            train = list(shuffle_indices[a[0]:a[i - 1]]) + list(shuffle_indices[a[i]:])
            val = shuffle_indices[a[i - 1]:a[i]]
        return train, val
