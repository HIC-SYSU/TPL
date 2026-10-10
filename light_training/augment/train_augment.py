import inspect
import multiprocessing
import os
import shutil
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import Union, Tuple, List
import numpy as np
import torch
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor


def get_train_transforms(patch_size, mirror_axes=None):
    tr_transforms = []
    patch_size_spatial = patch_size
    ignore_axes = None
    angle = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)

    tr_transforms.append(SpatialTransform(
    patch_size_spatial, patch_center_dist_from_border=None,
    do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
    do_rotation=True, angle_x=angle, angle_y=angle, angle_z=angle,
    p_rot_per_axis=1,  # todo experiment with this
    do_scale=True, scale=(0.7, 1.4),
    border_mode_data="constant", border_cval_data=0, order_data=3,
    border_mode_seg="constant", border_cval_seg=-1, order_seg=1,
    random_crop=False,  # random cropping is part of our dataloaders
    p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
    independent_scale_for_each_axis=False  # todo experiment with this
    ))

    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
    tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                            p_per_channel=0.5))
    tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
    tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                    p_per_channel=0.5,
                                                    order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                    ignore_axes=ignore_axes))
    tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1))
    tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))

    if mirror_axes is not None and len(mirror_axes) > 0:
        tr_transforms.append(MirrorTransform(mirror_axes))

    tr_transforms.append(RemoveLabelTransform(-1, 0))
    tr_transforms.append(NumpyToTensor(['data', 'seg'], 'float'))

    tr_transforms = Compose(tr_transforms)

    return tr_transforms

class LabelIntensityAdjustTransform(AbstractTransform):
    """
    向量化增强：利用 NumPy 广播机制并行调整 label 为 1 的位置。
    """
    def __init__(self, adjustment_range=0.3):
        """
        :param adjustment_range: 平均强度调整的范围。
        """
        self.adjustment_range = adjustment_range

    def __call__(self, **data_dict):
        """
        :param data_dict: 包含 'data' [B, C, D, H, W] 和 'seg' [B, 1, D, H, W]
        """
        data = data_dict["data"]
        seg = data_dict["seg"]
        B = data.shape[0]

        # 1. 为 Batch 中的每个样本生成一个独立的随机因子
        # 形状设为 (B, 1, 1, 1, 1) 以便在所有空间维度和通道维度上进行广播
        factors = np.random.uniform(
            1 - self.adjustment_range, 
            1 + self.adjustment_range, 
            size=(B, 1, 1, 1, 1)
        ).astype(data.dtype)

        # 2. 创建掩码 (找到 label == 1 的位置)
        # seg 是 (B, 1, D, H, W)，NumPy 会自动处理 bool 掩码
        mask = (seg == 1)

        # 3. 向量化操作：
        # np.where(condition, x, y): condition 为真选 x，否则选 y
        # 这里的计算是并行的，且 data * factors 利用了高性能广播
        data_dict["data"] = np.where(mask, data * factors, data)

        return data_dict

# 创建增强流水线
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.transforms.abstract_transforms import Compose

def get_custom_transforms():
    transforms = [
        GaussianNoiseTransform(noise_variance=(0, 0.01), p_per_sample=0.5),  # 高斯噪声
        LabelIntensityAdjustTransform(adjustment_range=0.2),  # 血管增强
        NumpyToTensor(['data', 'seg'], 'float')  # 转换为张量
    ]
    return Compose(transforms)

def get_train_transforms_noaug(patch_size, mirror_axes=None):
    tr_transforms = []
    tr_transforms.append(RemoveLabelTransform(-1, 0))
    tr_transforms.append(NumpyToTensor(['data', 'seg'], 'float'))
    tr_transforms = Compose(tr_transforms)

    return tr_transforms

def get_validation_transforms() -> AbstractTransform:
        val_transforms = []
        val_transforms.append(RemoveLabelTransform(-1, 0))
        val_transforms.append(NumpyToTensor(['data', 'seg'], 'float'))
        val_transforms = Compose(val_transforms)

        return val_transforms