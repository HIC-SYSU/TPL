import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


class SegmentationLoss(nn.Module):
    """Combine foreground-aware and all-class Dice-focal losses."""

    def __init__(
            self,
            num_classes: int = 2,
            alpha: float = 1.0,  
            beta: float = 1.0,  
            use_softmax: bool = True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.use_softmax = use_softmax

        try:
            from monai.losses import DiceFocalLoss

            
            self.loss_with_bg = DiceFocalLoss(
                include_background=True,
                to_onehot_y=True,
                sigmoid=False,
                softmax=use_softmax,
            )

            
            self.loss_without_bg = DiceFocalLoss(
                include_background=False,
                to_onehot_y=True,
                sigmoid=False,
                softmax=use_softmax,
            )

            self.use_monai = True

        except ImportError:
            print("Warning: MONAI not available, using fallback Dice + BCE loss")
            self.use_monai = False

    def forward(
            self,
            logits: torch.Tensor,  # [B, C, D, H, W]
            target: torch.Tensor,  # [B, 1, D, H, W]
            alpha: Optional[float] = None,
            beta: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the weighted segmentation objective."""
        alpha = alpha if alpha is not None else self.alpha
        beta = beta if beta is not None else self.beta

        
        if logits.shape[2:] != target.shape[2:]:
            target = F.interpolate(
                target.float(), size=logits.shape[2:], mode='nearest'
            )

        if self.use_monai:
            # MONAI DiceFocalLoss
            loss_with_bg = self.loss_with_bg(logits, target)
            loss_without_bg = self.loss_without_bg(logits, target)

            total = (alpha * loss_with_bg + beta * loss_without_bg) / (alpha + beta)

            return {
                'loss_with_bg': loss_with_bg,
                'loss_without_bg': loss_without_bg,
                'loss_seg': total,
            }
        else:
            raise ValueError