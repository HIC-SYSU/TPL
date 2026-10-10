"""TPL network with graph-guided prototype refinement.

The important ordering is::

    encoder -> decoder d4 -> embedding projection -> prototypes / graph
            -> GNN(d4) -> remaining decoder -> segmentation

TPL prototype assignment is computed from the
decoder's 1/4-resolution feature map, not from the encoder skip feature ``s4``.
The topology-consistent assignment term is disabled by fixing ``beta=0``.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .prototype import TopologicalPrototype
from .graph import (
    GraphConstructor,
    TopologicalPredictor,
    TopologySupervisionLoss,
)
from .topology_loss import TopologyAwareLoss
from .representation_loss import (
    InstanceToInstanceContrastive,
    InstanceToPrototypeContrastive,
    PrototypeDiversityLoss,
    InstanceToPrototypeEntropy,
)
from .segmentation_loss import SegmentationLoss


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ConvBlock(nn.Module):
    """MedNeXt/ConvNeXt-style residual 3D convolution block."""

    def __init__(
        self,
        channels: int,
        expansion: int = 2,
        kernel_size: int = 3,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.dw_conv = nn.Conv3d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False,
        )
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.expand = nn.Conv3d(channels, channels * expansion, kernel_size=1)
        self.act = nn.GELU()
        self.compress = nn.Conv3d(channels * expansion, channels, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.norm(self.dw_conv(x))
        x = self.compress(self.act(self.expand(x)))
        return identity + self.drop_path(x)


class DownBlock(nn.Module):
    """Two-fold spatial downsampling with doubled channels."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        self.dw_conv = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels,
            bias=False,
        )
        self.norm = nn.GroupNorm(min(32, in_channels), in_channels)
        self.expand = nn.Conv3d(
            in_channels, in_channels * expansion, kernel_size=1
        )
        self.act = nn.GELU()
        self.compress = nn.Conv3d(
            in_channels * expansion, out_channels, kernel_size=1
        )
        self.shortcut = nn.Conv3d(
            in_channels, out_channels, kernel_size=1, stride=2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)
        x = self.norm(self.dw_conv(x))
        x = self.compress(self.act(self.expand(x)))
        return x + shortcut


class UpBlock(nn.Module):
    """Two-fold spatial upsampling followed by skip fusion."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        self.up_conv = nn.ConvTranspose3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            output_padding=1,
            groups=in_channels,
            bias=False,
        )
        self.norm = nn.GroupNorm(min(32, in_channels), in_channels)
        self.expand = nn.Conv3d(
            in_channels, in_channels * expansion, kernel_size=1
        )
        self.act = nn.GELU()
        self.compress = nn.Conv3d(
            in_channels * expansion, out_channels, kernel_size=1
        )
        self.shortcut = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=1, stride=2, output_padding=1
        )
        self.skip_fusion = nn.Conv3d(
            out_channels * 2, out_channels, kernel_size=1
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)
        x = self.norm(self.up_conv(x))
        x = self.compress(self.act(self.expand(x)))
        x = x + shortcut
        return self.skip_fusion(torch.cat([x, skip], dim=1))


class LightweightGNN(nn.Module):
    """Local GCN implementing four layers on the dense 26-neighbor graph."""

    def __init__(
        self,
        channels: int,
        num_layers: int = 4,
        connectivity: int = 26,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.num_layers = num_layers
        self.connectivity = connectivity
        self.register_buffer('neighbor_offsets', self._get_offsets())
        self.gcn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(channels, channels, kernel_size=1),
                nn.GroupNorm(min(32, channels), channels),
                nn.GELU(),
            )
            for _ in range(num_layers)
        ])
        self.register_buffer('neighbor_kernel', self._create_neighbor_kernel())

    def _get_offsets(self) -> torch.Tensor:
        offsets = []
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dz != 0 or dy != 0 or dx != 0:
                        offsets.append([dz, dy, dx])
        return torch.tensor(offsets[:self.connectivity], dtype=torch.long)

    def _create_neighbor_kernel(self) -> torch.Tensor:
        kernel = torch.zeros(len(self.neighbor_offsets), 1, 3, 3, 3)
        for index, offset in enumerate(self.neighbor_offsets):
            z, y, x = offset + 1
            kernel[index, 0, z, y, x] = 1.0
        return kernel

    def _get_neighbor_features(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, depth, height, width = x.shape
        neighbors = F.conv3d(
            x.reshape(batch * channels, 1, depth, height, width),
            self.neighbor_kernel,
            padding=1,
        )
        return neighbors.reshape(
            batch,
            channels,
            len(self.neighbor_offsets),
            depth,
            height,
            width,
        )

    def _get_neighbor_scalar(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv3d(x, self.neighbor_kernel, padding=1).unsqueeze(1)

    def forward(
        self,
        features: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> torch.Tensor:
        h = features
        for layer in self.gcn_layers:
            neighbors = self._get_neighbor_features(h)
            weights = (
                edge_weights.permute(0, 4, 1, 2, 3)
                .unsqueeze(1)
                .clamp_min(0)
            )
            degree = weights.sum(dim=2) + 1.0
            neighbor_degree = self._get_neighbor_scalar(degree)
            normalized_weights = weights / torch.sqrt(
                degree.unsqueeze(2) * neighbor_degree + 1e-6
            )
            aggregated = (neighbors * normalized_weights).sum(dim=2)
            aggregated = aggregated + h / (degree + 1e-6)
            h = h + layer(aggregated)
        return h


class TPLEncoder(nn.Module):
    """Five-stage encoder at resolutions 1, 1/2, 1/4, 1/8 and 1/16."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_blocks: List[int] = [2, 2, 2, 2, 2],
        kernel_size: int = 5,
        expansion: int = 4,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        channels = [base_channels * (2 ** index) for index in range(5)]
        rates = [
            value.item()
            for value in torch.linspace(0, drop_path_rate, sum(num_blocks))
        ]
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, base_channels), base_channels),
            nn.GELU(),
        )
        self.stage1 = self._make_stage(
            channels[0], num_blocks[0], expansion, kernel_size,
            rates[:num_blocks[0]],
        )
        self.down1 = DownBlock(channels[0], channels[1], kernel_size, expansion)
        self.stage2 = self._make_stage(
            channels[1], num_blocks[1], expansion, kernel_size,
            rates[sum(num_blocks[:1]):sum(num_blocks[:2])],
        )
        self.down2 = DownBlock(channels[1], channels[2], kernel_size, expansion)
        self.stage3 = self._make_stage(
            channels[2], num_blocks[2], expansion, kernel_size,
            rates[sum(num_blocks[:2]):sum(num_blocks[:3])],
        )
        self.down3 = DownBlock(channels[2], channels[3], kernel_size, expansion)
        self.stage4 = self._make_stage(
            channels[3], num_blocks[3], expansion, kernel_size,
            rates[sum(num_blocks[:3]):sum(num_blocks[:4])],
        )
        self.down4 = DownBlock(channels[3], channels[4], kernel_size, expansion)
        self.stage5 = self._make_stage(
            channels[4], num_blocks[4], expansion, kernel_size,
            rates[sum(num_blocks[:4]):],
        )

    @staticmethod
    def _make_stage(
        channels: int,
        num_blocks: int,
        expansion: int,
        kernel_size: int,
        drop_path_rates: List[float],
    ) -> nn.Sequential:
        return nn.Sequential(*[
            ConvBlock(
                channels,
                expansion,
                kernel_size,
                drop_path=drop_path_rates[index],
            )
            for index in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        s1 = self.stage1(x)
        s2 = self.stage2(self.down1(s1))
        s4 = self.stage3(self.down2(s2))
        s8 = self.stage4(self.down3(s4))
        s16 = self.stage5(self.down4(s8))
        return {'s1': s1, 's2': s2, 's4': s4, 's8': s8, 's16': s16}


class QuarterResolutionTPLDecoder(nn.Module):
    """TPL decoder split around its 1/4-resolution stage."""

    def __init__(
        self,
        base_channels: int,
        num_prototypes: int,
        num_classes: int,
        num_blocks: List[int],
        use_gnn: bool,
    ) -> None:
        super().__init__()
        del num_prototypes  # Kept in the interface for configuration symmetry.
        channels = [base_channels * (2 ** index) for index in range(5)]
        self.use_gnn = use_gnn
        if use_gnn:
            self.gnn = LightweightGNN(channels[2], num_layers=4)

        self.up1 = UpBlock(channels[4], channels[3])
        self.stage1 = self._make_stage(channels[3], num_blocks[0])
        self.up2 = UpBlock(channels[3], channels[2])
        self.stage2 = self._make_stage(channels[2], num_blocks[1])
        self.up3 = UpBlock(channels[2], channels[1])
        self.stage3 = self._make_stage(channels[1], num_blocks[2])
        self.up4 = UpBlock(channels[1], channels[0])
        self.stage4 = self._make_stage(channels[0], num_blocks[3])

        self.output_head = nn.Sequential(
            nn.Conv3d(channels[0], channels[0], kernel_size=3, padding=1),
            nn.GroupNorm(min(32, channels[0]), channels[0]),
            nn.GELU(),
            nn.Conv3d(channels[0], num_classes, kernel_size=1),
        )
        self.ds_heads = nn.ModuleDict({
            's8': nn.Conv3d(channels[3], num_classes, kernel_size=1),
            's4': nn.Conv3d(channels[2], num_classes, kernel_size=1),
            's2': nn.Conv3d(channels[1], num_classes, kernel_size=1),
        })

    @staticmethod
    def _make_stage(channels: int, num_blocks: int) -> nn.Sequential:
        return nn.Sequential(*[
            ConvBlock(channels, expansion=2, kernel_size=3)
            for _ in range(num_blocks)
        ])

    def decode_to_quarter(
        self,
        encoder_features: Dict[str, torch.Tensor],
        return_deep_supervision: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Decode s16 -> d8 -> d4 and stop before graph refinement."""
        d8 = self.stage1(
            self.up1(encoder_features['s16'], encoder_features['s8'])
        )
        outputs: Dict[str, torch.Tensor] = {}
        if return_deep_supervision:
            outputs['ds_s8'] = self.ds_heads['s8'](d8)

        d4 = self.stage2(self.up2(d8, encoder_features['s4']))
        return d4, outputs

    def refine_quarter(
        self,
        d4: torch.Tensor,
        edge_weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply the four graph-convolution layers with H^(0) = F_1/4."""
        if not self.use_gnn or edge_weights is None:
            return d4

        graph_size = edge_weights.shape[1:4]
        feature_size = d4.shape[-3:]
        if graph_size != feature_size:
            edge_weights = F.interpolate(
                edge_weights.permute(0, 4, 1, 2, 3),
                size=feature_size,
                mode='trilinear',
                align_corners=False,
            ).permute(0, 2, 3, 4, 1)
        return self.gnn(d4, edge_weights)

    def decode_from_quarter(
        self,
        d4: torch.Tensor,
        encoder_features: Dict[str, torch.Tensor],
        outputs: Optional[Dict[str, torch.Tensor]] = None,
        return_deep_supervision: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Decode the refined d4 feature through d2 and d1."""
        outputs = {} if outputs is None else outputs
        if return_deep_supervision:
            outputs['ds_s4'] = self.ds_heads['s4'](d4)

        d2 = self.stage3(self.up3(d4, encoder_features['s2']))
        if return_deep_supervision:
            outputs['ds_s2'] = self.ds_heads['s2'](d2)

        d1 = self.stage4(self.up4(d2, encoder_features['s1']))
        outputs['logits'] = self.output_head(d1)
        return outputs


class TPL(nn.Module):
    """TPL with prototype learning integrated at decoder resolution 1/4."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 32,
        num_prototypes: int = 64,
        num_foreground: int = 60,
        encoder_blocks: List[int] = [2, 2, 2, 2, 2],
        decoder_blocks: List[int] = [2, 2, 2, 2],
        lambda_seg: float = 0.2,
        lambda_topo_sup: float = 1.0,
        lambda_geodesic: float = 3.0,
        lambda_spectral: float = 0.5,
        lambda_contrastive_ii: float = 1.0,
        lambda_contrastive_ip: float = 0.5,
        lambda_diversity: float = 1.0,
        lambda_entropy: float = 1.0,
        lambda_centroid: float = 0.5,
        lambda_topology_feature: float = 0.1,
        assignment_temperature: float = 0.1,
        num_sample_pairs: int = 16,
        use_gnn: bool = True,
        use_proto_attention: bool = False,
        dilation_radius: int = 2,
    ) -> None:
        super().__init__()
        del use_proto_attention  # Disabled in this configuration.

        self.num_prototypes = num_prototypes
        self.num_foreground = num_foreground
        self.num_classes = num_classes
        self.feature_dim = base_channels * 4
        self.dilation_radius = dilation_radius
        self.mode = None

        self.lambda_seg = lambda_seg
        self.lambda_topo_sup = lambda_topo_sup
        self.lambda_geodesic = lambda_geodesic
        self.lambda_spectral = lambda_spectral
        self.lambda_contrastive_ii = lambda_contrastive_ii
        self.lambda_contrastive_ip = lambda_contrastive_ip
        self.lambda_diversity = lambda_diversity
        self.lambda_entropy = lambda_entropy
        self.lambda_centroid = lambda_centroid
        self.lambda_topology_feature = lambda_topology_feature

        self.encoder = TPLEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            num_blocks=encoder_blocks,
            kernel_size=5,
            expansion=4,
            drop_path_rate=0.0,
        )
        self.prototype_proj = nn.Conv3d(
            self.feature_dim, self.feature_dim, kernel_size=1
        )
        self.prototype_module = TopologicalPrototype(
            num_prototypes=num_prototypes,
            num_foreground=num_foreground,
            feature_dim=self.feature_dim,
            beta=0.0,
            temperature=assignment_temperature,
        )
        self.graph_constructor = GraphConstructor(num_prototypes=num_prototypes)
        self.decoder = QuarterResolutionTPLDecoder(
            base_channels=base_channels,
            num_prototypes=num_prototypes,
            num_classes=num_classes,
            num_blocks=decoder_blocks,
            use_gnn=use_gnn,
        )
        self.topo_predictor = TopologicalPredictor(feature_dim=self.feature_dim)

        self.seg_loss = SegmentationLoss()
        self.topo_sup_loss = TopologySupervisionLoss(
            num_prototypes=num_prototypes,
            num_foreground=num_foreground,
        )
        self.topo_aware_loss = TopologyAwareLoss(
            num_prototypes=num_prototypes,
            num_sample_pairs=num_sample_pairs,
        )
        self.ii_contrastive = InstanceToInstanceContrastive()
        self.ip_contrastive = InstanceToPrototypeContrastive()
        self.pp_diversity = PrototypeDiversityLoss()
        self.ip_entropy = InstanceToPrototypeEntropy(num_foreground)

        self.register_buffer(
            'dilation_kernel', self._create_dilation_kernel(dilation_radius)
        )

        # Use latent-only prototype assignment.
        self.prototype_module.beta = 0.0

    def stage_switch(self, epoch: int) -> None:
        if epoch < 100:
            self.lambda_contrastive_ip = 0
            self.lambda_geodesic = 0
            self.lambda_spectral = 0
            self.lambda_topo_sup = 0
            self.mode = 0
            print(f"epoch{epoch},TPL pretrain stage")
        else:
            self.lambda_contrastive_ip = 1.0
            self.lambda_geodesic = 3.0
            self.lambda_spectral = 0.5
            self.lambda_seg = 0.2
            self.lambda_topo_sup = 1.0
            self.mode = 1
            print(f"epoch{epoch},TPL finetune stage")

    @staticmethod
    def _create_dilation_kernel(radius: int) -> torch.Tensor:
        size = 2 * radius + 1
        coordinates = torch.arange(size, dtype=torch.float32) - radius
        zz, yy, xx = torch.meshgrid(
            coordinates, coordinates, coordinates, indexing='ij'
        )
        kernel = (zz.square() + yy.square() + xx.square()).sqrt() <= radius
        return kernel.float().unsqueeze(0).unsqueeze(0)

    def dilate_skeleton(self, skeleton: torch.Tensor) -> torch.Tensor:
        squeeze_output = skeleton.dim() == 4
        if squeeze_output:
            skeleton = skeleton.unsqueeze(1)
        dilated = F.conv3d(
            skeleton.float(),
            self.dilation_kernel,
            padding=self.dilation_radius,
        )
        dilated = (dilated > 0).float()
        return dilated.squeeze(1) if squeeze_output else dilated

    @staticmethod
    def boost_gradient(x: torch.Tensor, multiplier: float) -> torch.Tensor:
        return x * multiplier + x.detach() * (1 - multiplier)

    @staticmethod
    def _downsample_skeleton(
        skeleton: torch.Tensor,
        spatial_size: Tuple[int, int, int],
    ) -> torch.Tensor:
        if skeleton.dim() == 5:
            skeleton = skeleton.squeeze(1)
        return F.adaptive_max_pool3d(
            skeleton.unsqueeze(1).float(), output_size=spatial_size
        ).squeeze(1)

    def _extract_decoder_quarter(
        self,
        x: torch.Tensor,
        return_deep_supervision: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        encoder_features = self.encoder(x)
        d4, decoder_outputs = self.decoder.decode_to_quarter(
            encoder_features,
            return_deep_supervision=return_deep_supervision,
        )
        return encoder_features, d4, decoder_outputs

    def assignment_centroids(
        self,
        features: torch.Tensor,
        Q: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return normalized Q-weighted feature centroids and their evidence."""
        feature_flat = F.normalize(features.float(), dim=1).flatten(2)
        q_flat = Q.float().flatten(2)
        evidence = q_flat.sum(dim=(0, 2))
        centroids = torch.einsum('bkn,bcn->kc', q_flat, feature_flat)
        centroids = centroids / evidence[:, None].clamp_min(1e-6)
        centroids = F.normalize(centroids, dim=1)
        return centroids, evidence

    def semantic_prototypes(
        self,
        features: torch.Tensor,
        Q: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Use on-manifold centroids for foreground topology prediction."""
        implicit = F.normalize(self.prototype_module.implicit_prototypes, dim=1)
        centroids, evidence = self.assignment_centroids(features, Q)
        semantic = implicit.clone()
        valid_fg = evidence[:self.num_foreground] > 1e-4
        semantic[:self.num_foreground] = torch.where(
            valid_fg[:, None],
            centroids[:self.num_foreground],
            implicit[:self.num_foreground],
        )
        return semantic, centroids, evidence

    def prototype_centroid_loss(
        self,
        centroids: torch.Tensor,
        evidence: torch.Tensor,
    ) -> torch.Tensor:
        """Align queries to assigned features without topology supervision."""
        implicit = F.normalize(
            self.prototype_module.implicit_prototypes[:self.num_foreground],
            dim=1,
        )
        target = centroids[:self.num_foreground].detach()
        valid = evidence[:self.num_foreground] > 1e-4
        if not valid.any():
            return implicit.sum() * 0.0
        return (1.0 - (implicit[valid] * target[valid]).sum(dim=1)).mean()

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch['cta']

        # 1. Backbone decoding reaches F_1/4 before TPL is invoked.
        encoder_features, d4, decoder_outputs = self._extract_decoder_quarter(
            x, return_deep_supervision=self.training
        )

        # 2. Project decoder features into dense embeddings.
        proto_feat = self.prototype_proj(d4)

        if self.training:
            skeleton = batch['skeleton']
            skeleton_ds = self._downsample_skeleton(
                skeleton, proto_feat.shape[-3:]
            )
            with torch.no_grad():
                skeleton_dilated = self.dilate_skeleton(skeleton)
            proto_outputs = self.prototype_module(
                proto_feat,
                skeleton_dilated=skeleton_dilated,
                skeleton=skeleton_ds,
                topk=3,
            )
        else:
            proto_outputs = self.prototype_module(proto_feat)

        Q = proto_outputs['Q']

        # 3. Prototype semantics instantiate the graph at exactly d4's grid.
        prototypes, centroids, centroid_evidence = self.semantic_prototypes(
            proto_feat, Q
        )
        prototypes_boost = self.boost_gradient(prototypes, multiplier=3)
        topo_pred = self.topo_predictor(prototypes_boost)
        # Supervision uses topo_pred['port_probs']; graph/path optimization
        # uses the fixed degree-conditioned probability budget.
        port_probs = topo_pred['graph_port_probs']

        # Keep this graph differentiable: segmentation through the GNN can then
        # update the decoder embeddings, assignments and prototype semantics.
        edge_weights = self.graph_constructor.build_graph_with_grad(
            Q, port_probs
        )

        # 4. H^(0)=F_1/4 is graph-refined, then passed to the remaining decoder.
        d4_refined = self.decoder.refine_quarter(d4, edge_weights)
        decoder_outputs = self.decoder.decode_from_quarter(
            d4_refined,
            encoder_features,
            outputs=decoder_outputs,
            return_deep_supervision=self.training,
        )
        logits = decoder_outputs['logits']

        if not self.training:
            return {'logits': logits}

        losses: Dict[str, torch.Tensor] = {}
        seg_losses = self.seg_loss(logits, batch['label'])
        losses['loss_seg'] = self.lambda_seg * seg_losses['loss_seg']
        losses['loss_ii'] = self.lambda_contrastive_ii * self.ii_contrastive(
            features=proto_feat,
            skeleton_dilated=skeleton_dilated,
            skeleton=skeleton_ds,
        )

        if self.mode == 0:
            losses['loss_total'] = losses['loss_seg'] + losses['loss_ii']
            return losses

        topo_sup_losses = self.topo_sup_loss(
            features=proto_feat,
            mlp=self.topo_predictor,
            Q=Q,
            prototype_topo_pred=topo_pred,
            num_fg_prototype=self.num_foreground,
            degree_gt=batch['degree_map'],
            neighbor_mask_gt=batch['neighbor_mask'],
            skeleton=skeleton_ds,
        )
        for name in ('degree', 'port', 'consistency', 'sparsity'):
            losses[f'loss_{name}'] = (
                self.lambda_topo_sup * topo_sup_losses[f'loss_{name}']
            )
        losses['loss_topology_feature'] = (
            self.lambda_topology_feature
            * topo_sup_losses['loss_topology_feature']
        )

        topo_aware_losses = self.topo_aware_loss(
            Q=Q,
            port_probs=port_probs,
            skeleton_gt=skeleton_ds.unsqueeze(1),
        )
        losses['loss_backbone'] = (
            self.lambda_geodesic * topo_aware_losses['loss_backbone']
        )
        losses['loss_shortcut'] = (
            self.lambda_geodesic * topo_aware_losses['loss_shortcut']
        )
        losses['loss_spectral'] = (
            self.lambda_spectral * topo_aware_losses['loss_spectral']
        )

        losses['loss_ip'] = self.lambda_contrastive_ip * self.ip_contrastive(
            features=proto_feat,
            prototypes=self.prototype_module.implicit_prototypes,
            Q=Q,
            skeleton=skeleton_ds,
        )
        losses['loss_pp'] = self.lambda_diversity * self.pp_diversity(
            self.prototype_module.implicit_prototypes
        )
        losses['loss_e'] = self.lambda_entropy * self.ip_entropy(
            proto_outputs['fg_soft_sim_prob']
        )
        losses['loss_centroid'] = (
            self.lambda_centroid
            * self.prototype_centroid_loss(centroids, centroid_evidence)
        )
        losses['loss_total'] = sum(
            value
            for name, value in losses.items()
            if name.startswith('loss_')
            and name != 'loss_total'
            and isinstance(value, torch.Tensor)
        )
        return losses

    @torch.no_grad()
    def get_prototype_assignment(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, d4, _ = self._extract_decoder_quarter(x)
        proto_outputs = self.prototype_module(self.prototype_proj(d4))
        return proto_outputs['Q'], proto_outputs['Q'].argmax(dim=1)

    @torch.no_grad()
    def get_proto_feat(self, x: torch.Tensor) -> torch.Tensor:
        _, d4, _ = self._extract_decoder_quarter(x)
        return self.prototype_proj(d4)

    @torch.no_grad()
    def get_graph(self, x: torch.Tensor, topk: Optional[int] = None):
        _, d4, _ = self._extract_decoder_quarter(x)
        proto_feat = self.prototype_proj(d4)
        kwargs = {} if topk is None else {'topk': topk}
        proto_outputs = self.prototype_module(proto_feat, **kwargs)
        Q = proto_outputs['Q']

        prototypes = F.normalize(
            self.prototype_module.implicit_prototypes, dim=1
        )
        topo_pred = self.topo_predictor(prototypes)
        port_probs = topo_pred['graph_port_probs']
        edge_weights = self.graph_constructor.build_dense_graph(Q, port_probs)
        return Q, edge_weights, port_probs, topo_pred['degree_logits']

    @torch.no_grad()
    def change_prototype(
        self,
        proto_degree: torch.Tensor,
        threshold: float = 0.5,
        num_candidates: int = 256,
    ) -> int:
        prototypes = self.prototype_module.implicit_prototypes
        num_foreground = self.num_foreground
        foreground_indices = torch.arange(
            num_foreground, device=proto_degree.device
        )
        zero_degree_indices = foreground_indices[
            proto_degree[foreground_indices] == 0
        ]
        if zero_degree_indices.numel() == 0:
            return 0

        candidates = F.normalize(
            torch.randn(
                num_candidates,
                prototypes.shape[1],
                device=prototypes.device,
            ),
            dim=1,
        )
        candidate_degree = self.topo_predictor(candidates)[
            'degree_logits'
        ].argmax(dim=1)
        candidates = candidates[candidate_degree > 0]
        if candidates.shape[0] == 0:
            return 0

        keep_mask = torch.ones(
            num_foreground, dtype=torch.bool, device=proto_degree.device
        )
        keep_mask[zero_degree_indices] = False
        kept_prototypes = prototypes[foreground_indices[keep_mask]]
        if kept_prototypes.numel() > 0:
            candidates = candidates[
                (candidates @ kept_prototypes.T).max(dim=1).values < threshold
            ]

        num_replace = min(zero_degree_indices.numel(), candidates.shape[0])
        if num_replace > 0:
            prototypes[zero_degree_indices[:num_replace]] = candidates[:num_replace]
        return num_replace

    @torch.no_grad()
    def replace_prototype_epoch(self) -> None:
        prototypes = self.prototype_module.implicit_prototypes
        topo_pred = self.topo_predictor(F.normalize(prototypes.detach(), dim=1))
        proto_degree = topo_pred['degree_logits'].argmax(dim=1)
        num_replace = self.change_prototype(proto_degree)
        if num_replace > 0:
            print(f"[Prototype Refresh] replaced {num_replace} prototypes")

    @torch.no_grad()
    def inference(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        logits = self.forward({'cta': x})['logits']
        if logits.shape[1] == 1:
            return torch.sigmoid(logits)
        probabilities = F.softmax(logits, dim=1)
        if logits.shape[1] == 2:
            return probabilities[:, 1:2]
        return probabilities[:, 1:]


