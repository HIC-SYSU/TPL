"""TPL-specific BF16 trainer built on light_training.Trainer."""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict

import torch

from model import TPL
from .trainer import Trainer



class TPLTrainingModule(torch.nn.Module):
    """Route both training phases through a single DDP forward call."""

    def __init__(self, model: TPL) -> None:
        super().__init__()
        self.tpl = model

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        phase: str,
    ) -> Dict[str, torch.Tensor]:
        if phase == 'warmup':
            return warmup_forward(self.tpl, batch)
        if phase == 'full':
            return self.tpl(batch)
        raise ValueError(f'unknown training phase: {phase}')


def unwrap_model(model: torch.nn.Module) -> TPL:
    """Return the TPL module from training and distributed wrappers."""
    module = model.module if hasattr(model, 'module') else model
    if isinstance(module, TPLTrainingModule):
        return module.tpl
    if isinstance(module, TPL):
        return module
    raise TypeError(f'unsupported model wrapper: {type(module).__name__}')

def warmup_forward(
    model: TPL,
    batch: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Train point semantics and prototypes before enabling path losses."""
    encoder_features, d4, decoder_outputs = model._extract_decoder_quarter(
        batch['cta'], return_deep_supervision=True
    )
    features = model.prototype_proj(d4)
    skeleton_ds = model._downsample_skeleton(
        batch['skeleton'], features.shape[-3:]
    )
    with torch.no_grad():
        skeleton_dilated = model.dilate_skeleton(batch['skeleton'])
    prototype_outputs = model.prototype_module(
        features,
        skeleton_dilated=skeleton_dilated,
        skeleton=skeleton_ds,
        topk=3,
    )
    q = prototype_outputs['Q']
    prototypes, centroids, evidence = model.semantic_prototypes(features, q)
    topology = model.topo_predictor(
        model.boost_gradient(prototypes, multiplier=3)
    )
    edges = model.graph_constructor.build_graph_with_grad(
        q, topology['graph_port_probs']
    )
    refined = model.decoder.refine_quarter(d4, edges)
    outputs = model.decoder.decode_from_quarter(
        refined,
        encoder_features,
        outputs=decoder_outputs,
        return_deep_supervision=True,
    )
    segmentation = model.seg_loss(outputs['logits'], batch['label'])['loss_seg']
    supervised = model.topo_sup_loss(
        features=features,
        mlp=model.topo_predictor,
        Q=q,
        prototype_topo_pred=topology,
        num_fg_prototype=model.num_foreground,
        degree_gt=batch['degree_map'],
        neighbor_mask_gt=batch['neighbor_mask'],
        skeleton=skeleton_ds,
    )
    losses = {
        'loss_seg': model.lambda_seg * segmentation,
        'loss_degree': supervised['loss_degree_fg'],
        'loss_port': supervised['loss_port_fg'],
        'loss_consistency': supervised['loss_consistency_fg'],
        'loss_sparsity': supervised['loss_sparsity_fg'],
        'loss_topology_feature': (
            model.lambda_topology_feature
            * supervised['loss_topology_feature']
        ),
        'loss_ip': model.lambda_contrastive_ip * model.ip_contrastive(
            features,
            model.prototype_module.implicit_prototypes,
            q,
            skeleton_ds,
        ),
        'loss_pp': model.lambda_diversity * model.pp_diversity(
            model.prototype_module.implicit_prototypes
        ),
        'loss_e': model.lambda_entropy * model.ip_entropy(
            prototype_outputs['fg_soft_sim_prob']
        ),
        'loss_centroid': model.lambda_centroid * model.prototype_centroid_loss(
            centroids, evidence
        ),
    }
    losses['loss_total'] = sum(losses.values())
    return losses


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    warmup_step: int,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save({
        'model': unwrap_model(model).state_dict(),
        'optimizer': optimizer.state_dict(),
        'warmup_step': warmup_step,
        'epoch': epoch,
        'global_step': global_step,
        'config': vars(args),
    }, temporary)
    temporary.replace(path)


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    phase: str,
    max_grad_norm: float,
) -> tuple[Dict[str, torch.Tensor], float]:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        losses = model(batch, phase=phase)
        total = losses['loss_total']
    if not torch.isfinite(total):
        raise RuntimeError(f'non-finite loss: {float(total.detach())}')
    total.backward()
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise RuntimeError('non-finite gradient')
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_grad_norm
    )
    optimizer.step()
    return losses, float(grad_norm)


def log_step(
    phase: str,
    step: int,
    losses: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    grad_norm: float,
    seconds: float,
) -> None:
    value = {
        key.removeprefix('loss_'): round(float(item.detach()), 5)
        for key, item in losses.items()
        if isinstance(item, torch.Tensor)
    }
    core_model = unwrap_model(model)
    contrast = core_model.topo_sup_loss.last_topology_contrast_metrics
    value.update({
        'phase': phase,
        'step': step,
        'seconds': round(seconds, 3),
        'grad_norm': round(grad_norm, 5),
        'sampled_degree_histogram': (
            core_model.topo_sup_loss.last_sampled_degree_histogram.tolist()
        ),
        'local_positive_pairs': int(contrast[0]),
        'local_negative_pairs': int(contrast[1]),
        'local_positive_similarity': round(float(contrast[2]), 5),
        'local_negative_similarity': round(float(contrast[3]), 5),
        'peak_memory_gib': round(
            torch.cuda.max_memory_allocated() / 1024 ** 3, 3
        ),
    })
    print(json.dumps(value, sort_keys=True), flush=True)


class TPLTrainer(Trainer):
    """Stable TPL objective on the copied light_training lifecycle."""

    def __init__(self, args: argparse.Namespace) -> None:
        if args.steps_per_epoch <= 0:
            raise ValueError('--steps-per-epoch must be positive')
        if args.warmup_steps % args.steps_per_epoch != 0:
            raise ValueError(
                '--warmup-steps must be divisible by --steps-per-epoch'
            )
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required')

        self.args = args
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.device_id = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(self.device_id)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('selected GPU does not support BF16')

        self.warmup_epochs = args.warmup_steps // args.steps_per_epoch
        total_epochs = self.warmup_epochs + args.epochs
        self.output = Path(args.output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.resume_step_offset = 0
        self.last_phase = None

        super().__init__(
            env_type='DDP',
            max_epochs=total_epochs,
            batch_size=1,
            num_step_per_epoch=args.steps_per_epoch * args.num_gpus,
            device=self.device_id,
            val_every=total_epochs + 1,
            num_gpus=args.num_gpus,
            logdir=str(self.output / 'tensorboard'),
            training_script=__file__,
            train_process=4,
            use_amp='bf16',
        )
        self.patch_size = [args.patch_size] * 3
        self.augmentation = False
        # Trainer owns data loading and epochs. This class keeps manual
        # optimization so every gradient is checked before the update.
        self.auto_optim = False
        tpl_model = TPL(
            base_channels=32,
            num_prototypes=64,
            num_foreground=60,
            assignment_temperature=0.1,
            lambda_topology_feature=0.1,
            num_sample_pairs=16,
        )
        tpl_model.stage_switch(100)
        self.model = TPLTrainingModule(tpl_model)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=args.lr, betas=(0.9, 0.95)
        )

    @property
    def absolute_step(self) -> int:
        return self.resume_step_offset + self.global_step

    def training_step(self, raw_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch = {
            'cta': raw_batch['data'].float(),
            'label': raw_batch['seg'].float(),
            'skeleton': raw_batch['skeleton'].float(),
            'degree_map': raw_batch['degree_map'].long(),
            'neighbor_mask': raw_batch['neighbor_mask'].float(),
        }
        phase = (
            'warmup'
            if self.absolute_step <= self.args.warmup_steps
            else 'full'
        )
        if phase != self.last_phase:
            if self.local_rank == 0:
                print(f'\n[phase] {phase} starts at step {self.absolute_step}')
            self.last_phase = phase

        start = time.perf_counter()
        losses, grad_norm = train_step(
            self.model,
            self.optimizer,
            batch,
            phase,
            self.args.max_grad_norm,
        )
        torch.cuda.synchronize(self.device_id)
        if self.local_rank == 0 and (
            self.absolute_step == 1
            or self.absolute_step % self.args.log_every == 0
        ):
            log_step(
                phase,
                self.absolute_step,
                losses,
                self.model,
                grad_norm,
                time.perf_counter() - start,
            )
            self.log(
                'training_loss',
                losses['loss_total'].item(),
                self.absolute_step,
            )
        return losses['loss_total']

    def validation_step(self, batch):
        raise RuntimeError('validation is disabled for TPL training')

    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device_id,
            weights_only=False,
        )
        core_model = unwrap_model(self.model)
        core_model.to(self.device_id)
        core_model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = int(checkpoint.get('epoch', 0))
        self.resume_step_offset = int(checkpoint.get('global_step', 0))
        if self.local_rank == 0:
            print(f'[resume] {checkpoint_path} at step {self.resume_step_offset}')

    def train_epoch(self, epoch: int) -> None:
        super().train_epoch(epoch)
        completed_epoch = epoch + 1
        if self.local_rank == 0:
            checkpoint_args = (
                self.model,
                self.optimizer,
                min(self.absolute_step, self.args.warmup_steps),
                completed_epoch,
                self.absolute_step,
                self.args,
            )
            save_checkpoint(self.output / 'latest.pt', *checkpoint_args)
            if completed_epoch == self.warmup_epochs and self.warmup_epochs > 0:
                save_checkpoint(self.output / 'warmup.pt', *checkpoint_args)
            if completed_epoch % self.args.save_every == 0:
                save_checkpoint(
                    self.output / f'epoch_{completed_epoch:03d}.pt',
                    *checkpoint_args,
                )
        torch.distributed.barrier(device_ids=[self.local_rank])


