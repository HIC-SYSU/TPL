import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from dataset import TPLTopologyDataset
from light_training.trainer_tpl import TPLTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', default='/runs/tpl')
    parser.add_argument(
        '--gpus',
        default='2,3',
        help='Comma-separated physical GPU IDs used by DDP',
    )
    parser.add_argument('--patch-size', type=int, default=160)
    parser.add_argument('--intensity-lower', type=float, default=-200.0)
    parser.add_argument('--intensity-upper', type=float, default=800.0)
    parser.add_argument('--warmup-steps', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--steps-per-epoch', type=int, default=250)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max-grad-norm', type=float, default=12.0)
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--save-every', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--resume', default=None)
    return parser.parse_args()


def parse_gpu_ids(value: str) -> list[int]:
    try:
        gpu_ids = [int(item.strip()) for item in value.split(',') if item.strip()]
    except ValueError as error:
        raise ValueError('--gpus must be a comma-separated list of integers') from error
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError('--gpus must contain unique GPU IDs')
    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError('--gpus cannot contain negative GPU IDs')
    return gpu_ids


def launch_ddp(args: argparse.Namespace) -> bool:
    """Launch one worker per requested GPU when called as a normal script."""
    if 'LOCAL_RANK' in os.environ:
        return False

    gpu_ids = parse_gpu_ids(args.gpus)
    environment = os.environ.copy()
    environment['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
    command = [
        sys.executable,
        '-m',
        'torch.distributed.run',
        '--standalone',
        '--nproc_per_node',
        str(len(gpu_ids)),
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    print(
        f"Launching DDP on physical GPUs {gpu_ids} "
        f"with {len(gpu_ids)} processes",
        flush=True,
    )
    subprocess.run(command, env=environment, check=True)
    return True


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    if launch_ddp(args):
        return

    gpu_ids = parse_gpu_ids(args.gpus)
    args.num_gpus = len(gpu_ids)
    rank = int(os.environ.get('RANK', 0))
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    trainer = TPLTrainer(args)
    if args.resume:
        trainer.load_checkpoint(args.resume)

    if rank == 0:
        print(json.dumps({
            'devices': gpu_ids,
            'world_size': args.num_gpus,
            'precision': trainer.use_amp,
            'trainer': 'light_training.Trainer',
            'patch_size': args.patch_size,
            'intensity_range': [args.intensity_lower, args.intensity_upper],
            'num_prototypes': 64,
            'num_foreground': 60,
            'beta': 0.0,
            'rare_degree_sampling': False,
            'cross_patch_queue': False,
            'warmup_steps': args.warmup_steps,
            'full_epochs': args.epochs,
            'steps_per_epoch': args.steps_per_epoch,
        }, sort_keys=True), flush=True)

    train_dataset = TPLTopologyDataset(
        is_train=True,
        fold_th=1,
        k_fold=4,
        dataset_txt_path=args.data,
        intensity_lower=args.intensity_lower,
        intensity_upper=args.intensity_upper,
    )
    torch.cuda.reset_peak_memory_stats(trainer.device_id)
    trainer.train(train_dataset=train_dataset, val_dataset=None)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
