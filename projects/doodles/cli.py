import argparse
from pathlib import Path

import torch

from logger import get_logger, LOG_FILE_NAME


def parse_args(archs):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--suffix',
        default=None,
        help='Suffix to append to the name of file with saved model'
    )
    parser.add_argument(
        '-ts', '--train-size',
        default=1000, type=int,
        help='Number of observations (per category) to use for training'
    )
    parser.add_argument(
        '-vs', '--valid-size',
        default=200, type=int,
        help='Number of observations (per category) to use for validation'
    )
    parser.add_argument(
        '-csv', '--csv-file',
        default=Path.home()/'data'/'doodle'/'prepared'
    )

    parser.add_argument(
        '-dev', '--device',
        default=1, type=int,
        help='PyTorch device used to train model'
    )
    parser.add_argument(
        '-arch', '--network',
        default='resnet34', choices=sorted(list(archs))
    )
    parser.add_argument(
        '-n', '--n-epochs',
        default=1, type=int,
        help='Number of training epochs'
    )
    parser.add_argument(
        '-bs', '--batch-size',
        default=256, type=int,
        help='Batch size'
    )
    parser.add_argument(
        '-sz', '--image-size',
        default=224, type=int,
        help='Size to rescale images before passing into model'
    )

    parser.add_argument(
        '-p', '--models-path',
        default=Path.home() / 'models' / 'doodle',
        help='Path to save trained models on learn.save() call'
    )
    parser.add_argument(
        '-log', '--logging',
        default=Path.home() / 'models' / 'doodle' / LOG_FILE_NAME
    )

    args = parser.parse_args()
    args.device = torch.device(f'cuda:{args.device}' if args.device >= 0 else 'cpu')
    args.arch_name = args.network
    args.network = archs[args.network]

    Path(args.models_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.logging).parent.mkdir(parents=True, exist_ok=True)

    args.log = get_logger(log_path=args.logging)

    return args
