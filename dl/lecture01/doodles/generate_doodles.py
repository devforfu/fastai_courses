import argparse
from pathlib import Path

from doodles_dataset import read_parallel


PATH = Path.home()/'data'/'doodle'


def main():
    args = parse_args()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--subset',
        default='train',
        choices=['train', 'test'],
        help=''
    )
    parser.add_argument(
        '--image-size',
        default=256, type=int,
        help=''
    )
    args = parser.parse_args()
    if args.subset == 'train':
        args.path = PATH/'train'
        args.train = True
    else:
        args.path = PATH/'test'/'test_simplified.csv'
        args.train = False
    return args


if __name__ == '__main__':
    main()
