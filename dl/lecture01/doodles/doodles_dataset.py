import sys
import argparse
from pathlib import Path
from typing import Union
from itertools import chain
from functools import partial
from multiprocessing import Pool, cpu_count

import feather
import numpy as np
import pandas as pd
from PIL import ImageDraw as PILDraw
from PIL import Image as PILImage

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from torchvision.models import resnet18, resnet34, resnet50, resnet101

from fastai import defaults
from fastai.callbacks.tracker import SaveModelCallback
from fastai.metrics import accuracy, error_rate
from fastai.vision import (
    ImageDataBunch,
    imagenet_stats,
    get_transforms,
    create_cnn,
    Image
)

from projects.logger import get_logger


PATH = Path.home()/'data'/'doodle'
TRAIN_DATA = PATH/'train'
PREPARED = PATH/'prepared'
DEBUG = False


ARCHS = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101
}


log = get_logger()


FloatOrInt = Union[float, int]


def main():
    args = parse_args()
    n, prefix = args['n_epochs'], args['prefix']
    bs, img_sz = args['batch_size'], args['image_size']
    prefix += '_' if prefix else ''

    bunch = create_data_bunch(bs, img_sz, args['train_size'], args['valid_size'], use_cache=args['use_cache'])
    train_sz, valid_sz = len(bunch.train_dl)/bunch.c, len(bunch.valid_dl)/bunch.c
    learn = create_cnn(bunch, args['network'])
    learn.metrics = [accuracy, error_rate]

    if args['continue']:
        log.info('Continue training using cached data')

    log.info('Epochs: %d', args['n_epochs'])
    log.info('Model: %s', args['network_name'])
    log.info('# of classes: %d', bunch.c)
    log.info('Train size (per class): %d', train_sz)
    log.info('Valid size (per class): %d', valid_sz)

    if args['continue']:
        cbs = [SaveModelCallback(learn, name='bestmodel_continue')]

        try:
            learn.load(f'{prefix}final_224')
        except Exception as e:
            log.error('Cannot restore model')
            log.error(e)
            sys.exit(1)

        learn.unfreeze()
        learn.fit_one_cycle(n, callabacks=cbs, max_lr=slice(3e-5, 3e-5))
        learn.save(f'{prefix}continued_224')

    else:
        cbs = [SaveModelCallback(learn)]

        learn.fit_one_cycle(1)
        learn.save(f'{prefix}one_224')

        learn.unfreeze()
        learn.freeze_to(-2)
        learn.fit_one_cycle(n - 2, max_lr=slice(1e-4, 1e-3))
        learn.save(f'{prefix}unfreeze_224')

        learn.unfreeze()
        learn.fit_one_cycle(1, callbacks=cbs, max_lr=slice(10e-5, 5e-5))
        learn.save(f'{prefix}final_224')

    log.info('Done!')


def parse_args():
    global DEBUG
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--prefix',
        default=None,
        help='Prefix to append to the name of file with saved model'
    )
    parser.add_argument(
        '-dev', '--device',
        default=None,
        help='PyTorch device used to train model'
    )
    parser.add_argument(
        '-bs', '--batch-size',
        default=256, type=int,
        help='Batch size'
    )
    parser.add_argument(
        '-sz', '--image-size',
        default=224, type=int,
        help='Image size'
    )
    parser.add_argument(
        '-t', '--train-size',
        default=1000, type=int,
        help='Number of observations (per category) to use for training'
    )
    parser.add_argument(
        '-v', '--valid-size',
        default=200, type=int,
        help='Number of observations (per category) to use for validation'
    )
    parser.add_argument(
        '-arch', '--network',
        default='resnet34', choices=sorted(list(ARCHS))
    )
    parser.add_argument(
        '-n', '--n-epochs',
        default=1, type=int,
        help='Number of training epochs'
    )
    parser.add_argument(
        '-cache', '--use-cache',
        action='store_true',
        help='Use previously parsed CSV data'
    )
    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help='Process file in single-threaded mode to simplify debugging'
    )
    parser.add_argument(
        '-c', '--continue',
        action='store_true',
        help='Loads previously trained model to continue training process; '
             'cache is enabled automatically.'
    )
    args = vars(parser.parse_args())
    args['network_name'] = args['network']
    args['network'] = ARCHS[args['network']]
    args['prefix'] = args['prefix'] or args['network_name']

    if args['device']:
        defaults.device = torch.device(args['device'])

    if args['continue']:
        args['use_cache'] = True

    DEBUG = args['debug']

    return args


def create_data_bunch(bs, img_sz, train_sz=None, valid_sz=None, use_cache=False):
    train_ds = QuickDraw(
        PREPARED,
        train=True,
        take_subset=True,
        use_cache=use_cache,
        subset_size=train_sz)

    valid_ds = QuickDraw(
        PREPARED,
        train=False,
        take_subset=True,
        use_cache=use_cache,
        subset_size=valid_sz)

    bunch = ImageDataBunch.create(
        train_ds, valid_ds, bs=bs, size=img_sz, ds_tfms=get_transforms())

    bunch.normalize(imagenet_stats)

    return bunch


def fastai_dataset(loss_func):
    """A class decorator to convert custom dataset into its fastai compatible version.

    The decorator attaches required properties to the dataset to use it with
    """
    def class_wrapper(dataset_cls):

        def get_n_classes(self):
            return len(self.classes)

        def get_loss_func(self):
            return loss_func

        dataset_cls.c = property(get_n_classes)
        dataset_cls.loss_func = property(get_loss_func)
        return dataset_cls

    return class_wrapper


@fastai_dataset(F.cross_entropy)
class QuickDraw(Dataset):

    img_size = (256, 256)

    def __init__(self, root: Path, train: bool=True, take_subset: bool=True,
                 subset_size: FloatOrInt=1000, bg_color='white',
                 stroke_color='black', lw=2, use_cache: bool=True):

        subfolder = root/('train' if train else 'valid')
        cache_file = subfolder.parent / 'cache' / f'{subfolder.name}.feather'

        if use_cache and cache_file.exists():
            log.info('Reading cached data from %s', cache_file)
            # walk around to deal with pd.read_feather nthreads error
            cats_df = feather.read_dataframe(cache_file)

        else:
            log.info('Parsing CSV files from %s...', subfolder)
            subset_size = subset_size if take_subset else None
            n_jobs = 1 if DEBUG else None
            cats_df = read_parallel(subfolder.glob('*.csv'), subset_size, n_jobs)
            if train:
                cats_df = cats_df.sample(frac=1)
            cats_df.reset_index(drop=True, inplace=True)
            log.info('Done! Parsed files saved into cache file')
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cats_df.to_feather(cache_file)

        targets = cats_df.word.values
        classes = np.unique(targets)
        class2idx = {v: k for k, v in enumerate(classes)}
        labels = np.array([class2idx[c] for c in targets])

        self.root = root
        self.train = train
        self.bg_color = bg_color
        self.stroke_color = stroke_color
        self.lw = lw
        self.data = cats_df.points.values
        self.classes = classes
        self.class2idx = class2idx
        self.labels = labels
        self._cached_images = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        points, target = self.data[item], self.labels[item]
        image = self.to_image_tensor(points)
        return image, target

    def to_image_tensor(self, points):
        img = to_pil_image(points, self.img_size, self.bg_color, self.stroke_color, self.lw)
        return Image(to_tensor(img))


def read_parallel(files, subset_size=None, n_jobs=None):
    worker = partial(read_csv, subset_size)
    n_jobs = n_jobs or cpu_count()
    with Pool(n_jobs) as pool:
        categories = pool.map(worker, files)
    df = pd.concat(categories)
    df.drop(columns=['countrycode', 'drawing', 'timestamp', 'recognized'], inplace=True)
    return df


def read_csv(subset_size, file):
    cat_df = pd.read_csv(file)
    cat_df = cat_df[cat_df.recognized]
    if subset_size is not None:
        n = _get_number_of_samples(cat_df, subset_size)
        cat_df = cat_df[:n]
    cat_df['points'] = cat_df.drawing.map(to_string)
    return cat_df


def _get_number_of_samples(df: pd.DataFrame, size: FloatOrInt):
    if isinstance(size, int):
        return size
    elif isinstance(size, float):
        if size <= 0:
            raise ValueError('sample size should be positive value')
        size = min(size, 1)
        return int(len(df) * size)

    raise ValueError(f'unexpected sample size value: {size}')


def to_pil_image(points, img_size, bg_color='white', stroke_color='black',
                 stroke_width=3):

    canvas = PILImage.new('RGB', img_size, color=bg_color)
    draw = PILDraw.Draw(canvas)
    for segment in points.split('|'):
        chunks = [int(x) for x in segment.split(',')]
        while len(chunks) >= 4:
            line, chunks = chunks[:4], chunks[2:]
            draw.line(tuple(line), fill=stroke_color, width=stroke_width)
    return canvas


def to_tuples(segments):
    return [list(zip(*segment)) for segment in eval(segments)]


def to_string(segments):
    strings = [','.join([str(x) for x in chain(*segment)]) for segment in to_tuples(segments)]
    string = '|'.join(strings)
    return string


if __name__ == '__main__':
    main()
