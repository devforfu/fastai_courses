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

from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.models import resnet34
from torchvision.transforms.functional import to_tensor
from fastai.vision import Image
from fastai.callbacks.tracker import SaveModelCallback
from fastai.vision import create_cnn, ImageDataBunch, imagenet_stats, get_transforms

from logger import get_logger


PATH = Path.home()/'data'/'doodle'
TRAIN_DATA = PATH/'train'
PREPARED = PATH/'prepared'
DEBUG = False


log = get_logger()


FloatOrInt = Union[float, int]


def main():
    args = parse_args()

    train_ds = QuickDraw(PREPARED, train=True, take_subset=True, use_cache=args['use_cache'])
    valid_ds = QuickDraw(PREPARED, train=False, take_subset=True, use_cache=args['use_cache'])
    bunch = ImageDataBunch.create(
        train_ds, valid_ds,
        bs=args['batch_size'], size=args['image_size'], ds_tfms=get_transforms())
    bunch.normalize(imagenet_stats)

    learn = create_cnn(bunch, resnet34)
    cbs = [SaveModelCallback(learn)]
    learn.fit_one_cycle(args['n_epochs'], callbacks=cbs)
    learn.save('sz_224')

    log.info('Done!')


def parse_args():
    global DEBUG
    parser = argparse.ArgumentParser()
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
    args = vars(parser.parse_args())
    DEBUG = args['debug']
    return args


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
        image = self.to_pil_image(points)
        return image, target

    def to_pil_image(self, points):
        canvas = PILImage.new('RGB', self.img_size, color=self.bg_color)
        draw = PILDraw.Draw(canvas)
        for segment in points.split('|'):
            chunks = [int(x) for x in segment.split(',')]
            while len(chunks) >= 4:
                line, chunks = chunks[:4], chunks[2:]
                draw.line(tuple(line), fill=self.stroke_color, width=self.lw)
        image = Image(to_tensor(canvas))
        return image


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


def to_pil_image(points, size, bg_color='white', stroke_color='black', lw=2):
    canvas = PILImage.new('RGB', size, color=bg_color)
    draw = PILDraw.Draw(canvas)
    chunks = [int(x) for x in points.split(',')]
    while len(chunks) >= 4:
        line, chunks = chunks[:4], chunks[2:]
        draw.line(tuple(line), fill=stroke_color, width=lw)
    image = Image(to_tensor(canvas))
    return image


def to_tuples(segments):
    return [list(zip(*segment)) for segment in eval(segments)]
    # return list(chain(*[zip(*segment) for segment in eval(segments)]))


def to_string(segments):
    strings = [','.join([str(x) for x in chain(*segment)]) for segment in to_tuples(segments)]
    string = '|'.join(strings)
    return string


    # return ','.join([str(x) for x in chain(*to_tuples(segments))])


if __name__ == '__main__':
    main()
