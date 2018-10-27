
from pathlib import Path
from PIL import ImageDraw as PILDraw
from PIL import Image as PILImage
from cycler import cycler
from itertools import chain
import pandas as pd
import numpy as np
from typing import Union
from multiprocessing import Pool, cpu_count

from torch.utils.data import Dataset
from torch.nn import functional as F
from functools import partial
from fastai.vision import Image

# from fastai import *
from fastai.vision import ConvLearner, ImageDataBunch, imagenet_stats, get_transforms
from sklearn.model_selection import train_test_split
from torchvision.transforms.functional import to_tensor
from torchvision.models import resnet34
from logger import get_logger
import feather

PATH = Path.home()/'data'/'doodle'/'prepared'


log = get_logger()


FloatOrInt = Union[float, int]


def main():
    train_ds = QuickDraw(PATH, train=True, take_subset=True)
    valid_ds = QuickDraw(PATH, train=False, take_subset=True)
    bunch = ImageDataBunch.create(train_ds, valid_ds, bs=40, size=224, ds_tfms=get_transforms())
    bunch.normalize(imagenet_stats)


    learn = ConvLearner(bunch, resnet34)

    learn.fit_one_cycle(1)


# def fastai_dataset(dataset_cls):
#
#     def as_tensor(self, )
#
#     dataset_cls.__getitem__ =



# @fastai_dataset
class QuickDraw(Dataset):

    img_size = (256, 256)

    def __init__(self, root: Path, train: bool=True, take_subset: bool=True,
                 subset_size: FloatOrInt=1000, bg_color='white',
                 stroke_color='black', lw=2, use_cache: bool=True):

        subfolder = root/('train' if train else 'valid')
        cache_file = subfolder.parent / 'cache' / f'{subfolder.name}.feather'

        if use_cache and cache_file.exists():
            log.info('Reading cached data')
            # walk around to deal with pd.read_feather nthreads error
            cats_df = feather.read_dataframe(cache_file)

        else:
            log.info('Parsing CSV files...')
            subset_size = subset_size if take_subset else None
            cats_df = _read_parallel(subfolder.glob('*.csv'), subset_size)
            if train:
                cats_df = cats_df.sample(frac=1)
            cats_df.reset_index(drop=True, inplace=True)
            log.info('Done! Parsed files saved into cache file')
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cats_df.to_feather(cache_file)

        self.root = root
        self.train = train
        self.bg_color = bg_color
        self.stroke_color = stroke_color
        self.lw = lw
        self.data = cats_df.points.values
        self.labels = cats_df.word.values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        points, target = self.data[item], self.labels[item]
        canvas = PILImage.new('RGB', self.img_size, color=self.bg_color)
        draw = PILDraw.Draw(canvas)
        chunks = [int(x) for x in points.split(',')]
        while len(chunks) >= 4:
            line, chunks = chunks[:4], chunks[2:]
            draw.line(tuple(line), fill=self.stroke_color, width=self.lw)
        image = Image(to_tensor(canvas))
        return image, target


def _read_parallel(files, subset_size=None, n_jobs=None):
    worker = partial(_read_csv, subset_size)
    n_jobs = n_jobs or cpu_count()
    with Pool(n_jobs) as pool:
        categories = pool.map(worker, files)
    df = pd.concat(categories)
    df.drop(columns=['countrycode', 'drawing', 'timestamp', 'recognized'], inplace=True)
    return df


def _read_csv(subset_size, file):
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


def to_tuples(segments):
    return list(chain(*[zip(*segment) for segment in eval(segments)]))


def to_string(segments):
    return ','.join([str(x) for x in chain(*to_tuples(segments))])


if __name__ == '__main__':
    main()
