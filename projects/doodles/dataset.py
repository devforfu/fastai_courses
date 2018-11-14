from functools import partial
from itertools import chain
from typing import Union
from pathlib import Path
from multiprocessing import Pool, cpu_count

import feather
from fastai.vision import Image
import numpy as np
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from torchvision.datasets.folder import pil_loader, has_file_allowed_extension, IMG_EXTENSIONS
import pandas as pd
from PIL import Image as PILImage
from PIL import ImageDraw as PILDraw

from logger import get_logger


FloatOrInt = Union[float, int]


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
                 subset_size: FloatOrInt=1000, bg_color='black',
                 stroke_color='white', lw=2, use_cache: bool=True,
                 parallel=True, log=None):

        log = log or get_logger()
        subfolder = root/('train' if train else 'valid')
        cache_file = subfolder.parent / 'cache' / f'{subfolder.name}_{subset_size}.feather'

        if use_cache and cache_file.exists():
            log.info('Reading cached data from %s', cache_file)
            # walk around to deal with pd.read_feather nthreads error
            cats_df = feather.read_dataframe(cache_file)

        else:
            log.info('Parsing CSV files from %s', subfolder)
            subset_size = subset_size if take_subset else None
            n_jobs = None if parallel else 1
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


class TestImagesFolder(Dataset):

    def __init__(self, path, loader=pil_loader, pseudolabel=0):
        path = Path(path)

        assert path.is_dir() and path.exists(), 'Not a directory!'
        assert path.stat().st_size > 0, 'Directory is empty'

        images = [
            file for file in path.iterdir()
            if has_file_allowed_extension(str(file), IMG_EXTENSIONS)]

        self.path = path
        self.loader = loader
        self.images = images
        self.pseudolabel = pseudolabel

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img = self.loader(self.images[item])
        return img, self.pseudolabel