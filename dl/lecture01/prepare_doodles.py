from pathlib import Path
from PIL import ImageDraw as PILDraw
from PIL import Image as PILImage
from cycler import cycler
from itertools import chain
import pandas as pd
import numpy as np
from typing import Union

from torch.utils.data import Dataset
from torch.nn import functional as F
from PIL import Image, ImageDraw

# from fastai import *
from fastai.vision import *
from sklearn.model_selection import train_test_split
from torchvision.models import resnet34


PATH = Path.home()/'data'/'doodle'
PREPARED = PATH/'prepared'
TRAIN_DATA = PATH/'train'
RANDOM_STATE = 1

np.random.seed(RANDOM_STATE)


FloatOrInt = Union[float, int]


def main():
    X_train, X_valid, y_train, y_valid = read_subset(PATH, sample_size=100)
    train_ds = QuickDraw(X_train, y_train)
    valid_ds = QuickDraw(X_valid, y_valid)
    bunch = ImageDataBunch.create(train_ds, valid_ds, bs=40, ds_tfms=get_transforms(), size=224)
    bunch.normalize(imagenet_stats)

    learn = ConvLearner(bunch, resnet34)

    learn.fit_one_cycle(4)


def read_subset(path: Path, sample: bool=True, sample_size: int=1000, random_state=1):
    categories = []

    for file in (path/'train').glob('*.csv'):
        if sample:
            chunks = pd.read_csv(file, chunksize=sample_size)
            recognized = []
            while True:
                chunk = next(chunks)
                chunk = chunk[chunk.recognized]
                recognized.extend(chunk.to_dict('records'))
                if len(recognized) >= sample_size:
                    break
            cat_df = pd.DataFrame(recognized)[:sample_size]
        else:
            cat_df = pd.read_csv(file)
        categories.append(cat_df)

    cats_df = pd.concat(categories)

    return train_test_split(
        cats_df.joined, cats_df.word,
        train_size=0.8, random_state=random_state)



# def fastai_dataset(f):
#     return None


# @fastai_dataset
class QuickDraw(Dataset):

    img_size = (256, 256)

    # def __init__(self, root: Path, train: bool=True, sample: bool=True,
    #              sample_size: FloatOrInt=1000, random_state=None,
    #              img_bg='white', img_fg='black', lw=2):
    #
    #     self.root = root
    #     self.train = train
    #     self.random_state = random_state
    #     self.img_bg = img_bg
    #     self.img_fg = img_fg
    #     self.lw = lw
    #
    #     subfolder = root/('train' if train else 'test')
    #     categories = []
    #
    #     for file in subfolder.glob('*.csv'):
    #         cat_df = pd.read_csv(file)
    #         cat_df = cat_df[cat_df.recognized]
    #         if sample:
    #             n = _get_number_of_samples(cat_df, sample_size)
    #             cat_df = cat_df.sample(n, random_state=random_state)
    #         cat_df['joined'] = cat_df.drawing.map(to_tuples)
    #         cat_df.drop(columns=['countrycode', 'drawing', 'timestamp', 'recognized'], inplace=True)
    #         categories.append(cat_df)
    #
    #     cats_df = pd.concat(categories)
    #     cats_df.sort_values(by=['key_id'], inplace=True)
    #     self.data = cats_df.joined.tolist()
    #     self.labels = cats_df.word.tolist()

    def __init__(self, strokes, words, img_bg='white', img_fg='black', lw=2):
        self.img_bg = img_bg
        self.img_fg = img_fg
        self.lw = lw
        self.data = strokes
        self.labels = words

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        points, target = self.data[item], self.labels[item]
        canvas = PILImage.new('RGB', self.img_size, color=self.img_bg)
        draw = ImageDraw.Draw(image)
        prev_pts, next_pts = points[:-1], points[1:]
        for (x1, y1), (x2, y2) in zip(prev_pts, next_pts):
            draw.line((x1, y1, x2, y2), fill=self.img_fg, width=self.lw)
        return canvas, target


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


if __name__ == '__main__':
    main()
