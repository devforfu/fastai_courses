"""
Converting FER2013 dataset from CSV representation into folder with images.

The dataset is taken from:
    https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge

Encoding:
    (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).

"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


PATH = Path.home()/'data'/'facial_expressions'/'fer2013'
INPUT_FILE = PATH/'fer2013.csv'
OUTPUT_DIR = PATH/'images'
IMG_SZ = 48

VERBOSE_NAMES = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'neutral'
}


def main():
    data = pd.read_csv(INPUT_FILE)
    data['pixels'] = data.pixels.str.split()
    data['emotion'] = data.emotion.map(VERBOSE_NAMES)
    data.loc[(data.Usage == 'Training') | (data.Usage == 'PublicTest'), 'Usage'] = 'train'
    data.loc[data.Usage == 'PrivateTest', 'Usage'] = 'valid'

    for subset, s_group in data.groupby('Usage'):
        for emotion, e_group in s_group.groupby('emotion'):
            path = OUTPUT_DIR/subset/emotion
            print('Creating %s' % path)
            path.mkdir(parents=True, exist_ok=True)
            for i, row in e_group.iterrows():
                np_pixels = np.array([float(p) for p in row.pixels])
                np_img = np_pixels.reshape(IMG_SZ, IMG_SZ)
                pil_img = Image.fromarray(np.uint8(np_img))
                pil_img.save(path/f'{i}.png', format='png')

    print('Done!')


if __name__ == '__main__':
    main()

