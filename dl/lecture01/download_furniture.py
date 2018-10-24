import os
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from functools import partial
from multiprocessing import Pool, cpu_count

import requests
import pandas as pd


PATH = Path.home()/'data'/'furniture'
IMAGES = PATH/'images'
TRAIN_IMAGES = IMAGES/'train'
VALID_IMAGES = IMAGES/'valid'
TEST_IMAGES = IMAGES/'test'
LABELS = PATH/'labels.csv'


def main():
    for dirname in (IMAGES, TRAIN_IMAGES, VALID_IMAGES, TEST_IMAGES):
        os.makedirs(dirname, exist_ok=True)

    paths = [
        ('train', TRAIN_IMAGES),
        ('valid', VALID_IMAGES),
        ('test', TEST_IMAGES)]

    for name, path in paths:
        json_file = PATH/f'{name}.json'
        print(f'Downloading {json_file}...')
        info = download(json_file, path)
        info.to_pickle(IMAGES/f'{name}_info.pickle')

    print('Done!')


@dataclass
class ImageInfo:
    path: Path
    label_id: int
    image_id: int


def download(json_file, path, pct=0.1):
    with json_file.open() as file:
        content = json.load(file)

    images, labels = content['images'], content['annotations']
    urls = [img['url'][0] for img in images]

    records = pd.DataFrame([
        {'url': url, **lbl}
        for url, lbl in zip(urls, labels)])

    if pct is not None:
        pct = max(0.0, min(pct, 1.0))
        subsets = []
        for key, group in records.groupby('label_id'):
            size = int(len(group) * pct)
            subsets.extend(group.sample(size).to_dict('records'))
        records = pd.DataFrame(subsets)

    worker = partial(download_single, path)
    items = records.to_dict('records')
    with Pool(cpu_count()) as pool:
        data = pool.map(worker, items)
    meta = pd.DataFrame([asdict(info) for info in data])
    return meta


def download_single(folder, info):
    url = info['url']
    r = requests.get(url)
    try:
        r.raise_for_status()
    except requests.HTTPError:
        print(f'Cannot download URL: {url}')
    img_name = str(info['image_id']) + ".jpg"
    path = folder/img_name
    with path.open('wb') as file:
        file.write(r.content)
    return ImageInfo(path, info['label_id'], info['image_id'])


if __name__ == '__main__':
    main()
