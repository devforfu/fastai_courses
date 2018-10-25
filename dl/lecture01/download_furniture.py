import os
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from functools import partial
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from fastai.core import partition

from logger import get_logger


PATH = Path.home()/'data'/'furniture'
IMAGES = PATH/'images'
TRAIN_IMAGES = IMAGES/'train'
VALID_IMAGES = IMAGES/'valid'
TEST_IMAGES = IMAGES/'test'
LABELS = PATH/'labels.csv'
RANDOM_STATE = 1


np.random.seed(RANDOM_STATE)
log = get_logger()


def main():
    for dirname in (IMAGES, TRAIN_IMAGES, VALID_IMAGES, TEST_IMAGES):
        os.makedirs(dirname, exist_ok=True)

    paths = [
        ('train', TRAIN_IMAGES),
        ('valid', VALID_IMAGES),
        ('test', TEST_IMAGES)]

    for name, path in paths:
        json_file = PATH/f'{name}.json'
        index_file = PATH/f'{name}_index.csv'

        if not index_file.exists():
            prepare_url_index(json_file, index_file)

        log.info(f'Downloading {json_file}...')
        index = pd.read_csv(index_file)
        info = download(index, path)
        info.to_pickle(IMAGES/f'{name}_info.pickle')

    log.info('Done!')


def prepare_url_index(json_file, index_file, pct=0.1):
    """Saves meta-information about images into CSV file.

    Args:
        json_file: Path to JSON file with dataset information.
        index_file: Path to CSV file to save image URL, label ID, and image ID
        pct: Percentage of dataset to take.

    """
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
            subsets.extend(group.sample(size, random_state=RANDOM_STATE).to_dict('records'))
        records = pd.DataFrame(subsets)

    records.to_csv(index_file, index=None)


@dataclass
class ImageInfo:
    path: Path
    label_id: int
    image_id: int


def download(index, path):
    """Downloads images with URLs from index dataframe."""

    n_cpu = cpu_count()
    worker = partial(download_single, path)
    queue = index.to_dict('records')
    meta = []

    with Pool(n_cpu) as pool:
        n = 1000
        chunk_size = len(queue)//n
        chunks = partition(queue, chunk_size)
        n_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            log.info('Downloading chunk %d of %d' % (i+1, n_chunks))
            data = [x for x in pool.imap_unordered(worker, chunk) if x]
            meta.extend([asdict(info) for info in data])

    return pd.DataFrame(meta)


def download_single(folder, info):
    url = info['url']
    img_name = str(info['image_id']) + ".jpg"
    path = folder / img_name
    info = ImageInfo(path, info['label_id'], info['image_id'])

    if path.exists():
        log.info('File already downloaded: %s', img_name)
        return info

    try:
        r = requests.get(url)
        r.raise_for_status()
    except requests.HTTPError:
        log.warning(f'Cannot download URL: {url}')
        return None
    except requests.ConnectionError:
        log.warning(f'Connection error! URL: {url}')
        return None
    except Exception:
        log.warning(f'Unexpected error while downloading URL: {url}')
        return None
    with path.open('wb') as file:
        file.write(r.content)

    return info


if __name__ == '__main__':
    main()
