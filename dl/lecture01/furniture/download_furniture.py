import os
import json
import argparse
from io import BytesIO
from pathlib import Path
from dataclasses import dataclass, asdict
from functools import partial
import configparser
from multiprocessing import Pool, cpu_count

import requests
import numpy as np
import pandas as pd
from PIL import Image
from fastai.core import partition

from logger import get_logger


PATH = Path.home()/'data'/'furniture'
IMAGES = PATH/'images'
TRAIN_IMAGES = IMAGES/'train'
VALID_IMAGES = IMAGES/'valid'
TEST_IMAGES = IMAGES/'test'
LABELS = PATH/'labels.csv'

HEADERS = {'User-Agent': 'Python3'}

RANDOM_STATE = 1
np.random.seed(RANDOM_STATE)

log = get_logger()


def main():
    args = parse_args()

    name = args.subset
    path = IMAGES/name
    os.makedirs(path, exist_ok=True)

    json_file = PATH/f'{name}.json'
    index_file = PATH/f'{name}_index.csv'
    prepare_url_index(json_file, index_file, pct=args.pct)

    log.info(f'Downloading {args.pct:2.2%} of {json_file}...')
    index = pd.read_csv(index_file)
    info = download(index, path, args.chunk_size, args.proxy)
    info.to_pickle(IMAGES/f'{name}_info.pickle')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--subset',
        default='train', choices=['train', 'validation', 'test'],
        help='Subset to download'
    )
    parser.add_argument(
        '--pct',
        default=0.1, type=float,
        help='Percent of images to download'
    )
    parser.add_argument(
        '--chunk-size',
        default=1000, type=int,
        help='Number of images to download per multi-threaded pool run'
    )
    parser.add_argument(
        '--proxy',
        default=None,
        help='proxy configuration (if required)'
    )

    args = parser.parse_args()

    if args.proxy is not None:
        conf = configparser.ConfigParser()
        conf.read(args.proxy)
        proxy = dict(conf['proxy'])
        url = 'socks5://{username}:{password}@{host}:{port}'.format(**proxy)
        args.proxy = {'http': url, 'https': url}

    return args


def prepare_url_index(json_file, index_file, pct=0.1):
    """Saves meta-information about images into CSV file.

    Args:
        json_file: Path to JSON file with dataset information.
        index_file: Path to CSV file to save image URL, label ID, and image ID
        pct: Percentage of dataset to take.

    """
    with json_file.open() as file:
        content = json.load(file)

    images = content['images']
    if 'annotations' in content:
        labels = content['annotations']
    else:
        labels = [
            {'image_id': img['image_id'], 'label_id': 0}
            for img in images]

    urls = [img['url'][0] for img in images]
    records = pd.DataFrame([
        {'url': url, **lbl}
        for url, lbl in zip(urls, labels)])

    if pct is not None and pct < 1:
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
    url: str
    failed: bool = False


def download(index, path, chunk_size: int=1000, proxy: dict=None):
    """Downloads images with URLs from index dataframe."""

    n_cpu = cpu_count()
    worker = partial(download_single, path, proxy)
    queue = index.to_dict('records')
    meta = []

    with Pool(n_cpu) as pool:
        chunks = partition(queue, chunk_size)
        n_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            log.info('Downloading chunk %d of %d' % (i+1, n_chunks))
            data = [x for x in pool.imap_unordered(worker, chunk) if not x.failed]
            meta.extend([asdict(info) for info in data])

    return pd.DataFrame(meta)


def download_single(folder, proxy, info):
    url = info['url']
    img_name = str(info['image_id']) + '.jpg'
    path = folder/img_name

    result = {
        'label_id': info['label_id'],
        'image_id': info['image_id'],
        'path': path,
        'url': url}

    if path.exists():
        return ImageInfo(**result)

    error, msg = True, ''
    try:
        r = requests.get(
            url, allow_redirects=True, timeout=60,
            headers=HEADERS, proxies=proxy)
        r.raise_for_status()
        error = False
    except requests.HTTPError:
        msg = 'HTTP error'
    except requests.ConnectionError:
        msg = 'Connection error'
    except requests.Timeout:
        msg = 'Waiting response too long'
    except Exception as e:
        msg = str(e)[:80]

    if error:
        log.warning('%s: %s', msg, url)
        return ImageInfo(failed=True, **result)

    try:
        pil_image = Image.open(BytesIO(r.content)).convert('RGB')
        pil_image.save(path, format='JPEG', quality=90)
    except Exception as e:
        log.warning('Cannot create PIL Image: %s', str(e))
        return ImageInfo(failed=True, **result)

    if os.stat(path).st_size <= 0:
        log.warning('Saved image file is emtpy: %s', path)
        return ImageInfo(failed=True, **result)

    return ImageInfo(**result)


if __name__ == '__main__':
    main()
