"""
The script takes original Quick Draw CSV and splits into training and validation
subsets to simplify training process.
"""
from pathlib import Path
from functools import partial
from multiprocessing import Pool, cpu_count

import pandas as pd

from projects.logger import get_logger


PATH = Path.home()/'data'/'doodle'


log = get_logger()


def main():
    split_dataset(PATH)


def split_dataset(path: Path, train_size: float=0.8, n_jobs=None):
    """Splits Quick Draw data into training and validation subsets.

    Args:
        path: Path to folder with original CSV files.
        train_size: Proportion of records to keep for model training.
        n_jobs: Number of workers for parallel execution pool.

    """
    n_jobs = n_jobs or cpu_count()
    worker = partial(parse_csv_file, train_size)
    plural = ['', 's'][n_jobs > 1]
    log.info('Running parallel processing pool with %d worker%s', n_jobs, plural)
    with Pool(n_jobs) as pool:
        files = list((path/'train').glob('*.csv'))
        pool.map(worker, files)
    log.info('Done!')


def parse_csv_file(train_size: float, path: Path):
    log.info('Parsing file %s', path)

    chunks = pd.read_csv(path, chunksize=10000)
    recognized = [chunk[chunk.recognized] for chunk in chunks]
    cat_df = pd.concat(recognized)

    total = len(cat_df)
    n = int(total * train_size)
    train, valid = cat_df[:n], cat_df[n:]
    log.info('%d train, %d valid instances for \'%s\'', len(train), len(valid), path.stem)

    for subset, df in [('train', train), ('valid', valid)]:
        file = path.parents[1]/'prepared'/subset/path.name
        file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file, index=None)
        log.info('Saved: %s' % file)


if __name__ == '__main__':
    main()
