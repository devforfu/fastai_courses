"""
Converts a subset of Quick Draw data from CSV strokes into PIL images.

The images are saved into separate folders, one folder per category. Therefore,
the tree structure generated with script looks like:

/images
    /train
        /helmet
            - <key_id_1>.png
            - <key_id_2>.png
            ...
        /cat
            - <key_id_k>.png
            ...

    /valid
        /helmet
            - <key_id_i>.png
            ...

"""
import os
import argparse
from pathlib import Path
from functools import partial
from multiprocessing import Pool, cpu_count

from tqdm import tqdm
import pandas as pd

from doodles_dataset import to_string, to_pil_image


IMG_SZ = 256, 256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--train',
        required=True,
        help='Path to folder with CSV file with training images'
    )
    parser.add_argument(
        '-v', '--valid',
        required=True,
        help='Path to folder with CSV file with validation images'
    )
    parser.add_argument(
        '-ts', '--train-size',
        default=1000, type=int,
        help='Number of samples to take from training file (per class)'
    )
    parser.add_argument(
        '-vs', '--valid-size',
        default=100, type=int,
        help='Number of samples to take from validation file (per class)'
    )
    parser.add_argument(
        '-o', '--output',
        default=Path.cwd(),
        help='Output directory to save training and validation images'
    )
    args = parser.parse_args()

    subsets = [
        (args.train, args.train_size, 'train'),
        (args.valid, args.valid_size, 'valid')
    ]

    for folder, size, subset_name in subsets:
        for filename in Path(folder).glob('*.csv'):
            print('Parsing subset of file %s' % filename)

            chunks = pd.read_csv(filename, chunksize=size)
            data = pd.DataFrame()
            while True:
                chunk = next(chunks)
                data = data.append(chunk[chunk.recognized])
                if len(data) >= size:
                    data = data[:size].copy()
                    break

            keys = data.key_id.tolist()
            strokes = data.drawing.map(to_string).tolist()
            path = Path(args.output) / subset_name / filename.stem
            os.makedirs(path, exist_ok=True)
            worker = partial(save_image, path)

            print('Saving images into folder', path)

            records = []
            with tqdm(total=len(strokes)) as bar:
                with Pool(cpu_count()) as pool:
                    pairs = zip(keys, strokes)
                    for record in pool.imap_unordered(worker, pairs):
                        records.append(record)
                        bar.update(1)

        print('Done!')


def save_image(output_folder, args):
    image_key, stroke = args
    img = to_pil_image(stroke, IMG_SZ)
    path = Path(output_folder)/f'{image_key}.png'
    img.save(path, 'png')
    return {'path': path.as_posix(), 'key': image_key}


if __name__ == '__main__':
    main()
