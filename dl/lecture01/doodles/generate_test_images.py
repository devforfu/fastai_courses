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
        '-f', '--file',
        required=True,
        help='Path to CSV file with images'
    )
    parser.add_argument(
        '-o', '--output',
        default=Path.cwd(),
        help='Output directory with test images'
    )
    args = parser.parse_args()

    test_data = pd.read_csv(args.file)
    keys = test_data.key_id.tolist()
    strokes = test_data.drawing.map(to_string).tolist()

    worker = partial(save_image, args.output)
    records = []
    with tqdm(total=len(strokes)) as bar:
        with Pool(cpu_count()) as pool:
            for record in pool.imap_unordered(worker, zip(keys, strokes)):
                records.append(record)
                bar.update(1)
        meta_df = pd.DataFrame(records)

    meta_path = Path(args.output)/'meta.csv'
    meta_df.to_csv(meta_path, index=None)
    print('Done! Meta information saved into file %s' % meta_path)


def save_image(output_folder, args):
    image_key, stroke = args
    img = to_pil_image(stroke, IMG_SZ)
    path = Path(output_folder)/f'{image_key}.png'
    img.save(path, 'png')
    return {'path': path.as_posix(), 'key': image_key}



if __name__ == '__main__':
    main()
