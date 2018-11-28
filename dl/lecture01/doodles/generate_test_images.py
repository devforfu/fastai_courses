"""
Converts image strokes from CSV file into folder with PIL images.

Each stroke is converted into a single PIL image file with key_id
used as a file name.
"""
import argparse
from pathlib import Path
from functools import partial
from multiprocessing import Pool, cpu_count

from tqdm import tqdm
import pandas as pd
from PIL import Image, ImageChops

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
    parser.add_argument(
        '-bg', '--background',
        default='black', choices=['black', 'white'],
        help='Black or white background'
    )
    parser.add_argument(
        '-lw', '--line-width',
        default=4, type=int,
        help='Stroke line width'
    )
    parser.add_argument(
        '-crop', '--crop-whitespace',
        default=False, action='store_true',
        help='Trip whitespace and resize images to match original size'
    )
    args = parser.parse_args()

    test_data = pd.read_csv(args.file)
    keys = test_data.key_id.tolist()
    strokes = test_data.drawing.map(to_string).tolist()

    bg = args.background
    fg = 'white' if bg == 'black' else 'black'
    lw = args.line_width
    crop = args.crop_whitespace

    worker = partial(save_image, args.output, bg, fg, lw, crop)
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


def save_image(output_folder, bg, fg, lw, crop, key_and_stroke):
    image_key, stroke = key_and_stroke
    img = to_pil_image(
        stroke, IMG_SZ, bg_color=bg, stroke_color=fg, stroke_width=lw)
    if crop:
        img = trim(img)
    path = Path(output_folder)/f'{image_key}.png'
    img.save(path, 'png')
    return {'path': path.as_posix(), 'key': image_key}


def trim(img):
    old_size = img.size
    bg = Image.new(img.mode, img.size, color='black')
    diff = ImageChops.difference(img, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        img = img.crop(bbox).resize(old_size)
    return img


if __name__ == '__main__':
    main()
