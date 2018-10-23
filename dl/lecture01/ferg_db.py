from fastai import *
from fastai.vision import *
from torchvision import models
import shutil


PATH = Path.home()/'data'/'FERG_DB_256'
PREPARED = PATH/'flatten'


def parse_label(filename):
    char, emotion, *_ = filename.stem.split('_')
    return f'{char}_{emotion}'


def main():
    files = []

    for name in PATH.ls():
        if name.endswith('.txt') or name.startswith('.'):
            continue
        char_dir = PATH / name
        for char_emotion in char_dir.ls():
            emotion_dir = char_dir / char_emotion
            files.extend([fname for fname in emotion_dir.glob('*')])

    # PREPARED.mkdir(parents=True, exist_ok=True)
    # for file in files:
    #     shutil.copy(file, PREPARED / file.name)

    data = ImageDataBunch.from_name_func(PREPARED, files, parse_label, ds_tfms=get_transforms(), size=224)
    data.normalize(imagenet_stats)


if __name__ == '__main__':
    main()
