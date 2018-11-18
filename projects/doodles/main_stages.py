from fastai import defaults
from fastai.vision import *
from fastai.metrics import accuracy
from fastai.callbacks import CSVLogger
from fastai.callbacks.tracker import EarlyStoppingCallback, SaveModelCallback
from torchvision.models import resnet18, resnet34, resnet50

from basedir import PREPARED, TEST, MODELS
from cli import parse_args
from dataset import QuickDraw, TestImagesFolder
from transforms import create_transforms
from metrics import map3


IMG_SIZE = 224, 224


def main():
    args = parse_args({
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50
    })

    img_sz, trn_sz, val_sz, n, bs = (
        args.image_size,
        args.train_size,
        args.valid_size,
        args.n_epochs,
        args.batch_size,
    )

    arch, log, device, path = (
        args.network,
        args.log,
        args.device,
        args.models_path
    )

    defaults.device = args.device

    tfms = create_transforms()
    bunch = create_bunch(subset_sizes=(trn_sz, val_sz), img_size=img_sz, bs=bs, ds_tfms=tfms, log=args.log)
    learn = create_cnn(bunch, arch)
    learn.metrics = [accuracy, map3]

    learn.fit_one_cycle(1, max_lr=1e-02)
    learn.fit_one_cycle(1, max_lr=1e-02 / 2)
    saved_path = learn.save('base', return_path=True)
    log.info('Intermediate model saved into %s', saved_path)

    learn.unfreeze()
    learn.freeze_to(-1)
    learn.fit_one_cycle(1, max_lr=slice(2e-04, 1e-03))
    saved_path = learn.save('unfreeze_one', return_path=True)
    log.info('Intermediate model saved into %s', saved_path)

    learn.unfreeze()
    learn.fit_one_cycle(1, max_lr=slice(1e-04, 3e-04))
    saved_path = learn.save('unfreeze_all', return_path=True)
    log.info('Intermediate model saved into %s', saved_path)

    learn.fit_one_cycle(1, max_lr=slice(1e-05, 5e-04))
    saved_path = learn.save('unfreeze_all', return_path=True)
    log.info('Done! The final model is saved into file %s', saved_path)



def create_bunch(train_path=PREPARED, test_path=TEST, subset_sizes=(200, 50),
                 img_size=IMG_SIZE, bs=800, ds_tfms=None, log=None):

    sz = (img_size, img_size)
    trn_sz, val_sz = subset_sizes
    trn_ds = QuickDraw(train_path, train=True, subset_size=trn_sz, img_size=sz, log=log)
    val_ds = QuickDraw(train_path, train=False, subset_size=val_sz, img_size=sz, log=log)
    tst_ds = TestImagesFolder(test_path, img_size=sz)
    bunch = ImageDataBunch.create(trn_ds, val_ds, tst_ds, bs=bs, num_workers=4, ds_tfms=ds_tfms)
    bunch.normalize(imagenet_stats)
    bunch.show_batch(rows=4)
    return bunch


def create_callbacks(learn):
    return [
        EarlyStoppingCallback(learn, patience=3),
        SaveModelCallback(learn),
        CSVLogger(learn)]


if __name__ == '__main__':
    main()
