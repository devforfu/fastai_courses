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

    defaults.device = device

    learn = train(arch=arch,
                  epochs=n,
                  lr=3e-2,
                  img_size=img_sz,
                  batch_size=bs,
                  train_size=trn_sz,
                  valid_size=val_sz,
                  log=log,
                  path=path)

    name = f'{args.arch_name}_{args.suffix}_{img_sz}_{n}'
    learn.save(name)

    # learn = train(arch=arch,
    #               preload=name,
    #               epochs=int(max(1, n//2)),
    #               lr=3e-2/3,
    #               img_size=img_sz*3//2,
    #               batch_size=bs*2,
    #               train_size=trn_sz,
    #               valid_size=val_sz,
    #               log=log,
    #               path=path)


    print('Done!')


def train(arch, epochs, lr, img_size, batch_size, train_size, valid_size, log,
          preload=None, path=MODELS):
    """Trains the model with specific set of parameters.

    The parameters are:

    * arch        backbone model (one of ResNets)
    * epochs      number of training epochs
    * lr          learning rates
    * img_size    training image size
    * batch_size  batch size
    * train_size  number of samples in training dataset
    * valid_size  number of samples in validation dataset
    * log         logger instance
    """
    sz = (img_size, img_size)
    bunch = create_data(sz, batch_size, train_size, valid_size, path, log)
    learn = create_cnn(bunch, arch, path=path)
    if preload is not None:
        learn.load(preload)
    learn.metrics = [accuracy, map3]
    callbacks = create_callbacks(learn)
    learn.fit_one_cycle(epochs, max_lr=lr, callbacks=callbacks)
    return learn


def create_data(img_size, bs, train_size, valid_size, path, log):
    trn_ds = QuickDraw(root=PREPARED,
                       train=True,
                       subset_size=train_size,
                       img_size=img_size,
                       log=log)
    val_ds = QuickDraw(root=PREPARED,
                       train=False,
                       subset_size=valid_size,
                       img_size=img_size,
                       log=log)
    tst_ds = TestImagesFolder(TEST, img_size=img_size)
    bunch = ImageDataBunch.create(
        trn_ds, val_ds, tst_ds,
        num_workers=4, path=path,
        bs=bs, ds_tfms=create_transforms())
    bunch.normalize(imagenet_stats)
    return bunch


def create_callbacks(learn):
    return [
        EarlyStoppingCallback(learn, patience=3),
        SaveModelCallback(learn),
        CSVLogger(learn)]


if __name__ == '__main__':
    main()
