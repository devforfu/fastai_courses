import argparse

from fastai import *
from fastai.vision import *
from fastai.metrics import accuracy, error_rate
from fastai.callbacks import CSVLogger
from fastai.callbacks.tracker import SaveModelCallback
from torchvision.models import resnet18, resnet34, resnet50

from projects.logger import get_logger, LOG_FILE_NAME


ARCHS = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
}

DATA_ROOT = Path.home()/'data'/'doodle'


def main():
    args = parse_args()

    logger = args.log
    logger.info('Training script parameters:')
    for k, v in vars(args).items():
        logger.info(f'\t {k} = {v}')

    logger.info('Reading training dataset...')
    train_ds = ImageClassificationDataset.from_folder(args.train)

    logger.info('Reading validation dataset...')
    valid_ds = ImageClassificationDataset.from_folder(args.valid, classes=train_ds.classes)

    logger.info('Reading testing dataset...')
    test_ds = ImageClassificationDataset.from_single_folder(args.test, classes=train_ds.classes)

    ds_tfms = get_transforms()
    train_tfms, valid_tfms = ds_tfms
    logger.info('Building data bunch with transformations')
    logger.info('Training transforms:')
    for trf in train_tfms:
        logger.info(f'\t {trf}')
    logger.info('Validation transforms:')
    for trf in valid_tfms:
        logger.info(f'\t {trf}')

    bunch = ImageDataBunch.create(
        train_ds, valid_ds, test_ds,
        bs=args.batch_size, size=args.image_size,
        ds_tfms=ds_tfms)

    logger.info(f'Normalizing data with ImageNet stats: {imagenet_stats}')
    bunch.normalize(imagenet_stats)

    logger.info('Datasets are prepared! The model is ready for training')
    learn = create_cnn(bunch, args.network, path=args.models_path)
    cbs = [SaveModelCallback(learn), CSVLogger(learn)]

    logger.info('Start model training...')
    learn.metrics = [accuracy, error_rate]
    learn.fit_one_cycle(args.n_epochs, callbacks=cbs, max_lr=args.learning_rates)
    path = learn.save(f'{args.arch_name}_{args.suffix}')
    logger.info('Done! Model saved into %s', path)

    logger.info('Interpreting results...')
    interp = ClassificationInterpretation.from_learner(learn)
    acc = (interp.pred_class == interp.y_true).float().mean().numpy().item()
    logger.info(f'Validation accuracy: {acc:2.2%}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--suffix',
        default=None,
        help='Suffix to append to the name of file with saved model'
    )
    parser.add_argument(
        '-r', '--data-root',
        default=DATA_ROOT,
        help='Parent folder to search dataset'
    )
    parser.add_argument(
        '-t', '--train',
        required=True,
        help='Path to folder with training images'
    )
    parser.add_argument(
        '-v', '--valid',
        required=True,
        help='Path to folder with validation images'
    )
    parser.add_argument(
        '-hd', '--test',
        required=True,
        help='Path to folder with test (holdout) images'
    )
    parser.add_argument(
        '-dev', '--device',
        default=1, type=int,
        help='PyTorch device used to train model'
    )
    parser.add_argument(
        '-arch', '--network',
        default='resnet34', choices=sorted(list(ARCHS))
    )
    parser.add_argument(
        '-n', '--n-epochs',
        default=1, type=int,
        help='Number of training epochs'
    )
    parser.add_argument(
        '-p', '--models-path',
        default=Path.home()/'models'/'doodle',
        help='Path to save trained models on learn.save() call'
    )
    parser.add_argument(
        '-log', '--logging',
        default=Path.home()/'models'/'doodle'/LOG_FILE_NAME
    )
    parser.add_argument(
        '-bs', '--batch-size',
        default=256, type=int,
        help='Batch size'
    )
    parser.add_argument(
        '-sz', '--image-size',
        default=224, type=int,
        help='Size to rescale images before passing into model'
    )
    parser.add_argument(
        '-lr', '--learning-rates',
        default=1e-4, type=valid_lr_list,
        help='Learning rates (single value, slice, or list)'
    )

    args = parser.parse_args()
    args.device = torch.device(f'cuda:{args.device}' if args.device >= 0 else 'cpu')
    args.arch_name = args.network
    args.network = ARCHS[args.network]
    args.train = Path(args.data_root)/args.train
    args.valid = Path(args.data_root)/args.valid
    args.test = Path(args.data_root)/args.test

    Path(args.models_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.logging).parent.mkdir(parents=True, exist_ok=True)

    args.log = get_logger(log_path=args.logging)
    defaults.device = args.device

    return args


def valid_lr_list(value):
    try:
        min_lr, max_lr = [float(x) for x in value.split(':')]
        return slice(max_lr, min_lr)
    except (TypeError, ValueError):
        try:
            lrs = [float(x) for x in value.split(',')]
            return lrs
        except (TypeError, ValueError):
            return float(value)


if __name__ == '__main__':
    main()
