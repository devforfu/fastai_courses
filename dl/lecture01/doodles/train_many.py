from fastai import *
from fastai.vision import *

from projects.logger import get_logger


ROOT = Path.home()/'data'/'doodle'
TRAIN_PATH = ROOT/'images_10000_200'/'train'
VALID_PATH = ROOT/'images_10000_200'/'valid'
TEST_PATH = ROOT/'test'/'images'
MODELS_PATH = Path.home()/'models'/'doodle'/'binary'


def main():
    logger = get_logger(log_path=MODELS_PATH/'log.txt')
    categories = [x.stem for x in TRAIN_PATH.ls()]
    n_cats = len(categories)

    tst_ds = ImageClassificationDataset.from_single_folder(
        folder=ROOT / 'test' / 'images', classes=sorted(categories))
    test_bunch = ImageDataBunch.create(tst_ds, tst_ds, tst_ds, size=224)
    test_bunch.normalize(imagenet_stats)
    preds = torch.zeros((len(tst_ds), len(categories)))

    for i, cat in enumerate(categories):
        logger.info(f'Train binary classifier for category \'{cat}\' ({i+1} of {n_cats})')
        learn = train_binary_classifier(cat)
        path = learn.save(f'{cat}_classifier', return_path=True)
        logger.info(f'Model was saved into: {path}')

        learn = sqeeze_net(test_bunch).load(f'{cat}_classifier')
        logger.info(f'Applying model to the testing set')
        y_hat, _ = learn.get_preds(DatasetType.Test)
        preds[:, i] = y_hat[:, 0]

    logger.info('Saving predicted losses')
    torch.save(preds, MODELS_PATH/'test_preds.pt')


def train_binary_classifier(cat: str):
    trn_fns, trn_lbs, val_fns, val_lbs = binary_problem(cat, TRAIN_PATH, VALID_PATH, valid_sz=2000)
    tfms = get_transforms(do_flip=True, max_zoom=1.05, max_warp=0.3, p_lighting=0)
    trn_ds = ImageClassificationDataset(trn_fns, trn_lbs)
    val_ds = ImageClassificationDataset(val_fns, val_lbs)
    bunch = ImageDataBunch.create(trn_ds, val_ds, bs=300, size=224, ds_tfms=tfms)
    bunch.normalize(imagenet_stats)
    learn = sqeeze_net(bunch)
    learn.fit_one_cycle(1, max_lr=4e-04)
    return learn


def sqeeze_net(data_bunch):
    body = tvm.squeezenet1_1(True).features
    nf = num_features_model(body) * 2
    head = create_head(nf, 2, [256, 128], 0.5)
    model = nn.Sequential(body, head)
    learn = ClassificationLearner(data_bunch, model, path=MODELS_PATH)
    learn.metrics = [accuracy]
    learn.split((model[0][8], model[1]))
    return learn


def binary_problem(category: str, train_path: Path, valid_path: Path, valid_sz: int = 2000):
    """Selects a subset of data to train binary classifier discriminating between class/not-class."""

    non_category = sorted([cat.stem for cat in train_path.ls() if cat.stem != category])
    pos_train_fns = (train_path / category).ls()
    pos_valid_fns = (valid_path / category).ls()

    assert valid_sz > len(pos_valid_fns), f"valid_sz should be > {len(valid_fs)}"

    neg_train_per_cat = int(math.ceil(len(pos_train_fns) / len(non_category)))
    neg_train_fns = []

    neg_valid_per_cat = (valid_sz - len(pos_valid_fns)) / len(non_category)
    neg_valid_fns = []

    for cat in non_category:
        neg_train_fns.extend([
            x for i, x in enumerate((train_path / cat).iterdir())
            if i < neg_train_per_cat])
        neg_valid_fns.extend([
            x for i, x in enumerate((valid_path / cat).iterdir())
            if i < neg_valid_per_cat])

    train_files = pos_train_fns + neg_train_fns
    valid_files = pos_valid_fns + neg_valid_fns

    train_labels = binary_labels(category, train_files)
    valid_labels = binary_labels(category, valid_files)

    return train_files, train_labels, valid_files, valid_labels


def binary_labels(cat, files):
    return [cat if x.parent.name == cat else f'not_{cat}' for x in files]



if __name__ == '__main__':
    main()
