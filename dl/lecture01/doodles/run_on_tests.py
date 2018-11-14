from fastai import *
from fastai.vision import *

from projects.logger import get_logger


defaults.device = torch.device('cuda:1')


ROOT = Path.home()/'data'/'doodle'
TRAIN_PATH = ROOT/'images_10000_200'/'train'
VALID_PATH = ROOT/'images_10000_200'/'valid'
TEST_PATH = ROOT/'test'/'images'
MODELS_PATH = Path.home()/'models'/'doodle'/'binary'


def main():
    logger = get_logger()

    categories = [x.stem for x in TRAIN_PATH.iterdir()]
    test_ds = ImageClassificationDataset.from_single_folder(
        folder=ROOT / 'test' / 'images',
        classes=sorted(categories))

    test_bunch = ImageDataBunch.create(test_ds, test_ds, test_ds)
    preds = torch.zeros((len(test_ds), len(categories)))

    for i, cat in enumerate(test_ds.classes):
        logger.info(f'Applying model for the category \'{cat}\'')
        learn = sqeeze_net(test_bunch)
        _ = learn.load(f'{cat}_classifier')
        y_hat, _ = learn.get_preds(DatasetType.Test)
        preds[:, i] = y_hat[:, 0]

    torch.save(preds, MODELS_PATH/'test_preds.pt')


def sqeeze_net(data_bunch):
    body = tvm.squeezenet1_1(True).features
    nf = num_features_model(body) * 2
    head = create_head(nf, 2, [256, 128], 0.5)
    model = nn.Sequential(body, head)
    learn = ClassificationLearner(data_bunch, model, path=MODELS_PATH)
    learn.metrics = [accuracy]
    learn.split((model[0][8], model[1]))
    return learn



if __name__ == '__main__':
    main()
