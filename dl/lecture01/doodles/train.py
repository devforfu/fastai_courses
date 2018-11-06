from fastai import *
from fastai.vision import *
from fastai.metrics import *
from torchvision import models


defaults.device = torch.device('cuda:0')


def main():
    path = untar_data(URLs.CIFAR)
    data = ImageDataBunch.from_folder(path, bs=4096, valid='test')
    learn = create_cnn(data, models.resnet18, metrics=accuracy)
    learn.fit(10)


if __name__ == '__main__':
    main()
