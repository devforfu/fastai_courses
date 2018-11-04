from fastai import *
from fastai.vision import *


def main():
    path = untar_data(URLs.CIFAR)
    data = ImageDataBunch.from_folder(path)
    learn = create_cnn()



if __name__ == '__main__':
    main()
