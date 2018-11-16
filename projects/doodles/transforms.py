from fastai.vision.image import TfmCrop
from fastai.vision.transform import *
from PIL import Image as PILImage
from PIL import ImageChops as PILImageChops
from torchvision.transforms import functional as vision_transform


@TfmCrop
def strip_empty_space(x, size, **kwargs):
    """Crops the image to exclude empty space from the image and rescale it to oritinal size."""

    img = vision_transform.to_pil_image(x)
    trimmed = trim(img).resize(size)
    new = vision_transform.to_tensor(trimmed).to(x)
    return new


def trim(img):
    bg = PILImage.new(img.mode, img.size, color='black')
    diff = PILImageChops.difference(img, bg)
    diff = PILImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        img = img.crop(bbox)
    return img


def create_transforms():
    train_tfms = [
        strip_empty_space(),
        rand_crop(padding_mode='zeros'),
        flip_affine(p=0.5),
        rotate(degrees=(-5, 5), p=0.75),
        rand_zoom(scale=(0.8, 1.1), p=0.5)
    ]

    valid_tfms = [
        crop_pad(padding_mode='zeros')
    ]

    return train_tfms, valid_tfms
