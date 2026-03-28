
# crop an image to the multiple of base
def crop_img(image, base=64):
    h = image.shape[0]
    w = image.shape[1]
    crop_h = h % base
    crop_w = w % base
    return image[crop_h // 2:h - crop_h + crop_h // 2, crop_w // 2:w - crop_w + crop_w // 2, :]

import numpy as np
import torch

def data_augmentation(image, mode):
    """
    image: numpy array, shape (H, W, C) or (H, W)
    mode: int in [0, 7]
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterclockwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.flipud(np.rot90(image))
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.flipud(np.rot90(image, k=2))
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.flipud(np.rot90(image, k=3))
    else:
        raise ValueError('Invalid choice of image transformation')

    return np.ascontiguousarray(out)


# def random_augmentation(*args):
#     out = []
#     flag_aug = random.randint(1, 7)
#     for data in args:
#         out.append(data_augmentation(data, flag_aug).copy())
#     return out