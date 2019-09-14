import random
import math
import numpy as np
import numbers
import collections
import cv2
import numbers
import collections
import torch
cv2.ocl.setUseOpenCL(False)


class Compose(object):
    """
    Composes several segsegtransforms together.

    Args:
        segtransforms (List[Transform]): list of segtransforms to compose.

    Example:
        segtransforms.Compose([
            segtransforms.RandScale([0.5, 2.0]),
            segtransforms.ToTensor()])
    """
    def __init__(self, segtransforms):
        self.segtransforms = segtransforms

    def __call__(self, image1, image2):
        for t in self.segtransforms:
            image1, image2 = t(image1, image2)
        return image1, image2


class ToTensor(object):
    # Converts numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).
    def __call__(self, image1, image2):
        if not isinstance(image1, np.ndarray) or not isinstance(image2, np.ndarray):
            raise (RuntimeError("segtransforms.ToTensor() only handle np.ndarray"
                                "[eg: data readed by cv2.imread()].\n"))
        if len(image1.shape) > 3 or len(image1.shape) < 2:
            raise (RuntimeError("segtransforms.ToTensor() only handle np.ndarray with 3 dims or 2 dims.\n"))
        if len(image2.shape) > 3 or len(image2.shape) < 2:
            raise (RuntimeError("segtransforms.ToTensor() only handle np.ndarray with 3 dims or 2 dims.\n"))

        image1 = torch.from_numpy(image1.transpose((2, 0, 1)))
        if not isinstance(image1, torch.FloatTensor):
            image1 = image1.float()
        image2 = torch.from_numpy(image2.transpose((2, 0, 1)))
        if not isinstance(image2, torch.FloatTensor):
            image2 = image2.float()
        return image1, image2


class Normalize(object):
    """
    Given mean and std of each channel
    Will normalize each channel of the torch.*Tensor (C*H*W), i.e.
    channel = (channel - mean) / std
    """
    def __init__(self, mean, std=None):
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def __call__(self, image1, image2):
        for t, m, s in zip(image1, self.mean, self.std):
            t.sub_(m).div_(s)
        for t, m, s in zip(image2, self.mean, self.std):
            t.sub_(m).div_(s)
        return image1, image2


class Resize(object):
    """
    Resize the input PIL Image to the given size.
    'size' is a 2-element tuple or list in the order of (h, w)
    """
    def __init__(self, size):
        assert (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size

    def __call__(self, image1, image2):
        image1 = cv2.resize(image1, self.size[::-1], interpolation=cv2.INTER_LINEAR)
        image2 = cv2.resize(image2, self.size[::-1], interpolation=cv2.INTER_LINEAR)
        return image1, image2



class Crop(object):
    """Crops the given ndarray image (H*W*C or H*W).
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
        int instead of sequence like (h, w), a square crop (size, size) is made.
    """
    def __init__(self, size, crop_type='center', padding=None, ignore_label=255):
        if isinstance(size, int):
            self.crop_h = size
            self.crop_w = size
        elif isinstance(size, collections.Iterable) and len(size) == 2 \
                and isinstance(size[0], int) and isinstance(size[1], int) \
                and size[0] > 0 and size[1] > 0:
            self.crop_h = size[0]
            self.crop_w = size[1]
        else:
            raise (RuntimeError("crop size error.\n"))
        if crop_type == 'center' or crop_type == 'rand':
            self.crop_type = crop_type
        else:
            raise (RuntimeError("crop type error: rand | center\n"))
        if padding is None:
            self.padding = padding
        elif isinstance(padding, list):
            if all(isinstance(i, numbers.Number) for i in padding):
                self.padding = padding
            else:
                raise (RuntimeError("padding in Crop() should be a number list\n"))
            if len(padding) != 3:
                raise (RuntimeError("padding channel is not equal with 3\n"))
        else:
            raise (RuntimeError("padding in Crop() should be a number list\n"))
        if isinstance(ignore_label, int):
            self.ignore_label = ignore_label
        else:
            raise (RuntimeError("ignore_label should be an integer number\n"))

    def __call__(self, image1, image2):
        h, w, c = image1.shape
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        if pad_h > 0 or pad_w > 0:
            if self.padding is None:
                raise (RuntimeError("segtransforms.Crop() need padding while padding argument is None\n"))
            image1 = cv2.copyMakeBorder(image1, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.padding)
        if self.crop_type == 'rand':
            h_off = random.randint(0, h - self.crop_h)
            w_off = random.randint(0, w - self.crop_w)
        else:
            h_off = (h - self.crop_h) // 2
            w_off = (w - self.crop_w) // 2
        image1 = image1[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
        
        h, w, c = image2.shape
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        if pad_h > 0 or pad_w > 0:
            if self.padding is None:
                raise (RuntimeError("segtransforms.Crop() need padding while padding argument is None\n"))
            image2 = cv2.copyMakeBorder(image2, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.padding)
        if self.crop_type == 'rand':
            h_off = random.randint(0, h - self.crop_h)
            w_off = random.randint(0, w - self.crop_w)
        else:
            h_off = (h - self.crop_h) // 2
            w_off = (w - self.crop_w) // 2
        image2 = image2[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
        
        return image1, image2



class RandomGaussianBlur(object):
    def __init__(self, radius=5):
        self.radius = radius

    def __call__(self, image1, image2):
        if random.random() < 0.5:
            image1 = cv2.GaussianBlur(image1, (self.radius, self.radius), 0)
            image2 = cv2.GaussianBlur(image2, (self.radius, self.radius), 0)
        return image1, image2


    
    
def saturation(pic, factor):
    grayimg = grayscale(pic)
    img = pic * factor + grayimg * (1.0 - factor)
    img = img.astype(pic.dtype)
    return img
def brightness(pic, factor):
    img = pic * factor
    return img
def grayscale(pic):
    assert (len(pic.shape) == 3) and (pic.shape[2] == 3), "input img for grayscale() should be H*W*3 ndarray"
    grayimg = 0.299 * pic[:, :, 2] + 0.587 * pic[:, :, 1] + 0.114 * pic[:, :, 0]
    grayimg = np.repeat(grayimg[:, :, np.newaxis], 3, axis=2)
    return grayimg
def contrast(pic, factor):
    grayimg = grayscale(pic)
    ave = grayimg[:, :, 0].mean()
    ave_img = np.ndarray(shape=pic.shape, dtype=float)
    ave_img.fill(ave)
    img = pic * factor + ave_img * (1 - factor)
    return img


class ColorJitter(object):
    """
    do ColorJitter for BGR ndarray image
    factor should be a number of list of three number, all numbers should be in (0,1)
    """

    def __init__(self, factor):
        if isinstance(factor, numbers.Number) and 0 < factor < 1:
            self.saturation_factor = factor
            self.brightness_factor = factor
            self.contrast_factor = factor
        elif isinstance(factor, collections.Iterable) and len(factor) == 3 \
                and isinstance(factor[0], numbers.Number) and 0 < factor[0] < 1 \
                and isinstance(factor[1], numbers.Number) and 0 < factor[1] < 1 \
                and isinstance(factor[2], numbers.Number) and 0 < factor[2] < 1:
            self.saturation_factor = factor[0]
            self.brightness_factor = factor[1]
            self.contrast_factor = factor[2]
        else:
            raise (RuntimeError("ColorJitter factor error.\n"))

    def __call__(self, img1, img2):
        ori_type = img1.dtype
        img1.astype('float32')
        this_saturation_factor = 1.0 + self.saturation_factor * random.uniform(-1.0, 1.0)
        this_brightness_factor = 1.0 + self.brightness_factor * random.uniform(-1.0, 1.0)
        this_contrast_factor = 1.0 + self.contrast_factor * random.uniform(-1.0, 1.0)
        funclist = [(saturation, this_saturation_factor),
                    (brightness, this_brightness_factor),
                    (contrast, this_contrast_factor)]
        random.shuffle(funclist)
        for func in funclist:
            img1 = (func[0])(img1, func[1])
        if ori_type == np.uint8:
            img1 = np.clip(img1, 0, 255)
            img1.astype('uint8')
        elif ori_type == np.uint16:
            img1 = np.clip(img1, 0, 65535)
            img1.astype('uint16')
            
        ori_type = img2.dtype
        img2.astype('float32')
        this_saturation_factor = 1.0 + self.saturation_factor * random.uniform(-1.0, 1.0)
        this_brightness_factor = 1.0 + self.brightness_factor * random.uniform(-1.0, 1.0)
        this_contrast_factor = 1.0 + self.contrast_factor * random.uniform(-1.0, 1.0)
        funclist = [(saturation, this_saturation_factor),
                    (brightness, this_brightness_factor),
                    (contrast, this_contrast_factor)]
        random.shuffle(funclist)
        for func in funclist:
            img2 = (func[0])(img2, func[1])
        if ori_type == np.uint8:
            img2 = np.clip(img2, 0, 255)
            img2.astype('uint8')
        elif ori_type == np.uint16:
            img2 = np.clip(img2, 0, 65535)
            img2.astype('uint16')            
            
        return img1, img2