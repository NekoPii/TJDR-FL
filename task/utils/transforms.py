#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: maojingxin
"""
import random
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import cv2
import imgviz
import math
import numpy as np
import torch
from PIL import Image as PImage
from torch import nn
from torchvision import transforms as T
from torchvision.transforms import functional as F
from tqdm import tqdm

from task.utils.loss import KLD

torch.set_num_threads(4)
cv2.setNumThreads(4)

InterpolationMode = F.InterpolationMode


def tensor_unsqueeze_(tensor: torch.Tensor, _from: int, _to=4):
    assert _from <= _to
    if len(tensor.shape) == _from:
        for _ in range(_to - _from):
            tensor.unsqueeze_(0)


def tensor_squeeze_(tensor: torch.Tensor, _to: int, _from=4):
    assert _to <= _from
    if len(tensor.shape) == _from:
        for _ in range(_from - _to):
            tensor.squeeze_(0)


class Resize(object):
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR):
        self._break = False
        if isinstance(size, int) or (isinstance(size, np.ndarray) and size.ndim == 0):
            self.new_h = int(size)
            self.new_w = int(size)
        elif isinstance(size, Iterable) and len(size) == 1:
            self.new_h = int(size[0])
            self.new_w = int(size[0])
        elif isinstance(size, Iterable) and len(size) == 2:
            self.new_h = int(size[0])
            self.new_w = int(size[1])
        else:
            warnings.warn("Resize() size param:{} valid.".format(size))
            self._break = True
        self.interpolation = interpolation

    def __call__(self, image, gt=None):
        if self._break:
            return image, gt

        w, h = F.get_image_size(image)
        if self.new_w == w and self.new_h == h:
            if gt is not None:
                gt_w, gt_h = F.get_image_size(gt)
                if gt_w == w and gt_h == h:
                    return image, gt
            else:
                return image, gt

        if torch.is_tensor(image) and len(image.shape) == 3:
            tensor_unsqueeze_(image, _from=3, _to=4)
        if gt is not None:
            if torch.is_tensor(gt) and len(gt.shape) == 2:
                tensor_unsqueeze_(gt, _from=2, _to=4)

        image = F.resize(image, [self.new_h, self.new_w], interpolation=self.interpolation)
        if gt is not None:
            gt = F.resize(gt, [self.new_h, self.new_w], interpolation=InterpolationMode.NEAREST)

        if torch.is_tensor(image) and len(image.shape) == 4:
            tensor_squeeze_(image, _from=4, _to=3)
        if gt is not None:
            if torch.is_tensor(gt) and len(gt.shape) == 4:
                tensor_squeeze_(gt, _from=4, _to=2)
        return image, gt


class CenterCrop(object):
    def __init__(self, size):
        if not isinstance(size, Iterable):
            self.new_h = int(size)
            self.new_w = int(size)
        elif isinstance(size, Iterable) and len(size) == 1:
            self.new_h = int(size[0])
            self.new_w = int(size[0])
        elif isinstance(size, Iterable) and len(size) == 2:
            self.new_h = int(size[0])
            self.new_w = int(size[1])
        else:
            raise RuntimeError("CenterCrop() size param error:{}.".format(size))

    def __call__(self, image, gt=None):
        image = F.center_crop(image, [self.new_h, self.new_w])
        if gt is not None:
            gt = F.center_crop(gt, [self.new_h, self.new_w])
        return image, gt


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, gt=None):
        if self.mean is None or self.std is None:
            return image, gt
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, gt


class Pad(object):
    def __init__(self, padding_n, padding_fill_value, padding_fill_gt_value):
        self.padding_n = padding_n
        self.padding_fill_value = padding_fill_value
        self.padding_fill_gt_value = padding_fill_gt_value

    def __call__(self, image, gt):
        image = F.pad(image, self.padding_n, self.padding_fill_value)
        if gt is not None:
            gt = F.pad(gt, self.padding_n, self.padding_fill_gt_value)
        return image, gt


class ToTensor(object):
    def __call__(self, image, gt):
        image = F.to_tensor(image)
        if gt is not None:
            gt = torch.as_tensor(np.array(gt), dtype=torch.float)
        return image, gt


class Compose(object):
    def __init__(self, transforms: List):
        self.transforms = transforms

    def append(self, transform):
        self.transforms.append(transform)

    def __call__(self, image, gt=None):
        for t in self.transforms:
            image, gt = t(image, gt)
        return image, gt


class Identity(object):
    def __init__(self):
        pass

    def __call__(self, image, gt=None):
        return image, gt


class RandomChoice(object):
    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, image, gt=None):
        trans = T.RandomChoice(self.transforms)
        image, gt = trans(image, gt)
        return image, gt


class TransformTwice(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, gt=None):
        image1, gt1 = self.transforms(image, gt)
        image2, gt2 = self.transforms(image, gt)
        return [image1, gt1], [image2, gt2]


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, gt=None):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            if gt is not None:
                gt = F.hflip(gt)
        return image, gt


class RandomVerticalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, gt=None):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            if gt is not None:
                gt = F.vflip(gt)
        return image, gt


class RandomResize(object):
    def __init__(self, prob=0.5, scale=(0.5, 2.0), aspect_ratio=None):
        self.prob = prob
        self.scale = scale
        self.aspect_ratio = aspect_ratio

    def __call__(self, image, gt=None):
        if random.random() < 0.5:
            temp_scale = self.scale[0] + (1.0 - self.scale[0]) * random.random()
        else:
            temp_scale = 1.0 + (self.scale[1] - 1.0) * random.random()
        temp_aspect_ratio = 1.0
        if self.aspect_ratio is not None:
            temp_aspect_ratio = (self.aspect_ratio[0] + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random())
            temp_aspect_ratio = math.sqrt(temp_aspect_ratio)
        scale_factor_w = temp_scale * temp_aspect_ratio
        scale_factor_h = temp_scale / temp_aspect_ratio
        w, h = F.get_image_size(image)
        new_w = int(w * scale_factor_w)
        new_h = int(h * scale_factor_h)

        image, gt = Resize(size=[new_h, new_w])(image, gt)
        return image, gt


class RandomCrop(object):
    def __init__(self, size, crop_type="center", padding_fill_value=0, padding_fill_gt_value=0):
        if isinstance(size, int):
            self.crop_h = size
            self.crop_w = size
        elif (
                isinstance(size, Iterable)
                and len(size) == 2
                and isinstance(size[0], int)
                and isinstance(size[1], int)
                and size[0] > 0
                and size[1] > 0
        ):
            self.crop_h = size[0]
            self.crop_w = size[1]
        else:
            raise RuntimeError("RandomCrop() size error.")
        if crop_type == "center" or crop_type == "rand":
            self.crop_type = crop_type
        else:
            raise RuntimeError("crop type error: rand | center")
        self.padding_fill_value = padding_fill_value
        self.padding_fill_gt_value = padding_fill_gt_value

    def __call__(self, image, gt=None):
        w, h = F.get_image_size(image)
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        if pad_h > 0 or pad_w > 0:
            pad_left = pad_w_half
            pad_right = pad_w - pad_w_half
            pad_top = pad_h_half
            pad_bottom = pad_h - pad_h_half
            image, gt = Pad(
                padding_n=[pad_left, pad_top, pad_right, pad_bottom],
                padding_fill_value=self.padding_fill_value,
                padding_fill_gt_value=self.padding_fill_gt_value
            )(image, gt)
        w, h = F.get_image_size(image)
        if self.crop_type == "rand":
            h_off = random.randint(0, h - self.crop_h)
            w_off = random.randint(0, w - self.crop_w)
        else:
            h_off = (h - self.crop_h) // 2
            w_off = (w - self.crop_w) // 2

        image = F.crop(image, top=h_off, left=w_off, height=self.crop_h, width=self.crop_w)
        if gt is not None:
            gt = F.crop(gt, top=h_off, left=w_off, height=self.crop_h, width=self.crop_w)
        return image, gt


class RandomResizedCrop(object):
    def __init__(self, size, prob=1.0, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=InterpolationMode.BILINEAR):
        if isinstance(size, int):
            self.new_h = size
            self.new_w = size
        elif (
                isinstance(size, Iterable)
                and len(size) == 2
                and isinstance(size[0], int)
                and isinstance(size[1], int)
                and size[0] > 0
                and size[1] > 0
        ):
            self.new_h = size[0]
            self.new_w = size[1]
        else:
            raise RuntimeError("Resize() size param error.")

        self.prob = prob
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, image, gt=None):
        if self.prob is None or (isinstance(self.prob, float) and random.random() < self.prob):
            rrc_params = T.RandomResizedCrop.get_params(image, self.scale, self.ratio)
            image = F.resized_crop(image, *rrc_params, size=[self.new_h, self.new_w], interpolation=self.interpolation)
            if gt is not None:
                gt = F.resized_crop(
                    gt, *rrc_params, size=[self.new_h, self.new_w],
                    interpolation=InterpolationMode.NEAREST
                )
        else:
            image, gt = RandomCrop(size=[self.new_h, self.new_w], crop_type="center", padding_fill_value=0)(image, gt)
            image, gt = Resize([self.new_h, self.new_w], self.interpolation)(image, gt)
        return image, gt


class RandomAffine(object):
    def __init__(self, degree: List[float] = (.0, .0), translate: List[float] = None, scale: List[float] = None, shear=None):
        """
        Args:
            degree: eg: (-10,10)
            translate: eg: (0.05,0.05)
            scale: eg: (0.95,1.02)
            shear: default is not use
        """
        self.degree = degree
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def __call__(self, image, gt=None):
        affine_params = T.RandomAffine.get_params(
            self.degree, self.translate, self.scale, self.shear,
            img_size=F.get_image_size(image)
        )
        image = F.affine(image, *affine_params, interpolation=InterpolationMode.BILINEAR)
        if gt is not None:
            gt = F.affine(gt, *affine_params, interpolation=InterpolationMode.NEAREST)
        return image, gt


class RandomColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        """

        Args:
            brightness: [max(0,1-brightness),1+brightness]
            contrast: [max(0,1-contrast),1+contrast]
            saturation: [max(0,1-saturation),1+saturation]
            hue: [-hue,hue] range in [0.0.5]
        """
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image, gt=None):
        color_trans = T.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        image = color_trans(image)
        return image, gt


class RandomGaussianNoise(object):
    def __init__(self, mean, std, amplitude=1):
        self.mean = mean
        self.std = std
        self.amplitude = amplitude

    def __call__(self, image, gt):
        np_img = np.array(image)
        h, w, c = np_img.shape
        gauss_noise = self.amplitude * np.random.normal(self.mean, self.std, size=(h, w, 1))
        gauss_noise = np.repeat(gauss_noise, c, axis=2)
        np_img = np.clip(np_img + gauss_noise, 0, 255)
        pil_img = PImage.fromarray(np_img.astype("uint8"))
        return pil_img, gt


class RandomGaussianBlur(object):
    def __init__(self, kernel_size, min_std, max_std):
        self.kernel_size = kernel_size
        self.min_std = min_std
        self.max_std = max_std
        assert min_std <= max_std, "RandomGaussianBlur() std random range valid"

    def __call__(self, image, gt=None):
        trans = T.GaussianBlur(self.kernel_size, (self.min_std, self.max_std))
        image = trans(image)
        return image, gt


class RandomEqualize(object):
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image, gt=None):
        trans = T.RandomEqualize(self.prob)
        image = trans(image)
        return image, gt


class RandomSolarize(object):
    def __init__(self, threshold: int, prob: float = 0.5):
        assert isinstance(threshold, int) and 0 <= threshold <= 256
        self.threshold = threshold
        self.prob = prob

    def __call__(self, image, gt=None):
        threshold = self.threshold
        if isinstance(image, torch.Tensor):
            threshold /= 255

        trans = T.RandomSolarize(threshold, self.prob)
        image = trans(image)
        return image, gt


class RandomPosterize(object):
    def __init__(self, bits: int, prob: float = 0.5):
        assert isinstance(bits, int) and 0 <= bits <= 8
        self.bits = bits
        self.prob = prob

    def __call__(self, image, gt=None):
        trans = T.RandomPosterize(self.bits, self.prob)
        image = trans(image)
        return image, gt


class RandomSharpness(object):
    def __init__(self, sharpness_factor: float, prob: float = 0.5):
        assert sharpness_factor >= 0
        self.sharpness_factor = sharpness_factor
        self.prob = prob

    def __call__(self, image, gt=None):
        trans = T.RandomAdjustSharpness(self.sharpness_factor, self.prob)
        image = trans(image)
        return image, gt


class DR_transform(object):
    def __init__(self, alpha=4, beta=-4, sigmaX=10):
        self.alpha = alpha
        self.beta = beta
        self.sigmaX = sigmaX

    def __call__(self, image, gt=None):
        if image is None:
            return image, gt
        if isinstance(image, PImage.Image):
            np_img = np.array(image)
            np_img = cv2.addWeighted(np_img, self.alpha, cv2.GaussianBlur(np_img, (0, 0), self.sigmaX), self.beta, 128)
            pil_image = PImage.fromarray(np_img)
            return pil_image, gt
        else:
            raise ValueError("image must be PIL.Image for DR_transform")


class CircleCrop(object):
    def __init__(self, threshold: int = 7):
        self.threshold = threshold

    def crop_image_from_gray(self, img, mask=None):
        if img.ndim == 2:
            mask = img > self.threshold if mask is None else mask
            return img[np.ix_(mask.any(1), mask.any(0))], mask
        elif img.ndim == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            mask = gray_img > self.threshold if mask is None else mask

            check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
            if check_shape == 0:  # image is too dark so that we crop out everything,
                return img  # return original image
            else:
                img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
                img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
                img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
                img = np.stack([img1, img2, img3], axis=-1)
            return img, mask

    def __call__(self, image, gt=None):
        if gt is not None:
            warnings.warn("CircleCrop may cause loss of details for segmentation")
        if isinstance(image, PImage.Image):
            np_img = np.asarray(image)
            np_img, np_img_mask1 = self.crop_image_from_gray(np_img)

            height, width, depth = np_img.shape

            x = int(width / 2)
            y = int(height / 2)
            r = np.amin((x, y))

            circle_img = np.zeros((height, width), np.uint8)
            cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
            np_img = cv2.bitwise_and(np_img, np_img, mask=circle_img)
            np_img, np_img_mask2 = self.crop_image_from_gray(np_img)
            pil_image = PImage.fromarray(np_img)
            if gt is not None and isinstance(gt, PImage.Image):
                np_gt = np.asarray(gt)
                np_gt, _ = self.crop_image_from_gray(np_gt, mask=np_img_mask1)
                np_gt = cv2.bitwise_and(np_gt, np_gt, mask=circle_img)
                np_gt, _ = self.crop_image_from_gray(np_gt, mask=np_img_mask2)
                gt = PImage.fromarray(np_gt, mode="P")
                gt.putpalette(imgviz.label_colormap())
            return pil_image, gt
        else:
            raise ValueError("image must be PIL.Image for CircleCrop")


class FGSMGenerator(object):
    def __init__(self, net: nn.Module, eps: float = 0.05):
        self.net = net
        self.eps = eps

    def __call__(self, img, gt, sup_loss_f):
        assert torch.is_tensor(img) and torch.is_tensor(gt)
        img.requires_grad = True
        if img.grad is not None:
            img.grad.zero_()
        self.net.zero_grad()
        logits = self.net(img)
        loss = sup_loss_f(logits, gt)
        loss.backward()
        adv_img, noise = self.fgsm(img, img.grad, eps=self.eps)
        self.net.zero_grad()
        img.grad.zero_()
        return adv_img, noise

    @staticmethod
    def fgsm(img, img_grad, eps):
        sign_grad = img_grad.sign()
        noise = (eps * sign_grad).detach_()
        adversarial_img = (img + noise).detach_()
        return adversarial_img, noise


class VATGenerator(object):
    def __init__(self, net: nn.Module, xi=1e1, eps=1e0, ip=1):
        """VAT generator based on https://arxiv.org/abs/1704.03976
        Args:
            net:
            xi: hyperparameter of VAT (default: 10.0)
            eps: hyperparameter of VAT (default: 1.0)
            ip: iteration times of computing adv noise (default: 1)
        """
        self.net = net
        self.kl_div_loss = KLD(dim=1, reduction=True, logits=True)
        self.xi = xi
        self.eps = eps
        self.ip = ip

    @staticmethod
    def _l2_normalize(d):
        d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
        d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-16
        assert torch.allclose(d.view(d.shape[0], -1).norm(dim=1), torch.ones(d.shape[0]).to(d.device), rtol=1e-3)
        return d

    def __call__(self, img):
        with torch.no_grad():
            logit = self.net(img)
        d = torch.Tensor(img.size()).normal_()
        # d = self._l2_normalize(d).to(img.device)
        self.net.zero_grad()
        for _ in range(self.ip):
            d = self.xi * self._l2_normalize(d).to(img.device)
            d.requires_grad = True
            y_hat = self.net(img + d)
            loss = self.kl_div_loss(y_hat, logit.detach())
            loss.backward()
            d = d.grad.data.clone().cpu()
            self.net.zero_grad()
        d = self._l2_normalize(d).to(img.device)
        adv_r = (self.eps * d).detach_()
        adv_img = (img + adv_r).detach_()
        return adv_img, adv_r


class RandAugment(T.RandAugment):
    def __init__(
            self, num_ops: int = 2, num_magnitude_bins: int = 31, magnitude: Optional[int] = 9,
            interpolation: InterpolationMode = InterpolationMode.BILINEAR,
            fill: Optional[List[float]] = 0
    ) -> None:
        magnitude = torch.randint(num_magnitude_bins, (1,), dtype=torch.long) if magnitude is None else magnitude
        super().__init__(num_ops, magnitude, num_magnitude_bins, interpolation, fill)

    def _augmentation_space(self, num_bins: int, image_size: List[int]) -> Dict[str, Tuple[torch.Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

    def __call__(self, image, gt=None):
        image = self.forward(image)
        return image, gt


class TrivialAugment(T.TrivialAugmentWide):
    def __init__(
            self, num_magnitude_bins: int = 31,
            interpolation: InterpolationMode = InterpolationMode.NEAREST, fill: Optional[List[float]] = None
    ) -> None:
        # op's num always equal to 1
        super().__init__(num_magnitude_bins, interpolation, fill)

    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[torch.Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.99, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.99, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 32.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, 32.0, num_bins), True),
            "Rotate": (torch.linspace(0.0, 135.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.99, num_bins), True),
            "Color": (torch.linspace(0.0, 0.99, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.99, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.99, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

    def __call__(self, image, gt=None):
        image = self.forward(image)
        return image, gt


def preprocess_resize(img_dir: Path, img_suffix: str, target_dir: Path = None, target_suffix: str = None, h=1024, w=1024, interpolation=InterpolationMode.BILINEAR):
    imgs_path = sorted(list(img_dir.glob("*{}".format(img_suffix))))

    target_dir = img_dir.parent / "resize" / img_dir.name if target_dir is None else target_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(imgs_path, desc="resize:{}".format(img_dir), unit="img"):
        img_name = "{}{}".format(img_path.stem, img_path.suffix if target_suffix is None else target_suffix)
        img = PImage.open(img_path)
        img, _ = Resize(size=(h, w), interpolation=interpolation)(img)
        img.save(target_dir / img_name)


class KaggleDRPreprocess(object):
    def __init__(self, numOfBlurred=1):
        self.numOfBlurred = numOfBlurred

    def clahe(self, img, clipLimit=2.0, gridsize=8):
        # Converting image to LAB Color model
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # Splitting the LAB image to different channels
        lab_planes = list(cv2.split(lab))
        # Creating and applying clahe to luminance channel
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(gridsize, gridsize))
        lab_planes[0] = clahe.apply(lab_planes[0])
        # Merge planes together
        lab = cv2.merge(lab_planes)
        # Convert image from LAB color plane to BGR color plane
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return img

    def subtract_gaussian_blur(self, img, b=5):
        gb_img = cv2.GaussianBlur(img, (0, 0), b)
        return cv2.addWeighted(img, 4, gb_img, -4, 128)

    def remove_outer_circle(self, a, p, r):
        b = np.zeros(a.shape, dtype=a.dtype)
        cv2.circle(b, (a.shape[1] // 2, a.shape[0] // 2), int(r * p), (1, 1, 1), -1, 8, 0)
        return a * b + 128 * (1 - b)

    def crop_img(self, img):
        non_zeros = img.nonzero()  # Find indices of non zero elements
        non_zero_rows = [min(np.unique(non_zeros[0])), max(np.unique(non_zeros[0]))]  # Find the first and last row with non zero elements
        non_zero_cols = [min(np.unique(non_zeros[1])), max(np.unique(non_zeros[1]))]  # Find the first and last row with non zero elements
        crop_img = img[non_zero_rows[0]:non_zero_rows[1], non_zero_cols[0]:non_zero_cols[1], :]  # Crop the image
        return crop_img

    def make_square(self, img, min_size=256):
        x, y, z = img.shape
        size = max(min_size, x, y)
        new_img = np.zeros((size, size, z), dtype=img.dtype)
        for i in range(z):
            new_img[:, :, i] = img[0, 0, i]
        new_img[int((size - x) / 2):int((size - x) / 2 + x), int((size - y) / 2):int((size - y) / 2 + y), :] = img
        return np.array(new_img)

    def preprocess(self, img, numOfBlurred):
        dtype = img.dtype
        img = self.crop_img(img)
        blurred_imgs = np.zeros([img.shape[0], img.shape[1], numOfBlurred], dtype=dtype)
        if numOfBlurred > 1:
            for row in range(numOfBlurred):
                img2 = self.subtract_gaussian_blur(img, (row + 1) * 5)
                blurred_imgs[:, :, row] = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            blurred_imgs = self.subtract_gaussian_blur(img, 5)
        img = self.make_square(blurred_imgs)
        img = self.remove_outer_circle(img, 0.97, img.shape[0] // 2)
        img = self.clahe(img)
        return img

    def __call__(self, image, gt=None):
        if isinstance(image, PImage.Image):
            image = image.convert("RGB")
            image = np.asarray(image)[:, :, ::-1]
        image = self.preprocess(image, self.numOfBlurred)
        image = PImage.fromarray(image[:, :, ::-1])
        return image, gt
