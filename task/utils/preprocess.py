#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: maojingxin
"""
__all__ = ["select_preprocess", "select_normalize"]

from pathlib import Path

import numpy as np
import torch
from PIL import Image as PImage

from gutils import constants as C
from task.utils import transforms as T


def toTensor_normalize_trans(dataset_name: str):
    normalize_trans = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(*select_normalize(dataset_name)),
        ]
    )
    return normalize_trans


def DDR_cls_preprocess(img: PImage.Image, gt: [int, None], img_size, is_train: bool, label_type: str, dataset_name: str):
    gt = torch.tensor(int(gt)).long() if gt is not None else gt
    onebatch = dict()
    init_trans = toTensor_normalize_trans(dataset_name)

    if is_train:
        trans = T.Compose(
            [
                # T.RandAugment(num_ops=4),
                init_trans,
                # T.Resize(240),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomResizedCrop(img_size, prob=1.0, scale=(0.5, 1.0), ratio=(0.8, 1.2)),
                # T.RandomCrop(size=img_size, crop_type="rand", padding_fill_value=0),
            ]
        )
    else:
        trans = T.Compose(
            [
                init_trans,
                T.Resize(img_size),
            ]
        )
    tensor_img, _ = trans(img)
    onebatch["t_img"] = tensor_img
    onebatch["t_gt"] = gt
    return onebatch


def APTOS2019_preprocess(img: PImage.Image, gt: [int, None], img_size, is_train: bool, label_type: str, dataset_name: str):
    gt = torch.tensor(int(gt)).long() if gt is not None else gt
    onebatch = dict()
    init_trans = toTensor_normalize_trans(dataset_name)

    if is_train:
        trans = T.Compose(
            [
                # T.RandAugment(num_ops=4),
                # T.DR_transform(),
                init_trans,
                # T.Resize(np.array(img_size) * 5 // 4),
                # T.CLAHE(clip_limit=4.0),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                # T.RandomResizedCrop(img_size, prob=1.0, scale=(0.8, 1.2), ratio=(0.8, 1.2)),
                T.RandomResizedCrop(img_size, prob=1.0, scale=(0.5, 1.0), ratio=(0.8, 1.2)),
                # T.RandomCrop(size=img_size, crop_type="rand", padding_fill_value=0),
            ]
        )
    else:
        trans = T.Compose(
            [
                # T.DR_transform(),
                init_trans,
                T.Resize(img_size),
            ]
        )
    tensor_img, _ = trans(img)
    onebatch["t_img"] = tensor_img
    onebatch["t_gt"] = gt
    return onebatch


def DDR_seg_preprocess(img: PImage.Image, gt: [PImage.Image, None], img_size, is_train: bool, label_type: str, dataset_name: str):
    pil_img = img.convert("RGB")
    pil_gt = gt.convert("P")
    onebatch = dict()

    init_trans = toTensor_normalize_trans(dataset_name)

    if is_train:
        trans = T.Compose(
            [
                init_trans,
                T.RandomHorizontalFlip(),
                T.RandomCrop(img_size, crop_type="rand", padding_fill_value=0, padding_fill_gt_value=0),
            ]
        )
    else:
        trans = T.Compose(
            [
                init_trans,
                T.Resize(img_size)
            ]
        )

    tensor_img, tensor_gt = trans(pil_img, pil_gt)

    onebatch["t_img"] = tensor_img
    if label_type == C.LABELED:
        onebatch["t_gt"] = tensor_gt
    else:
        weakly_aug_trans = T.RandomChoice(
            [
                # T.RandomColorJitter(brightness=0.02, contrast=0.02),
                T.RandomGaussianBlur(kernel_size=5, min_std=0.1, max_std=0.5),
            ]
        )
        strong_aug_trans = T.Compose(
            [
                # T.RandomColorJitter(brightness=0.25, contrast=0.25),
                T.RandomGaussianBlur(kernel_size=5, min_std=0.5, max_std=1.5),
            ]
        )
        wa_tensor_img, _ = weakly_aug_trans(tensor_img)
        sa_tensor_img, _ = strong_aug_trans(tensor_img)
        onebatch["t_a_img"] = wa_tensor_img
        onebatch["t_A_img"] = sa_tensor_img
    return onebatch


def TJDR_preprocess(img: PImage.Image, gt: [PImage.Image, None], img_size, is_train: bool, label_type: str, dataset_name: str):
    pil_img = img.convert("RGB")
    pil_gt = gt.convert("P")
    onebatch = dict()

    init_trans = toTensor_normalize_trans(dataset_name)

    if is_train:
        trans = T.Compose(
            [
                # T.DR_transform(),
                init_trans,
                T.RandomHorizontalFlip(),
                T.RandomCrop(img_size, crop_type="rand", padding_fill_value=0, padding_fill_gt_value=0),
            ]
        )
    else:
        trans = T.Compose(
            [
                # T.DR_transform(),
                init_trans,
                T.Resize(img_size)
            ]
        )

    tensor_img, tensor_gt = trans(pil_img, pil_gt)

    onebatch["t_img"] = tensor_img
    onebatch["t_gt"] = tensor_gt
    return onebatch


def IDRiD_preprocess(img: PImage.Image, gt: [PImage.Image, None], img_size, is_train: bool, label_type: str, dataset_name: str):
    pil_img = img.convert("RGB")
    pil_gt = gt.convert("P")
    onebatch = dict()

    init_trans = toTensor_normalize_trans(dataset_name)

    if is_train:
        trans = T.Compose(
            [
                # T.DR_transform(),
                init_trans,
                T.RandomHorizontalFlip(),
                T.RandomCrop(img_size, crop_type="rand", padding_fill_value=0, padding_fill_gt_value=0),
            ]
        )
    else:
        trans = T.Compose(
            [
                # T.DR_transform(),
                init_trans,
                T.Resize(img_size)
            ]
        )

    tensor_img, tensor_gt = trans(pil_img, pil_gt)

    onebatch["t_img"] = tensor_img
    onebatch["t_gt"] = tensor_gt
    return onebatch


def select_preprocess(dataset_name: str, img: [PImage.Image or str], gt: [int or str], img_size, is_train: bool, label_type: str):
    if (isinstance(img, str) and Path(img).exists()) or (isinstance(img, Path) and img.exists()):
        img = PImage.open(img)
    if (isinstance(gt, str) and Path(gt).exists()) or (isinstance(gt, Path) and gt.exists()):
        gt = PImage.open(gt)
    if isinstance(img_size, int):
        img_size = [img_size, img_size]
    # Todo: add dataset , modify below
    if dataset_name in [C.DDR_SEG]:
        return DDR_seg_preprocess(img, gt, img_size, is_train, label_type, dataset_name)
    elif dataset_name in [C.TJDR]:
        return TJDR_preprocess(img, gt, img_size, is_train, label_type, dataset_name)
    elif dataset_name in [C.IDRiD]:
        return IDRiD_preprocess(img, gt, img_size, is_train, label_type, dataset_name)
    elif dataset_name in [C.DDR_GRADING]:
        return DDR_cls_preprocess(img, gt, img_size, is_train, label_type, dataset_name)
    elif dataset_name in [C.APTOS2019]:
        return APTOS2019_preprocess(img, gt, img_size, is_train, label_type, dataset_name)


def select_normalize(dataset_name):
    # Todo: add dataset , modify below
    if dataset_name in [C.DDR_SEG]:
        return C.DDR_SEG_MEAN, C.DDR_SEG_STD
    elif dataset_name in [C.TJDR]:
        return C.TJDR_MEAN, C.TJDR_STD
    elif dataset_name in [C.IDRiD]:
        return C.IDRiD_MEAN, C.IDRiD_STD
    elif dataset_name in [C.DDR_GRADING]:
        return C.DDR_CLS_MEAN, C.DDR_CLS_STD
    elif dataset_name in [C.APTOS2019]:
        return C.APTOS2019_MEAN, C.APTOS2019_STD
