#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:maojingxin
"""

__all__ = ["dataset_init", "minibatch_init", "sup_loss_init"]

from pathlib import Path

import numpy as np
import torch
from torch import nn

from gutils import constants as C, Logger
from task.utils.datasets import ListIMGDataset, MixUp, CutMix, CutOut
from task.utils.loss import FocalLoss, DiceLoss


def dataset_init(cfg: dict, dataset_path: str, **kwargs):
    assert Path(dataset_path).exists(), "{} not exist"

    dataset_name = cfg[C.NAME_DATASET]
    return ListIMGDataset(
        txt_path=dataset_path,
        dataset_name=dataset_name,
        num_classes=cfg[C.NUM_CLASSES],
        img_size=kwargs["img_size"],
        is_train=kwargs["is_train"],
    )


def minibatch_init(task: str, minibatch: dict, device=None, **kwargs):
    if task in [C.IMG_CLASSIFICATION, C.IMG_SEGMENTATION]:
        X, Y = minibatch[kwargs.get("X", "t_img")], minibatch[kwargs.get("X", "t_gt")]
    else:
        raise ValueError(task)

    if device:
        X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
    if kwargs.get("mixup", False):
        alpha = kwargs.get("mixup_alpha", 1.0)
        mixed_X, Y_a, Y_b, lamb = MixUp(X, Y, alpha)
        return mixed_X, (Y_a, Y_b), lamb
    if kwargs.get("cutmix", False) and task in [C.IMG_CLASSIFICATION, C.IMG_SEGMENTATION]:
        alpha = kwargs.get("cutmix_alpha", 1.0)
        mixed_X, Y_a, Y_b, lamb = CutMix(X, Y, alpha)
        return mixed_X, (Y_a, Y_b), lamb
    if kwargs.get("cutout", False) and task in [C.IMG_CLASSIFICATION, C.IMG_SEGMENTATION]:
        n_holes = kwargs.get("n_holes", 3)
        length = kwargs.get("length", (X.size()[2:]) // 8)
        X, Y = CutOut(X, Y, n_holes=n_holes, length=length)
        return X, Y
    return X, Y


def sup_loss_init(cfg: dict, logger: Logger, **kwargs):
    sup_loss_f = []
    ignore_label = cfg.get(C.IGNORE_LABEL, C.DEFAULT_IGNORE_LABEL)
    dataset_name = cfg[C.NAME_DATASET]
    num_classes = cfg[C.NUM_CLASSES]
    device = kwargs.get("device")
    loss_fn_cfg = dict()

    # Todo: add dataset , modify belows
    sup_loss_weight = 1
    if dataset_name in [C.DDR_GRADING]:
        alpha = np.array([1.0, 2.0, 1.0, 1.2, 1.0, 1.0])
        size_average = False
        loss_fn_cfg.update(dict(FocalLoss=dict(alpha=alpha, size_average=size_average)))

        sup_loss_f.append(nn.CrossEntropyLoss())
        sup_loss_f.append(FocalLoss(alpha=alpha, size_average=size_average))

        sup_loss_weight = [0.1, 1.0]
        # class_weights = torch.FloatTensor([1., 10., 1.3, 20., 8., 6.]).to(device)
        # sup_loss_f.append(nn.CrossEntropyLoss(weight=class_weights))
        # logger.info("loss_fn: nn.CrossEntropyLoss(weight={})".format(class_weights))
        # sup_loss_f.append(nn.CrossEntropyLoss())
        # logger.info("loss_fn: nn.CrossEntropyLoss()")
    elif dataset_name in [C.APTOS2019]:
        # sup_loss_f.append(nn.CrossEntropyLoss())

        # class_weights = torch.FloatTensor([1., 5., 2., 9., 6.]).to(device)
        # label_smoothing = 0.1
        # loss_fn_cfg.update({"nn.CrossEntropyLoss": dict(
        #     class_weights=class_weights,
        #     label_smoothing=label_smoothing
        # )})
        # sup_loss_f.append(nn.CrossEntropyLoss(
        #     weight=class_weights,
        #     label_smoothing=label_smoothing
        # ))

        alpha = np.array([1.0, 2.0, 1.2, 2.5, 2.5])
        size_average = True
        smooth = 1e-5
        loss_fn_cfg.update(dict(FocalLoss=dict(alpha=alpha, size_average=size_average, smooth=smooth)))

        sup_loss_f.append(nn.CrossEntropyLoss())
        sup_loss_f.append(FocalLoss(alpha=alpha, size_average=size_average, smooth=smooth))
        sup_loss_weight = [0.1, 1.0]
    elif dataset_name in [C.IDRiD,
                          # C.TJDR
                          ]:
        sup_loss_f.append(DiceLoss())

        alpha = np.array([1.0, 3.0, 3.0, 3.0, 3.0])
        sup_loss_f.append(FocalLoss(alpha=alpha))
        loss_fn_cfg.update(dict(FocalLoss=dict(alpha=alpha)))

        sup_loss_weight = [0.1, 1.0]
    elif dataset_name in [
        C.TJDR,
        C.DDR_SEG
    ]:
        sup_loss_f.append(DiceLoss())

        alpha = np.array([1.0, 3.0, 3.0, 9.0, 3.0])
        # alpha = np.array([1.0, 3.0, 3.0, 3.0, 3.0])
        sup_loss_f.append(FocalLoss(alpha=alpha))
        loss_fn_cfg.update(dict(FocalLoss=dict(alpha=alpha)))

        sup_loss_weight = [0.1, 1.0]
    else:
        raise ValueError("dataset error:{}".format(dataset_name))
    assert sup_loss_weight is None \
           or isinstance(sup_loss_weight, (int, float)) \
           or isinstance(sup_loss_weight, (list, tuple, np.ndarray)) and (len(sup_loss_f) == len(sup_loss_weight))
    logger.info("sup_loss_f:{}".format(sup_loss_f))
    for k, v in loss_fn_cfg.items():
        logger.info("{}:{}".format(k, v))
    logger.info("sup_loss_weight:{}".format(sup_loss_weight))
    return sup_loss_f, sup_loss_weight
