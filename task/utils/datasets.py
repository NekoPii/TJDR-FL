#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: maojingxin
"""
import copy
import random
import warnings
from pathlib import Path
from typing import List

import cv2 as cv
import imgviz
import numpy as np
import torch
from PIL import Image as PImage
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from gutils import constants as C, gutil
from task.utils import transforms as T
from task.utils.preprocess import select_preprocess


def dataset_txt_generate(
        img_dir: Path,
        img_suffix: str,
        gts_dir: Path or None,
        gt_suffix: str or None,
        txt_file_path: Path,
        is_augment: bool,
        **kwargs,
):
    img_paths = sorted(list(img_dir.glob("*{}".format(img_suffix))))
    img_names = [img_path.name for img_path in img_paths]
    if gts_dir is not None:
        gt_paths = sorted(list(gts_dir.glob("*{}".format(gt_suffix))))
        gt_names = [gt_path.name for gt_path in gt_paths]

    if is_augment:
        # .../augment/...
        is_dr_aug = kwargs.get("enable_dr_aug", False)

        aug_img_dict = {}
        aug_img_dir = img_dir / "augment"
        aug_img_paths = list(aug_img_dir.glob("*{}".format(img_suffix)))
        if is_dr_aug:
            dr_aug_img_dir = aug_img_dir / "dr"
            aug_img_paths += list(dr_aug_img_dir.glob("*{}".format(img_suffix)))
        aug_img_paths = sorted(aug_img_paths)

        for aug_img_path in aug_img_paths:
            ori_img_name = aug_img_path.name.split("_", 1)[1]
            if ori_img_name in aug_img_dict:
                aug_img_dict[ori_img_name].append(aug_img_path)
            else:
                aug_img_dict[ori_img_name] = [aug_img_path]
            aug_img_dict[ori_img_name] = sorted(aug_img_dict[ori_img_name])

        if gts_dir is not None:
            aug_gts_dict = {}
            aug_gts_dir = gts_dir / "augment"
            aug_gts_paths = list(aug_gts_dir.glob("*{}".format(gt_suffix)))
            if is_dr_aug:
                dr_aug_gts_dir = aug_gts_dir / "dr"
                aug_gts_paths += list(dr_aug_gts_dir.glob("*{}".format(gt_suffix)))
            aug_gts_paths = sorted(aug_gts_paths)

            for aug_gt_path in aug_gts_paths:
                ori_gt_name = aug_gt_path.name.split("_", 1)[1]
                if ori_gt_name in aug_gts_dict:
                    aug_gts_dict[ori_gt_name].append(aug_gt_path)
                else:
                    aug_gts_dict[ori_gt_name] = [aug_gt_path]
                aug_gts_dict[ori_gt_name] = sorted(aug_gts_dict[ori_gt_name])

    lines_dict = {}
    aug_lines_dict = {}
    if gts_dir is not None:
        if len(img_paths) > len(gt_paths) > 0:
            warnings.warn("len(img_paths):{} > len(gt_paths):{}".format(len(img_paths), len(gt_paths)))
            gt_stems = sorted(list(set([gt_path.stem for gt_path in gt_paths])))
            tmp_img_names = [str(gt_stem) + img_suffix for gt_stem in gt_stems]
            img_paths = sorted(filter(lambda x: str(x.name) in tmp_img_names, img_paths))
        assert len(img_paths) == len(gt_paths), "len(img_paths):{} != len(gt_paths):{}".format(len(img_paths), len(gt_paths))
        for img_path, gt_path in zip(img_paths, gt_paths):
            lines_dict[img_path.name] = str(img_path) + " " + str(gt_path) + "\n"
        if is_augment:
            for img_name, gt_name in zip(img_names, gt_names):
                aug_lines_dict[img_name] = [str(aug_img_path) + " " + str(aug_gt_path) + "\n" for aug_img_path, aug_gt_path in zip(aug_img_dict[img_name], aug_gts_dict[gt_name])]
    else:
        for img_path in img_paths:
            lines_dict[img_path.name] = str(img_path) + " " + img_path.stem.rsplit("_", 1)[1] + "\n"
        if is_augment:
            for img_name in img_names:
                aug_lines_dict[img_name] = [str(aug_img_path) + " " + aug_img_path.stem.rsplit("_", 1)[1] + "\n" for aug_img_path in aug_img_dict[img_name]]
    if txt_file_path is not None:
        with open(txt_file_path, "w+") as f:
            all_lines = list(lines_dict.values())
            if is_augment:
                for aug_lines in aug_lines_dict.values():
                    all_lines += aug_lines
            f.writelines(all_lines)
    return lines_dict, aug_lines_dict


def iid_dataset_txt_generate(
        img_dir: Path,
        img_suffix: str,
        gts_dir: Path or None,
        gt_suffix: str or None,
        dataset_split: int,
        txt_file_paths: list,
        is_augment: bool,
        drop_split: float,
        **kwargs
):
    if len(txt_file_paths) == 0 or dataset_split == 0:
        return
    assert 0 <= drop_split <= 1, "drop_split:{} valid".format(drop_split)

    lines_dict, aug_lines_dict = dataset_txt_generate(img_dir, img_suffix, gts_dir, gt_suffix, None, is_augment, **kwargs)
    img_names = sorted(list(lines_dict.keys()))
    split_num = dataset_split
    split_dataset_lines = []
    per_split_data_size = len(img_names) // split_num
    for i in range(split_num - 1):
        now_split_names = sorted(np.random.choice(img_names, per_split_data_size, replace=False))
        img_names = sorted(list(set(img_names) - set(now_split_names)))

        now_split_lines = sorted([lines_dict[img_name] for img_name in now_split_names])
        if is_augment:
            aug_lines = []
            for img_name in now_split_names:
                aug_lines += aug_lines_dict[img_name]
            now_split_lines += sorted(np.random.choice(aug_lines, int(len(aug_lines) * (1 - drop_split)), replace=False))
        split_dataset_lines.append(now_split_lines)

    now_split_lines = sorted([lines_dict[img_name] for img_name in img_names])
    if is_augment:
        aug_lines = []
        for img_name in img_names:
            aug_lines += aug_lines_dict[img_name]
        now_split_lines += sorted(np.random.choice(aug_lines, int(len(aug_lines) * (1 - drop_split)), replace=False))
    split_dataset_lines.append(now_split_lines)

    alloc_dataset_lines_ids = list(np.random.choice(list(range(split_num)), len(txt_file_paths), replace=False))
    for i, txt_file_path in enumerate(txt_file_paths):
        with open(txt_file_path, "w+") as f:
            now_split_lines = split_dataset_lines[alloc_dataset_lines_ids[i]]
            """
            old drop_split
            """
            # _len = len(split_dataset_lines[alloc_dataset_lines_ids[i]])
            # now_split_lines = sorted(np.random.choice(split_dataset_lines[alloc_dataset_lines_ids[i]], int(_len * (1 - drop_split)), replace=False))
            random.shuffle(now_split_lines)
            f.writelines(now_split_lines)


def non_iid_dataset_txt_generate(
        img_dir: Path,
        img_suffix: str,
        gts_dir: Path or None,
        gt_suffix: str or None,
        dataset_split: int,
        txt_file_paths: list,
        is_augment: bool,
        classes: list,
        gamma: float,
        **kwargs
):
    if len(txt_file_paths) == 0 or dataset_split == 0:
        return
    if gts_dir is not None or gt_suffix is not None:
        raise TypeError("non_iid dataset split not supports for segmentation task")

    lines_dict, aug_lines_dict = dataset_txt_generate(img_dir, img_suffix, None, None, None, is_augment, **kwargs)

    split_num = dataset_split
    split_dataset_names = [[] for _ in range(split_num)]
    split_dataset_lines = [[] for _ in range(split_num)]
    num_classes = len(classes)
    non_iid_label_distribu = np.random.dirichlet([gamma] * split_num, num_classes)
    classes2names = {cls: [] for cls in range(num_classes)}

    img_names = sorted(list(lines_dict.keys()))
    for img_name in img_names:
        gt = int(lines_dict[img_name].split(" ", 1)[1].strip())
        classes2names[gt].append(img_name)

    for cls in range(num_classes):
        cur_cls_split = np.split(classes2names[cls], (np.cumsum(non_iid_label_distribu[cls])[:-1] * len(classes2names[cls])).astype(int))
        for i in range(split_num):
            split_dataset_names[i] += list(cur_cls_split[i])

    for i in range(split_num):
        split_dataset_lines[i] += [lines_dict[name] for name in split_dataset_names[i]]
        if is_augment:
            for name in split_dataset_names[i]:
                split_dataset_lines[i] += aug_lines_dict[name]

    alloc_dataset_lines_ids = list(np.random.choice(list(range(split_num)), len(txt_file_paths), replace=False))
    for i, txt_file_path in enumerate(txt_file_paths):
        with open(txt_file_path, "w+") as f:
            now_split_lines = split_dataset_lines[alloc_dataset_lines_ids[i]]
            random.shuffle(now_split_lines)
            f.writelines(now_split_lines)


def dataset_augment(
        dataset_name: str,
        img_dir: Path,
        img_suffix: str,
        gt_dir: Path = None,
        gt_suffix: str = None,
        force=False,
        **kwargs,
):
    img_paths = sorted([img_path for img_path in img_dir.glob("*{}".format(img_suffix))])
    augment_img_dir = img_dir / "augment"
    augment_img_dir.mkdir(exist_ok=True, parents=True)

    is_dr = kwargs.get("enable_dr_aug", False) and dataset_name in [C.TJDR, C.DDR_SEG, C.APTOS2019, C.DDR_GRADING]
    drp_trans = T.DR_transform()

    if force or len([img_path for img_path in augment_img_dir.glob("*{}".format(img_suffix))]) != len(img_paths) * (5 + is_dr):
        for img_path in tqdm(img_paths, desc="imgs_augment"):
            img_pil = PImage.open(img_path)
            img_pil.transpose(PImage.ROTATE_90).save(augment_img_dir / "90_{}".format(img_path.name))
            img_pil.transpose(PImage.ROTATE_180).save(augment_img_dir / "180_{}".format(img_path.name))
            img_pil.transpose(PImage.ROTATE_270).save(augment_img_dir / "270_{}".format(img_path.name))
            img_pil.transpose(PImage.FLIP_LEFT_RIGHT).save(augment_img_dir / "h_{}".format(img_path.name))
            img_pil.transpose(PImage.FLIP_TOP_BOTTOM).save(augment_img_dir / "v_{}".format(img_path.name))
            if is_dr:
                dr_aug_dir = augment_img_dir / "dr"
                dr_aug_dir.mkdir(parents=True, exist_ok=True)
                drp_img_pil = drp_trans(image=img_pil)[0]
                drp_img_pil.save(dr_aug_dir / "drp_{}".format(img_path.name))

    if gt_dir is not None:
        gt_paths = sorted([gt_path for gt_path in gt_dir.glob("*{}".format(gt_suffix))])
        augment_gt_dir = gt_dir / "augment"
        augment_gt_dir.mkdir(exist_ok=True, parents=True)
        if len(img_paths) != len(gt_paths):
            warnings.warn("dataset augment -- len(img_paths):{}!=len(gt_paths):{}".format(len(img_paths), len(gt_paths)))

        if force or len([gt_path for gt_path in augment_gt_dir.glob("*{}".format(gt_suffix))]) != len(gt_paths) * (5 + is_dr):
            for gt_path in tqdm(gt_paths, desc="gts_augment"):
                gt_pil = PImage.open(gt_path)
                gt_pil.transpose(PImage.ROTATE_90).save(augment_gt_dir / "90_{}".format(gt_path.name))
                gt_pil.transpose(PImage.ROTATE_180).save(augment_gt_dir / "180_{}".format(gt_path.name))
                gt_pil.transpose(PImage.ROTATE_270).save(augment_gt_dir / "270_{}".format(gt_path.name))
                gt_pil.transpose(PImage.FLIP_LEFT_RIGHT).save(augment_gt_dir / "h_{}".format(gt_path.name))
                gt_pil.transpose(PImage.FLIP_TOP_BOTTOM).save(augment_gt_dir / "v_{}".format(gt_path.name))
                if is_dr:
                    dr_aug_dir = augment_img_dir / "dr"
                    dr_aug_dir.mkdir(parents=True, exist_ok=True)
                    drp_gt_pil = drp_trans(image=None, gt=gt_pil)[1]
                    drp_gt_pil.save(dr_aug_dir / "drp_{}".format(gt_path.name))


def labels2annotations(
        img_dir: Path,
        gt_dir: Path,
        ann_dir: Path,
        img_suffix: str,
        gt_suffix: str,
        classes: list,
        dataset_type: str,
        force=False,
        ignore_class_index=0,
):
    """
    Attention:img_name should be equal to gt_name
    :param img_dir:.../img_files
    :param gt_dir:.../class/gt_files
    :param ann_dir:.../ann_files
    :param img_suffix: include dot
    :param gt_suffix: include dot
    :param classes:the list of classes
    :param dataset_type:
    :param force: force to re-generate annotations
    """
    print("classes:{}".format(classes))
    ann_dir.mkdir(parents=True, exist_ok=True)
    img_paths = sorted(list(img_dir.glob("*{}".format(img_suffix))))
    for img_path in tqdm(img_paths, desc="{}_labels_to_annotations".format(dataset_type)):
        img_name = img_path.stem
        gt_name = img_name

        ann_path = ann_dir / "{}.png".format(img_name)
        if not force and ann_path.exists():
            continue
        label_paths = []
        for idx, now_class in enumerate(classes):
            if idx == ignore_class_index:
                continue
            label_path = gt_dir / now_class / "{}{}".format(gt_name, gt_suffix)
            label_paths.append(label_path)
        label2annotation(img_path, label_paths, ann_path)


def label2annotation(img_path: Path, label_paths: List[Path], ann_path: Path):
    img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
    shape = img.shape
    del img
    ann = np.zeros(shape, dtype=np.int32)

    for i, label_path in enumerate(label_paths):
        if not label_path.exists():
            continue
        label = cv.imread(str(label_path), cv.IMREAD_GRAYSCALE)
        ann[label > 0] = i + 1
    ann_pil = PImage.fromarray(ann.astype(np.uint8), mode="P")
    color_map = imgviz.label_colormap()
    ann_pil.putpalette(color_map)
    ann_pil.save(ann_path)
    return ann_pil


class ListIMGDataset(Dataset):
    def __init__(self, txt_path: str, dataset_name: str, num_classes: int, img_size, is_train: bool):
        txt_path = Path(txt_path)
        assert txt_path.exists(), "{} not exist".format(txt_path)
        with txt_path.open("r") as f:
            row_datas = f.readlines()

        self.dataset_name = dataset_name
        self.task = C.TASK.get(self.dataset_name)
        self.num_classes = num_classes
        self.img_size = [img_size, img_size] if isinstance(img_size, int) else img_size
        self.is_train = is_train

        self.row_datas = []
        for row_data in row_datas:
            # remove augment for eval dataset
            if self.is_train or "augment" not in row_data:
                self.row_datas.append(row_data.strip().split(" ", 1))

        self.imgs, self.gts = self._preload()

    def _preload(self):
        imgs, gts = [], []
        for img_p, gt_p in self.row_datas:
            if self.task == C.IMG_CLASSIFICATION:
                if np.all(np.array(self.img_size) < np.array([128, 128])):
                    img_pil = PImage.open(img_p)
                    if gutil.get_img_size(img_pil) != self.img_size:
                        img_pil, _ = T.Resize(self.img_size)(img_pil)
                        imgs.append(copy.deepcopy(img_pil))
                else:
                    imgs.append(img_p)
                gts.append(int(gt_p))
            else:
                imgs.append(img_p)
                gts.append(gt_p)
        return imgs, gts

    def __getitem__(self, index):
        img, gt = copy.deepcopy(self.imgs[index]), copy.deepcopy(self.gts[index])
        minibatch = select_preprocess(self.dataset_name, img, gt, self.img_size, self.is_train, C.LABELED)
        return minibatch

    def __len__(self):
        return len(self.row_datas)


class MaskGenerator(object):
    """
    Mask Generator
    """

    def generate_params(self, n_masks, mask_shape, rng=None):
        raise NotImplementedError

    def append_to_batch(self, *batch):
        x = batch[0]
        params = self.generate_params(len(x), x.shape[2:4])
        return batch + (params,)


class BoxMaskGenerator(MaskGenerator):
    def __init__(self, prop_range, n_boxes=1, random_aspect_ratio=True, prop_by_area=True, within_bounds=True, invert=False):
        if isinstance(prop_range, float):
            prop_range = [prop_range, prop_range]
        self.prop_range = prop_range
        self.n_boxes = n_boxes
        self.random_aspect_ratio = random_aspect_ratio
        self.prop_by_area = prop_by_area
        self.within_bounds = within_bounds
        self.invert = invert

    def generate_params(self, n_masks, mask_shape, rng=None):
        """
        Box masks can be generated quickly on the CPU so do it there.
        :param n_masks: number of masks to generate (batch size)
        :param mask_shape: Mask shape as a `(height, width)` tuple
        :param rng: [optional] np.random.RandomState instance
        :return: masks: masks as a `(N, 1, H, W)` array
        """
        if rng is None:
            rng = np.random

        if self.prop_by_area:
            # Choose the proportion of each mask that should be above the threshold
            mask_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))

            # Zeros will cause NaNs, so detect and suppres them
            zero_mask = mask_props == 0.0

            if self.random_aspect_ratio:
                y_props = np.exp(rng.uniform(low=0.0, high=1.0, size=(n_masks, self.n_boxes)) * np.log(mask_props))
                x_props = mask_props / y_props
            else:
                y_props = x_props = np.sqrt(mask_props)
            fac = np.sqrt(1.0 / self.n_boxes)
            y_props *= fac
            x_props *= fac

            y_props[zero_mask] = 0
            x_props[zero_mask] = 0
        else:
            if self.random_aspect_ratio:
                y_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
                x_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
            else:
                x_props = y_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
            fac = np.sqrt(1.0 / self.n_boxes)
            y_props *= fac
            x_props *= fac

        sizes = np.round(np.stack([y_props, x_props], axis=2) * np.array(mask_shape)[None, None, :])

        if self.within_bounds:
            positions = np.round((np.array(mask_shape) - sizes) * rng.uniform(low=0.0, high=1.0, size=sizes.shape))
            rectangles = np.append(positions, positions + sizes, axis=2)
        else:
            centres = np.round(np.array(mask_shape) * rng.uniform(low=0.0, high=1.0, size=sizes.shape))
            rectangles = np.append(centres - sizes * 0.5, centres + sizes * 0.5, axis=2)

        if self.invert:
            masks = np.zeros((n_masks, 1) + mask_shape)
        else:
            masks = np.ones((n_masks, 1) + mask_shape)
        for i, sample_rectangles in enumerate(rectangles):
            for y0, x0, y1, x1 in sample_rectangles:
                masks[i, 0, int(y0):int(y1), int(x0):int(x1)] = 1 - masks[i, 0, int(y0):int(y1), int(x0):int(x1)]
        return masks


class AddMaskParamsToBatch(object):
    """
    We add the cut-and-paste parameters to the mini-batch within the collate function,
    (we pass it as the `batch_aug_fn` parameter to the `SegCollate` constructor)
    as the collate function pads all samples to a common size
    """

    def __init__(self, mask_gen):
        self.mask_gen = mask_gen

    def __call__(self, batch):
        sample = batch[0]
        mask_size = sample["t_img"].shape[1:3]
        params = self.mask_gen.generate_params(len(batch), mask_size)
        for sample, p in zip(batch, params):
            sample["masks"] = p.astype(np.float32)
        return batch


class SegCollate(object):
    def __init__(self, batch_aug_fn=None):
        self.batch_aug_fn = batch_aug_fn

    def __call__(self, batch):
        if self.batch_aug_fn is not None:
            batch = self.batch_aug_fn(batch)
        return default_collate(batch)


def _rand_bbox(size, lamb, cut_w=None, cut_h=None):
    W = size[2]
    H = size[3]

    if lamb is not None:
        cut_rat = np.sqrt(1. - lamb)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def MixUp(X, Y, alpha=1.0):
    if alpha > 0:
        lamb = np.random.beta(alpha, alpha)
    else:
        lamb = 1

    batch_size = X.size(0)
    index = torch.randperm(batch_size)

    mixed_X = lamb * X + (1 - lamb) * X[index, :]
    Y_a, Y_b = Y, Y[index]
    return mixed_X, Y_a, Y_b, lamb


def CutMix(X, Y, alpha=1.0):
    if alpha > 0:
        lamb = np.random.beta(alpha, alpha)
    else:
        lamb = 1

    batch_size = X.size(0)
    index = torch.randperm(batch_size)

    bbx1, bby1, bbx2, bby2 = _rand_bbox(X.size(), lamb)
    X[:, :, bbx1:bbx2, bby1:bby2] = X[index, :, bbx1:bbx2, bby1:bby2]
    Y_a, Y_b = Y, Y[index]
    lamb = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (X.size(-1) * X.size(-2)))
    return X, Y_a, Y_b, lamb


def CutOut(X, Y, n_holes, length):
    if isinstance(length, int):
        cut_h = length
        cut_w = length
    else:
        cut_h, cut_w = length

    if n_holes > 0 and length > 0:
        for n in range(n_holes):
            bbx1, bby1, bbx2, bby2 = _rand_bbox(X.size(), lamb=None, cut_h=cut_h, cut_w=cut_w)
            X[:, :, bbx1:bbx2, bby1:bby2] = 0
    return X, Y
