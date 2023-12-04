#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:maojingxin
"""
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

root_dir = Path(__file__).resolve().parent.parent.parent  # ..TJDR-FL/
if str(root_dir) != sys.path[0]:
    sys.path[0] = str(root_dir)

from gutils import gutil, constants as C
from task.utils import transforms as T, datasets

_dir = root_dir / "task" / "datas" / "IDRiD"


def gen_ann():
    labels_dir = _dir / "All Segmentation Groundtruths"
    train_img_dir = _dir / "Original Images" / "TrainSet"
    test_img_dir = _dir / "Original Images" / "TestSet"

    train_labels_dir = labels_dir / "TrainSet"
    test_labels_dir = labels_dir / "TestSet"

    train_ann_dir = train_labels_dir / "annotation"
    test_ann_dir = test_labels_dir / "annotation"

    datasets.labels2annotations(train_img_dir, train_labels_dir, train_ann_dir, ".jpg", ".tif", C.DATASET_CLASSES[C.IDRiD], C.TRAIN, force=True)
    datasets.labels2annotations(test_img_dir, test_labels_dir, test_ann_dir, ".jpg", ".tif", C.DATASET_CLASSES[C.IDRiD], C.TEST, force=True)


def circle_crop(size=1024):
    trans = T.Compose(
        [
            T.CircleCrop(),
            T.Resize(size, interpolation=T.InterpolationMode.BICUBIC)
        ]
    )

    train_img_dir = _dir / "Original Images" / "TrainSet"
    train_img_paths = sorted(list(train_img_dir.glob("*.jpg")))
    train_ann_dir = _dir / "All Segmentation Groundtruths" / "TrainSet" / "annotation"
    train_ann_paths = sorted(list(train_ann_dir.glob("*.png")))

    test_img_dir = _dir / "Original Images" / "TestSet"
    test_img_paths = sorted(list(test_img_dir.glob("*.jpg")))
    test_ann_dir = _dir / "All Segmentation Groundtruths" / "TestSet" / "annotation"
    test_ann_paths = sorted(list(test_ann_dir.glob("*.png")))

    new_train_img_dir = _dir / "circle_crop_resize" / "train" / "image"
    new_train_gt_dir = _dir / "circle_crop_resize" / "train" / "annotation"
    new_test_img_dir = _dir / "circle_crop_resize" / "test" / "image"
    new_test_gt_dir = _dir / "circle_crop_resize" / "test" / "annotation"

    new_train_img_dir.mkdir(parents=True, exist_ok=True)
    new_train_gt_dir.mkdir(parents=True, exist_ok=True)
    new_test_img_dir.mkdir(parents=True, exist_ok=True)
    new_test_gt_dir.mkdir(parents=True, exist_ok=True)

    for img_fp, gt_fp in tqdm(list(zip(train_img_paths, train_ann_paths))):
        img, gt = trans(Image.open(img_fp), Image.open(gt_fp))
        img.save(new_train_img_dir / img_fp.name)
        gt.save(new_train_gt_dir / gt_fp.name)

    for img_fp, gt_fp in tqdm(list(zip(test_img_paths, test_ann_paths))):
        img, gt = trans(Image.open(img_fp), Image.open(gt_fp))
        img.save(new_test_img_dir / img_fp.name)
        gt.save(new_test_gt_dir / gt_fp.name)


if __name__ == "__main__":
    gutil.set_all_seed(831)
    gen_ann()
    circle_crop()
