#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:maojingxin
"""
import shutil
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

root_dir = Path(__file__).resolve().parent.parent.parent  # ..TJDR-FL/
if str(root_dir) != sys.path[0]:
    sys.path[0] = str(root_dir)

from gutils import constants as C
from gutils import gutil
from task.utils import transforms as T
from task.utils import datasets

seg_dir = root_dir / "task" / "datas" / "DDR-seg"


def seg_init(_dir=seg_dir):
    lesions = ["EX", "HE", "MA", "SE"]
    _types = [
        C.TRAIN,
        C.VALIDATION,
        C.TEST
    ]

    for _type in _types:
        cur_img_dir = _dir / _type / "image"
        cur_label_dir = _dir / _type / "label"
        cur_ann_dir = _dir / _type / "annotation"
        cur_ann_dir.mkdir(parents=True, exist_ok=True)

        datasets.labels2annotations(cur_img_dir, cur_label_dir, cur_ann_dir, ".jpg", ".tif", lesions, _type, ignore_class_index=-1)
        crop_contour_resize(cur_img_dir, ".jpg", cur_ann_dir, ".png")


def crop_contour_resize(img_dir: Path, img_suffix: str = ".jpg", gt_dir: Path = None, gt_suffix: str = None, img_size=1024, threshold=20):
    img_paths = sorted(list(img_dir.glob("*{}".format(img_suffix))))
    crop_contour_img_dir = img_dir.parent / "crop_contour_resize" / img_dir.name
    crop_contour_img_dir.mkdir(parents=True, exist_ok=True)

    if gt_dir is not None:
        gt_paths = sorted(list(gt_dir.glob("*{}".format(gt_suffix))))
        crop_contour_gt_dir = gt_dir.parent / "crop_contour_resize" / gt_dir.name
        crop_contour_gt_dir.mkdir(parents=True, exist_ok=True)

    contour_dir = img_dir.parent / "contour"
    contour_dir.mkdir(parents=True, exist_ok=True)

    error_list = []
    for i, img_path in enumerate(tqdm(img_paths)):
        pil_img = Image.open(img_path)
        # while background ( for DDR-cls )
        if img_path.stem in ["007-7249-400_4", "007-7252-400_4", "007-7255-400_4", "007-7256-400_4", "007-7257-400_4", "007-7258-400_4", "007-7259-400_4"]:
            size = min(pil_img.size)
        # full black img ( for DDR-cls )
        elif img_path.stem in ["007-8707-605_5", "007-8846-605_5"]:
            error_list.append(str(img_path))
            continue
        else:
            contour_img_path = contour_dir / img_path.name
            size, contours = gutil.getContours(str(img_path), str(contour_img_path), threshold)
        if size is None:
            error_list.append(str(img_path))
            continue
        img_trans = T.Compose(
            [
                T.CenterCrop(size),
                T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC)
            ]
        )
        pil_img = img_trans(pil_img)[0]
        pil_img.save(crop_contour_img_dir / img_path.name)
        if gt_dir is not None:
            gt_trans = T.Compose(
                [
                    T.CenterCrop(size),
                    T.Resize(img_size, interpolation=T.InterpolationMode.NEAREST)
                ]
            )
            pil_gt = Image.open(gt_paths[i])
            pil_gt = gt_trans(pil_gt)[0]
            pil_gt.save(crop_contour_gt_dir / gt_paths[i].name)

    print("error_list:{}".format(error_list))
    with open(img_dir.parent / "error_list.txt", "a+") as f:
        f.writelines(gutil.get_now_daytime())
        f.writelines(error_list)
        f.writelines("\n--------------------------------------\n")


def merge_train_val_seg():
    train_dir = seg_dir / C.TRAIN / "crop_contour_resize"
    val_dir = seg_dir / C.VALIDATION / "crop_contour_resize"
    train_val_dir = seg_dir / "train+val" / "crop_contour_resize"

    train_image_dir = train_dir / "image"
    val_image_dir = val_dir / "image"
    train_val_image_dir = train_val_dir / "image"
    train_val_image_dir.mkdir(parents=True, exist_ok=True)

    train_annotation_dir = train_dir / "annotation"
    val_annotation_dir = val_dir / "annotation"
    train_val_annotation_dir = train_val_dir / "annotation"
    train_val_annotation_dir.mkdir(parents=True, exist_ok=True)

    for img in tqdm(list(train_image_dir.glob("*.jpg")) + list(val_image_dir.glob("*.jpg"))):
        shutil.copy(img, train_val_image_dir / img.name)

    for img in tqdm(list(train_annotation_dir.glob("*.png")) + list(val_annotation_dir.glob("*.png"))):
        shutil.copy(img, train_val_annotation_dir / img.name)


if __name__ == "__main__":
    seg_init(seg_dir)
    merge_train_val_seg()
