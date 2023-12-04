#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:maojingxin
"""
import shutil
import sys
from pathlib import Path

import torchvision.transforms as T
import tqdm
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

root_dir = Path(__file__).resolve().parent.parent.parent  # ..TJDR-FL/
if str(root_dir) != sys.path[0]:
    sys.path[0] = str(root_dir)

from gutils import gutil

_dir = root_dir / "task" / "datas" / "DDR-grading"


def init(DDR_grad_dir: Path):
    _dir = dict(
        train=DDR_grad_dir / "train",
        valid=DDR_grad_dir / "valid",
        test=DDR_grad_dir / "test",
    )
    train_txt = DDR_grad_dir / "train.txt"
    val_txt = DDR_grad_dir / "valid.txt"
    test_txt = DDR_grad_dir / "test.txt"

    lines = dict()
    with train_txt.open("r") as f:
        lines["train"] = f.readlines()
    with val_txt.open("r") as f:
        lines["valid"] = f.readlines()
    with test_txt.open("r") as f:
        lines["test"] = f.readlines()

    for _type in ["train", "valid", "test"]:
        cur_dir = _dir[_type]
        cur_lines = lines[_type]
        for line in tqdm.tqdm(cur_lines):
            name, label = line.strip().split(" ", 1)
            cur_img = cur_dir / name
            new_name = "{}{}".format("{}_{}".format(cur_img.stem, label), cur_img.suffix)
            cur_img.rename(cur_dir / new_name)


def crop_contour_resize(img_dir: Path, img_suffix: str = ".jpg", img_size=1024, threshold=20):
    img_paths = sorted(list(img_dir.glob("*{}".format(img_suffix))))
    crop_contour_dir = img_dir.parent / "crop_contour_resize" / img_dir.name
    crop_contour_dir.mkdir(parents=True, exist_ok=True)
    error_list = []
    for img_path in tqdm(img_paths):
        contour_img_path = crop_contour_dir / img_path.name
        size, contours = gutil.getContours(str(img_path), str(contour_img_path), threshold)
        if size is None:
            error_list.append(img_path)
            continue
        trans = T.Compose(
            [
                T.CenterCrop(size),
                T.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC)
            ]
        )
        pil_img = Image.open(img_path)
        pil_img = trans(pil_img)
        pil_img.save(contour_img_path)

    print("error_list:{}".format(error_list))


def merge_train_val(_dir: Path, suffix=".jpg"):
    train_dir = _dir / "train"
    val_dir = _dir / "valid"
    train_val_dir = _dir / "train+val"
    train_val_dir.mkdir(exist_ok=True, parents=True)

    trains = list(train_dir.glob("*{}".format(suffix)))
    vals = list(val_dir.glob("*{}".format(suffix)))
    print("len(train):{}".format(len(trains)))
    print("len(val):{}".format(len(vals)))

    train_vals = trains + vals
    for train_val in train_vals:
        shutil.copy(train_val, train_val_dir / train_val.name)


if __name__ == "__main__":
    init(_dir)
    crop_contour_resize(_dir / "train")
    crop_contour_resize(_dir / "valid")
    crop_contour_resize(_dir / "test")

    merge_train_val(_dir / "crop_contour_resize")
