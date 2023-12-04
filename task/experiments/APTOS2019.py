#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:maojingxin
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tqdm
from PIL import Image
from tqdm import tqdm

from gutils import gutil
from task.utils import transforms as T

torch.set_num_threads(4)

root_dir = Path(__file__).resolve().parent.parent.parent  # ..TJDR-FL/

_dir = root_dir / "task" / "datas" / "APTOS2019"


def init():
    image_dir = _dir / "train_images"
    image_csv = _dir / "train.csv"

    df = pd.read_csv(image_csv)
    df.index = df["id_code"]

    print(df.diagnosis.value_counts())
    # diagnosis
    # 0    1805
    # 2     999
    # 1     370
    # 4     295
    # 3     193
    # Name: count, dtype: int64
    # len(images):3662

    images = list(image_dir.glob("*.png"))
    print("len(images):{}".format(len(images)))

    for image in images:
        label = df.loc[image.stem].diagnosis
        new_name = "{}_{}{}".format(image.stem, label, image.suffix)
        image.rename(image_dir / new_name)


def circle_crop_resize(h=2048, w=2048):
    trans = T.Compose(
        [
            T.CircleCrop(),
            T.Resize((h, w), interpolation=T.InterpolationMode.BICUBIC)
        ]
    )

    img_dir = _dir / "train_images"
    new_img_dir = _dir / "crop_resize" / "image"

    new_img_dir.mkdir(parents=True, exist_ok=True)

    img_fps = sorted(list(img_dir.glob("*.png")))
    print("img:{}".format(len(img_fps)))
    for img_fp in tqdm(list(img_fps)):
        img, _ = trans(Image.open(img_fp))
        img.save(new_img_dir / img_fp.name)


def split_train_val(train_rate=0.75):
    train_dir = _dir / "train"
    val_dir = _dir / "val"
    train_dir.mkdir(exist_ok=True, parents=True)
    val_dir.mkdir(exist_ok=True, parents=True)

    images = list((_dir / "crop_resize" / "image").glob("*.png"))
    np.random.shuffle(images)

    train_len = int(len(images) * train_rate)

    train_imgs = images[:train_len]
    val_imgs = images[train_len:]

    print("len(train):{}".format(len(train_imgs)))
    print("len(val):{}".format(len(val_imgs)))

    for train_img in train_imgs:
        shutil.copy(train_img, train_dir / train_img.name)

    for val_img in val_imgs:
        shutil.copy(val_img, val_dir / val_img.name)


if __name__ == "__main__":
    gutil.set_all_seed(1234)
    init()
    circle_crop_resize(h=512, w=512)
    split_train_val()
