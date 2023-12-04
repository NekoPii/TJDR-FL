#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:maojingxin
"""
__all__ = ["IMGDrawer"]

import sys
from pathlib import Path
from typing import List

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

cur_dir = Path(__file__).resolve().parent  # ..components/
root_dir = cur_dir.parent  # ..TJDR-FL/
if str(root_dir) != sys.path[0]:
    sys.path[0] = str(root_dir)

from gutils import Logger, constants as C, gutil


class Drawer(object):
    def __init__(self, logger: Logger = None):
        self.logger = Logger() if logger is None else logger

    def plot_train_heatmap(self, clients_classes_array, save_path: Path, **kwargs):
        self.logger.info("clients_classes_array:\n{}".format(clients_classes_array))
        self.logger.info("num:{}".format(np.sum(clients_classes_array)))

        # sns.set(font_scale=1.4)
        fig, ax = plt.subplots()
        cmap = kwargs.get("cmap", sns.cubehelix_palette(n_colors=6, rot=-0.3, gamma=0.5, as_cmap=True, light=0.95, dark=0.3))
        sns.heatmap(
            clients_classes_array, cmap=cmap, ax=ax, annot=True, fmt="g", annot_kws={"size": 13}
            # cbar_kws=dict(use_gridspec=False, location="top")
        )
        ax.invert_yaxis()
        x_ticks = np.arange(1, 1 + len(clients_classes_array[0]))
        y_ticks = np.arange(0, len(clients_classes_array))
        ax.set_xticklabels(x_ticks, size=15)
        ax.set_yticklabels(y_ticks, size=15)
        ax.set_xlabel("Client ID", fontsize=16)
        ax.set_ylabel("Class ID", fontsize=16)
        ax.collections[0].colorbar.ax.tick_params(labelsize=14)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        # gutil.get_func_params(inspect.currentframe(), f_path=save_path.with_name("params.txt"))
        fig.savefig(save_path, dpi=500, bbox_inches="tight")
        # fig.savefig(save_path.with_suffix(".eps"), dpi=500, format="eps", bbox_inches="tight")
        self.logger.info("plot_train_dataset_distribu_heatmap completed.")

    def plot_train_hist(self, dataset_label_lists, num_classes, num_clients, save_path: Path):
        fig, ax = plt.subplots()
        plt.hist(
            dataset_label_lists, stacked=True,
            bins=np.arange(min(min(dataset_label_lists)) - 0.5, max(max(dataset_label_lists)) + 1.5, 1),
            # label=["Client_{}".format(i) for i in range(num_clients)],
            rwidth=0.5,
            orientation="vertical"
        )
        # ax.invert_yaxis()
        ax.set_xlabel("Class ID", fontsize=12)
        ax.set_xticks(np.arange(num_classes))
        ax.set_ylabel("Numbers", fontsize=12)
        # ax.set_yticks(fontsize=12)
        plt.legend(labels=["Client_{}".format(i) for i in range(num_clients)], bbox_to_anchor=(1, 1))

        save_path.parent.mkdir(parents=True, exist_ok=True)
        # gutil.get_func_params(inspect.currentframe(), f_path=save_path.with_name("params.txt"))
        fig.savefig(save_path, dpi=500, bbox_inches="tight")
        self.logger.info("plot_train_dataset_distribu_hist completed.")


class IMGDrawer(Drawer):
    def __init__(self, logger: Logger = None):
        self.default_save_dir = root_dir / "task" / "saves" / "figs" / gutil.get_now_daytime()
        super().__init__(logger)

    def draw_dataset_distribu_heatmap(self, dataset_txts: List[Path] or None, dataset_txt_dir: Path or None, dataset_type: str, dataset_name: str, num_classes: int, save_path: Path = None, **kwargs):
        if not isinstance(dataset_name, str):
            return
        task = C.TASK.get(dataset_name)
        assert dataset_type in [C.TRAIN, C.TEST, C.VALIDATION]
        if task != C.IMG_CLASSIFICATION:
            self.logger.warn("dataset_name:{} task:{} is invalid for draw_{}_dataset_distribu_heatmap.".format(dataset_name, task, dataset_type))
            return

        dataset_txts = sorted([f for f in dataset_txt_dir.glob("*.txt") if dataset_type in f.stem]) if dataset_txts is None else dataset_txts
        clients_classes_array = []
        for dataset_txt in dataset_txts:
            cur_classes = np.zeros(num_classes, dtype=int)
            with open(dataset_txt, "r") as f:
                lines = f.readlines()
            for line in lines:
                cur_classes[int(line.rsplit(".", 1)[0].rsplit("_", 1)[1])] += 1
            clients_classes_array.append(cur_classes)
        clients_classes_array = np.array(clients_classes_array)
        clients_classes_array = clients_classes_array.T

        if save_path is None:
            date_time = "{}-{}".format(dataset_txts[0].parent.parent.name, dataset_txts[0].parent.name)
            save_path = self.default_save_dir / dataset_name / "distribu_heatmap" / "{}.png".format(date_time)
        self.plot_train_heatmap(clients_classes_array, save_path, **kwargs)

    def draw_train_dataset_distribu_hist(self, dataset_txts: List[Path] or None, dataset_txt_dir: Path or None, dataset_type: str, dataset_name: str, num_classes: int, save_path: Path = None):
        if not isinstance(dataset_name, str):
            return
        task = C.TASK.get(dataset_name)
        assert dataset_type in [C.TRAIN, C.TEST, C.VALIDATION]
        if task != C.IMG_CLASSIFICATION:
            self.logger.warn("dataset_name:{} task:{} is invalid for draw_{}_dataset_distribu_hist.".format(dataset_name, task, dataset_type))
            return

        dataset_txts = sorted([f for f in dataset_txt_dir.glob("*.txt") if dataset_type in f.stem]) if dataset_txts is None else dataset_txts
        _len = len(dataset_txts)  # num_clients
        dataset_label_lists = []
        for dataset_txt in dataset_txts:
            with open(dataset_txt, "r") as f:
                lines = f.readlines()
            cur_labels = []
            for line in lines:
                cur_labels.append(int(line.rsplit(".", 1)[0].rsplit("_", 1)[1]))
            dataset_label_lists.append(cur_labels)

        if save_path is None:
            date_time = "{}-{}".format(dataset_txts[0].parent.parent.name, dataset_txts[0].parent.name)
            save_path = self.default_save_dir / dataset_name / "distribu_hist" / "{}.png".format(date_time)
        self.plot_train_hist(dataset_label_lists, num_classes, _len, save_path)
