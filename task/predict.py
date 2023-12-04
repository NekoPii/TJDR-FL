#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: maojingxin
"""
import argparse
import copy
import os
import sys
from collections.abc import Iterable
from pathlib import Path

import imgviz
import numpy as np
import torch
from PIL import Image as PImage
from matplotlib import pyplot as plt
from tqdm import tqdm

cur_dir = Path(__file__).resolve().parent  # ..task/
root_dir = cur_dir.parent  # ..TJDR-FL/
if str(root_dir) != sys.path[0]:
    sys.path[0] = str(root_dir)

from components import Models, Evaler
from task.utils.preprocess import select_preprocess, select_normalize
from task.utils import transforms as T
from gutils import Logger, gutil, constants as C, metrics

torch.set_num_threads(16)


class Predictor(object):
    def __init__(self, config: dict, weights_path: str, draw: bool = False, gpu: str = None, multi_dataset: str = None):
        now_day = gutil.get_now_day()

        self.img_size = config[C.IMG_SIZE]
        assert isinstance(self.img_size, int) or (isinstance(self.img_size, Iterable) and len(self.img_size) == 2)
        self.dataset_name = config[C.NAME_DATASET] if multi_dataset is None else multi_dataset
        self.task = C.TASK.get(self.dataset_name)
        model_name = config[C.NAME_MODEL]
        self.num_classes = config[C.NUM_CLASSES]
        self.ignore_label = config.get(C.IGNORE_LABEL, C.DEFAULT_IGNORE_LABEL)
        self.classes = config.get(C.CLASSES, C.DATASET_CLASSES[self.dataset_name])
        self.weights_path = weights_path
        self.draw = draw
        self.logger = Logger()

        os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu"]) if gpu is None else gpu

        if C.SLIDE_INFERENCE in config:
            self.is_slide_inference = True
            self.slide_crop_size = config[C.SLIDE_INFERENCE][C.SLIDE_CROP_SIZE]
            self.slide_stride = config[C.SLIDE_INFERENCE][C.SLIDE_STRIDE]
        else:
            self.is_slide_inference = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.np_dtype = np.int8 if self.num_classes < 255 else np.int16

        self.predict_dir = Path(config[C.DIR_PREDICT]) / now_day
        if multi_dataset:
            self.predict_dir = self.predict_dir / multi_dataset
        self.batch_save_dir = None
        self.save_dir = None

        self.now_weight_name = "_".join(self.weights_path.rsplit("/", 4)[1:])

        # self.model = getattr(Models, model_name)(config, only_init_net=True)

        if self.task in [C.IMG_CLASSIFICATION]:
            self.model = Models.cls_model_gen(
                model_name,
                ptr_weights=config.get(C.PTR_WEIGHTS),
                dropout=config.get("dropout", 0.2)
            )(config, logger=self.logger, only_init_net=True)
        elif self.task in [C.IMG_SEGMENTATION]:
            self.model = Models.seg_model_gen(model_name, ptr_weights=config.get(C.PTR_WEIGHTS))(config, logger=self.logger, only_init_net=True)
        else:
            raise ValueError(self.task)

        self.model.set_weights(weights_path)
        self.net = self.model.net
        self.net = self.net.to(self.device)
        self.net.eval()

    @staticmethod
    def draw_dr_seg_predict(
            classes: list, img: PImage.Image, gt_mask: PImage.Image, pred_one_hot: np.ndarray,
            save_dir: Path, draw_gt_one_hot=False, ignore_index=-1, ignore_label=255,
            verbose=True, color_map=None, figsize=(30, 30), remove_class=True
    ):
        color_map = imgviz.label_colormap() if color_map is None else color_map
        predict_mask_h, predict_mask_w = pred_one_hot.shape[-2:]
        w, h = img.size
        if predict_mask_h > h or predict_mask_w > w:
            pred_one_hot, _ = T.CenterCrop(size=(h, w))(torch.from_numpy(pred_one_hot))
            pred_one_hot = pred_one_hot.numpy()
        num_classes = len(classes)
        assert pred_one_hot.shape[0] == num_classes, "{}!={}".format(pred_one_hot.shape[0], num_classes)
        save_dir.mkdir(exist_ok=True, parents=True)

        gt_mask.putpalette(color_map)
        gt_mask.save(save_dir / "gt.png")
        pil_predict_mask = PImage.fromarray(np.argmax(pred_one_hot, axis=0).astype(np.uint8))
        pil_predict_mask.putpalette(color_map)
        pil_predict_mask.save(save_dir / "pred.png")

        new_classes = list(range(num_classes))
        for i in range(num_classes):
            if remove_class and ignore_index != i:
                if np.sum(pred_one_hot[i, :, :]) == 0:
                    # if gutil.check_nparray_same(pred_one_hot[i, :, :], np.zeros_like(pred_one_hot[i, :, :])):
                    new_classes.remove(i)

        new_num_classes = len(new_classes)
        gray_color_map = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
        for i in range(new_num_classes):
            if ignore_index != new_classes[i]:
                one_hot_pred = PImage.fromarray(pred_one_hot[new_classes[i], :, :].astype(np.uint8))
                one_hot_pred.putpalette(gray_color_map)
                one_hot_pred_dir = save_dir / "pred"
                one_hot_pred_dir.mkdir(exist_ok=True, parents=True)
                one_hot_pred.save(one_hot_pred_dir / "{}.png".format(classes[new_classes[i]]))
        if draw_gt_one_hot:
            one_hot_gts = gutil.one_hot(np.asarray(gt_mask, dtype=np.int_), num_classes, ignore_label=ignore_label)
            for i in range(new_num_classes):
                if ignore_index != new_classes[i]:
                    one_hot_gt = PImage.fromarray(one_hot_gts[new_classes[i], :, :].astype(np.uint8))
                    one_hot_gt.putpalette(gray_color_map)
                    one_hot_gt_dir = save_dir / "gt"
                    one_hot_gt_dir.mkdir(exist_ok=True, parents=True)
                    one_hot_gt.save(one_hot_gt_dir / "{}.png".format(classes[new_classes[i]]))
        if verbose:
            print("[Info]:predict result saved : {}".format(save_dir))

    @staticmethod
    def draw_seg_predict(
            classes: list, img: PImage.Image, gt_mask: PImage.Image, pred_one_hot: np.ndarray,
            save_path: Path, draw_gt_one_hot=False, ignore_index=-1, ignore_label=255,
            verbose=True, color_map=None, figsize=(30, 30), remove_class=True
    ):
        color_map = imgviz.label_colormap() if color_map is None else color_map
        predict_mask_h, predict_mask_w = pred_one_hot.shape[-2:]
        w, h = img.size
        if predict_mask_h > h or predict_mask_w > w:
            pred_one_hot, _ = T.CenterCrop(size=(h, w))(torch.from_numpy(pred_one_hot))
            pred_one_hot = pred_one_hot.numpy()
        num_classes = len(classes)
        assert pred_one_hot.shape[0] == num_classes, "{}!={}".format(pred_one_hot.shape[0], num_classes)
        plt.figure(figsize=figsize)
        if num_classes > 1:
            flag = ignore_index in range(num_classes)
            plt.subplots_adjust(top=0.96, bottom=0.02, left=0.02, right=0.98, hspace=0.05, wspace=0.02)
            ax = plt.subplot(2 + draw_gt_one_hot, 3, 1)
            ax.set_title("ori img", fontsize=35)
            ax.imshow(img)
            ax.axis("off")
            ax = plt.subplot(2 + draw_gt_one_hot, 3, 2)
            ax.set_title("gt color_map", fontsize=35)
            gt_mask.putpalette(color_map)
            ax.imshow(gt_mask)
            ax.axis("off")
            ax = plt.subplot(2 + draw_gt_one_hot, 3, 3)
            ax.set_title("predict color_map", fontsize=35)
            pil_predict_mask = PImage.fromarray(np.argmax(pred_one_hot, axis=0).astype(np.uint8))
            pil_predict_mask.putpalette(color_map)
            ax.imshow(pil_predict_mask)
            ax.axis("off")

            new_classes = list(range(num_classes))
            for i in range(num_classes):
                if remove_class and ignore_index != i:
                    if np.sum(pred_one_hot[i, :, :]) == 0:
                        # if gutil.check_nparray_same(pred_one_hot[i, :, :], np.zeros_like(pred_one_hot[i, :, :])):
                        new_classes.remove(i)

            new_num_classes = len(new_classes)
            for i in range(new_num_classes):
                if ignore_index != new_classes[i]:
                    ax = plt.subplot(
                        2 + draw_gt_one_hot, new_num_classes - flag,
                        new_num_classes + i + 1 - flag - (flag and new_classes[i] > ignore_index)
                    )
                    ax.set_title("segmap(class {})".format(new_classes[i]) if classes is None else classes[new_classes[i]], fontsize=30)
                    ax.imshow(pred_one_hot[new_classes[i], :, :], cmap="gray")
                    ax.axis("off")
            if draw_gt_one_hot:
                for i in range(new_num_classes):
                    if ignore_index != new_classes[i]:
                        one_hot_gt = gutil.one_hot(np.asarray(gt_mask, dtype=np.int_), num_classes, ignore_label=ignore_label)
                        ax = plt.subplot(
                            2 + draw_gt_one_hot, new_num_classes - flag,
                            2 * (new_num_classes - flag) + i + 1 - (flag and new_classes[i] > ignore_index)
                        )
                        ax.set_title("gt(class {})".format(new_classes[i]) if classes is None else "gt({})".format(classes[new_classes[i]]), fontsize=30)
                        ax.imshow(one_hot_gt[new_classes[i], :, :], cmap="gray")
                        ax.axis("off")
        elif num_classes == 1:
            plt.subplots_adjust(top=0.96, bottom=0.01, left=0.02, right=0.98, hspace=0.01, wspace=0.01)
            ax = plt.subplot(1, 3, 1)
            ax.set_title("ori img", fontsize=30)
            ax.imshow(img)
            ax.axis("off")
            ax = plt.subplot(1, 3, 2)
            ax.set_title("ground truth", fontsize=30)
            ax.imshow(gt_mask, cmap="gray")
            ax.axis("off")
            ax = plt.subplot(1, 3, 3)
            ax.set_title("segmap" if classes is None else classes[0], fontsize=30)
            ax.imshow(pred_one_hot[0, :, :], cmap="gray")
            ax.axis("off")
        else:
            raise AssertionError("num_classes({}) >= 1".format(num_classes))
        plt.xticks([]), plt.yticks([])
        plt.savefig(save_path)
        if verbose:
            print("[Info]:predict result saved : {}".format(save_path))
        plt.close()

    @torch.no_grad()
    def batch_inference(self, img_dir: Path, img_suffix, gt_dir: Path = None, gt_suffix=None, num: int = None, **kwargs):
        self.logger.info("kwargs:{}".format(kwargs))
        now_time = gutil.get_now_time()
        addition_name = kwargs.get("name", "")
        if len(addition_name) > 0:
            addition_name = "[{}]".format(addition_name)
        if C.TRAIN == img_dir.stem:
            self.batch_save_dir = self.predict_dir / (addition_name + self.now_weight_name) / C.TRAIN / "batch" / now_time
        elif C.VALIDATION == img_dir.stem:
            self.batch_save_dir = self.predict_dir / (addition_name + self.now_weight_name) / C.VALIDATION / "batch" / now_time
        elif C.TEST == img_dir.stem:
            self.batch_save_dir = self.predict_dir / (addition_name + self.now_weight_name) / C.TEST / "batch" / now_time
        else:
            self.batch_save_dir = self.predict_dir / (addition_name + self.now_weight_name) / "batch" / now_time
        self.batch_save_dir.mkdir(parents=True, exist_ok=True)

        img_paths = kwargs.get("img_paths", sorted(list(img_dir.glob("*{}".format(img_suffix)))))
        _len = len(img_paths)
        num = _len if (num is None or num > _len) else num

        gt_paths = kwargs.get("gt_paths")
        if gt_paths is None:
            if gt_dir is not None and gt_suffix is not None:
                gt_paths = sorted(list(gt_dir.glob("*{}".format(gt_suffix))))
                assert len(img_paths) == len(gt_paths), "{}!={}".format(len(img_paths), len(gt_paths))

        all_preds, all_gts, all_probs = Evaler.init_placeholder(
            num, self.num_classes,
            img_size=self.img_size,
            np_dtype=self.np_dtype,
            task=self.task,
            seg_cal_auc=True
        )

        choice_list = np.random.choice(range(_len), num, replace=False)
        for idx, choice_idx in enumerate(tqdm(choice_list, desc="predicting ...", unit="img")):
            img_path = img_paths[choice_idx]
            gt = img_path.stem.rsplit("_")[1] if gt_paths is None else gt_paths[choice_idx]
            pred, gt, prob = self.inference(img_path, gt, is_batch=True, **kwargs)
            all_preds[idx:idx + 1] = pred
            all_gts[idx:idx + 1] = gt
            # if self.task in [C.IMG_CLASSIFICATION]:
            all_probs[idx:idx + 1] = prob

        metric = metrics.new_cal_metric(
            all_preds, all_gts, all_probs,
            num_classes=self.num_classes,
            ignore_label=self.ignore_label,
            multi=100,
            dot=2,
            mean_type="macro",
            seg_cal_auc=True
        )

        eval_acc = metrics.acc_i2cls(metric, self.classes)
        summary_path = self.batch_save_dir / "summay.txt"
        f_log = Logger(f_path=summary_path)
        self.logger.info("Predict Summary. Result saved in {}".format(summary_path))
        f_log.info("Predict Summary.")
        gutil.log_acc(f_log, acc=eval_acc, classes=self.classes)
        gutil.log_acc(self.logger, acc=eval_acc, classes=self.classes)

    @torch.no_grad()
    def inference(self, predict_img_path: Path, ground_truth_path, is_batch=False, **kwargs):
        now_time = gutil.get_now_time()
        if is_batch:
            save_path = self.batch_save_dir / "{}.jpg".format(predict_img_path.stem)
        else:
            if C.TRAIN == predict_img_path.parent.stem:
                self.save_dir = self.predict_dir / self.now_weight_name / C.TRAIN
            elif C.VALIDATION == predict_img_path.parent.stem:
                self.save_dir = self.predict_dir / self.now_weight_name / C.VALIDATION
            elif C.TEST == predict_img_path.parent.stem:
                self.save_dir = self.predict_dir / self.now_weight_name / C.TEST
            else:
                self.save_dir = self.predict_dir / self.now_weight_name
            self.save_dir.mkdir(parents=True, exist_ok=True)
            save_path = self.save_dir / "t{}_{}.jpg".format(now_time, predict_img_path.stem)

        img = copy.deepcopy(PImage.open(predict_img_path))
        gt = copy.deepcopy(PImage.open(ground_truth_path)) if Path(ground_truth_path).exists() else int(ground_truth_path)
        onebatch = select_preprocess(self.dataset_name, img, gt, self.img_size, is_train=False, label_type=C.LABELED)

        if self.task == C.IMG_CLASSIFICATION:
            return self.class_inference(onebatch, save_path, is_batch=is_batch)
        elif self.task == C.IMG_SEGMENTATION:
            return self.seg_inference(onebatch, save_path, is_batch=is_batch, **kwargs)
        else:
            raise ValueError(self.task)

    @torch.no_grad()
    def class_inference(self, onebatch, save_path: Path, is_batch: bool, out_threshold=0.5):
        batch_img = onebatch["t_img"].unsqueeze(0)  # (1,Channel,H,W)
        gt = onebatch["t_gt"].long().cpu().numpy().astype(self.np_dtype)  # ()

        batch_img = batch_img.to(self.device)
        output = self.net(batch_img).detach().cpu()

        if self.num_classes == 1:
            prob = torch.sigmoid(output)  # (1,1)
            pred = (prob > out_threshold).long().squeeze()  # (1,)
        else:
            prob = torch.softmax(output, dim=1)  # (1,Classes,)
            pred = torch.argmax(prob, dim=1).long()  # (1,)

        prob, pred = prob.numpy(), pred.numpy().astype(self.np_dtype)

        if self.draw:
            pil_onebatch_img = PImage.fromarray(
                gutil.tensor2npimg(
                    onebatch["t_img"],
                    mean=select_normalize(self.dataset_name)[0],
                    std=select_normalize(self.dataset_name)[1],
                    img_type=np.uint8
                )
            )
            gutil.draw_class_predict(
                classes=self.classes,
                img=pil_onebatch_img,
                prob=prob[0],
                gt=gt.item(),
                pred=pred.item(),
                save_path=save_path,
                verbose=not is_batch,
            )

        return pred, gt, prob

    @torch.no_grad()
    def seg_inference(self, onebatch, save_path: Path, is_batch: bool, out_threshold=0.5, **kwargs):
        batch_img = onebatch["t_img"].unsqueeze(0)  # (1,Channel,H,W)
        gt = onebatch["t_gt"].long().cpu().numpy().astype(self.np_dtype)  # (H,W)

        batch_img = batch_img.to(self.device)
        if self.is_slide_inference:
            slide_size = kwargs.get("slide_size", self.slide_crop_size)
            slide_stride = kwargs.get("slide_stride", self.slide_stride)
            output = gutil.slide_inference(batch_img, self.net, self.num_classes, slide_size, slide_stride)
        else:
            output = self.net(batch_img)
        output = output.detach().cpu()

        if self.num_classes == 1:
            prob = torch.sigmoid(output)  # (1,1,H,W)
            pred = (prob > out_threshold).squeeze(dim=1).long().numpy()  # (1,H,W)
            pred_one_hot = gutil.one_hot(pred[0], 2, ignore_label=self.ignore_label)  # (2,H,W)
        else:
            prob = torch.softmax(output, dim=1)  # (1,C,H,W)
            pred = torch.argmax(prob, dim=1).long().numpy()  # (1,H,W)
            pred_one_hot = gutil.one_hot(pred[0], self.num_classes, ignore_label=self.ignore_label)  # (C,H,W)

        prob = prob.numpy()
        pred = pred.astype(self.np_dtype)
        pred_one_hot = (pred_one_hot * 255).astype(np.uint8)

        if self.draw:
            pil_onebatch_img = PImage.fromarray(
                gutil.tensor2npimg(
                    onebatch["t_img"],
                    mean=select_normalize(self.dataset_name)[0],
                    std=select_normalize(self.dataset_name)[1],
                    img_type=np.uint8
                )
            )
            pil_onebatch_gt = PImage.fromarray(gt.astype(np.uint8))

            # Todo: add segmentation dataset , modify belows
            if self.dataset_name in [C.DDR_SEG, C.TJDR, C.IDRiD]:
                # Multi-Classes
                dr_save_dir = save_path.with_name(save_path.stem)
                self.draw_dr_seg_predict(
                    self.classes, pil_onebatch_img, pil_onebatch_gt, pred_one_hot, dr_save_dir,
                    draw_gt_one_hot=True,
                    ignore_index=0,  # ignore bg
                    verbose=not is_batch,
                    remove_class=False
                )
            else:
                raise AssertionError("no such dataset:{}".format(self.dataset_name))
        return pred, gt, prob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--cloud_config_path", dest="cloud_config_path", type=str, help="path of cloud config", required=True
    )
    parser.add_argument(
        "-t", "--predict_type", dest="predict_type", type=str, required=True, help="predict type", choices=[C.TRAIN, C.VALIDATION, C.TEST]
    )
    parser.add_argument(
        "-n", "--num", dest="num", help="the number of data to predict, default is \"all\" if not specified",
    )
    parser.add_argument(
        "-w", "--weights_path", dest="weights_path", type=str, help="path of weights to perform evaluation, default is best_weights_path of server config"
    )
    parser.add_argument(
        "-d", "--draw", dest="draw", action="store_true", help="draw predict image for each predicted instance"
    )
    parser.add_argument(
        "-g", "--gpu", dest="gpu", type=str, help="specified gpu to run, default is determined by the gpu of server config"
    )
    parser.add_argument(
        "-slide_size", dest="slide_size", help="the size of sliding window inference"
    )
    parser.add_argument(
        "-slide_stride", dest="slide_stride", help="the stride of sliding window inference"
    )
    parser.add_argument(
        "--multi_dataset", dest="multi_dataset", type=str, help="one of the name for the multi-dataset predict, default is None for single-dataset predict"
    )
    parser.add_argument(
        "--img_paths", dest="img_paths", type=str, nargs="*", help="specified img_paths to predict"
    )
    parser.add_argument(
        "--gt_paths", dest="gt_paths", type=str, nargs="*", help="specified gt_paths to predict"
    )
    parser.add_argument(
        "--name", dest="name", type=str, help="additional info"
    )
    args = parser.parse_args()
    cloud_config_path = args.cloud_config_path.strip()
    assert Path(cloud_config_path).exists(), "{} not exists.".format(cloud_config_path)
    cloud_cfg = gutil.load_json(cloud_config_path)
    predict_info = cloud_cfg["predict_info"]
    predict_type = args.predict_type
    best_weights_path = cloud_cfg[C.PATH_BEST_WEIGHTS][predict_type]
    dataset_name = cloud_cfg[C.NAME_DATASET]
    weights_path = best_weights_path if args.weights_path is None else args.weights_path.strip()
    weights_path = Path(weights_path)
    num = int(args.num) if args.num is not None else args.num

    if not weights_path.exists():
        weights_path = weights_path.with_name("{}{}".format("edge#1-{}-best".format(predict_type), weights_path.suffix))
    weights_path = str(weights_path)

    slide_size = args.slide_size
    slide_stride = args.slide_stride
    multi_dataset = args.multi_dataset
    img_paths = args.img_paths
    gt_paths = args.gt_paths
    name = args.name
    kwargs = dict()
    if slide_size is not None:
        kwargs.update({"slide_size": slide_size})
    if slide_size is not None:
        kwargs.update({"slide_stride": slide_stride})
    if img_paths is not None:
        kwargs.update({"img_paths": [Path(img_path) for img_path in img_paths]})
    if gt_paths is not None:
        kwargs.update({"gt_paths": [Path(gt_path) for gt_path in gt_paths]})
    if name is not None:
        kwargs.update({"name": name})

    if predict_type in predict_info:
        predict_img_dir = Path(predict_info[predict_type]["image_dir"]) if multi_dataset is None else Path(predict_info[predict_type]["image_dir"][multi_dataset])
        predict_img_suffix = predict_info[predict_type]["image_suffix"] if multi_dataset is None else predict_info[predict_type]["image_suffix"][multi_dataset]
        predict_gt_dir = predict_info[predict_type].get("gt_dir")
        if multi_dataset:
            predict_gt_dir = predict_gt_dir[multi_dataset]
        predict_gt_dir = Path(predict_gt_dir) if predict_gt_dir is not None else predict_gt_dir
        predict_gt_suffix = predict_info[predict_type].get("gt_suffix")
        if multi_dataset:
            predict_gt_suffix = predict_gt_suffix[multi_dataset]
        predictor = Predictor(cloud_cfg, weights_path, draw=args.draw, gpu=args.gpu, multi_dataset=multi_dataset)
        predictor.batch_inference(
            img_dir=predict_img_dir,
            img_suffix=predict_img_suffix,
            gt_dir=predict_gt_dir,
            gt_suffix=predict_gt_suffix,
            num=num,
            **kwargs
        )
        predictor.logger.info("predict over.")
    else:
        raise ValueError(predict_type)
