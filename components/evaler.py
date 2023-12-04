#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: maojingxin
"""
import copy
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

cur_dir = Path(__file__).resolve().parent  # ..components/
root_dir = cur_dir.parent  # ..TJDR-FL/
if str(root_dir) != sys.path[0]:
    sys.path[0] = str(root_dir)

from gutils import constants as C, Logger, metrics, task_init, gutil


class Evaler(object):
    def __init__(
            self,
            config,
            logger: Logger,
            loss_f,
            loss_weight
    ):
        super(Evaler, self).__init__()
        self.config = config
        self.logger = logger

        self.num_classes = self.config[C.NUM_CLASSES]
        self.dataset_name = self.config[C.NAME_DATASET]
        self.classes = self.config.get(C.CLASSES, C.DATASET_CLASSES[self.dataset_name])
        self.ignore_label = self.config.get(C.IGNORE_LABEL, C.DEFAULT_IGNORE_LABEL)
        self.task = C.TASK.get(self.dataset_name)
        self.mean_type = self.config.get("mean_type", "macro")

        if C.SLIDE_INFERENCE in self.config:
            self.is_slide_inference = True
            self.slide_crop_size = self.config[C.SLIDE_INFERENCE][C.SLIDE_CROP_SIZE]
            self.slide_stride = self.config[C.SLIDE_INFERENCE][C.SLIDE_STRIDE]
        else:
            self.is_slide_inference = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.multi_gpu = torch.cuda.device_count() > 1
        self.np_dtype = np.int8 if self.num_classes < 255 else np.int16

        self.eval_dataset = dict()
        self.eval_dataloader = dict()
        self.eval_contribution = dict()
        self.eval_dataloader_init()

        self.sup_loss_f, self.sup_loss_weight = loss_f, loss_weight

    def get_eval_dataset_len(self, _type):
        return len(self.eval_dataset[_type])

    def eval_dataloader_init(self):
        self.logger.info("Eval-DataLoader Init ......")
        for _type in [C.TRAIN, C.VALIDATION, C.TEST]:
            if self.config.get(_type):
                self.eval_dataset[_type] = task_init.dataset_init(
                    self.config, self.config[_type],
                    img_size=self.config.get(C.IMG_SIZE),
                    is_train=False
                )
                self.eval_dataloader[_type] = DataLoader(
                    self.eval_dataset[_type],
                    batch_size=self.config[C.EVAL_BATCH_SIZE],
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True
                )
                self.eval_contribution[_type] = torch.tensor(len(self.eval_dataset[_type]))
            else:
                self.eval_dataset[_type] = None
                self.eval_dataloader[_type] = None
                self.eval_contribution[_type] = None
        self.logger.info("Eval-DataLoader Init Completed.")

    def get_eval_contrib(self, _type):
        assert _type in [C.TRAIN, C.VALIDATION, C.TEST], "Evaler type:{} valid.".format(_type)
        return self.eval_contribution[_type]

    def cal_sup_loss(self, pred, gt):
        if self.sup_loss_weight is not None:
            if isinstance(self.sup_loss_weight, (int, float)):
                weight = [self.sup_loss_weight] * len(self.sup_loss_f)
            else:
                weight = self.sup_loss_weight
            weight = torch.as_tensor(weight)
            loss = self.sup_loss_f[0](pred, gt) * weight[0]
            for i, loss_func in enumerate(self.sup_loss_f[1:], start=1):
                loss += loss_func(pred, gt) * weight[i]
        else:
            loss = self.sup_loss_f[0](pred, gt)
            for i, loss_func in enumerate(self.sup_loss_f[1:], start=1):
                loss += loss_func(pred, gt)
        return loss

    @staticmethod
    def init_placeholder(_n, _c, **kwargs):
        prob_shape = [_n, _c]
        pred_shape = [_n]
        _img_size = kwargs.get(C.IMG_SIZE)
        task = kwargs.get("task")
        if _img_size is not None and task == C.IMG_SEGMENTATION:
            if isinstance(_img_size, int):
                _img_size = (_img_size, _img_size)
            pred_shape += list(_img_size)
            prob_shape += list(_img_size)

        np_dtype = kwargs.get("np_dtype", np.int16)
        all_preds = np.zeros(pred_shape, dtype=np_dtype)
        all_gts = np.zeros(pred_shape, dtype=np_dtype)

        all_probs = None
        if task in [C.IMG_CLASSIFICATION] or kwargs.get("seg_cal_auc", False):
            all_probs = np.zeros(prob_shape, dtype=np.float_)

        return all_preds, all_gts, all_probs

    @torch.no_grad()
    def evaluate(self, eval_net: nn.Module, eval_type: str):
        assert eval_type in [C.TRAIN, C.VALIDATION, C.TEST], "Evaler eval_type:{} valid.".format(eval_type)
        eval_net = copy.deepcopy(eval_net)
        if self.multi_gpu:
            eval_net = eval_net.module
        eval_net = eval_net.to(self.device)
        eval_net.eval()

        eval_dataloader = self.eval_dataloader[eval_type]
        assert eval_dataloader is not None, "{} dataset_path in the config is necessity".format(eval_type)
        _n = self.get_eval_dataset_len(eval_type)

        all_preds, all_gts, all_probs = self.init_placeholder(
            _n, self.num_classes,
            img_size=self.config.get(C.IMG_SIZE),
            np_dtype=self.np_dtype,
            task=self.task
        )

        eval_loss = []
        cur_idx = 0
        for iter, minibatch in enumerate(eval_dataloader, start=1):
            X, Y = task_init.minibatch_init(self.task, minibatch, device=self.device)
            X, Y = X.float(), Y.long()
            _len = len(X)

            if self.is_slide_inference:
                logits = gutil.slide_inference(X, eval_net, self.num_classes, self.slide_crop_size, self.slide_stride)
            else:
                logits = eval_net(X)

            assert logits.shape[1] == self.num_classes, "{}!={}".format(logits.shape[1], self.num_classes)
            loss = self.cal_sup_loss(logits, Y)
            eval_loss.append(loss.item())

            logits = logits.detach().cpu()
            Y = Y.detach().cpu().numpy().astype(self.np_dtype)
            if self.num_classes == 1:
                probs = torch.sigmoid(logits)  # (N,C,)
                preds = (probs > 0.5).squeeze(dim=1)  # (N,)
            else:
                probs = torch.softmax(logits, dim=1)  # (N,C,)
                preds = torch.argmax(probs, dim=1)  # (N,)

            probs, preds = probs.numpy(), preds.numpy().astype(self.np_dtype)

            all_preds[cur_idx:cur_idx + _len] = preds
            all_gts[cur_idx:cur_idx + _len] = Y

            # Todo: Not store all_probs for segmentation
            if self.task in [C.IMG_CLASSIFICATION]:
                all_probs[cur_idx:cur_idx + _len] = probs

            cur_idx += _len

        eval_loss = np.array(eval_loss)
        eval_loss = eval_loss[~np.isnan(eval_loss)]
        eval_loss = 0 if len(eval_loss) == 0 else eval_loss.mean()
        metric = metrics.new_cal_metric(
            all_preds, all_gts, all_probs,
            num_classes=self.num_classes if self.num_classes > 1 else 2,
            ignore_label=self.ignore_label,
            multi=100,
            dot=3,
            mean_type=self.mean_type
        )

        eval_acc = metrics.acc_i2cls(metric, self.classes)

        del eval_net
        return eval_loss, eval_acc
