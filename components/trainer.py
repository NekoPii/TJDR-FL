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

from gutils import Logger, Epoch, gutil
from gutils import constants as C, task_init
from gutils.scheduler import Scheduler
from task.utils.datasets import BoxMaskGenerator, AddMaskParamsToBatch, SegCollate

g = torch.Generator()
g.manual_seed(0)


class Trainer(object):
    def __init__(
            self,
            config,
            _id,
            net: nn.Module,
            logger: Logger,
            last_epoch: -1
    ):
        super(Trainer, self).__init__()

        self.config = config
        self.id = _id
        self.net = net
        self.logger = logger
        self.last_epoch = last_epoch

        self.fed_mode = self.config[C.FED_MODE]
        self.fed_params = self.config.get(C.FED_PARAMS, {})
        self.dataset_name = self.config[C.NAME_DATASET]
        self.task = C.TASK[self.dataset_name]
        self.classes = self.config.get(C.CLASSES, C.DATASET_CLASSES[self.dataset_name])
        self.num_classes = len(self.classes)
        assert self.num_classes == self.config[C.NUM_CLASSES], "len(classes){} != num_classes{}".format(self.num_classes, self.config[C.NUM_CLASSES])

        self.log_iter = self.config.get(C.LOG_ITER, 10)

        self.gpu_count = torch.cuda.device_count()
        self.multi_gpu = self.gpu_count > 1

        self.use_amp = self.config.get(C.AMP, False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        if C.SLIDE_INFERENCE in self.config:
            self.is_slide_inference = True
            self.slide_crop_size = self.config[C.SLIDE_INFERENCE][C.SLIDE_CROP_SIZE]
            self.slide_stride = self.config[C.SLIDE_INFERENCE][C.SLIDE_STRIDE]
        else:
            self.is_slide_inference = False

        self.total_epoch = int(self.config[C.TOTAL_EPOCH])
        self.ignore_label = self.config.get(C.IGNORE_LABEL, C.DEFAULT_IGNORE_LABEL)

        self.train_dataset, self.train_contribution = self.train_dataset_init()
        self.train_dataloader, self.train_per_epoch_iter_num = self.train_dataloader_init(self.train_dataset)

        self.optim_cfg = self.config[C.OPTIMIZER]
        self.lr_sch_cfg = self.config.get(C.LR_SCHEDULE)

        self.optimizer = self.optimizer_init(self.net, self.optim_cfg)
        self.lr_schedule = Scheduler(
            self.optimizer, self.lr_sch_cfg, self.logger,
            total_epoch=self.total_epoch,
            train_per_epoch_iter_num=self.train_per_epoch_iter_num,
            last_epoch=self.last_epoch
        )

        self.sup_loss_f, self.sup_loss_weight = self.loss_init()

    def init_info_log(self, name, **kwargs):
        for k, v in kwargs.items():
            self.logger.info("Init[{}] | {}:{}".format(name, k, v))

    def ep_info_log(self, name: str, ep: Epoch, **kwargs):
        for k, v in kwargs.items():
            self.logger.info("ClientEpoch:{} | [{}] | {}:{}".format(ep.cec_to_str(), name, k, v))

    def iter_info_log(self, name: str, ep: Epoch, iter: int, **kwargs):
        for k, v in kwargs.items():
            self.logger.info("ClientEpoch:{} | [{}] | iter:{} | {}:{}".format(ep.cec_to_str(), name, iter, k, v))

    def cal_eploss(self, iter_losses: list):
        if len(iter_losses) == 2 and isinstance(iter_losses[0], list) and isinstance(iter_losses[1], list):
            iter_losses = iter_losses[0] + iter_losses[1]
        iter_losses = np.array(iter_losses)
        iter_losses = iter_losses[~np.isnan(iter_losses)]
        ep_loss = 0 if len(iter_losses) == 0 else iter_losses.mean()
        return ep_loss

    def optimizer_init(self, net: nn.Module, optim_cfg):
        if optim_cfg is None:
            self.logger.info("Optimizer Init None.")
            return None

        self.logger.info("Optimizer Config:{}".format(optim_cfg))
        self.logger.info("Optimizer Component Init ......")

        optim_type = optim_cfg.get(C.TYPE, C.Adam)

        init_lr = float(optim_cfg.get(C.LR, 1e-3))
        momentum = float(optim_cfg.get(C.MOMENTUM, 0.9))
        weight_decay = float(optim_cfg.get(C.WEIGHT_DECAY, 5e-4))

        # Todo:add more optimizer
        if optim_type == C.Adam:
            optimizer = torch.optim.Adam(
                params=[{"params": net.parameters(), "initial_lr": init_lr}],
                lr=init_lr,
                betas=(0.9, 0.999),
                weight_decay=weight_decay
            )
        elif optim_type == C.SGD:
            optimizer = torch.optim.SGD(
                params=[{"params": net.parameters(), "initial_lr": init_lr}],
                lr=init_lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError("optim type:{} error".format(optim_type))

        self.logger.info("Optimizer Component Init Completed.")

        return optimizer

    def get_optim_params(self):
        return copy.deepcopy(self.optimizer.state_dict())

    def set_optim_params(self, optim_params):
        self.optimizer.load_state_dict(optim_params)

    def train_dataset_init(self):
        self.logger.info("Train-Dataset Init ......")
        # self.logger.info("Add DR_preprocess")
        dataset_path = self.config.get(C.TRAIN)
        drop_split = self.config.get("drop_split")
        self.logger.info("drop_split:{}".format(drop_split))
        assert dataset_path, "train_dataset_init() dataset_path valid"
        train_dataset = task_init.dataset_init(
            self.config,
            dataset_path,
            img_size=self.slide_crop_size if self.is_slide_inference else self.config.get(C.IMG_SIZE),
            is_train=True,
        )
        train_contribution = len(train_dataset)
        self.logger.info("Train-Dataset Init Completed.")
        return train_dataset, train_contribution

    def train_dataloader_init(self, train_dataset):
        self.logger.info("Train-DataLoader Init ......")
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config[C.BATCH_SIZE] * (self.gpu_count if self.multi_gpu else 1),
            shuffle=True,
            num_workers=self.config[C.NUM_WORKERS],
            worker_init_fn=gutil.seed_worker,
            generator=g,
            drop_last=self.task in [C.IMG_SEGMENTATION],
            pin_memory=True
        )  # drop last for BN
        train_per_epoch_iter_num = len(train_dataloader)
        self.logger.info("Train-DataLoader Init Completed.")
        return train_dataloader, train_per_epoch_iter_num

    def cutmix_train_dataloader_init(self):
        cutmix_cfg = self.config.get("cutmix")
        if cutmix_cfg.get(C.ENABLE, False) and self.task in [C.IMG_SEGMENTATION]:
            self.logger.info("CutMix Config:{}".format(cutmix_cfg))
            self.logger.info("CutMix Dataloader Init ...")
            mask_generator = BoxMaskGenerator(
                prop_range=list(cutmix_cfg.get("prop_range", [0.25, 0.5])),
                n_boxes=cutmix_cfg.get("n_boxes", 3),
                random_aspect_ratio=cutmix_cfg.get("random_aspect_ratio", True),
                prop_by_area=cutmix_cfg.get("prop_by_area", True),
                within_bounds=cutmix_cfg.get("within_bounds", True),
                invert=cutmix_cfg.get("invert", True)
            )
            add_mask_params_to_batch = AddMaskParamsToBatch(mask_generator)
            mask_collate_fn = SegCollate(batch_aug_fn=add_mask_params_to_batch)
            cutmix_train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.config[C.BATCH_SIZE] * (self.gpu_count if self.multi_gpu else 1),
                shuffle=True,
                num_workers=self.config[C.NUM_WORKERS],
                worker_init_fn=gutil.seed_worker,
                generator=g,
                drop_last=True,
                pin_memory=True,
                collate_fn=mask_collate_fn,
            )
            self.logger.info("CutMix Dataloader Init Completed.")
        else:
            cutmix_train_dataloader = None
        return cutmix_train_dataloader

    def get_train_contrib(self, _type: str):
        assert _type in [C.TRAIN], "{} valid.".format(_type)
        return torch.tensor(self.train_contribution)

    def loss_init(self):
        self.logger.info("Loss Component Init ......")
        sup_loss_f, sup_loss_weight = task_init.sup_loss_init(self.config, logger=self.logger, device=self.device)
        self.logger.info("Loss Component Init Completed.")
        return sup_loss_f, sup_loss_weight

    def cal_sup_loss(self, pred, gt, lamb=None):
        if lamb is not None:
            loss = lamb * self.cal_sup_loss(pred, gt[0]) + (1 - lamb) * self.cal_sup_loss(pred, gt[1])
            return loss
        if self.sup_loss_weight is not None:
            if isinstance(self.sup_loss_weight, (int, float)):
                weight = [self.sup_loss_weight] * len(self.sup_loss_f)
            else:
                weight = self.sup_loss_weight
            weight = torch.as_tensor(weight, dtype=torch.float)
            loss = self.sup_loss_f[0](pred, gt) * weight[0]
            for i, loss_func in enumerate(self.sup_loss_f[1:], start=1):
                loss += loss_func(pred, gt) * weight[i]
        else:
            loss = self.sup_loss_f[0](pred, gt)
            for i, loss_func in enumerate(self.sup_loss_f[1:], start=1):
                loss += loss_func(pred, gt)
        return loss

    def get_train_dataset_len(self):
        return len(self.train_dataset)

    def get_weights(self, to_cpu=True, keep_vars=True, remove_requires_false=False):
        weights = copy.deepcopy(self.net).module if self.multi_gpu else copy.deepcopy(self.net)
        weights = weights.cpu().state_dict(keep_vars=keep_vars) if to_cpu else weights.state_dict(keep_vars=keep_vars)
        if remove_requires_false:
            weights = gutil.remove_requires_false(weights)
        return weights

    def set_weights(self, weights, strict: bool = True):
        if gutil.is_weights_path(weights):
            # .pt/.pth file
            weights = torch.load(weights, map_location=self.device)
        # state_dict file
        if self.multi_gpu:
            weights = gutil.state_dict2multigpu(weights)
        self.net.load_state_dict(weights, strict)

    def reset_lr_scheduler(self):
        if self.lr_schedule and "lr_decay" in self.optim_cfg:
            lr_decay = self.optim_cfg["lr_decay"]
            for group in self.optimizer.param_groups:
                group["initial_lr"] *= lr_decay
                group["lr"] = group["initial_lr"]
            self.lr_schedule = Scheduler(
                self.optimizer, self.lr_sch_cfg, self.logger,
                total_epoch=self.total_epoch,
                train_per_epoch_iter_num=self.train_per_epoch_iter_num,
                last_epoch=self.last_epoch
            )

    def train(self, num_client_ep: int, ep: Epoch, sio=None, **kwargs):
        self.net = self.net.to(self.device)
        self.net.train()

        ep_losses = []
        for cur_client_ep in range(1, num_client_ep + 1):
            ep.client_epoch_plus()
            iter_losses = []
            self.ep_info_log(
                "Trainer", ep,
                lr=self.optimizer.param_groups[0]["lr"],
            )
            for iter, minibatch in enumerate(self.train_dataloader, start=1):
                if self.config.get("mixup"):
                    alpha = self.config.get("mixup").get("alpha", 1.0)
                    X, Y, lamb = task_init.minibatch_init(self.task, minibatch, self.device, mixup=True, mixup_alpha=alpha)
                elif self.config.get("cutmix"):
                    alpha = self.config.get("cutmix").get("alpha", 1.0)
                    X, Y, lamb = task_init.minibatch_init(self.task, minibatch, self.device, cutmix=True, cutmix_alpha=alpha)
                else:
                    if self.config.get("cutout"):
                        X, Y = task_init.minibatch_init(self.task, minibatch, self.device, cutout=True)
                    else:
                        X, Y = task_init.minibatch_init(self.task, minibatch, self.device)
                    X, Y = X.float(), Y.long()
                    lamb = None

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    logits = self.net(X)
                    loss = self.cal_sup_loss(logits, Y, lamb)

                iter_losses.append(loss.item())

                if self.log_iter > 0:
                    if iter == 1 or iter == len(self.train_dataloader) or iter % self.log_iter == 0:
                        self.iter_info_log("Trainer", ep, iter, loss=loss.item())

                self.scaler.scale(loss).backward()
                # grad_accumulate，increase batch_size without increasing gpu
                if C.GRAD_ACCUMULATE in self.config and self.config[C.GRAD_ACCUMULATE] > 0 and iter % self.config[
                    C.GRAD_ACCUMULATE] == 0 or C.GRAD_ACCUMULATE not in self.config or self.config[C.GRAD_ACCUMULATE] <= 0:
                    if C.GRAD_ACCUMULATE in self.config and self.config[C.GRAD_ACCUMULATE] > 0:
                        self.logger.info("Accumulate Grad : batch_size*{}".format(self.config[C.GRAD_ACCUMULATE]))
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                    self.lr_schedule.step(ep.total_client_ep(ignore_cloud=True), True)

                    if sio is not None:
                        sio.emit(
                            "client_train_process", {
                                C.FID: self.id.fid,
                                "ep": ep.serialize(),
                                "process": "{:.1f}%".format(iter / self.train_per_epoch_iter_num * 100)}
                        )

            ep_loss = self.cal_eploss(iter_losses)
            ep_losses.append(ep_loss)
            self.ep_info_log(
                "Trainer", ep,
                loss=ep_loss
            )
            self.lr_schedule.step(ep.total_client_ep(ignore_cloud=True), False)

        res_dict = dict(
            losses=ep_losses
        )
        return res_dict
