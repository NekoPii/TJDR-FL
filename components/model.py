#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: maojingxin
"""
import copy
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

cur_dir = Path(__file__).resolve().parent  # ..components/
root_dir = cur_dir.parent  # ..TJDR-FL/
if str(root_dir) != sys.path[0]:
    sys.path[0] = str(root_dir)

from gutils import Logger, Epoch, ID
from gutils import gutil, constants as C
from components.models import ClassificationModel, TimmClassificationModel, MLPModel, SegmentationModel, UNet, ConvMLP
from components.trainer import Trainer
from components.evaler import Evaler

torch.set_num_threads(C.NUM_THREADS)


class BaseModel(object):
    def __init__(self, config: dict, client_id: ID = None, logger=None, only_init_net=False, last_epoch=-1):
        self.config = config
        self.logger = Logger(logger)
        self.only_init_net = only_init_net
        self.last_epoch = last_epoch

        self.model_mode = self.config.get(C.MODEL_MODE, C.CNN)  # Compatible with the old configs
        self.model_name = self.config[C.NAME_MODEL]
        self.dataset_name = self.config[C.NAME_DATASET]
        self.task = C.TASK[self.dataset_name[0]] if isinstance(self.dataset_name, list) else C.TASK[self.dataset_name]
        self.num_classes = self.config[C.NUM_CLASSES]
        self.num_channels = self.config[C.NUM_CHANNELS]
        self.fed_mode = self.config[C.FED_MODE]
        self.fed_params = self.config.get(C.FED_PARAMS)

        self.gpu_count = torch.cuda.device_count()
        self.multi_gpu = self.gpu_count > 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger.info("Model:{} init ......".format(self.model_name))
        if self.only_init_net:
            # for cloud
            self.net = self.net_init(multi_gpu=False, model_mode=self.model_mode)
            self.logger.info("Model:{} Construct and Init Completed.".format(self.model_name))
        else:
            self.client_id = client_id
            self.classes = self.config.get(C.CLASSES, C.DATASET_CLASSES[self.dataset_name])
            self.net = self.net_init(multi_gpu=self.multi_gpu, model_mode=self.model_mode)
            self.logger.info("Model:{} Construct and Init Completed.".format(self.model_name))

            self.logger.info("Trainer Init ......")
            self.trainer = Trainer(self.config, self.client_id, self.net, self.logger, last_epoch)
            self.logger.info("Trainer Init Completed.")

            self.logger.info("Evaler Init ......")
            self.evaler = Evaler(self.config, self.logger, self.trainer.sup_loss_f, self.trainer.sup_loss_weight)
            self.logger.info("Evaler Init Completed.")

    def __del__(self):
        del self.net
        if not self.only_init_net:
            del self.trainer

    def get_weights(self, to_cpu=True, keep_vars=True):
        if self.only_init_net:
            # Todo: add model_mode , modify belows
            if self.model_mode in [C.SEG, C.CNN]:
                weights = copy.deepcopy(self.net)
                weights = weights.cpu().state_dict(keep_vars=keep_vars) if to_cpu else weights.state_dict(keep_vars=keep_vars)
                return weights
        else:
            return self.trainer.get_weights(to_cpu, keep_vars=keep_vars)

    def set_weights(self, weights, strict: bool = True):
        if self.only_init_net:
            if gutil.is_weights_path(weights):
                # .pt/.pth file
                weights = torch.load(weights, map_location=self.device)
            else:
                # state_dict file
                weights = copy.deepcopy(weights)
            if self.multi_gpu:
                weights = gutil.state_dict2multigpu(weights)
            self.net.load_state_dict(weights, strict=strict)
        else:
            self.trainer.set_weights(weights, strict=strict)

    def get_optim_state_dict(self):
        return self.trainer.get_optim_params()

    def set_optim_state_dict(self, params):
        self.trainer.set_optim_params(params)

    def get_contrib(self, _type, is_eval=False):
        if not is_eval:
            return self.trainer.get_train_contrib(_type)
        else:
            return self.evaler.get_eval_contrib(_type)

    def get_dataset_len(self, _type):
        assert _type in [C.TRAIN, C.VALIDATION, C.TEST]
        if _type == C.TRAIN:
            return self.trainer.get_train_dataset_len()
        else:
            return self.evaler.get_eval_dataset_len(_type)

    def reset_lr_scheduler(self):
        if self.model_mode in [C.CNN]:
            self.trainer.reset_lr_scheduler()

    def net_init(self, multi_gpu, model_mode) -> nn.Module:
        """
        init self.net:
        """
        # Todo: add model_mode , modify belows
        if model_mode in [C.CNN]:
            return self.cnn_net_init(multi_gpu)
        elif model_mode in [C.SEG]:
            return self.seg_net_init(multi_gpu)
        else:
            raise TypeError("net_init() model_mode:{} valid.".format(model_mode))

    def cnn_net_init(self, multi_gpu):
        raise NotImplementedError

    def seg_net_init(self, multi_gpu, **kwargs):
        raise NotImplementedError

    def train(self, num_client_ep: int, ep: Epoch, sio=None, **kwargs):
        train_res = self.trainer.train(num_client_ep, ep, sio=sio, **kwargs)
        return train_res

    @torch.no_grad()
    def eval(self, eval_type):
        assert self.trainer.net == self.net
        return self.evaler.evaluate(self.net, eval_type)


def classification_model_gen(name: str, weights: Optional[str], **kwargs):
    class classification_model(BaseModel):
        def cnn_net_init(self, multi_gpu):
            self.logger.info("Model Config:{}".format(kwargs))
            if name.startswith("tm_"):
                net = TimmClassificationModel(
                    name=name[3:],
                    pretrain=weights is not None,
                    in_channels=self.num_channels,
                    classes=self.num_classes,
                    dropout=kwargs.get("dropout", 0.2)
                )
            else:
                net = ClassificationModel(
                    name=name,
                    weights=weights,
                    in_channels=self.num_channels,
                    classes=self.num_classes,
                    dropout=kwargs.get("dropout", 0.2)
                )
            if multi_gpu:
                net = nn.DataParallel(net)
            return net

    return classification_model


def segmentation_model_gen(encoder_name: str, decoder_name: str, weights: Optional[str] = "imagenet", **kwargs):
    if len(encoder_name) == 0 and decoder_name == "unet":
        class unet(BaseModel):
            def seg_net_init(self, multi_gpu, **kwargs):
                self.logger.warn("UNet always not loads pretrain_weights")
                net = UNet(self.num_channels, self.num_classes, up_sampling=True, batch_norm=True)
                if multi_gpu:
                    net = nn.DataParallel(net)
                return net

        return unet
    else:
        class segmentation_model(BaseModel):
            def seg_net_init(self, multi_gpu, projection_params=None):
                net = SegmentationModel(
                    encoder_name=encoder_name,
                    decoder_name=decoder_name,
                    encoder_weights=weights,
                    in_channels=self.num_channels,
                    classes=self.num_classes,
                    projection_params=projection_params,
                    **kwargs
                )

                if multi_gpu:
                    net = nn.DataParallel(net)
                return net

        return segmentation_model


class Models:
    # resnet18 = classification_model_gen("resnet18", "imagenet")
    # resnet34 = classification_model_gen("resnet34", "imagenet")
    # resnet50 = classification_model_gen("resnet50", "imagenet")
    # resnet101 = classification_model_gen("resnet101", "imagenet")
    # densenet121 = classification_model_gen("densenet121", "imagenet")
    # densenet169 = classification_model_gen("densenet169", "imagenet")
    # densenet201 = classification_model_gen("densenet201", "imagenet")
    # densenet161 = classification_model_gen("densenet161", "imagenet")
    # mobilenet_v2 = classification_model_gen("mobilenet_v2", "imagenet")
    #
    # mlp_1024 = classification_model_gen("mlp_1024_512x2_128")
    # mlp_256 = classification_model_gen("mlp_256x4")
    # mlp_128 = classification_model_gen("mlp_128x4")
    # mlp_20 = classification_model_gen("mlp_20x3")
    #
    # _unet = segmentation_model_gen(None, "unet")
    # resnet50_unet = segmentation_model_gen("resnet50", "unet", "imagenet")
    # densenet169_unet = segmentation_model_gen("densenet169", "unet", "imagenet")
    # resnet50_unetplusplus = segmentation_model_gen("resnet50", "unetpp", "imagenet")
    # resnet101_deeplabv3 = segmentation_model_gen("resnet101", "deeplabv3", "imagenet")
    # resnet101_deeplabv3plus = segmentation_model_gen("resnet101", "deeplabv3p", "imagenet")
    # resnet50_fpn = segmentation_model_gen("resnet50", "fpn", "imagenet")
    # resnet50_pspnet = segmentation_model_gen("resnet50", "pspnet", "imagenet")
    #
    # res_unet = resnet50_unet
    # dense_unet = densenet169_unet
    # unetplusplus = resnet50_unetplusplus
    # deeplabv3 = resnet101_deeplabv3
    # deeplabv3plus = resnet101_deeplabv3plus

    @staticmethod
    def cls_model_gen(model_name, ptr_weights, **kwargs):
        return classification_model_gen(model_name, ptr_weights, **kwargs)

    @staticmethod
    def seg_model_gen(model_name, ptr_weights):
        encoder_name, decoder_name = model_name.split("_", 1)
        return segmentation_model_gen(encoder_name, decoder_name, ptr_weights)
