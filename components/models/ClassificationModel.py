#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: maojingxin
"""
from typing import Optional

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from segmentation_models_pytorch.base.modules import Activation
import ssl
import timm

ssl._create_default_https_context = ssl._create_unverified_context


def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class ClassificationHead(nn.Sequential):

    def __init__(self, in_channels, classes, pooling="avg", dropout=0.2, activation=None):
        if pooling not in ("max", "avg"):
            raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        pool = nn.AdaptiveAvgPool2d(1) if pooling == 'avg' else nn.AdaptiveMaxPool2d(1)
        flatten = nn.Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout is not None else nn.Identity()
        linear = nn.Linear(in_channels, classes, bias=True)
        activation = Activation(activation)
        super().__init__(pool, flatten, dropout, linear, activation)


class ClassificationModel(nn.Module):
    def __init__(
            self,
            name: str,
            weights: Optional[str] = "imagenet",
            in_channels: int = 3,
            classes: int = 1,
            dropout: float = 0.2
    ):
        super().__init__()
        self.name = name
        self.weights = weights
        self.in_channels = in_channels
        self.classes = classes
        self.dropout = dropout
        self._init()

    def _init(self):
        self.encoder = smp.encoders.get_encoder(
            name=self.name,
            in_channels=self.in_channels,
            weights=self.weights
        )

        self.classification_head = ClassificationHead(
            in_channels=self.encoder.out_channels[-1],
            classes=self.classes,
            pooling="avg",
            dropout=self.dropout,
            activation=None
        )

        initialize_head(self.classification_head)

    def set_encoder_weights(self, encoder_weights, strict: bool = True):
        self.encoder.load_state_dict(encoder_weights, strict=strict)

    def set_class_head_weights(self, class_head_weights, strict: bool = True):
        self.classification_head.load_state_dict(class_head_weights, strict=strict)

    def forward(self, x, return_type="class", in_type=None):
        if in_type == "encoder":
            features = x
        else:
            features = self.encoder(x)
        if return_type == "class":
            classes = self.classification_head(features[-1])
            return classes
        elif return_type == "encoder":
            return features

    @torch.no_grad()
    def predict(self, x):
        if self.training:
            self.eval()
        x = self.forward(x)
        return x


class TimmClassificationModel(ClassificationModel):
    def __init__(
            self,
            name: str,
            pretrain: bool = True,
            in_channels: int = 3,
            classes: int = 1,
            dropout: float = 0.2
    ):
        self.pretrain = pretrain
        super().__init__(name, in_channels=in_channels, classes=classes, dropout=dropout)

    def _init(self):
        self.encoder = timm.create_model(
            model_name=self.name,
            pretrained=self.pretrain,
            in_chans=self.in_channels,
            features_only=True,
        )

        self.classification_head = ClassificationHead(
            in_channels=self.encoder.feature_info.channels()[-1],
            classes=self.classes,
            pooling="avg",
            dropout=self.dropout,
            activation=None
        )

        initialize_head(self.classification_head)


if __name__ == "__main__":
    """"""
    model = ClassificationModel(
        # name="timm-regnetx_002",
        # name="timm-regnety_002",
        # name="xception",
        # name="efficientnet-b5",
        # name="resnet50",
        name="resnet34",
        # name="timm-mobilenetv3_small_minimal_100",
        # name="resnet34",
        weights="imagenet", in_channels=3, classes=5)
    from torchkeras import summary

    # summary(model, input_shape=(1, 7, 7), batch_size=128)

    summary(model, input_shape=(3, 512, 512), batch_size=16)
    """"""

    ###################################################################

    timm_model = TimmClassificationModel(
        # name="efficientnet_b4",
        # name="resnet50",
        name="resnet34",
        pretrain=True,
        classes=5,
        in_channels=3
    )

    from torchkeras import summary

    # summary(model, input_shape=(1, 7, 7), batch_size=128)

    summary(timm_model, input_shape=(3, 512, 512), batch_size=16)
