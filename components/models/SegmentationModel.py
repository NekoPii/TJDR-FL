#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:maojingxin
"""
from typing import Optional

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class ProjectionHead(nn.Sequential):

    def __init__(self, in_channels, out_channels=256, proj="convmlp"):
        if proj == "linear":
            proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        elif proj == "convmlp":
            proj = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),  # proj channel not equal encoder.out_channel[-1]
                nn.ReLU(),  # proj channel not equal encoder.out_channel[-1]
                nn.Conv2d(out_channels, in_channels, kernel_size=1)  # proj channel not equal encoder.out_channel[-1]
            )
        super().__init__(proj)


class SegmentationModel(nn.Module):
    def __init__(
            self,
            encoder_name: str,
            decoder_name: str,
            encoder_weights: Optional[str] = "imagenet",
            in_channels: int = 3,
            classes: int = 1,
            projection_params: Optional[dict] = None,
            **kwargs
    ):
        super().__init__()

        assert decoder_name in ["unet", "unetpp", "deeplabv3", "deeplabv3p", "fpn", "pspnet"], "decoder_name:{} valid".format(decoder_name)

        if decoder_name == "unet":
            init_net = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                **kwargs
            )
        elif decoder_name == "unetpp":
            init_net = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                **kwargs
            )
        elif decoder_name == "deeplabv3":
            init_net = smp.DeepLabV3(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                **kwargs
            )
        elif decoder_name == "deeplabv3p":
            init_net = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                **kwargs
            )
        elif decoder_name == "fpn":
            init_net = smp.FPN(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                **kwargs
            )
        elif decoder_name == "pspnet":
            init_net = smp.PSPNet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                **kwargs
            )
        else:
            raise ValueError(decoder_name)

        self.encoder = init_net.encoder
        self.decoder = init_net.decoder
        self.segmentation_head = init_net.segmentation_head

        del init_net

        if projection_params is not None:
            self.projection_head = ProjectionHead(
                in_channels=self.encoder.out_channels[-1],
                **projection_params
            )
            initialize_head(self.projection_head)
        else:
            self.projection_head = None

    def set_encoder_weights(self, encoder_weights, strict: bool = True):
        self.encoder.load_state_dict(encoder_weights, strict=strict)

    def set_decoder_weights(self, decoder_weights, strict: bool = True):
        self.decoder.load_state_dict(decoder_weights, strict=strict)

    def set_seg_head_weights(self, seg_head_weights, strict: bool = True):
        self.segmentation_head.load_state_dict(seg_head_weights, strict=strict)

    def set_proj_head_weights(self, proj_head_weights, strict: bool = True):
        self.projection_head.load_state_dict(proj_head_weights, strict=strict)

    def forward(self, x, return_type="seg"):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)

        if return_type == "seg":
            decoder_output = self.decoder(*features)
            masks = self.segmentation_head(decoder_output)
            return masks
        elif return_type == "proj":
            proj = self.projection_head(features[-1])
            return proj
        elif return_type == "encoder":
            return features

    @torch.no_grad()
    def predict(self, x):
        if self.training:
            self.eval()
        x = self.forward(x)
        return x


if __name__ == "__main__":
    net = SegmentationModel(encoder_name="resnext50_32x4d", decoder_name="deeplabv3p", encoder_weights="imagenet", in_channels=3, classes=5)
    from torchkeras import summary

    summary(net, input_shape=(3, 224, 224), batch_size=8)
