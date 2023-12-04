#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:maojingxin
"""
from typing import List

import torch.nn as nn

from segmentation_models_pytorch.base.modules import Activation


class L_R_BoD(nn.Sequential):

    def __init__(self, in_dim: int, out_dim: int, bn=False, dropout=0.2):
        assert dropout is None or (isinstance(dropout, float) and 0 <= dropout <= 1)
        linear = nn.Linear(in_dim, out_dim, bias=True)
        relu = nn.ReLU()
        bn = nn.BatchNorm1d(out_dim) if bn else nn.Identity()
        dropout = nn.Dropout(p=dropout) if dropout else nn.Identity()
        super().__init__(linear, relu, bn, dropout)


class L_Head(nn.Sequential):
    def __init__(self, in_dim: int, out_dim: int, activation=None):
        linear = nn.Linear(in_dim, out_dim, bias=True)
        activation = Activation(activation)
        super().__init__(linear, activation)


class Conv1dReLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel_size, padding=0, stride=1, use_batchnorm=True):
        super().__init__()
        self.conv = nn.Conv1d(
            in_dim,
            out_dim,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batchnorm,
        )
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_dim) if use_batchnorm else nn.Identity()

    def forward(self, x):
        out = self.bn(self.relu(self.conv(x)))
        return out


class MLPModel(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, inner_dims: List[int] = None, bn=False, dropout=.0):
        super().__init__()
        dims = [in_dim] + ([] if inner_dims is None else inner_dims) + [out_dim]
        self.mlp = []
        self.head = None
        for i, (i_d, o_d) in enumerate(zip(dims[:-1], dims[1:])):
            if i == len(dims) - 2:
                self.head = L_Head(i_d, o_d, activation=None)
            else:
                self.mlp.append(L_R_BoD(i_d, o_d, bn=bn, dropout=dropout))
        self.mlp = nn.Sequential(*self.mlp)
        print(in_dim, out_dim, inner_dims, bn, dropout)

    @classmethod
    def parse_name(cls, name: str):
        ss = name.split("_")
        assert ss[0] == "mlp"
        ss = ss[1:]
        inner_dims = []
        bn = False
        dropout = None
        for s in ss:
            if s.startswith("b"):
                bn = True
                continue
            elif s.startswith("d"):
                dropout = float(s[1:])
                continue
            else:
                if "x" in s:
                    n, c = s.split("x")
                    inner_dims += [int(n)] * int(c)
                else:
                    inner_dims += [int(s)]
        return inner_dims, bn, dropout

    def forward(self, x, return_type="class", in_type=None):
        if in_type == "encoder":
            rep = x
        else:
            rep = self.mlp(x)
        if return_type == "encoder":
            return rep
        elif return_type == "class":
            out = self.head(rep)
            return out


class ConvMLP1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation=None):
        super().__init__()
        self.encoder = nn.Sequential(
            Conv1dReLU(in_channels, 64, kernel_size=3, padding=0, stride=1, use_batchnorm=False),
            Conv1dReLU(64, 128, kernel_size=3, padding=0, stride=1, use_batchnorm=False),
            nn.MaxPool1d(kernel_size=2, stride=2),
            Conv1dReLU(128, 256, kernel_size=3, padding=0, stride=1, use_batchnorm=False),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.5)
        )
        self.head = L_Head(256, out_channels, activation=activation)

    def forward(self, x, return_type="class", in_type=None):
        if in_type == "encoder":
            rep = x
        else:
            rep = self.encoder(x)
        if return_type == "encoder":
            return rep
        elif return_type == "class":
            out = self.head(rep)
            return out


class ConvMLP2(ConvMLP1):
    def __init__(self, in_channels: int, out_channels: int, activation=None):
        super().__init__(in_channels, out_channels, activation)
        self.encoder = nn.Sequential(
            Conv1dReLU(in_channels, 32, kernel_size=3, padding=0, stride=1, use_batchnorm=False),
            Conv1dReLU(32, 64, kernel_size=3, padding=0, stride=1, use_batchnorm=False),
            nn.MaxPool1d(kernel_size=2, stride=2),
            Conv1dReLU(64, 128, kernel_size=3, padding=0, stride=1, use_batchnorm=False),
            Conv1dReLU(128, 256, kernel_size=3, padding=0, stride=1, use_batchnorm=False),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            # nn.Dropout(0.5),
            nn.Linear(256, 1024)
        )
        self.head = L_Head(1024, out_channels, activation=activation)


class ConvMLP:
    convmlp1 = ConvMLP1
    convmlp2 = ConvMLP2


if __name__ == "__main__":
    # mlp = MLPModel(42, 10, [256, 256, 256, 256, 256], bn=True, dropout=0.3)
    name = "mlp_256x4_128"
    mlp = MLPModel(42, 10, *MLPModel.parse_name(name))

    conv_mlp = ConvMLP2(1, 10)
    from torchkeras import summary

    # summary(mlp, input_shape=(42,), batch_size=2000)
    summary(conv_mlp, input_shape=(1, 42), batch_size=128)
