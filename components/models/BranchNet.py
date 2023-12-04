#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: maojingxin
"""

from collections.abc import Iterable
from typing import TypeVar

import torch
import torch.nn as nn

from gutils import constants as C

T = TypeVar('T', bound='Module')


class BranchNet(nn.Module):
    def __init__(self, net1: nn.Module, net2: nn.Module):
        super(BranchNet, self).__init__()
        self.branch1 = net1
        self.branch2 = net2

    def forward(self, data, branch=1):
        if not self.training:
            out1 = self.branch1(data)
            return out1

        if branch == 1:
            return self.branch1(data)
        elif branch == 2:
            return self.branch2(data)

    def train(self, mode: bool = True):
        super(BranchNet, self).train(mode)
        self.branch1.train(mode)
        self.branch2.train(mode)

    def eval(self):
        super(BranchNet, self).eval()
        self.branch1.eval()
        self.branch2.eval()

    def to_(self, device):
        self.branch1 = self.branch1.to(device)
        self.branch2 = self.branch2.to(device)

    def set_param(self, param, branch: int, strict: bool = True):
        if branch == 1:
            self.branch1.load_state_dict(param, strict)
        elif branch == 2:
            self.branch2.load_state_dict(param, strict)

    def get_param(self, branch: int):
        if branch == 1:
            return self.branch1.parameters()
        elif branch == 2:
            return self.branch2.parameters()

    def load_state_dict(self, state_dict, strict: bool = True):
        assert isinstance(state_dict, Iterable) and len(state_dict) == 2
        self.branch1.load_state_dict(state_dict[0], strict)
        self.branch2.load_state_dict(state_dict[1], strict)


class AdditionNet(BranchNet):
    def __init__(self, body: nn.Module, addition: nn.Module, kappa: float):
        super(AdditionNet, self).__init__(net1=body, net2=addition)
        self.kappa = kappa

    @torch.no_grad()
    def init_addition_weights(self):
        for param_2 in self.branch2.parameters():
            param_2.data.mul_(self.kappa)

    @torch.no_grad()
    def add(self, sign=1.0, branch=1):
        if branch == 1:
            for param_1, param_2 in zip(self.branch1.parameters(), self.branch2.parameters()):
                param_1.data.add_(sign * param_2.data)
        elif branch == 2:
            for param_1, param_2 in zip(self.branch1.parameters(), self.branch2.parameters()):
                param_2.data.add_(sign * param_1.data)
        else:
            raise ValueError(branch)

    def forward(self, data, _type=C.LABELED):
        if not self.training:
            self.add(sign=1.0, branch=1)
            out = self.branch1(data)
            self.add(sign=-1.0, branch=1)
            return out

        if _type == C.LABELED:
            self.add(sign=1.0, branch=1)
            out1 = self.branch1(data)
            self.add(sign=-1.0, branch=1)
            self.add(sign=1.0, branch=2)
            out2 = self.branch2(data)
            self.add(sign=-1.0, branch=2)
            out = out1, out2
        elif _type == C.UNLABELED:
            out = self.branch2(data)
        else:
            raise TypeError(_type)
        return out
