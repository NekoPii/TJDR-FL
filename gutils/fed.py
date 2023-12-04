#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: maojingxin
"""
import abc
import copy
from typing import List

import numpy as np
import torch
from torch import nn

from gutils import gutil


@torch.no_grad()
def FedAvg(clients_weights, clients_contribution: List[torch.Tensor] or List[List[torch.Tensor]] = None, ignore_bn=False):
    if isinstance(clients_contribution, list) and len(clients_contribution) == 0:
        clients_contribution = None

    if isinstance(clients_weights[0], (tuple, list)) and len(clients_weights[0]) == 2:
        clients_weights1 = [clients_weight[0] for clients_weight in clients_weights]
        clients_weights2 = [clients_weight[1] for clients_weight in clients_weights]
        if clients_contribution is not None:
            if len(clients_contribution) == 2 and isinstance(clients_contribution[0], list) and isinstance(clients_contribution[1], list):
                return FedAvg(clients_weights1, clients_contribution[0], ignore_bn=ignore_bn), FedAvg(clients_weights2, clients_contribution[1], ignore_bn=ignore_bn)
            else:
                return FedAvg(clients_weights1, clients_contribution, ignore_bn=ignore_bn), FedAvg(clients_weights2, clients_contribution, ignore_bn=ignore_bn)
        else:
            return FedAvg(clients_weights1, ignore_bn=ignore_bn), FedAvg(clients_weights2, ignore_bn=ignore_bn)
    if clients_contribution is not None:
        assert len(clients_weights) == len(clients_contribution)
    total_size = len(clients_weights) if clients_contribution is None else sum(clients_contribution)

    avg_weights = copy.deepcopy(clients_weights[0])

    if len(clients_weights) == 1:
        return avg_weights

    if ignore_bn:
        avg_weights = gutil.remove_layer(avg_weights, "bn")

    for k in avg_weights.keys():
        assert not torch.any(torch.isnan(clients_weights[0][k])), "client[0] weights contains nan"
        if clients_contribution is not None:
            avg_weights[k] = torch.mul(avg_weights[k], clients_contribution[0])
        for i in range(1, len(clients_weights)):
            assert not torch.any(torch.isnan(clients_weights[i][k])), "client[{}] weights contains nan".format(i)
            avg_weights[k] += clients_weights[i][k] if clients_contribution is None else torch.mul(clients_weights[i][k], clients_contribution[i])
        avg_weights[k] = torch.div(avg_weights[k], total_size)
        assert not torch.any(torch.isnan(avg_weights[k])), "{} contains nan".format(avg_weights[k])
    return avg_weights


def cal_alphaFed_contrib(n, alpha, clients_contribution: List[torch.Tensor] = None):
    if clients_contribution is None:
        return clients_contribution

    if isinstance(alpha, float) and 0 < alpha < 1:
        alpha = [alpha] * n
    assert isinstance(alpha, (list, np.ndarray, tuple)) and len(alpha) == n

    alpha = np.array(alpha)
    alpha = alpha.reshape(n, 1)
    alpha = np.tile(alpha, n)
    alpha = np.eye(n) * alpha + (1 - np.eye(n)) * (1 - alpha)
    alpha = torch.Tensor(alpha)

    alphaFed_clients_contribs = torch.Tensor(clients_contribution)
    alphaFed_clients_contribs = torch.repeat_interleave(alphaFed_clients_contribs.unsqueeze_(0), repeats=n, dim=0)

    if alphaFed_clients_contribs.ndim == 3:
        temp = (1 - torch.eye(n)) * alphaFed_clients_contribs
        alphaFed_clients_contribs = (torch.eye(n) + temp / torch.sum(temp, dim=1).reshape(-1, 1)) * alpha
    else:
        temp = (1 - torch.eye(n)) * alphaFed_clients_contribs
        alphaFed_clients_contribs = (torch.eye(n) + temp / torch.sum(temp, dim=1).reshape(-1, 1)) * alpha
    alphaFed_clients_contribs = list(alphaFed_clients_contribs)
    alphaFed_clients_contribs = [list(cd_clients_contrib) for cd_clients_contrib in alphaFed_clients_contribs]

    return alphaFed_clients_contribs


@torch.no_grad()
def alphaFedAvg(alpha, clients_weights, clients_contribution: List[torch.Tensor] = None, ignore_bn=False):
    _len = len(clients_weights)
    alphaFed_contribs = cal_alphaFed_contrib(_len, alpha, clients_contribution)
    fed_weights = []
    for i in range(_len):
        fed_weights.append(FedAvg(clients_weights, alphaFed_contribs[i], ignore_bn))
    return fed_weights
