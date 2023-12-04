#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: maojingxin
"""
__all__ = ["GaussRampUp", "LinearRampUp"]

import numpy as np


class RampUp(object):
    def __init__(self):
        pass

    def value(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError


class GaussRampUp(RampUp):
    def __init__(self, begin_epoch, end_epoch, eta_max=1, exp_coeff=5, last_epoch=0):
        super().__init__()
        self.begin_epoch = int(begin_epoch)
        self.end_epoch = int(end_epoch)
        self.eta_max = float(eta_max)
        self.exp_coeff = float(exp_coeff)
        self.last_epoch = int(last_epoch)
        assert self.begin_epoch <= self.end_epoch, "begin_epoch > end_epoch is valid"
        if self.last_epoch == -1:
            self.step()

    def step(self):
        self.last_epoch += 1

    @property
    def value(self):
        return self.gauss_ramp_up(self.last_epoch, self.begin_epoch, self.end_epoch, self.eta_max, self.exp_coeff)

    def get_value(self, epoch):
        return self.gauss_ramp_up(epoch, self.begin_epoch, self.end_epoch, self.eta_max, self.exp_coeff)

    @staticmethod
    def gauss_ramp_up(cur_epoch, begin_epoch, end_epoch, eta_max, exp_coeff):
        if cur_epoch < begin_epoch:
            return 0
        elif cur_epoch >= end_epoch:
            return eta_max
        return eta_max * np.exp(
            -1.0 * exp_coeff * (1 - float((cur_epoch - begin_epoch + 1) / (end_epoch - begin_epoch + 1))) ** 2
        )


class LinearRampUp(RampUp):
    def __init__(self, begin_epoch, end_epoch, eta_max=1, last_epoch=0):
        super().__init__()
        self.begin_epoch = int(begin_epoch)
        self.end_epoch = int(end_epoch)
        self.eta_max = float(eta_max)
        self.last_epoch = int(last_epoch)
        assert self.begin_epoch <= self.end_epoch, "begin_epoch > end_epoch is valid"
        if self.last_epoch == -1:
            self.step()

    def step(self):
        self.last_epoch += 1

    @property
    def value(self):
        return self.linear_ramp_up(self.last_epoch, self.begin_epoch, self.end_epoch, self.eta_max)

    def get_value(self, epoch):
        return self.linear_ramp_up(epoch, self.begin_epoch, self.end_epoch, self.eta_max)

    @staticmethod
    def linear_ramp_up(cur_epoch, begin_epoch, end_epoch, eta_max):
        if cur_epoch < begin_epoch:
            return 0
        elif cur_epoch >= end_epoch:
            return eta_max
        return eta_max * float((cur_epoch - begin_epoch + 1) / (end_epoch - begin_epoch + 1))
