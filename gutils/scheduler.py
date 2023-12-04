#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:maojingxin
"""
__all__ = ["Scheduler"]

from collections.abc import Iterable

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

from gutils import gutil, constants as C


class Scheduler(object):
    def __init__(self, optimizer, lr_sch_cfg, logger, **kwargs):
        self.optimizer = optimizer
        self.lr_sch_cfg = lr_sch_cfg
        self.logger = logger

        if self.optimizer is None or self.lr_sch_cfg is None:
            self.logger.info("LR_Scheduler Init None.")
            self.lr_scheduler = None

        else:
            self.logger.info("LR_Scheduler Config:{}".format(lr_sch_cfg))
            self.logger.info("LR_Scheduler Component Init ......")

            # for Adam , don't use lr_schedule
            if C.Adam in str(optimizer):
                self.lr_scheduler = None
            else:
                self.lr_mode = lr_sch_cfg.get(C.MODE, C.POLY)  # default is poly scheduler
                total_epoch = kwargs.get("total_epoch", np.inf)
                self.lr_begin_epoch = np.clip(int(lr_sch_cfg.get("begin_epoch", 0)), 0, total_epoch).item()
                self.lr_end_epoch = np.clip(int(lr_sch_cfg.get("end_epoch", total_epoch)), 0, total_epoch).item()
                assert self.lr_begin_epoch < self.lr_end_epoch, "lr_begin_epoch({}) >= lr_end_epoch({}) is valid".format(self.lr_begin_epoch, self.lr_end_epoch)
                self.lr_epoch = self.lr_end_epoch - self.lr_begin_epoch
                self.lr_iter_step = lr_sch_cfg.get("iter_step", False)  # lr scheduler step per iter

                train_per_epoch_iter_num = kwargs.get("train_per_epoch_iter_num")
                last_epoch = kwargs.get("last_epoch", -1)
                self.lr_scheduler = self._lr_scheduler_init(self.lr_mode, train_per_epoch_iter_num, last_epoch)

    def _lr_scheduler_init(self, lr_mode, train_per_epoch_iter_num, last_epoch=-1):
        # Todo:add more lr_scheduler
        if lr_mode == C.POLY:
            lr_schedule = PolyLR(
                self.optimizer,
                power=float(self.lr_sch_cfg.get("power", 0.9)),
                nums_epoch=self.lr_epoch * max(self.lr_iter_step * train_per_epoch_iter_num, 1),
                min_lr=float(self.lr_sch_cfg.get("min_lr", 1e-5)),
                last_epoch=last_epoch,
            ).get_schedule()
        elif lr_mode == C.COS:
            lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.lr_epoch * max(self.lr_iter_step * train_per_epoch_iter_num, 1),
                eta_min=float(self.lr_sch_cfg.get("min_lr", 1e-5)),
                last_epoch=last_epoch
            )
        elif lr_mode == C.WARMUP_POLY:
            warmup_epoch = self.lr_sch_cfg.get("warmup_epoch", self.lr_epoch // 10)
            lr_schedule = WarmUpPolyLR(
                self.optimizer,
                power=float(self.lr_sch_cfg.get("power", 0.9)),
                nums_epoch=self.lr_epoch * max(self.lr_iter_step * train_per_epoch_iter_num, 1),
                warmup_epoch=warmup_epoch * max(self.lr_iter_step * train_per_epoch_iter_num, 1),
                min_lr=float(self.lr_sch_cfg.get("min_lr", 1e-5)),
                last_epoch=last_epoch,
            ).get_schedule()
        elif lr_mode == C.CONSTANT:
            lr_schedule = torch.optim.lr_scheduler.ConstantLR(
                self.optimizer,
                factor=1,
                total_iters=last_epoch,
                last_epoch=last_epoch
            )
        elif lr_mode == "step":
            step_size = self.lr_sch_cfg.get("step_size", self.lr_epoch // 4)
            gamma = self.lr_sch_cfg.get("gamma", 0.1)
            lr_schedule = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma,
                last_epoch=last_epoch
            )
        elif lr_mode == "onecycle":
            lr_schedule = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.optimizer.param_groups[0]["initial_lr"],
                total_steps=self.lr_epoch * max(self.lr_iter_step * train_per_epoch_iter_num, 1),
                pct_start=float(self.lr_sch_cfg.get("pct_start", 0.3)),
                div_factor=float(self.lr_sch_cfg.get("div_factor", 25)),
                final_div_factor=float(self.lr_sch_cfg.get("final_div_factor", 1e4)),
            )
        else:
            raise ValueError("lr_scheduler mode:{} error".format(lr_mode))

        # Sequential LR only for labeled-client
        milestone = self.lr_sch_cfg.get(C.MILESTONE)
        if milestone is not None:
            assert isinstance(milestone, int) or isinstance(milestone, Iterable), "milestone:{} valid".format(milestone)
            if isinstance(milestone, int):
                milestone = [milestone]
            milestone = list(milestone)

            begin_milestone_end = milestone.copy()
            begin_milestone_end.insert(0, self.lr_begin_epoch)
            begin_milestone_end.append(self.lr_end_epoch)
            begin_milestone_end = list(np.array(begin_milestone_end) * max(self.lr_iter_step * train_per_epoch_iter_num, 1))

            milestone = list((np.array(milestone) - self.lr_begin_epoch) * max(self.lr_iter_step * train_per_epoch_iter_num, 1))

            assert gutil.check_list_mono(begin_milestone_end, increase=True, strict=True), "begin_milestone_end:{} is not strict mono-increase".format(begin_milestone_end)

            lr_schedules = []

            for i in range(len(begin_milestone_end) - 1):
                now_ep_gap = begin_milestone_end[i + 1] - begin_milestone_end[i]
                if lr_mode == C.POLY:
                    lr_schedules.append(
                        PolyLR(
                            self.optimizer,
                            power=float(self.lr_sch_cfg.get("power", 0.9)),
                            nums_epoch=now_ep_gap,
                            min_lr=float(self.lr_sch_cfg.get("min_lr", 1e-5)),
                            last_epoch=last_epoch
                        ).get_schedule()
                    )
                elif lr_mode == C.COS:
                    lr_schedules.append(
                        torch.optim.lr_scheduler.CosineAnnealingLR(
                            self.optimizer,
                            T_max=now_ep_gap,
                            eta_min=float(self.lr_sch_cfg.get("min_lr", 1e-5)),
                            last_epoch=last_epoch
                        )
                    )
                else:
                    raise ValueError("lr_scheduler mode:{} error".format(lr_mode))

            lr_schedule = torch.optim.lr_scheduler.SequentialLR(
                optimizer=self.optimizer,
                schedulers=lr_schedules,
                milestones=milestone,
                last_epoch=last_epoch
            )

        # For disconnected clients
        if lr_schedule and last_epoch > -1:
            for _ in range((last_epoch + 1) * max(self.lr_iter_step * train_per_epoch_iter_num, 1)):
                lr_schedule.step()

        self.logger.info("LR_Scheduler Component Init Completed.")

        return lr_schedule

    def step(self, cur_ep: int, is_iter: bool):
        if self.lr_scheduler is not None and is_iter == self.lr_iter_step and cur_ep >= self.lr_begin_epoch:
            self.lr_scheduler.step()


class PolyLR(object):
    def __init__(self, optimizer, nums_epoch: int, power=0.9, min_lr=1e-5, last_epoch=-1):
        self.optimizer = optimizer
        self.nums_epoch = nums_epoch
        self.power = power
        self.min_lr = min_lr
        self.init_lr = self.optimizer.param_groups[0]["initial_lr"]
        self.last_epoch = last_epoch

    def get_schedule(self):
        def poly(cur_epoch):
            if cur_epoch >= self.nums_epoch:
                return self.min_lr / self.init_lr
            return (1 - (cur_epoch / self.nums_epoch)) ** self.power

        return LambdaLR(self.optimizer, poly, self.last_epoch)


class WarmUpPolyLR(object):
    def __init__(self, optimizer, nums_epoch: int, warmup_epoch: int, power=0.9, min_lr=1e-5, last_epoch=-1):
        self.optimizer = optimizer
        self.nums_epoch = nums_epoch
        self.warmup_epoch = warmup_epoch
        assert self.warmup_epoch < self.nums_epoch, "warmup epoch:{} valid , nums_epoch:{}".format(warmup_epoch, nums_epoch)
        self.power = power
        self.min_lr = min_lr
        self.init_lr = self.optimizer.param_groups[0]["initial_lr"]
        self.last_epoch = last_epoch

    def get_schedule(self):
        def warmup_poly(cur_epoch):
            if cur_epoch < self.warmup_epoch:
                return cur_epoch / self.warmup_epoch
            elif cur_epoch >= self.nums_epoch:
                return self.min_lr / self.init_lr
            return (1 - ((cur_epoch - self.warmup_epoch) / (self.nums_epoch - self.warmup_epoch))) ** self.power

        return LambdaLR(self.optimizer, warmup_poly, self.last_epoch)
