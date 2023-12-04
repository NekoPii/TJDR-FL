#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: maojingxin
"""
__all__ = ["MSE2D", "Entropy", "KLD", "F_JSD", "IRM_KLD", "FocalLoss", "DiceLoss", "weight_reduce_loss"]

import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import segmentation_models_pytorch as smp

from gutils import gutil


class MSE2D(nn.Module):
    def __init__(self, dim=1, reduction=True, logits=False):
        super().__init__()
        self.dim = dim
        self.reduction = reduction
        self.logits = logits

    def forward(self, pred, gt):
        if self.logits:
            pred = F.softmax(pred, self.dim)
            gt = F.softmax(gt, self.dim)
        assert gutil.check_same_shape(pred, gt)
        mse_loss = (pred - gt) ** 2
        if self.reduction:
            mse_loss = mse_loss.sum(dim=self.dim)
            mse_loss = mse_loss.mean()
        return mse_loss


class Entropy(nn.Module):
    def __init__(self, dim=1, eps=1e-8, logits=False, reduction=False):
        super(Entropy, self).__init__()
        self.dim = dim
        self.eps = eps
        self.logits = logits
        self.reduction = reduction

    def forward(self, prob):
        if self.logits:
            prob = F.softmax(prob, self.dim)
        entropy = -1.0 * prob * (prob + self.eps).log()
        entropy = entropy.sum(dim=self.dim)
        if self.reduction:
            entropy = entropy.mean()
        return entropy


class KLD(nn.Module):
    def __init__(self, dim=1, reduction=False, logits=False, eps=1e-10):
        super(KLD, self).__init__()
        self.dim = dim
        self.reduction = reduction
        self.logits = logits
        self.eps = eps

    def forward(self, p_prob, q_prob):
        if self.logits:
            p_prob = F.softmax(p_prob, self.dim)
            q_prob = F.softmax(q_prob, self.dim)

        assert gutil.check_prob(p_prob, dim=self.dim), "{} is not prob".format(p_prob)
        assert gutil.check_prob(q_prob, dim=self.dim), "{} is not prob".format(q_prob)
        assert gutil.check_same_shape(p_prob, q_prob), "the shape of p is not same as q"

        log_p = (p_prob + self.eps).log()
        log_q = (q_prob + self.eps).log()

        qlog_q = (q_prob * log_q).sum(dim=self.dim)
        qlog_p = (q_prob * log_p).sum(dim=self.dim)

        kld = qlog_q - qlog_p

        return kld.mean() if self.reduction else kld


class JSD(nn.Module):
    def __init__(self, dim=1, reduction=False, logits=False, eps=1e-10):
        super(JSD, self).__init__()
        self.dim = dim
        self.reduction = reduction
        self.logits = logits
        self.eps = eps
        self.entropy = Entropy(dim=self.dim, logits=False, eps=self.eps)

    def forward(self, *probs):
        for prob in probs:
            if self.logits:
                prob = F.softmax(prob, self.dim)
            assert gutil.check_prob(prob, dim=self.dim), "{} is not prob".format(prob)
        assert gutil.check_same_shape(*probs), "the shape of probs is not same"

        prob_mean = sum(probs) / len(probs)
        prob_mean_entropy = self.entropy(prob_mean)

        prob_entropy_mean = sum([self.entropy(prob) for prob in probs]) / len(probs)

        jsd = prob_mean_entropy - prob_entropy_mean

        return jsd.mean() if self.reduction else jsd


class F_JSD(nn.Module):
    def __init__(self, dim=1, reduction="batchmean", logits=False, eps=1e-8):
        super(F_JSD, self).__init__()
        self.dim = dim
        self.reduction = reduction
        self.logits = logits
        self.eps = eps

    def forward(self, p, q):
        if self.logits:
            p = F.softmax(p, self.dim)
            q = F.softmax(q, self.dim)
        mean_pq = (p + q) / 2
        jsd = F.kl_div((mean_pq + self.eps).log(), p, reduction=self.reduction) / 2 + F.kl_div((mean_pq + self.eps).log(), q, reduction=self.reduction) / 2
        return jsd


class IRM_KLD(nn.Module):
    def __init__(self, num_classes: int, reduction="batchmean", eps=1e-6):
        super(IRM_KLD, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, gt):
        with torch.no_grad():
            pred_nan_mask = torch.isnan(pred)
            gt_nan_mask = torch.isnan(gt)

            mask = (~pred_nan_mask) * (~gt_nan_mask)

        mask_pred = pred[mask].view(-1, self.num_classes)
        mask_gt = gt[mask].view(-1, self.num_classes)

        mask_pred = F.softmax(mask_pred + self.eps, dim=1)
        mask_gt = F.softmax(mask_gt + self.eps, dim=1)

        jsd = F.kl_div(mask_pred.log(), mask_gt, reduction=self.reduction) / 2 + F.kl_div(mask_gt.log(), mask_pred, reduction=self.reduction) / 2
        return jsd


def get_enum(reduction: str) -> int:
    if reduction == 'none':
        ret = 0
    elif reduction == 'mean':
        ret = 1
    elif reduction == 'elementwise_mean':
        warnings.warn("reduction='elementwise_mean' is deprecated, please use reduction='mean' instead.")
        ret = 1
    elif reduction == 'sum':
        ret = 2
    else:
        ret = -1
        raise ValueError("{} is not a valid value for reduction".format(reduction))
    return ret


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction="mean", avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor or None): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float or None): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        if weight.dim() > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == "mean":
            loss = loss.sum() / avg_factor
        # if reduction is "none", then do nothing, otherwise raise an error
        elif reduction != "none":
            raise ValueError("avg_factor can not be used with reduction='sum'")
    return loss


class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(
            self,
            alpha=np.array([1.0, 3.0]),
            gamma=2,
            balance_index=0,
            smooth=1e-5,
            size_average=True
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, gt, **kwargs):
        logit = F.softmax(logit, dim=1)

        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        # gt = torch.squeeze(gt, 1)
        gt = gt.view(-1, 1)
        # print(logit.shape, gt.shape)
        #
        # alpha = np.array([0.5, 2.5])
        alpha = self.alpha
        # alpha = None

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = gt.cpu().long()

        one_hot_key = torch.FloatTensor(gt.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth
            )
        pt = (one_hot_key * logit).sum(1) + self.smooth
        log_pt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]  # N,H,W,1
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * log_pt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class DiceLoss(nn.Module):
    """DiceLoss.

    This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
    Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.

    Args:
        smooth (float): A float number to smooth loss, and avoid NaN error.
            Default: 1
        exponent (float): An float number to calculate denominator
            value: \\sum{x^exponent} + \\sum{y^exponent}. Default: 2.
        reduction (str, optional): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_image is True. Default: "mean".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Default to 1.0.
        ignore_index (int | None): The label index to be ignored. Default: 255.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to "loss_dice".
    """

    def __init__(self, smooth=1, exponent=2, reduction="mean", class_weight=None, loss_weight=1.0, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.reduction = reduction
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, gt, avg_factor=None, reduction_override=None):
        """
        :param pred: [N,C,*]
        :param gt: [N,*] values in [0,num_classes)
        :param avg_factor:
        :param reduction_override:
        :return:
        """
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = pred.new_tensor(self.class_weight)
        else:
            class_weight = None

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        one_hot_gt = F.one_hot(torch.clamp(gt.long(), 0, num_classes - 1), num_classes=num_classes)
        valid_mask = (gt != self.ignore_index).long()

        loss = self.loss_weight * self.dice_loss(
            pred,
            one_hot_gt,
            valid_mask=valid_mask,
            reduction=reduction,
            avg_factor=avg_factor,
            smooth=self.smooth,
            exponent=self.exponent,
            class_weight=class_weight,
            ignore_index=self.ignore_index
        )
        return loss

    @classmethod
    def dice_loss(cls, pred, gt, valid_mask, weight=None, reduction="mean", avg_factor=None, smooth=1., exponent=2., class_weight=None, ignore_index=255):
        assert pred.shape[0] == gt.shape[0]
        total_loss = 0
        num_classes = pred.shape[1]
        for i in range(num_classes):
            if i != ignore_index:
                dice_loss = cls.binary_dice_loss(
                    pred[:, i], gt[..., i], valid_mask=valid_mask, smooth=smooth,
                    exponent=exponent
                )
                if class_weight is not None:
                    dice_loss *= class_weight[i]
                total_loss += dice_loss
        loss = total_loss / num_classes
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    @classmethod
    def binary_dice_loss(cls, pred, gt, valid_mask, weight=None, reduction="mean", avg_factor=None, smooth=1., exponent=2.):
        assert pred.shape[0] == gt.shape[0]
        pred = pred.reshape(pred.shape[0], -1)
        gt = gt.reshape(gt.shape[0], -1)
        valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

        num = torch.sum(torch.mul(pred, gt) * valid_mask, dim=1) * 2 + smooth
        den = torch.sum(pred.pow(exponent) + gt.pow(exponent), dim=1) + smooth

        loss = 1 - num / den
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)

        return loss


if __name__ == "__main__":
    diceloss1 = DiceLoss(exponent=1, smooth=0)
    diceloss2 = smp.losses.DiceLoss(mode="multiclass", from_logits=False)
    diceloss3 = smp.losses.DiceLoss(mode="multiclass", from_logits=True)

    _input = torch.randn((4, 5, 3, 3))
    _softmax = torch.softmax(_input, dim=1)
    _target = torch.randint(0, 5, (4, 3, 3))

    print(_input)
    print(_softmax)
    print(torch.argmax(_softmax, dim=1))
    print(_target)

    dls1 = diceloss1(_input, _target)
    dls2 = diceloss2(_softmax, _target)
    dls3 = diceloss3(_input, _target)
    print("dls1:{}".format(dls1))
    print("dls2:{}".format(dls2))
    print("dls3:{}".format(dls3))
