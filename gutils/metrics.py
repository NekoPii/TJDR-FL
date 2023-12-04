#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: maojingxin
"""
import copy
import warnings
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, multilabel_confusion_matrix, precision_recall_curve

from gutils import gutil

np.seterr(divide="ignore", invalid="ignore")
warnings.filterwarnings("ignore")


def new_cal_confusion_matrix(pred, label, num_classes, ignore_label):
    mask = (label != ignore_label)
    pred = pred[mask]
    label = label[mask]

    """
    [[[TN,FP],
    [FN,TP]],]
    """
    mcm = multilabel_confusion_matrix(label, pred, labels=np.arange(num_classes))  # (num_class,2,2)
    TP = mcm[:, 1, 1]
    TN = mcm[:, 0, 0]
    FP = mcm[:, 0, 1]
    FN = mcm[:, 1, 0]

    weight = TP + FN

    """
    class=3
                  pred
            0  1  2  3  4  5      
          0         
          1   TN    FP   TN
    label 2      
          3   FN    TP   FN
          4
          5   TN    FP   TN
    """

    return TP, TN, FP, FN, weight


def cal_ROC_AUC(probs, gts, num_classes, ignore_label=-1, multi=1.0, dot=5, mean_type="macro"):
    if not isinstance(probs, (torch.Tensor, np.ndarray)) and isinstance(probs, list):
        probs = torch.stack(probs, dim=0) if isinstance(probs[0], torch.Tensor) else np.array(probs)
    gts_onehot = gutil.one_hot(gts, num_classes, ignore_label)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for cid in range(num_classes):
        fpr[cid], tpr[cid], _ = roc_curve(gts_onehot[:, cid], probs[:, cid])
        fpr[cid], tpr[cid] = np.nan_to_num(fpr[cid]), np.nan_to_num(tpr[cid])
        roc_auc[cid] = np.nan_to_num(np.array(auc(fpr[cid], tpr[cid])))

    if mean_type == "macro":
        all_fpr = np.unique(np.concatenate([fpr[cid] for cid in range(num_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for cid in range(num_classes):
            mean_tpr += np.interp(all_fpr, fpr[cid], tpr[cid])
        mean_tpr /= num_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        # roc_auc["macro"] = np.array(auc(fpr["macro"], tpr["macro"]))
        roc_auc["macro"] = np.mean([roc_auc[cid] for cid in range(num_classes)])

    if mean_type == "micro":
        fpr["micro"], tpr["micro"], _ = roc_curve(gts_onehot.flatten(), probs.flatten())
        roc_auc["micro"] = np.array(auc(fpr["micro"], tpr["micro"]))

    return_auc = copy.deepcopy(roc_auc)
    for k in return_auc.keys():
        return_auc[k] = np.around(return_auc[k] * multi, dot)
    return return_auc, roc_auc, fpr, tpr


def cal_PR_AUC(probs, gts, num_classes, ignore_label=-1, multi=1.0, dot=5, mean_type="macro"):
    if not isinstance(probs, (torch.Tensor, np.ndarray)) and isinstance(probs, list):
        probs = torch.stack(probs, dim=0) if isinstance(probs[0], torch.Tensor) else np.array(probs)
    gts_onehot = gutil.one_hot(gts, num_classes, ignore_label)

    precision = dict()
    recall = dict()
    pr_auc = dict()
    for cid in range(num_classes):
        precision[cid], recall[cid], _ = precision_recall_curve(gts_onehot[:, cid], probs[:, cid])
        precision[cid], recall[cid] = np.nan_to_num(precision[cid]), np.nan_to_num(recall[cid])
        pr_auc[cid] = np.nan_to_num(np.array(auc(recall[cid], precision[cid])))

    if mean_type == "macro":
        all_recall = np.unique(np.concatenate([recall[cid] for cid in range(num_classes)]))
        mean_precision = np.zeros_like(all_recall)
        for cid in range(num_classes):
            mean_precision += np.interp(all_recall, recall[cid], precision[cid])
        mean_precision /= num_classes

        recall["macro"] = all_recall
        precision["macro"] = mean_precision
        # pr_auc["macro"] = np.array(auc(recall["macro"], precision["macro"]))
        pr_auc["macro"] = np.mean([pr_auc[cid] for cid in range(num_classes)])

    if mean_type == "micro":
        precision["micro"], recall["micro"], _ = precision_recall_curve(gts_onehot.flatten(), probs.flatten())
        pr_auc["micro"] = np.array(auc(recall["micro"], precision["micro"]))

    return_auc = copy.deepcopy(pr_auc)
    for k in return_auc.keys():
        return_auc[k] = np.around(return_auc[k] * multi, dot)
    return return_auc, pr_auc, precision, recall


def draw_ROC(probs, gts, num_classes, ignore_label: int, save_path: str):
    _, roc_auc, fpr, tpr = cal_ROC_AUC(probs, gts, num_classes, ignore_label)

    fig = plt.figure(figsize=[30, 30])
    plt.plot(
        fpr["macro"], tpr["macro"],
        label="macro-average ROC curve (AUC = {:.2f})".format(roc_auc["macro"]),
        color="darkorange", lw=2
    )
    plt.plot(
        fpr["micro"], tpr["micro"],
        label="micro-average ROC curve (AUC = {:.2f})".format(roc_auc["micro"]),
        color="cornflowerblue", lw=2
    )
    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-AUC")
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close(fig)


@torch.no_grad()
def new_cal_metric(preds, gts, probs, num_classes, ignore_label=-1, nan_to_num=0.0, multi=1.0, dot=3, mean_type="macro", **kwargs):
    assert mean_type in ["macro", "micro", "weighted"], "mean_type:{} valid".format(mean_type)
    assert preds.shape == gts.shape and preds.shape[0] == gts.shape[0], "preds.shape:{} gts.shape:{}".format(preds.shape, gts.shape)
    if len(preds.shape) > 2:
        # n, h, w = preds.shape
        # preds = preds.reshape(-1)
        # gts = gts.reshape(-1)
        # probs = probs.reshape(n * h * w, num_classes)
        task = "seg"
    else:
        task = "cls"

    TP, TN, FP, FN, weight = new_cal_confusion_matrix(preds, gts, num_classes, ignore_label=ignore_label)

    Precision = TP / (TP + FP) * multi
    Recall = TP / (TP + FN) * multi
    Acc = TP.sum() / (TP + FN).sum() * multi
    F1 = (2.0 * TP / (2.0 * TP + FN + FP)) * multi

    F1 = np.around(np.nan_to_num(F1, nan=nan_to_num), dot)
    Acc = np.around(np.nan_to_num(Acc, nan=nan_to_num), dot)
    Precision = np.around(np.nan_to_num(Precision, nan=nan_to_num), dot)
    Recall = np.around(np.nan_to_num(Recall, nan=nan_to_num), dot)

    if task == "cls":
        AUC = cal_ROC_AUC(probs, gts, num_classes, ignore_label, multi=multi, dot=dot, mean_type=mean_type)[0]
        class_AUC = np.array([AUC[cid] for cid in range(num_classes)])

        if mean_type == "macro":
            mAUC = np.around(AUC["macro"], dot)
            mF1 = np.around(np.mean(F1), dot)
            mPrecision = np.around(np.mean(Precision), dot)
            mRecall = np.around(np.mean(Recall), dot)
        elif mean_type == "micro":
            mAUC = np.around(AUC["micro"], dot)
            mF1 = np.around(np.nan_to_num((2 * TP.sum() / (2 * TP + FN + FP).sum()) * multi, nan=nan_to_num), dot)
            mPrecision = np.around(np.nan_to_num(TP.sum() / (TP + FP).sum() * multi, nan=nan_to_num), dot)
            mRecall = np.around(np.nan_to_num(TP.sum() / (TP + FN).sum() * multi, nan=nan_to_num), dot)
        else:
            weight = weight / sum(weight)
            mAUC = np.around(sum(class_AUC * weight), dot)
            mF1 = np.around(sum(F1 * weight), dot)
            mPrecision = np.around(sum(Precision * weight), dot)
            mRecall = np.around(sum(Recall * weight), dot)

        metric = dict(
            mean_type=mean_type,
            AUC=dict(
                cls=class_AUC,
                mean=mAUC,
            ),
            F1=dict(
                cls=F1,
                mean=mF1,
            ),
            Precision=dict(
                cls=Precision,
                mean=mPrecision,
            ),
            Recall=dict(
                cls=Recall,
                mean=mRecall,
            ),
            Acc=dict(
                mean=np.around(Acc, dot),
            ),
        )
    else:
        seg_cal_auc = kwargs.get("seg_cal_auc", False)
        if seg_cal_auc:
            flatten_gts = gts.reshape(-1)
            flatten_probs = probs.swapaxes(1, 2).swapaxes(2, 3).reshape(-1, num_classes)
            ROC_AUC = cal_ROC_AUC(flatten_probs, flatten_gts, num_classes, ignore_label, multi=multi, dot=dot, mean_type=mean_type)[0]
            PR_AUC = cal_PR_AUC(flatten_probs, flatten_gts, num_classes, ignore_label, multi=multi, dot=dot, mean_type=mean_type)[0]
            class_ROC_AUC = np.array([ROC_AUC[cid] for cid in range(num_classes)])
            class_PR_AUC = np.array([PR_AUC[cid] for cid in range(num_classes)])

        IoU = TP / (TP + FP + FN) * multi
        IoU = np.around(np.nan_to_num(IoU, nan=nan_to_num), dot)
        # SP = TN / (TN + FP) * multi
        # SP = np.around(np.nan_to_num(SP, nan=nan_to_num), dot)
        # AC = (TP + TN) / (TP + FP + TN + FN) * multi
        # AC = np.around(np.nan_to_num(AC, nan=nan_to_num), dot)

        Dice = F1
        PA = Acc

        if mean_type == "macro":
            mIoU = np.around(np.mean(IoU), dot)
            mDice = np.around(np.mean(Dice), dot)
            mPrecision = np.around(np.mean(Precision), dot)
            mRecall = np.around(np.mean(Recall), dot)
            if seg_cal_auc:
                mROC_AUC = np.around(ROC_AUC["macro"], dot)
                mPR_AUC = np.around(PR_AUC["macro"], dot)
        elif mean_type == "micro":
            mIoU = np.around(np.nan_to_num((TP.sum() / (TP + FN + FP).sum()) * multi, nan=nan_to_num), dot)
            mDice = np.around(np.nan_to_num((2 * TP.sum() / (2 * TP + FN + FP).sum()) * multi, nan=nan_to_num), dot)
            mPrecision = np.around(np.nan_to_num(TP.sum() / (TP + FP).sum() * multi, nan=nan_to_num), dot)
            mRecall = np.around(np.nan_to_num(TP.sum() / (TP + FN).sum() * multi, nan=nan_to_num), dot)
            if seg_cal_auc:
                mROC_AUC = np.around(ROC_AUC["micro"], dot)
                mPR_AUC = np.around(PR_AUC["micro"], dot)
        else:
            weight = weight / sum(weight)
            mIoU = np.around(sum(F1 * weight), dot)
            mDice = np.around(sum(Dice * weight), dot)
            mPrecision = np.around(sum(Precision * weight), dot)
            mRecall = np.around(sum(Recall * weight), dot)
            if seg_cal_auc:
                mROC_AUC = np.around(sum(class_ROC_AUC * weight), dot)
                mPR_AUC = np.around(sum(class_PR_AUC * weight), dot)

        metric = dict(
            mean_type=mean_type,
            IoU=dict(
                cls=IoU,
                mean=mIoU,
            ),
            Dice=dict(
                cls=Dice,
                mean=mDice,
            ),
            Precision=dict(
                cls=Precision,
                mean=mPrecision,
            ),
            Recall=dict(
                cls=Recall,
                mean=mRecall,
            ),
            PA=dict(
                mean=np.around(PA, dot),
            ),
        )
        if seg_cal_auc:
            metric.update(dict(
                AUC_PR=dict(
                    cls=class_PR_AUC,
                    mean=mPR_AUC,
                ),
                AUC_ROC=dict(
                    cls=class_ROC_AUC,
                    mean=mROC_AUC,
                )
            ))
    return metric


def acc_i2cls(acc: dict, classes: List[str]):
    eval_acc = dict()
    for k, v in acc.items():
        if k == "mean_type":
            eval_acc[k] = v
        else:
            eval_acc[k] = dict()
            if "cls" in v:
                for i, now_class in enumerate(classes):
                    eval_acc[k][now_class] = v["cls"][i]
            if "mean" in v:
                eval_acc[k]["mean"] = v["mean"]
    return eval_acc
