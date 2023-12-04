#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: maojingxin
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

root_dir = Path(__file__).resolve().parent  # ..TJDR-FL/
if str(root_dir) != sys.path[0]:
    sys.path[0] = str(root_dir)

from components import Models
from gutils import Epoch, ID
from gutils import constants as C, gutil


class Client(object):
    def __init__(self, config: dict, client_id: ID, last_epoch=-1, sio=None):
        self.config = config
        self.id = client_id
        self.sio = sio

        self.num_client_ep = self.config[C.EPOCH]
        self.fed_mode = self.config[C.FED_MODE]
        self.ep = Epoch(0, 0, 0, num_client_ep=self.num_client_ep)
        self.dataset_name = self.config[C.NAME_DATASET]
        self.classes = self.config.get(C.CLASSES, C.DATASET_CLASSES[self.dataset_name])
        self.task = C.TASK.get(self.dataset_name)
        self.eval_cfg = self.config.get(C.CLIENT_EVAL)
        self.record_file_dir = Path(self.config[C.DIR_RECORD_FILE]) if C.DIR_RECORD_FILE in self.config else None

        self.logger = gutil.init_log("client#{}[{}]".format(self.id.nid, self.id.sid), self.config[C.PATH_LOGFILE], debug=C.DEBUG)
        self.logger.info("=" * 100)
        self.logger.info(json.dumps(self.config, indent=4))

        if self.task in [C.IMG_CLASSIFICATION]:
            self.model = Models.cls_model_gen(
                self.config[C.NAME_MODEL],
                ptr_weights=self.config.get(C.PTR_WEIGHTS),
                dropout=self.config.get("dropout", 0.2)
            )(config, self.id, self.logger, only_init_net=False, last_epoch=last_epoch)
        elif self.task in [C.IMG_SEGMENTATION]:
            self.model = Models.seg_model_gen(self.config[C.NAME_MODEL], ptr_weights=self.config.get(C.PTR_WEIGHTS))(config, self.id, self.logger, only_init_net=False, last_epoch=last_epoch)
        else:
            raise ValueError(self.dataset_name, self.task)
        # self.model = getattr(Models, self.config[C.NAME_MODEL])(config, self.id, self.logger, only_init_net=False, last_epoch=last_epoch)

        self.stats = {
            C.TRAIN: {C.ACC: [], C.LOSS: []},
            C.VALIDATION: {C.ACC: [], C.LOSS: []},
            C.TEST: {C.ACC: [], C.LOSS: []}
        }

        self.best = {
            C.TRAIN: {C.LOSS: None, C.ACC: None, C.WEIGHTS: None, C.EPOCH: Epoch(0, 0, 0)},
            C.VALIDATION: {C.LOSS: None, C.ACC: None, C.WEIGHTS: None, C.EPOCH: Epoch(0, 0, 0)},
            C.TEST: {C.LOSS: None, C.ACC: None, C.WEIGHTS: None, C.EPOCH: Epoch(0, 0, 0)}
        }

    def get_weights(self, to_cpu=True, keep_vars=True):
        return self.model.get_weights(to_cpu, keep_vars=keep_vars)

    def get_optim_weights(self):
        return self.model.get_optim_state_dict()

    def update_weights(self, new_weights, weight_type: str, strict: bool = True):
        if weight_type == C.EDGE:
            self.logger.debug("ClientEpoch:{} | [Update Client Weights with Edge  Weights Completed.]".format(self.ep.cec_to_str()))
        elif weight_type == C.CLOUD:
            self.logger.debug("ClientEpoch:{} | [Update Client Weights with Cloud Weights Completed.]".format(self.ep.cec_to_str()))
        else:
            raise ValueError(weight_type)
        self.model.set_weights(copy.deepcopy(new_weights), strict)

    def update_optim_weights(self, params):
        self.model.set_optim_state_dict(params)

    def get_contribution(self, contribution_type: str, is_eval: bool):
        return self.model.get_contrib(contribution_type, is_eval)

    def get_dataset_len(self, _type: str):
        return self.model.get_dataset_len(_type)

    def train(self, num_client_ep: int = None, **kwargs):
        num_client_ep = self.num_client_ep if num_client_ep is None else num_client_ep
        self.ep.reset(client_epoch=True)

        client_train_loss = []
        res = None
        for _ in range(num_client_ep):
            res = self.model.train(1, self.ep, sio=self.sio, **kwargs)
            client_train_loss += res["losses"]
            if self.eval_cfg is not None:
                for eval_type, eval_data in self.eval_cfg.items():
                    if eval_data[C.NUM] > 0 and self.ep.client_epoch % eval_data[C.NUM] == 0:
                        self.client_eval(eval_type)
        loss = np.mean(client_train_loss) if client_train_loss else np.nan
        contrib = self.get_contribution(C.TRAIN, is_eval=False)
        return self.get_weights(to_cpu=True), loss, contrib

    def reset_lr_scheduler(self):
        self.model.reset_lr_scheduler()

    @torch.no_grad()
    def eval(self, eval_type, is_record=False):
        loss, acc = self.model.eval(eval_type)
        contrib = self.get_contribution(eval_type, is_eval=True)
        if is_record:
            self.stats[eval_type][C.LOSS].append(loss)
            self.stats[eval_type][C.ACC].append(acc)
            self.update_best(eval_type)
        return loss, acc, contrib

    def update_best(self, best_type: str):
        self.logger.debug("best_type:{}".format(best_type))
        assert best_type in [C.TRAIN, C.VALIDATION, C.TEST], "best_type:{} error".format(best_type)
        now_client_loss = self.stats[best_type][C.LOSS][-1]
        now_client_acc = self.stats[best_type][C.ACC][-1]
        client_metric = self.eval_cfg[best_type].get(C.METRIC)

        if client_metric:
            # init
            if self.best[best_type][C.ACC] is None:
                self.best[best_type][C.ACC] = now_client_acc
            if self.best[best_type][C.LOSS] is None:
                self.best[best_type][C.LOSS] = now_client_loss
            if self.best[best_type][C.EPOCH].cloud_epoch == self.best[best_type][C.EPOCH].edge_epoch == self.best[best_type][C.EPOCH].client_epoch == 0:
                self.best[best_type][C.EPOCH] = copy.deepcopy(self.ep)
            # if self.best[best_type][C.WEIGHTS] is None:
            #     self.best[best_type][C.WEIGHTS] = copy.deepcopy(self.weights)
            # compare with client_eval metric

            if isinstance(client_metric, (list, tuple)):
                cur_value = 1
                cur_best_value = 1
                cloud_metric = list(set(client_metric))
                for m in cloud_metric:
                    assert m[0] == "m"
                    metric_type = m[1:] if m[0] == "m" else m
                    assert metric_type in now_client_acc.keys()
                    cur_value *= now_client_acc[metric_type]["mean"]
                    cur_best_value *= self.best[best_type][C.ACC][metric_type]["mean"]
                if cur_value > cur_best_value:
                    self.best[best_type][C.LOSS] = now_client_loss
                    self.best[best_type][C.ACC] = now_client_acc
                    self.best[best_type][C.EPOCH] = copy.deepcopy(self.ep)
            else:
                assert client_metric[0] == "m" or client_metric == C.LOSS
                metric_type = client_metric[1:] if client_metric[0] == "m" else client_metric
                if metric_type in now_client_acc.keys():
                    if now_client_acc[metric_type]["mean"] > self.best[best_type][C.ACC][metric_type]["mean"]:
                        self.best[best_type][C.LOSS] = now_client_loss
                        self.best[best_type][C.ACC] = now_client_acc
                        # self.best[best_type][C.WEIGHTS] = copy.deepcopy(self.weights)
                        self.best[best_type][C.EPOCH] = copy.deepcopy(self.ep)
                else:
                    if now_client_loss < self.best[best_type][C.LOSS]:
                        self.best[best_type][C.LOSS] = now_client_loss
                        self.best[best_type][C.ACC] = now_client_acc
                        # self.best[best_type][C.WEIGHTS] = copy.deepcopy(self.weights)
                        self.best[best_type][C.EPOCH] = copy.deepcopy(self.ep)

    def record_metric(self, record_file_dir: Path, record_type: str, record_interval: int):
        if record_file_dir is None:
            return
        record_acc = dict()
        record_epoch = []
        record_file_dir.mkdir(exist_ok=True, parents=True)
        record_file_path = record_file_dir / "{}.json".format(record_type)
        mean_type = "mean"
        for i, acc in enumerate(self.stats[record_type][C.ACC]):
            for k, v in acc.items():
                if k == "mean_type":
                    mean_type = v
                    continue
                if self.task in [C.IMG_SEGMENTATION]:
                    record_acc_val = v
                else:
                    record_acc_val = v["mean"]
                if k in record_acc:
                    record_acc[k].append(record_acc_val)
                else:
                    record_acc[k] = [record_acc_val]
            record_epoch.append(record_interval * (i + 1))
        record_json = gutil.load_json(record_file_path)
        record = {
            "client#{}[{}]".format(self.id.nid, self.id.sid): {
                "epoch": record_epoch,
                "loss": self.stats[record_type][C.LOSS],
                "acc": record_acc,
                "mean_type": mean_type
            }
        }
        record_json.update(record)
        gutil.write_json(record_file_path, record_json, mode="w+", indent=4)

    def record_best_acc(self, record_file_dir, record_type: str):
        if isinstance(record_file_dir, str):
            record_file_dir = Path(record_file_dir)
        best_acc = self.best[record_type][C.ACC]
        if isinstance(best_acc, dict):
            mean_type = "mean"
            metric_val = dict()
            for k, v in best_acc.items():
                if k == "mean_type":
                    mean_type = v
                    continue
                if k == "Acc" or k == "PA":
                    metric_val[k] = v
                else:
                    metric_val["{}_{}".format(mean_type, k)] = v["mean"]
            df = pd.DataFrame(metric_val)
            record_file_dir.mkdir(exist_ok=True, parents=True)
            df.to_csv(record_file_dir / "client#{}_best_{}.csv".format(self.id.nid, record_type), index=False)

    def client_eval(self, eval_type: str, emit_data: dict = None):
        assert eval_type in [C.TRAIN, C.VALIDATION, C.TEST], "eval_type:{} error".format(eval_type)
        self.logger.info("ClientEpoch:{} | [Client-Eval-{}] | Start ...".format(self.ep.cec_to_str(), eval_type))
        loss, acc, contrib = self.eval(eval_type, is_record=True)
        if emit_data is not None:
            emit_data[eval_type] = {
                C.LOSS: loss,
                C.ACC: acc,
                C.CONTRIB: contrib
            }
        self.logger.info("ClientEpoch:{} | [Client-Eval-{}] | Loss:{:.4f}".format(self.ep.cec_to_str(), eval_type, loss))
        gutil.log_acc(logger=self.logger, acc=acc, classes=self.classes)
        self.logger.info("ClientEpoch:{} | [Client-Eval-{}] | Done.".format(self.ep.cec_to_str(), eval_type))

    def edge_eval(self, eval_type: str, emit_data):
        assert eval_type in [C.TRAIN, C.VALIDATION, C.TEST], "eval_type:{} error".format(eval_type)
        if emit_data[C.EDGE_EVAL] is True:
            self.logger.info("EdgeEpoch:{} | [Edge-Eval-{}] | Start ...".format(self.ep.ce_to_str(), eval_type))
            loss, acc, contrib = self.eval(eval_type)
            emit_data[eval_type] = {
                C.LOSS: loss,
                C.ACC: acc,
                C.CONTRIB: contrib
            }
            self.logger.info("EdgeEpoch:{} | [Edge-Eval-{}] | Loss:{:.4f}".format(self.ep.ce_to_str(), eval_type, loss))
            gutil.log_acc(logger=self.logger, acc=acc, classes=self.classes)
            self.logger.info("EdgeEpoch:{} | [Edge-Eval-{}] | Done.".format(self.ep.ce_to_str(), eval_type))
        else:
            self.logger.info("The Current Client is not selected for edge evaluation. Skip [Edge-Eval-{}].".format(eval_type))
            emit_data[eval_type] = {
                C.LOSS: None,
                C.ACC: None,
                C.CONTRIB: 0
            }

    def cloud_eval(self, eval_type: str, emit_data):
        assert eval_type in [C.TRAIN, C.VALIDATION, C.TEST], "eval_type:{} error".format(eval_type)
        if emit_data[C.CLOUD_EVAL] is True:
            self.logger.info("CloudEpoch:{} | [Cloud-Eval-{}] | Start ...".format(self.ep.c_to_str(), eval_type))
            loss, acc, contrib = self.eval(eval_type)
            emit_data[eval_type] = {
                C.LOSS: loss,
                C.ACC: acc,
                C.CONTRIB: contrib
            }
            self.logger.info("CloudEpoch:{} | [Cloud-Eval-{}] | Loss:{:.4f}".format(self.ep.c_to_str(), eval_type, loss))
            gutil.log_acc(logger=self.logger, acc=acc, classes=self.classes)
            self.logger.info("CloudEpoch:{} | [Cloud-Eval-{}] | Done.".format(self.ep.c_to_str(), eval_type))
        else:
            self.logger.info("The Current Client is not selected for cloud evaluation. Skip [Cloud-Eval-{}].".format(eval_type))
            emit_data[eval_type] = {
                C.LOSS: None,
                C.ACC: None,
                C.CONTRIB: 0
            }

    def fin(self):
        if self.eval_cfg:
            for client_eval_type in self.eval_cfg.keys():
                assert client_eval_type in [C.TRAIN, C.VALIDATION, C.TEST], "client eval type:{} error".format(client_eval_type)
                self.record_metric(self.record_file_dir, client_eval_type, self.eval_cfg[client_eval_type][C.NUM])
                self.record_best_acc(self.record_file_dir, client_eval_type)
                now_best = self.best[client_eval_type]
                self.logger.info("[Client-Summary-{}] | Metrics:{}".format(client_eval_type, self.eval_cfg[client_eval_type].get(C.METRIC)))
                self.logger.info("[Client-Summary-{}] | Best ClientEpoch:{}".format(client_eval_type, now_best[C.EPOCH].cec_to_str()))
                self.logger.info("[Client-Summary-{}] | Best Loss:{}".format(client_eval_type, now_best[C.LOSS]))
                gutil.log_acc(logger=self.logger, acc=now_best[C.ACC], classes=self.classes)
                # if now_best[C.WEIGHTS]:
                #     self.logger.info("[Client-Summary-{}] | Save Best EdgeWeights : {}".format(client_eval_type, self.best_weights_path[client_eval_type]))
                #     gutil.save_weights(now_best[C.WEIGHTS], self.best_weights_path[client_eval_type])
        self.logger.info("Federated Learning Fin.")
