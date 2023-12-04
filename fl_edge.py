#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: maojingxin
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from threading import Timer
from typing import List

import numpy as np
import pandas as pd
import socketio
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

root_dir = Path(__file__).resolve().parent  # ..TJDR-FL/
if str(root_dir) != sys.path[0]:
    sys.path[0] = str(root_dir)

from fl_client import Client
from gutils import Epoch, ID, Logger, FedAvg
from gutils import endecrypt, gutil, constants as C

ENABLE_ENCRYPT_EMIT = C.ENABLE_ENCRYPT_EMIT

torch.set_num_threads(C.NUM_THREADS)


class Edge(object):
    def __init__(self, edge_cfg: dict, edge_id: ID):
        self.edge_cfg = edge_cfg
        self.id = edge_id
        self._children = []

        self.num_edge_ep = self.edge_cfg[C.EPOCH]
        self.num_clients = self.edge_cfg[C.NUM_CLIENTS]
        self.frac_join = float(self.edge_cfg.get(C.FRAC_JOIN, 1.))
        self.num_choice_clients = int(self.num_clients * self.frac_join)
        self.fed_mode = self.edge_cfg.get(C.FED_MODE, C.FedAvg)
        self.fed_params = self.edge_cfg.get(C.FED_PARAMS, {})
        self.ep = Epoch(0, 0, None, num_edge_ep=self.num_edge_ep)
        self.now_tolerate = 0

        self.weights_dir = self.edge_cfg[C.DIR_WEIGHTS]
        self.best_weights_path = self.edge_cfg[C.PATH_BEST_WEIGHTS]
        self.record_file_dir = Path(self.edge_cfg[C.DIR_RECORD_FILE])
        self.eval_cfg = self.edge_cfg[C.EDGE_EVAL]
        self.eval_types = list(self.eval_cfg.keys())
        self.tolerate = self.edge_cfg.get("tolerate")
        self.dataset_name = self.edge_cfg[C.NAME_DATASET]
        self.task = C.TASK.get(self.dataset_name)
        self.classes = self.edge_cfg.get(C.CLASSES, C.DATASET_CLASSES[self.dataset_name])

        self.logger = gutil.init_log("edge#{}[{}]".format(self.id.nid, self.id.sid), self.edge_cfg[C.PATH_LOGFILE], debug=C.DEBUG)
        tbX_dir = Path(self.edge_cfg[C.DIR_TBX_LOGFILE])
        tbX_dir.mkdir(exist_ok=True, parents=True)
        self.tbX = SummaryWriter(logdir=tbX_dir)

        self.early_stop = False

        self.weights = None

        self.stats = {
            C.TRAIN: {C.ACC: [], C.LOSS: []},
            C.VALIDATION: {C.ACC: [], C.LOSS: []},
            C.TEST: {C.ACC: [], C.LOSS: []}
        }

        self.prev = {
            C.TRAIN: {C.LOSS: None, C.ACC: None},
            C.VALIDATION: {C.LOSS: None, C.ACC: None},
            C.TEST: {C.LOSS: None, C.ACC: None}
        }
        self.best = {
            C.TRAIN: {C.LOSS: None, C.ACC: None, C.WEIGHTS: None, C.EPOCH: Epoch(0, 0, None)},
            C.VALIDATION: {C.LOSS: None, C.ACC: None, C.WEIGHTS: None, C.EPOCH: Epoch(0, 0, None)},
            C.TEST: {C.LOSS: None, C.ACC: None, C.WEIGHTS: None, C.EPOCH: Epoch(0, 0, None)}
        }

        self.logger.info("=" * 100)
        self.logger.info(json.dumps(self.edge_cfg, indent=4))

    def add_child(self, child: Client):
        self._children.append(child)

    def get_weights(self):
        return self.weights

    def update_weights(self, new_weights, weight_type: str):
        if weight_type == C.EDGE:
            self.logger.debug("EdgeEpoch:{} | [Update Edge Weights with FedAggre Clients Weights Completed.]".format(self.ep.ce_to_str()))
        elif weight_type == C.CLOUD:
            self.logger.debug("EdgeEpoch:{} | [Update Edge Weights with Cloud Weights Completed.]".format(self.ep.ce_to_str()))
        else:
            raise ValueError(weight_type)
        self.weights = copy.deepcopy(new_weights)

    def update_children_weights(self, weight_type: str, new_weights=None, children: List[Client] = None, strict=True):
        _weights = self.weights if new_weights is None else new_weights
        _children = self._children if children is None else children
        assert not isinstance(_weights, list) or (isinstance(_weights, list) and len(_weights) == len(_children))
        for ci, client in enumerate(_children):
            if isinstance(_weights, list):
                client.update_weights(_weights[ci], weight_type=weight_type, strict=strict)
            else:
                client.update_weights(_weights, weight_type=weight_type, strict=strict)

    def get_stats(self):
        return {
            "edge_stats": self.stats,
        }

    def train(self, num_edge_ep: int = None):
        edge_train_loss = []
        num_edge_ep = self.num_edge_ep if num_edge_ep is None else num_edge_ep
        self.ep.reset(edge_epoch=True)
        # if self.ep.cloud_epoch > 1:
        #     for client in self._children:
        #         client.reset_lr_scheduler()
        for _ in tqdm(
                range(num_edge_ep),
                desc="Edge#{}[{}]-{}-CloudEpoch:{}".format(self.id.nid, self.id.sid, self.fed_mode, self.ep.cloud_epoch),
                unit="EdgeEpoch",
                leave=False,
                file=sys.stdout
        ):
            self.ep.edge_epoch_plus()
            self.logger.info("EdgeEpoch:{} | [Train] | Start ...".format(self.ep.ce_to_str()))

            clients_weights = []
            clients_train_loss = []
            clients_train_contrib = []

            choice_client_ids = np.random.choice(np.arange(len(self._children)), self.num_choice_clients, replace=False).tolist()
            # Init when edge_epoch==1
            if self.ep.edge_epoch == 1:
                self.update_children_weights(weight_type=C.CLOUD)

            for c_id in range(len(self._children)):
                client = self._children[c_id]

                if c_id in choice_client_ids:
                    client.ep.update(self.ep)
                    cpu_weights, loss, contrib = client.train()
                    clients_weights.append(cpu_weights)
                    clients_train_loss.append(loss)
                    clients_train_contrib.append(contrib)
                else:
                    clients_weights.append(copy.deepcopy(self.weights))
                    clients_train_loss.append(None)
                    clients_train_contrib.append(client.get_contribution(C.TRAIN, is_eval=False))

            edge_loss, _ = self.get_edge_loss_acc(C.TRAIN, clients_train_loss, None, clients_train_contrib, is_record=False)
            self.logger.info("EdgeEpoch:{} | [Train] | Contrib:{}".format(self.ep.ce_to_str(), clients_train_contrib))
            self.logger.info("EdgeEpoch:{} | [Train] | Loss:{:.4f}".format(self.ep.ce_to_str(), edge_loss))
            self.tbX.add_scalars("edge-train/loss", {"#{}[{}]".format(self.id.nid, self.id.sid): edge_loss}, self.ep.total_edge_ep())
            edge_train_loss.append(edge_loss)

            edge_fed_w = FedAvg(clients_weights, clients_train_contrib)
            self.logger.info("EdgeEpoch:{} | [Aggre] | Completed.".format(self.ep.ce_to_str()))

            self.update_weights(edge_fed_w, weight_type=C.EDGE)
            self.update_children_weights(weight_type=C.EDGE)

            self.save_ckpt(self.edge_cfg.get("save_ckpt_epoch"))
            self.logger.info("EdgeEpoch:{} | [Train] | Done.".format(self.ep.ce_to_str()))

            self.edge_eval()

            if self.early_stop:
                # self.fin_summary()
                self.logger.debug("EdgeEpoch:{} | Early Stopping.".format(self.ep.ce_to_str()))
                return self.weights, np.mean(edge_train_loss), self.get_contribution(C.TRAIN, is_eval=False)
            else:
                self.logger.debug("Start Next EdgeEpoch Training ...")

        return self.weights, np.mean(edge_train_loss), self.get_contribution(C.TRAIN, is_eval=False)

    def eval(self, eval_type: str, _type: str, vt_eval_client_num=1, **kwargs):
        assert eval_type in [C.TRAIN, C.VALIDATION, C.TEST]
        assert _type in [C.CLOUD, C.EDGE]

        choice_client_idx = np.random.choice(list(range(len(self._children))), vt_eval_client_num, replace=False)
        clients_cloud_eval_datas = dict()
        for idx, client in enumerate(self._children):
            eval_data = dict()
            if _type == C.CLOUD:
                eval_data[C.CLOUD_EVAL] = True if eval_type == C.TRAIN or idx in choice_client_idx else False
                client.cloud_eval(eval_type, eval_data)
            else:
                eval_data[C.EDGE_EVAL] = True if eval_type == C.TRAIN or idx in choice_client_idx else False
                client.edge_eval(eval_type, eval_data)
            clients_cloud_eval_datas["#{}[{}]".format(client.id.nid, client.id.sid)] = eval_data
        eval_loss, eval_acc, eval_contrib = self.aggre_eval(eval_type, clients_cloud_eval_datas, _type == C.EDGE or kwargs.get("is_multi_dataset", False))
        return eval_loss, eval_acc, eval_contrib

    def cloud_eval(self, eval_type, emit_data, vt_eval_client_num=1, **kwargs):
        assert eval_type in [C.TRAIN, C.VALIDATION, C.TEST], "eval_type:{} error".format(eval_type)
        if emit_data[eval_type] is True:
            self.logger.info("EdgeEpoch:{} | [Cloud-Eval-{}] | Start ...".format(self.ep.ce_to_str(), eval_type))
            eval_loss, eval_acc, eval_contrib = self.eval(eval_type, C.CLOUD, vt_eval_client_num, **kwargs)
            self.logger.info("EdgeEpoch:{} | [Cloud-Eval-{}] | Loss:{:.4f}".format(self.ep.ce_to_str(), eval_type, eval_loss))
            self.logger.info("EdgeEpoch:{} | [Cloud-Eval-{}] | Contrib:{}".format(self.ep.ce_to_str(), eval_type, eval_contrib))
            gutil.log_acc(logger=self.logger, acc=eval_acc, classes=self.classes)
            self.logger.info("EdgeEpoch:{} | [Cloud-Eval-{}] | Done.".format(self.ep.ce_to_str(), eval_type))
            emit_data[eval_type] = {
                C.LOSS: eval_loss,
                C.ACC: eval_acc,
                C.CONTRIB: sum(eval_contrib).item()
            }
        else:
            self.logger.info("The Current Edge is not selected for cloud evaluation. Skip [Cloud-Eval-{}].".format(eval_type))
            emit_data[eval_type] = {
                C.LOSS: None,
                C.ACC: None,
                C.CONTRIB: 0
            }

    def edge_eval(self, vt_eval_client_num=1):
        for eval_type in self.eval_types:
            if self.eval_cfg[eval_type][C.NUM] > 0 and self.ep.edge_epoch % self.eval_cfg[eval_type][C.NUM] == 0:
                self.logger.info("EdgeEpoch:{} | [Edge-Eval-{}] | Start ...".format(self.ep.ce_to_str(), eval_type))
                eval_loss, eval_acc, eval_contrib = self.eval(eval_type, C.EDGE, vt_eval_client_num)
                self.logger.info("EdgeEpoch:{} | [Edge-Eval-{}] | Loss:{:.4f}".format(self.ep.ce_to_str(), eval_type, eval_loss))
                self.logger.info("EdgeEpoch:{} | [Edge-Eval-{}] | Contrib:{}".format(self.ep.ce_to_str(), eval_type, eval_contrib))
                gutil.log_acc(logger=self.logger, acc=eval_acc, classes=self.classes)
                if self.tbX is not None:
                    self.tbX.add_scalars("edge-eval/loss", {"{}-#{}[{}]".format(eval_type, self.id.nid, self.id.sid): eval_loss}, self.ep.total_edge_ep())
                    for k, v in eval_acc.items():
                        if k == "mean_type":
                            continue
                        self.tbX.add_scalars(
                            "edge-eval/m{}".format(k), {"{}-#{}[{}]".format(eval_type, self.id.nid, self.id.sid): v["mean"]}, self.ep.total_edge_ep()
                        )
                        if self.task in [C.IMG_SEGMENTATION]:
                            for name, value in v.items():
                                self.tbX.add_scalars("edge-eval/{}/{}".format(k, name), {"{}-#{}[{}]".format(eval_type, self.id.nid, self.id.sid): value}, self.ep.total_edge_ep())
                        else:
                            self.tbX.add_scalars("edge-eval/m{}".format(k), {"{}-#{}[{}]".format(eval_type, self.id.nid, self.id.sid): v["mean"]}, self.ep.total_edge_ep())

                self.logger.info("EdgeEpoch:{} | [Edge-Eval-{}] | Done.".format(self.ep.ce_to_str(), eval_type))
                self.update_best(eval_type)
                tolerate_res = self.update_tolerate(eval_type)
                if isinstance(tolerate_res, bool):
                    self.early_stop = tolerate_res

    def get_contribution(self, contribution_type: str, is_eval: bool, return_contrib_type="sum"):
        assert return_contrib_type in [None, "sum", "avg"]
        contrib = []
        for client in self._children:
            contrib.append(client.get_contribution(contribution_type, is_eval))
        if return_contrib_type == "sum":
            return sum(contrib)
        elif return_contrib_type == "avg":
            return sum(contrib) / len(contrib)
        else:
            return contrib

    def get_edge_loss_acc(self, eval_type: str, client_losses: list, client_acc: List[dict] or None, client_contributions: list, is_record: bool):
        assert eval_type in [C.TRAIN, C.VALIDATION, C.TEST], "edge eval_type:{} error".format(eval_type)
        now_edge_loss = gutil.list_mean(client_losses, client_contributions)

        now_edge_acc = None
        now_acc_client_contributions = []
        if client_acc is not None:
            now_edge_acc = dict()
            metric_client_acc = dict()
            for c_acc, c_contrib in zip(client_acc, client_contributions):
                if c_acc is not None:
                    for k, v in c_acc.items():
                        if k == "mean_type":
                            if k not in now_edge_acc.keys():
                                now_edge_acc[k] = v
                        else:
                            if k in metric_client_acc:
                                metric_client_acc[k].append(v)
                            else:
                                metric_client_acc[k] = [v]
                    now_acc_client_contributions.append(c_contrib)

            for metric_type in metric_client_acc.keys():
                now_edge_acc[metric_type] = gutil.dict_list_mean(metric_client_acc[metric_type], now_acc_client_contributions)

        if is_record:
            self.stats[eval_type][C.LOSS].append(now_edge_loss)
            self.stats[eval_type][C.ACC].append(now_edge_acc)

        return now_edge_loss, now_edge_acc

    def aggre_eval(self, eval_type, client_update_datas, is_record=False, return_contrib_type: str = None):
        assert eval_type in [C.TRAIN, C.VALIDATION, C.TEST], "eval_type:{} error".format(eval_type)
        assert return_contrib_type in [None, "sum", "avg"]
        edge_loss, edge_acc = self.get_edge_loss_acc(
            eval_type,
            [client_data[eval_type][C.LOSS] for client_data in client_update_datas.values()],
            [client_data[eval_type][C.ACC] for client_data in client_update_datas.values()],
            [client_data[eval_type][C.CONTRIB] for client_data in client_update_datas.values()],
            is_record
        )
        contrib = [client_data[eval_type][C.CONTRIB] for client_data in client_update_datas.values()]
        if return_contrib_type == "sum":
            contrib = sum(contrib)
        elif return_contrib_type == "avg":
            contrib = sum(contrib) / len(contrib)
        return edge_loss, edge_acc, contrib

    def update_tolerate(self, now_type):
        self.logger.debug("tolerate:{}".format(self.tolerate))
        if self.tolerate is None:
            return None
        assert len(self.tolerate.keys()) == 1, "tolerate parameter must just have one"
        tolerate_type = list(self.tolerate.keys())[0]
        assert tolerate_type in [C.TRAIN, C.VALIDATION, C.TEST]
        if now_type == tolerate_type:
            tolerate_metric = self.tolerate[tolerate_type][C.METRIC]
            tolerate_num = self.tolerate[tolerate_type][C.NUM]

            now_stats = {
                C.LOSS: self.stats[tolerate_type][C.LOSS][-1],
                C.ACC: self.stats[tolerate_type][C.ACC][-1]
            }
            assert tolerate_metric == C.LOSS or (tolerate_metric[0] == "m" and tolerate_metric[1:] in now_stats[C.ACC].keys()), "metric_tolerate error:{}".format(tolerate_metric)
            if tolerate_metric == C.LOSS:
                delta = self.tolerate[tolerate_type].get("delta", 0)
                preLoss = self.prev[tolerate_type][C.LOSS]
                nowLoss = now_stats[C.LOSS]
                if preLoss and nowLoss - preLoss > -delta:
                    self.now_tolerate += 1
                else:
                    self.now_tolerate = 0
            else:
                delta = self.tolerate[tolerate_type].get("delta", 0)
                preAcc = self.prev[tolerate_type][C.ACC]
                nowAcc = now_stats[C.ACC]
                if preAcc and nowAcc[tolerate_metric[1:]]["mean"] - preAcc[tolerate_metric[1:]]["mean"] < delta:
                    self.now_tolerate += 1
                else:
                    self.now_tolerate = 0

            self.prev[tolerate_type][C.LOSS] = self.stats[tolerate_type][C.LOSS][-1]
            self.prev[tolerate_type][C.ACC] = self.stats[tolerate_type][C.ACC][-1]

            if self.now_tolerate >= tolerate_num > 0:
                self.logger.info("{}(metric:{},delta:{}) Early Stopping.".format(tolerate_type, tolerate_metric, delta))
                return True
            return False
        return None

    def update_best(self, best_type: str):
        self.logger.debug("best_type:{}".format(best_type))
        assert best_type in [C.TRAIN, C.VALIDATION, C.TEST], "best_type:{} error".format(best_type)
        now_edge_loss = self.stats[best_type][C.LOSS][-1]
        now_edge_acc = self.stats[best_type][C.ACC][-1]
        edge_metric = self.eval_cfg[best_type][C.METRIC]
        # init
        if self.best[best_type][C.ACC] is None:
            self.best[best_type][C.ACC] = now_edge_acc
        if self.best[best_type][C.LOSS] is None:
            self.best[best_type][C.LOSS] = now_edge_loss
        if self.best[best_type][C.WEIGHTS] is None:
            self.best[best_type][C.WEIGHTS] = copy.deepcopy(self.weights)
        if self.best[best_type][C.EPOCH].cloud_epoch == self.best[best_type][C.EPOCH].edge_epoch == 0:
            self.best[best_type][C.EPOCH] = copy.deepcopy(self.ep)
        # compare with edge_eval metric

        if isinstance(edge_metric, (list, tuple)):
            cur_value = 1
            cur_best_value = 1
            cloud_metric = list(set(edge_metric))
            for m in cloud_metric:
                assert m[0] == "m"
                metric_type = m[1:] if m[0] == "m" else m
                assert metric_type in now_edge_acc.keys()
                cur_value *= now_edge_acc[metric_type]["mean"]
                cur_best_value *= self.best[best_type][C.ACC][metric_type]["mean"]
            if cur_value > cur_best_value:
                self.best[best_type][C.LOSS] = now_edge_loss
                self.best[best_type][C.ACC] = now_edge_acc
                self.best[best_type][C.WEIGHTS] = copy.deepcopy(self.weights)
                self.best[best_type][C.EPOCH] = copy.deepcopy(self.ep)
        else:
            assert edge_metric[0] == "m" or edge_metric == C.LOSS
            metric_type = edge_metric[1:] if edge_metric[0] == "m" else edge_metric
            if metric_type in now_edge_acc.keys():
                if now_edge_acc[metric_type]["mean"] > self.best[best_type][C.ACC][metric_type]["mean"]:
                    self.best[best_type][C.LOSS] = now_edge_loss
                    self.best[best_type][C.ACC] = now_edge_acc
                    self.best[best_type][C.WEIGHTS] = copy.deepcopy(self.weights)
                    self.best[best_type][C.EPOCH] = copy.deepcopy(self.ep)
            else:
                if now_edge_loss < self.best[best_type][C.LOSS]:
                    self.best[best_type][C.LOSS] = now_edge_loss
                    self.best[best_type][C.ACC] = now_edge_acc
                    self.best[best_type][C.WEIGHTS] = copy.deepcopy(self.weights)
                    self.best[best_type][C.EPOCH] = copy.deepcopy(self.ep)

    def save_ckpt(self, save_ckpt_epoch: int):
        if save_ckpt_epoch is not None and isinstance(save_ckpt_epoch, int) and save_ckpt_epoch > 0:
            if self.ep.edge_epoch % save_ckpt_epoch == 0:
                ckpt_record_path = Path(self.weights_dir, "record", "edge#{}[{}]".format(self.id.nid, self.id.sid), "ep[{}].pt".format(self.ep.ce_to_str()))
                ckpt_record_path.parent.mkdir(exist_ok=True, parents=True)
                gutil.save_weights(self.weights, ckpt_record_path)
                self.logger.info("EdgeEpoch:{} | Save Record EdgeWeights : {}".format(self.ep.ce_to_str(), ckpt_record_path))

    def record_metric(self, record_file_dir: Path, record_type: str, record_interval: int):
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
            "edge#{}[{}]".format(self.id.nid, self.id.sid): {
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
            df.to_csv(record_file_dir / "edge#{}_best_{}.csv".format(self.id.nid, record_type), index=False)

    def fin_summary(self, edge_eval_types=None):
        if edge_eval_types is None:
            edge_eval_types = self.eval_types
        self.logger.info("Federated Learning Summary ...")
        for edge_eval_type in edge_eval_types:
            assert edge_eval_type in [C.TRAIN, C.VALIDATION, C.TEST], "edge eval type:{} error".format(edge_eval_types)
            self.record_metric(self.record_file_dir, edge_eval_type, self.eval_cfg[edge_eval_type][C.NUM])
            self.record_best_acc(self.record_file_dir, edge_eval_type)
            now_best = self.best[edge_eval_type]
            self.logger.info("[Edge-Summary-{}] | Metrics:{}".format(edge_eval_type, self.eval_cfg[edge_eval_type].get(C.METRIC)))
            self.logger.info("[Edge-Summary-{}] | Best EdgeEpoch:{}".format(edge_eval_type, now_best[C.EPOCH].ce_to_str()))
            self.logger.info("[Edge-Summary-{}] | Best Loss:{}".format(edge_eval_type, now_best[C.LOSS]))
            gutil.log_acc(logger=self.logger, acc=now_best[C.ACC], classes=self.classes)
            if now_best[C.WEIGHTS]:
                self.logger.info("[Edge-Summary-{}] | Save Best EdgeWeights : {}".format(edge_eval_type, self.best_weights_path[edge_eval_type]))
                gutil.save_weights(now_best[C.WEIGHTS], self.best_weights_path[edge_eval_type])
        for client in self._children:
            client.fin()
        self.tbX.close()


class HeartBeatWorker(object):
    def __init__(self, sio, interval=60):
        self.sio = sio
        self.interval = interval
        self.t = None
        self.run = True

    def work(self):
        if self.run:
            self.sio.emit("heartbeat")
            self.t = Timer(self.interval, function=self.work)
            self.t.start()

    def stop(self):
        self.run = False
        if self.t:
            self.t.cancel()
            del self.t


class EdgeDevice(Edge):
    def __init__(self, edge_cfg: dict, edge_id: ID, clients_cfg: List[dict], clients_id: List[ID], cloud_host: str = None, cloud_port: int = None):
        super().__init__(edge_cfg, edge_id)
        self.clients_cfg = clients_cfg
        self.clients_id = clients_id
        assert len(self.clients_cfg) == len(self.clients_id)
        self.cloud_host = edge_cfg.get(C.HOST, "127.0.0.1") if cloud_host is None else cloud_host
        self.cloud_port = edge_cfg.get(C.PORT, "9191") if cloud_port is None else cloud_port
        self.pubkey, self.privkey = endecrypt.newkey(512)
        self.socketio = socketio.Client(logger=False, engineio_logger=False)
        self.cloud_pubkey = None
        self.register_handles()
        self.socketio.connect("ws://{}:{}".format(self.cloud_host, self.cloud_port))
        self.ignore_loadavg = self.edge_cfg.get("ignore_loadavg", True)

        self.tmp_weights_dir = Path(self.weights_dir) / "tmp"
        self.tmp_weights_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_weights_path = self.tmp_weights_dir / "edge#{}[{}].pkl".format(self.id.nid, self.id.sid)

        self.heartbeat_worker = HeartBeatWorker(sio=self.socketio, interval=30)

    def wakeup(self):
        self.logger.info("Edge#{}[{}] connect {}:{}".format(self.id.nid, self.id.sid, self.cloud_host, self.cloud_port))
        emit_data = {"edge_pubkey": {"n": str(self.pubkey.n), "e": str(self.pubkey.e)}, C.FID: self.id.fid}
        self.socketio.emit("edge_wakeup", emit_data)
        self.socketio.start_background_task(target=self.heartbeat_worker.work)
        self.socketio.wait()

    def rsaEncrypt(self, data, dumps=True, enable=True):
        """
        rsaEncrypt data
        :param data: the data will encrypt
        :param dumps: default is True , whether data need to serialize before encrypt
        :param enable: default is True, enable rsaEncrypt
        :return:
        """
        if not enable:
            return data
        if self.cloud_pubkey is None:
            retry = 10
            while retry > 0:
                self.socketio.emit("get_cloud_pubkey")
                time.sleep(3)
                if self.cloud_pubkey is not None:
                    break
                retry -= 1
        res_data = endecrypt.rsaEncrypt(self.cloud_pubkey, data, dumps)
        return res_data

    def rsaDecrypt(self, data, loads=True, enable=True):
        """
        rsaDecrypt data
        :param data: the data will decrypt
        :param loads: default is True , whether decrypt data need to deserialize
        :param enable: default is True, enable rsaDecrypt
        :return:
        """
        if not enable:
            return data
        res_data = endecrypt.rsaDecrypt(self.privkey, data, loads)
        return res_data

    def register_handles(self):
        @self.socketio.event
        def connect():
            self.logger.info("Connect")

        @self.socketio.event
        def connect_error(e):
            self.logger.error(e)

        @self.socketio.event
        def disconnect():
            self.logger.info("Close Connect.")
            self.socketio.disconnect()
            pid = self.edge_cfg[C.PID] if os.getpid() == self.edge_cfg[C.PID] else os.getpid()
            gutil.kill(pid)

        @self.socketio.on("reconnect")
        def reconnect():
            self.logger.info("Re Connect")
            self.wakeup()

        @self.socketio.on("re_heartbeat")
        def re_heartbeat():
            self.logger.debug("HeartBeat Complete. Keep Connecting")

        @self.socketio.on("get_edge_pubkey")
        def send_edge_pubkey():
            emit_data = {"edge_pubkey": {"n": str(self.pubkey.n), "e": str(self.pubkey.e)}, C.FID: self.id.fid}
            self.socketio.emit("edge_pubkey", emit_data)

        @self.socketio.on("cloud_pubkey")
        def get_cloud_pubkey(data):
            self.cloud_pubkey = endecrypt.toPubkey(int(data["cloud_pubkey"]["n"]), int(data["cloud_pubkey"]["e"]))

        @self.socketio.on("edge_init")
        def client_init():
            self.logger.info("Clients Init ...")
            for client_cfg, client_id in zip(self.clients_cfg, self.clients_id):
                client = Client(client_cfg, client_id, last_epoch=-1, sio=self.socketio)
                assert client.id in self.id.children_id
                client.id.set_parent_id(self.id)
                self.add_child(client)
            self.logger.info("Clients Init Completed.")
            self.logger.info("Clients Join Frac:{} , the Number of Join Clients:{}".format(self.frac_join, self.num_choice_clients))

            emit_data = {C.NAME_DATASET: self.dataset_name}
            self.socketio.emit("edge_ready", emit_data)

        @self.socketio.on("edge_check_resource")
        def edge_check_resource(data):
            self.logger.debug("Start Check Resource ...")
            self.logger.debug("before decrypt data={}".format(data))
            data = self.rsaDecrypt(data, enable=ENABLE_ENCRYPT_EMIT)
            self.logger.debug("decrypt data={}".format(data))

            self.cloud_pubkey = endecrypt.toPubkey(int(data["cloud_pubkey"]["n"]), int(data["cloud_pubkey"]["e"]))

            is_halfway = data.get("halfway", False)
            if self.ignore_loadavg:
                self.logger.debug("Ignore Loadavg")
                loadavg = 0.15
            else:
                loadavg_data = {}
                with open("/proc/loadavg") as f:
                    loadavg_raw_data = f.read().split()
                    loadavg_data["loadavg_1min"] = loadavg_raw_data[0]
                    loadavg_data["loadavg_5min"] = loadavg_raw_data[1]
                    loadavg_data["loadavg_15min"] = loadavg_raw_data[2]
                    loadavg_data["loadavg_rate"] = loadavg_raw_data[3]
                    loadavg_data["last_pid"] = loadavg_raw_data[4]

                loadavg = loadavg_data["loadavg_15min"]
                self.logger.debug("Loadavg : {}".format(loadavg))
            emit_data = {"loadavg": loadavg}
            if is_halfway:
                self.socketio.emit("halfway_edge_check_resource_complete", self.rsaEncrypt(emit_data, enable=ENABLE_ENCRYPT_EMIT))
            else:
                self.socketio.emit("edge_check_resource_complete", self.rsaEncrypt(emit_data, enable=ENABLE_ENCRYPT_EMIT))
            self.logger.debug("Check Resource Completed.")

        @self.socketio.on("edge_train")
        def edge_train(data):
            self.logger.debug("Edge Train Receiving ...")

            self.logger.debug("before decrypt data={}".format(data))
            data = self.rsaDecrypt(data, enable=ENABLE_ENCRYPT_EMIT)
            self.logger.debug("decrypt data={}".format(data))

            cloud_ep = Epoch(**data["ep"])
            self.ep.update(cloud_ep)

            self.logger.debug("Edge Train Start ...")

            # update with cloud weights
            if "weights" in data:
                self.logger.debug("Receive Weights ...")
                is_multi_weights = data.get("multi_weights")
                weights = gutil.pickle2obj(data["weights"])
                if is_multi_weights and self.fed_mode in [C.alphaFed]:
                    weights = weights[self.id.fid]
                gutil.obj2pickle(weights, self.tmp_weights_path)
                self.update_weights(weights, weight_type=C.CLOUD)
                self.logger.debug("Update Weights Completed")

            # train num_lep
            cpu_weights, loss, contrib = self.train()

            pickle_weights = gutil.obj2pickle(cpu_weights, self.tmp_weights_path)  # pickle weights path

            emit_data = {
                "ep": self.ep.serialize(),
                "weights": pickle_weights,
                C.TRAIN_LOSS: loss,
                C.TRAIN_ACC: None,
                C.TRAIN_CONTRIB: contrib.item(),
            }

            self.logger.debug("Edge Train Completed.")
            self.logger.debug("Emit Updates To Cloud ...")
            self.socketio.emit("edge_update_complete", self.rsaEncrypt(emit_data, enable=ENABLE_ENCRYPT_EMIT))
            self.logger.debug("Emit Updates Completed.")

        @self.socketio.on("eval_with_cloud_weights")
        def eval_with_cloud_weights(data):
            self.logger.debug("Receive FedAggre Weights From Cloud ...")

            self.logger.debug("before decrypt data={}".format(data))
            data = self.rsaDecrypt(data, enable=ENABLE_ENCRYPT_EMIT)
            self.logger.debug("decrypt data={}".format(data))

            cloud_ep = Epoch(**data["ep"])
            is_multi_weights = data["multi_weights"]

            assert cloud_ep.cloud_epoch == self.ep.cloud_epoch, "cloud_ep:{}!=now edge cloud_ep:{}".format(cloud_ep.cloud_epoch, self.ep.cloud_epoch)

            cloud_weights = gutil.pickle2obj(data["weights"])
            if is_multi_weights and self.fed_mode in [C.alphaFed]:
                cloud_weights = cloud_weights[self.id.fid]
            gutil.obj2pickle(cloud_weights, self.tmp_weights_path)  # save global weights to local weights path

            self.update_weights(cloud_weights, weight_type=C.CLOUD)
            self.update_children_weights(weight_type=C.CLOUD)
            self.logger.debug("Update FedAggre Weights Completed.")

            emit_data = {}

            if C.TRAIN in data:
                emit_data[C.TRAIN] = data[C.TRAIN]
                self.cloud_eval(C.TRAIN, emit_data, is_multi_dataset=is_multi_weights)
                if is_multi_weights:
                    self.update_best(C.TRAIN)

            if C.VALIDATION in data:
                emit_data[C.VALIDATION] = data[C.VALIDATION]
                self.cloud_eval(C.VALIDATION, emit_data, is_multi_dataset=is_multi_weights)
                if is_multi_weights:
                    self.update_best(C.VALIDATION)

            if C.TEST in data:
                emit_data[C.TEST] = data[C.TEST]
                self.cloud_eval(C.TEST, emit_data, is_multi_dataset=is_multi_weights)
                if is_multi_weights:
                    self.update_best(C.TEST)

            self.socketio.emit("eval_with_cloud_weights_complete", self.rsaEncrypt(emit_data, enable=ENABLE_ENCRYPT_EMIT))

        @self.socketio.on("fin")
        def fin():
            self.fin_summary()
            os.remove(self.tmp_weights_path)
            emit_data = {C.FID: self.id.fid}
            self.heartbeat_worker.stop()
            self.socketio.emit("edge_fin", self.rsaEncrypt(emit_data, enable=ENABLE_ENCRYPT_EMIT))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--edge_config_path", type=str, dest="edge_config_path", required=True, help="path of edge config")
    parser.add_argument("-c", "--client_config_paths", type=str, dest="client_config_paths", nargs="+", required=True, help="paths of client config")
    parser.add_argument("-g", "--gpu", default="", dest="gpu", type=str, help="optional,specified gpu to run", required=False)
    parser.add_argument("--host", type=str, dest="host", help="optional cloud host , 'configs/base_config.yaml' has inited host")
    parser.add_argument("--port", type=int, dest="port", help="optional cloud port , 'configs/base_config.yaml' has inited port")

    args = parser.parse_args()
    edge_config_path = args.edge_config_path
    client_config_paths = args.client_config_paths
    host, port = args.host, args.port
    logger = Logger()

    assert Path(edge_config_path).exists(), "{} not exist".format(edge_config_path)
    for client_config_path in client_config_paths:
        assert Path(client_config_path).exists(), "{} not exist".format(client_config_path)

    edge_cfg = gutil.load_json(edge_config_path)
    seed = edge_cfg.get(C.SEED, C.DEFAULT_SEED)
    run_seed = edge_cfg.get(C.RUN_SEED, seed)
    logger.info("run_seed:{}".format(run_seed))
    gutil.set_all_seed(run_seed)
    gpu = args.gpu if len(args.gpu) != 0 else str(edge_cfg.get("gpu", "0"))

    edge_cfg[C.PID] = os.getpid()
    gutil.write_json(edge_config_path, edge_cfg, mode="w+", indent=4)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    clients_cfg = [gutil.load_json(client_config_path) for client_config_path in client_config_paths]

    edge_id, clients_id = gutil.build_id_connect(edge_cfg, clients_cfg)
    edge = EdgeDevice(edge_cfg, edge_id, clients_cfg, clients_id, host, port)
    edge.logger.info("gpu-id:{}".format(gpu))
    edge.wakeup()
