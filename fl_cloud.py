#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: maojingxin
"""
import argparse
import copy
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from flask import Flask, request, render_template, redirect
from flask_socketio import SocketIO, emit, disconnect
from tensorboardX import SummaryWriter
from tqdm import tqdm

root_dir = Path(__file__).resolve().parent  # ..TJDR-FL/
if str(root_dir) != sys.path[0]:
    sys.path[0] = str(root_dir)

from components import Models
from fl_edge import Edge
from gutils import Epoch, ID, Logger, FedAvg, alphaFedAvg
from gutils import endecrypt, gutil, constants as C

ENABLE_ENCRYPT_EMIT = C.ENABLE_ENCRYPT_EMIT

torch.set_num_threads(C.NUM_THREADS)


class Cloud(object):
    def __init__(self, config: dict, cloud_id: ID):
        self.config = config
        self.id = cloud_id
        self._children = []

        self.num_cloud_ep = self.config[C.EPOCH]
        self.fed_mode = self.config.get(C.FED_MODE, C.FedAvg)
        self.fed_params = self.config.get(C.FED_PARAMS)
        self.ep = Epoch(0, None, None, num_cloud_ep=self.num_cloud_ep)
        self.now_tolerate = 0

        self.weights_dir = self.config[C.DIR_WEIGHTS]
        self.best_weights_path = self.config[C.PATH_BEST_WEIGHTS]
        self.record_file_dir = self.config[C.DIR_RECORD_FILE]
        self.init_weights_path = self.config.get(C.PATH_INIT_WEIGHTS)
        self.eval_cfg = self.config[C.CLOUD_EVAL]
        self.eval_types = list(self.eval_cfg.keys())
        self.tolerate = self.config.get("tolerate")

        self.dataset_name = self.config[C.NAME_DATASET]
        self.multi_dataset = self.config.get(C.MULTI_DATASET, False)
        if isinstance(self.dataset_name, list):
            self.task = list(set([C.TASK.get(d_n) for d_n in self.dataset_name]))
            assert len(self.task) == 1, "multi dataset: task: {} error.".format(self.task)
            self.task = self.task[0]
            self.classes = self.config.get(C.CLASSES, C.DATASET_CLASSES[self.dataset_name[0]])
        else:
            self.task = C.TASK[self.dataset_name]
            self.classes = self.config.get(C.CLASSES, C.DATASET_CLASSES[self.dataset_name])

        self.logger = gutil.init_log("cloud", self.config[C.PATH_LOGFILE], debug=C.DEBUG)
        self.tbX_dir = Path(self.config[C.DIR_TBX_LOGFILE])
        self.tbX_dir.mkdir(exist_ok=True, parents=True)
        self.tbX = SummaryWriter(logdir=self.tbX_dir)

        self.logger.info("=" * 100)
        self.logger.info(json.dumps(self.config, indent=4))

        self.fin = False

        self.weights = self.get_init_weights()

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
            C.TRAIN: {C.LOSS: None, C.ACC: None, C.WEIGHTS: None, C.EPOCH: Epoch(0, None, None)},
            C.VALIDATION: {C.LOSS: None, C.ACC: None, C.WEIGHTS: None, C.EPOCH: Epoch(0, None, None)},
            C.TEST: {C.LOSS: None, C.ACC: None, C.WEIGHTS: None, C.EPOCH: Epoch(0, None, None)}
        }

    def add_child(self, child: Edge):
        self._children.append(child)

    def check_child_id_by_fid(self, fid):
        for child_id in self.id.children_id:
            if child_id.fid == fid:
                return True
        return False

    def get_weights(self):
        return self.weights

    def update_weights(self, new_weights):
        self.weights = copy.deepcopy(new_weights)
        self.logger.debug("CloudEpoch:{} | [Update Cloud Weights with FedAggre Weights Completed.]".format(self.ep.c_to_str()))

    def update_children_weights(self, new_weights=None, children: List[Edge] = None):
        _weights = self.weights if new_weights is None else new_weights
        _children = self._children if children is None else children
        assert not isinstance(_weights, list) or (isinstance(_weights, list) and len(_weights) == len(_children))
        for ei, edge in enumerate(_children):
            if isinstance(_weights, list):
                edge.update_weights(_weights[ei], weight_type=C.CLOUD)
            else:
                edge.update_weights(_weights, weight_type=C.CLOUD)

    def get_init_weights(self):
        if self.init_weights_path is None or not gutil.is_weights_path(self.init_weights_path):
            self.logger.info("Train from scratch.")

            if self.task in [C.IMG_CLASSIFICATION]:
                model = Models.cls_model_gen(
                    self.config[C.NAME_MODEL],
                    ptr_weights=self.config.get(C.PTR_WEIGHTS),
                    dropout=self.config.get("dropout", 0.2)
                )(self.config, logger=self.logger, only_init_net=True)
            else:
                model = Models.seg_model_gen(self.config[C.NAME_MODEL], ptr_weights=self.config.get(C.PTR_WEIGHTS))(self.config, logger=self.logger, only_init_net=True)
            # model = getattr(Models, self.config[C.NAME_MODEL])(self.config, self.logger, only_init_net=True)
            init_weights = copy.deepcopy(model.get_weights(to_cpu=True))
            self.logger.info("Init Weights Completed.")
            del model
        else:
            self.logger.info("Load Weights From:{}".format(self.init_weights_path))
            init_weights = torch.load(self.init_weights_path, map_location="cpu")
            self.logger.info("Load Weights Completed.")
        return init_weights

    def get_stats(self):
        return {
            "cloud_stats": self.stats,
        }

    def start(self):
        for _ in tqdm(
                range(self.num_cloud_ep),
                desc="Cloud#{}[{}]-{}".format(self.id.nid, self.id.sid, self.fed_mode),
                unit="CloudEpoch",
                file=sys.stdout
        ):
            self.ep.cloud_epoch_plus()
            self.logger.info("CloudEpoch:{} | [Train] | Start ...".format(self.ep.c_to_str()))

            edges_weights = []
            edges_train_loss = []
            edges_train_contrib = []

            # Init when cloud_epoch==1
            if self.ep.cloud_epoch == 1:
                self.update_children_weights()
            for edge in self._children:
                edge.ep.update(self.ep)

                cpu_weights, loss, contrib = edge.train()

                edges_weights.append(cpu_weights)
                edges_train_loss.append(loss)
                edges_train_contrib.append(contrib)

            cloud_loss, _ = self.get_cloud_loss_acc(C.TRAIN, edges_train_loss, None, edges_train_contrib, is_record=False)
            self.logger.info("CloudEpoch:{} | [Train] | Contrib:{}".format(self.ep.c_to_str(), edges_train_contrib))
            self.logger.info("CloudEpoch:{} | [Train] | Loss:{:.4f}".format(self.ep.c_to_str(), cloud_loss))
            self.tbX.add_scalars("cloud-train", {C.LOSS: cloud_loss}, self.ep.total_cloud_ep())

            if self.fed_mode == C.alphaFed and self.multi_dataset:
                alpha = self.fed_params.get("alpha", 0.9)
                cloud_fed_w = alphaFedAvg(alpha, edges_weights, edges_train_contrib)
                self.logger.info("CloudEpoch:{} | [alphaFed-Aggre:{}] | Completed.".format(self.ep.c_to_str(), alpha))
            else:
                cloud_fed_w = FedAvg(edges_weights, edges_train_contrib)
                self.logger.info("CloudEpoch:{} | [Aggre] | Completed.".format(self.ep.c_to_str()))

            self.update_weights(cloud_fed_w)
            self.update_children_weights()

            self.save_ckpt(self.config.get("save_ckpt_epoch"))

            self.logger.info("CloudEpoch:{} | [Train] | Done.".format(self.ep.c_to_str()))

            self.eval()
            self.check_summary()

    def eval(self, vt_eval_edge_num=1, vt_eval_client_num=1):
        if isinstance(self.dataset_name, list):
            vt_eval_edge_num = len(self.dataset_name)
            choice_edge_idx = []
            for d_n in self.dataset_name:
                child_idxs = []
                for child_idx, child in enumerate(self._children):
                    if child.dataset_name == d_n:
                        child_idxs.append(child_idx)
                choice_edge_idx += np.random.choice(child_idxs, 1, replace=False).tolist()
            assert len(choice_edge_idx) == vt_eval_edge_num
        else:
            choice_edge_idx = np.random.choice(list(range(len(self._children))), vt_eval_edge_num, replace=False)
        for eval_type in self.eval_types:
            if self.eval_cfg[eval_type][C.NUM] > 0 and self.ep.cloud_epoch % self.eval_cfg[eval_type][C.NUM] == 0:
                self.logger.info("CloudEpoch:{} | [Cloud-Eval-{}] | Start ...".format(self.ep.c_to_str(), eval_type))
                edges_cloud_eval_datas = dict()
                for idx, edge in enumerate(self._children):
                    eval_data = dict()
                    eval_data[eval_type] = True if (eval_type == C.TRAIN or idx in choice_edge_idx) else False
                    edge.cloud_eval(eval_type, eval_data, vt_eval_client_num)
                    edges_cloud_eval_datas["#{}[{}]".format(edge.id.nid, edge.id.sid)] = eval_data

                cloud_loss, cloud_acc, contrib = self.aggre_eval(eval_type, edges_cloud_eval_datas, is_record=True)

                self.logger.info("CloudEpoch:{} | [Cloud-Eval-{}] | Contrib:{}".format(self.ep.c_to_str(), eval_type, contrib))
                self.logger.info("CloudEpoch:{} | [Cloud-Eval-{}] | Loss:{:.4f}".format(self.ep.c_to_str(), eval_type, cloud_loss))
                gutil.log_acc(logger=self.logger, acc=cloud_acc, classes=self.classes)

                if self.tbX is not None:
                    self.tbX.add_scalars("cloud-eval/loss", {eval_type: cloud_loss}, self.ep.total_cloud_ep())
                    for k, v in cloud_acc.items():
                        if k == "mean_type":
                            continue
                        if self.task in [C.IMG_SEGMENTATION]:
                            for name, value in v.items():
                                self.tbX.add_scalars("cloud-eval/{}/{}".format(k, name), {eval_type: value}, self.ep.total_cloud_ep())
                        else:
                            self.tbX.add_scalars("cloud-eval/m{}".format(k), {eval_type: v["mean"]}, self.ep.total_cloud_ep())

                self.logger.info("CloudEpoch:{} | [Cloud-Eval-{}] | Done.".format(self.ep.c_to_str(), eval_type))
                self.update_best(eval_type)
                tolerate_res = self.update_tolerate(eval_type)
                if isinstance(tolerate_res, bool):
                    self.fin = tolerate_res

    def get_cloud_loss_acc(self, eval_type: str, edges_loss: list, edges_acc: List[dict] or None, edges_contrib: list, is_record: bool):
        assert eval_type in [C.TRAIN, C.VALIDATION, C.TEST], "cloud eval_type:{} error".format(eval_type)
        now_cloud_loss = gutil.list_mean(edges_loss, edges_contrib)

        now_cloud_acc = None
        now_acc_edge_contributions = []
        if edges_acc is not None:
            now_cloud_acc = dict()
            metric_edge_acc = dict()
            for e_acc, e_contrib in zip(edges_acc, edges_contrib):
                if e_acc is not None:
                    for k, v in e_acc.items():
                        if k == "mean_type":
                            if k not in now_cloud_acc:
                                now_cloud_acc[k] = v
                        else:
                            if k in metric_edge_acc:
                                metric_edge_acc[k].append(v)
                            else:
                                metric_edge_acc[k] = [v]
                    now_acc_edge_contributions.append(e_contrib)

            for metric_type in metric_edge_acc.keys():
                now_cloud_acc[metric_type] = gutil.dict_list_mean(metric_edge_acc[metric_type], now_acc_edge_contributions)

        if is_record:
            self.stats[eval_type][C.LOSS].append(now_cloud_loss)
            self.stats[eval_type][C.ACC].append(now_cloud_acc)

        return now_cloud_loss, now_cloud_acc

    def aggre_eval(self, eval_type, edge_update_datas, is_record=False, return_contrib_type=None):
        assert eval_type in [C.TRAIN, C.VALIDATION, C.TEST], "eval_type:{} error".format(eval_type)
        assert return_contrib_type in [None, "sum", "avg"]
        contrib = [edge_data[eval_type][C.CONTRIB] if isinstance(edge_data[eval_type][C.CONTRIB], torch.Tensor)
                   else torch.tensor(edge_data[eval_type][C.CONTRIB]) for edge_data in edge_update_datas.values()]

        cloud_loss, cloud_acc = self.get_cloud_loss_acc(
            eval_type,
            [edge_data[eval_type][C.LOSS] for edge_data in edge_update_datas.values()],
            [edge_data[eval_type][C.ACC] for edge_data in edge_update_datas.values()],
            contrib,
            is_record
        )

        if return_contrib_type == "sum":
            contrib = sum(contrib)
        elif return_contrib_type == "avg":
            contrib = sum(contrib) / len(contrib)
        return cloud_loss, cloud_acc, contrib

    def update_tolerate(self, now_type: str):
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
        now_cloud_loss = self.stats[best_type][C.LOSS][-1]
        now_cloud_acc = self.stats[best_type][C.ACC][-1]
        cloud_metric = self.eval_cfg[best_type][C.METRIC]
        # init
        if self.best[best_type][C.ACC] is None:
            self.best[best_type][C.ACC] = now_cloud_acc
        if self.best[best_type][C.LOSS] is None:
            self.best[best_type][C.LOSS] = now_cloud_loss
        if self.best[best_type][C.WEIGHTS] is None:
            self.best[best_type][C.WEIGHTS] = copy.deepcopy(self.weights)
        if self.best[best_type][C.EPOCH].cloud_epoch == 0:
            self.best[best_type][C.EPOCH] = copy.deepcopy(self.ep)
        # compare with cloud_eval metric
        if isinstance(cloud_metric, (list, tuple)):
            cur_value = 1
            cur_best_value = 1
            cloud_metric = list(set(cloud_metric))
            for m in cloud_metric:
                assert m[0] == "m"
                metric_type = m[1:] if m[0] == "m" else m
                assert metric_type in now_cloud_acc.keys()
                cur_value *= now_cloud_acc[metric_type]["mean"]
                cur_best_value *= self.best[best_type][C.ACC][metric_type]["mean"]
            if cur_value > cur_best_value:
                self.best[best_type][C.LOSS] = now_cloud_loss
                self.best[best_type][C.ACC] = now_cloud_acc
                self.best[best_type][C.WEIGHTS] = copy.deepcopy(self.weights)
                self.best[best_type][C.EPOCH] = copy.deepcopy(self.ep)
        else:
            assert cloud_metric[0] == "m" or cloud_metric == C.LOSS
            metric_type = cloud_metric[1:] if cloud_metric[0] == "m" else cloud_metric
            if metric_type in now_cloud_acc.keys():
                if now_cloud_acc[metric_type]["mean"] > self.best[best_type][C.ACC][metric_type]["mean"]:
                    self.best[best_type][C.LOSS] = now_cloud_loss
                    self.best[best_type][C.ACC] = now_cloud_acc
                    self.best[best_type][C.WEIGHTS] = copy.deepcopy(self.weights)
                    self.best[best_type][C.EPOCH] = copy.deepcopy(self.ep)
            else:
                if now_cloud_loss < self.best[best_type][C.LOSS]:
                    self.best[best_type][C.LOSS] = now_cloud_loss
                    self.best[best_type][C.ACC] = now_cloud_acc
                    self.best[best_type][C.WEIGHTS] = copy.deepcopy(self.weights)
                    self.best[best_type][C.EPOCH] = copy.deepcopy(self.ep)

    def save_ckpt(self, save_ckpt_epoch: int):
        if save_ckpt_epoch is not None and isinstance(save_ckpt_epoch, int) and save_ckpt_epoch > 0:
            if self.ep.cloud_epoch % save_ckpt_epoch == 0:
                ckpt_record_path = Path(self.weights_dir, "record", "cloud", "ep[{}].pt".format(self.ep.c_to_str()))
                ckpt_record_path.parent.mkdir(exist_ok=True, parents=True)
                gutil.save_weights(self.weights, ckpt_record_path)
                self.logger.info("CloudEpoch:{} | Save Record CloudWeights : {}".format(self.ep.c_to_str(), ckpt_record_path))

    def record_metric(self, record_file_dir: str, record_type: str, record_interval: int):
        record_acc = dict()
        record_epoch = []
        record_file_path = Path(record_file_dir, "{}.json".format(record_type))
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
            "cloud": {
                "epoch": record_epoch,
                "loss": self.stats[record_type][C.LOSS],
                "acc": record_acc,
                "mean_type": mean_type
            }
        }
        record_json.update(record)
        gutil.write_json(record_file_path, record, mode="w+", indent=4)

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
            df.to_csv(record_file_dir / "cloud_best_{}.csv".format(record_type), index=False)

    def check_summary(self):
        if self.ep.cloud_epoch >= self.num_cloud_ep:
            self.logger.info("Go to NUM_CLOUD_EPOCH:{}".format(self.num_cloud_ep))
            self.fin = True

        if not self.fin:
            # next global epoch
            self.logger.info("Start Next CloudEpoch Training ...")
        else:
            self.fin_summary()

    def fin_summary(self, cloud_eval_types=None):
        if cloud_eval_types is None:
            cloud_eval_types = self.eval_types
        self.logger.info("Federated Learning Summary ...")
        for cloud_eval_type in cloud_eval_types:
            assert cloud_eval_type in [C.TRAIN, C.VALIDATION, C.TEST], "cloud eval type:{} error".format(cloud_eval_types)
            self.record_metric(self.record_file_dir, cloud_eval_type, self.eval_cfg[cloud_eval_type][C.NUM])
            self.record_best_acc(self.record_file_dir, cloud_eval_type)
            now_best = self.best[cloud_eval_type]
            self.logger.info("[Cloud-Summary-{}] | Metrics:{}".format(cloud_eval_type, self.eval_cfg[cloud_eval_type].get(C.METRIC)))
            self.logger.info("[Cloud-Summary-{}] | Best CloudEpoch:{}".format(cloud_eval_type, now_best[C.EPOCH].c_to_str()))
            self.logger.info("[Cloud-Summary-{}] | Best Loss:{}".format(cloud_eval_type, now_best[C.LOSS]))
            gutil.log_acc(logger=self.logger, acc=now_best[C.ACC], classes=self.classes)
            if now_best[C.WEIGHTS]:
                self.logger.info("[Cloud-Summary-{}] | Save Best CloudWeights : {}".format(cloud_eval_type, self.best_weights_path[cloud_eval_type]))
                gutil.save_weights(now_best[C.WEIGHTS], self.best_weights_path[cloud_eval_type])
        for edge in self._children:
            edge.fin_summary()
        self.tbX.close()
        exit()


class CloudDevice(Cloud):
    def __init__(self, config: dict, cloud_id: ID, config_dir: Path, host: str = None, port: int = None):
        super().__init__(config, cloud_id)
        self.config_dir = config_dir
        self.host = config.get(C.HOST, "127.0.0.1") if host is None else host
        self.port = config.get(C.PORT, "9191") if port is None else port
        self.app = Flask(
            __name__, template_folder=C.ROOT_PATH / "static" / "templates",
            static_folder=C.ROOT_PATH / "static"
        )
        logging.getLogger("werkzeug").setLevel(logging.WARNING)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", ping_timeout=360000, ping_interval=300)

        self.pubkey, self.privkey = endecrypt.newkey(512)
        self.num_edges = self.config[C.NUM_EDGES]
        self.tmp_weights_dir = Path(self.weights_dir) / "tmp"
        self.tmp_weights_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_weights_path = self.tmp_weights_dir / "cloud.pkl"
        self.tbx_proc = None

        self.wait_time = 0
        self.sleep_time = 0
        self.vt_eval_edge_num = 1
        self.SINGLE_EDGE_MAX_LOADAVG = self.config.get("per_edge_max_loadavg", 1.0)

        self.ready_edge_fids = set()
        self.running_edge_fids = set()
        self.edge_resource = dict()
        self.edge_update_datas = dict()  # now cloud epoch , all edge-update datas
        self.edge_eval_datas = dict()  # now cloud epoch , all edge-eval datas
        self.edge_pubkeys = dict()
        self.fid2dataset_name = dict()

        self.sid_fid = gutil.generate_bidict("sid", "fid")

        self.bar = tqdm(total=self.num_cloud_ep, unit="CloudEpoch", desc="Cloud-{}".format(self.fed_mode))
        self.register_handles()

        @self.app.route("/")
        def home_page():
            return render_template("dashboard.html", async_mode=self.socketio.async_mode)

        @self.app.route("/stats")
        def stats_page():
            return json.dumps(self.get_stats())

        @self.app.route("/tbx")
        def tensorboard_page():
            self.tbx_proc = subprocess.Popen(["tensorboard", "--logdir", self.tbX_dir, "--port", "6006"], preexec_fn=os.setsid)
            return redirect(location="http://{}:16006".format(self.host))

        @self.app.route("/tbx_fin")
        def tensorboard_proc_close():
            try:
                if self.tbx_proc is not None:
                    self.tbx_proc.terminate()
                    self.tbx_proc.wait()
                    # gutil.kill(self.tbx_proc.pid)
            except Exception as e:
                self.logger.error(e)
            finally:
                return redirect("/")

    def check_child_id_by_sid(self, sid):
        fid = self.sid_fid.fid_for.get(sid)
        if fid is not None:
            return self.check_child_id_by_fid(fid)

    def rsaEncrypt(self, fid, data, dumps=True, enable=True):
        """
        rsaEncrypt data
        :param fid: the edge fid
        :param data: the data will encrypt
        :param dumps: default is True , whether data need to serialize before encrypt
        :param enable: default is True, enable rsaEncrypt
        :return:
        """
        if not enable:
            return data
        if fid not in self.edge_pubkeys or self.edge_pubkeys[fid] is None:
            retry = 10
            while retry > 0:
                emit("get_edge_pubkey", broadcast=True)
                self.socketio.sleep(3)
                if fid in self.edge_pubkeys and self.edge_pubkeys[fid] is not None:
                    break
                retry -= 1
        res_data = endecrypt.rsaEncrypt(self.edge_pubkeys[fid], data, dumps)
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

    def start(self):
        self.logger.info("Cloud Start {}:{}".format(self.host, self.port))
        self.socketio.run(self.app, host=self.host, port=self.port)
        emit("ui_cloud_connect", broadcast=True, namespace="/ui")  # for ui

    def edges_check_resource(self):
        self.edge_resource = dict()
        if self.fin:
            ready_edge_fids = copy.deepcopy(self.ready_edge_fids)
            for ready_edge_fid in ready_edge_fids:
                self.socketio.sleep(3)
                emit("fin", room=self.sid_fid.sid_for[ready_edge_fid])
        else:
            self.running_edge_fids = set(np.random.choice(list(self.ready_edge_fids), self.num_edges, replace=False))
            for fid in self.running_edge_fids:
                sid = self.sid_fid.sid_for[fid]
                emit("ui_edge_check_resource", {C.SID: sid, C.FID: fid}, broadcast=True, namespace="/ui")  # for ui
                self.socketio.sleep(self.sleep_time)
                emit_data = {"cloud_pubkey": {"n": str(self.pubkey.n), "e": str(self.pubkey.e)}}
                emit("edge_check_resource", self.rsaEncrypt(fid, emit_data, enable=ENABLE_ENCRYPT_EMIT), room=sid)

    def halfway_edge_check_resource(self, fid):
        sid = self.sid_fid.sid_for[fid]
        self.running_edge_fids.add(fid)
        emit("ui_edge_check_resource", {C.SID: sid, C.FID: fid}, broadcast=True, namespace="/ui")  # for ui
        self.socketio.sleep(self.sleep_time)
        emit_data = {"halfway": True, "cloud_pubkey": {"n": str(self.pubkey.n), "e": str(self.pubkey.e)}}
        emit("edge_check_resource", self.rsaEncrypt(fid, emit_data, enable=ENABLE_ENCRYPT_EMIT), room=sid)

    def cloud_train_next_epoch(self, runnable_edge_fids: List[str]):
        self.ep.cloud_epoch_plus()
        # if self.ep.cloud_epoch > 1:
        self.bar.update(1)
        self.edge_update_datas = dict()
        self.logger.info("GlobalEpoch : {}".format(self.ep.c_to_str()))
        self.logger.debug("Edges : [{}]".format(",".join(runnable_edge_fids)))

        emit_data = {"ep": self.ep.serialize()}
        # Init when cloud_epoch==1
        if self.ep.cloud_epoch == 1:
            weights_pickle = gutil.obj2pickle(self.weights, self.tmp_weights_path)
            emit_data["weights"] = weights_pickle

        for fid in runnable_edge_fids:
            assert self.check_child_id_by_fid(fid), "{} not in children_id".format(fid)
            sid = self.sid_fid.sid_for[fid]
            emit(
                "ui_edge_train", {C.SID: sid, "ep": self.ep.cloud_epoch}, broadcast=True,
                namespace="/ui"
            )  # for ui
            emit("edge_train", self.rsaEncrypt(fid, emit_data, enable=ENABLE_ENCRYPT_EMIT), room=sid)

    def halfway_train(self, runnable_edge_fid: str):
        self.logger.info("CloudEpoch : {}".format(self.ep.c_to_str()))
        self.logger.info("Edge: [{}]".format(runnable_edge_fid))

        weights_pickle = gutil.obj2pickle(self.weights, self.tmp_weights_path)
        emit_data = {"ep": self.ep.serialize(), "weights": weights_pickle}
        sid = self.sid_fid.sid_for[runnable_edge_fid]
        emit(
            "ui_edge_train", {C.SID: sid, "ep": self.ep.cloud_epoch}, broadcast=True,
            namespace="/ui"
        )  # for ui
        emit("edge_train", self.rsaEncrypt(runnable_edge_fid, emit_data, enable=ENABLE_ENCRYPT_EMIT), room=sid)

    def fin_summary(self, cloud_eval_types=None):
        if cloud_eval_types is None:
            cloud_eval_types = self.eval_types

        emit("ui_cloud_summary", broadcast=True, namespace="/ui")  # for ui
        self.socketio.sleep(self.sleep_time)
        self.logger.info("Federated Learning Summary ...")
        for cloud_eval_type in cloud_eval_types:
            assert cloud_eval_type in [C.TRAIN, C.VALIDATION, C.TEST], "cloud eval type:{} error".format(cloud_eval_types)
            self.record_metric(self.record_file_dir, cloud_eval_type, self.eval_cfg[cloud_eval_type][C.NUM])
            self.record_best_acc(self.record_file_dir, cloud_eval_type)
            now_best = self.best[cloud_eval_type]
            self.logger.info("[Cloud-Summary-{}] | Metrics:{}".format(cloud_eval_type, self.eval_cfg[cloud_eval_type].get(C.METRIC)))
            self.logger.info("[Cloud-Summary-{}] | Best CloudEpoch:{}".format(cloud_eval_type, now_best[C.EPOCH].c_to_str()))
            gutil.log_acc(logger=self.logger, acc=now_best[C.ACC], classes=self.classes)
            if now_best[C.WEIGHTS]:
                self.logger.info("[Cloud-Summary-{}] | Save Best CloudWeights : {}".format(cloud_eval_type, self.best_weights_path[cloud_eval_type]))
                gutil.save_weights(now_best[C.WEIGHTS], self.best_weights_path[cloud_eval_type])

    def register_handles(self):
        @self.socketio.on("connect")
        def connect_handle():
            self.logger.info("[{}] Connect".format(request.sid))
            emit("ui_edge_connect", {"sid": request.sid}, broadcast=True, namespace="/ui")  # for ui

        @self.socketio.on("connect", namespace="/ui")
        def ui_connect_handle():
            self.logger.info("ui [{}] Connect".format(request.sid))

        @self.socketio.on("reconnect")
        def reconnect_handle():
            self.logger.info("[{}] Re Connect".format(request.sid))
            emit("ui_edge_reconnect", {"sid": request.sid}, broadcast=True, namespace="/ui")  # for ui

        @self.socketio.on("reconnect", namespace="/ui")
        def ui_reconnect_handle():
            self.logger.info("ui [{}] Re Connect".format(request.sid))

        @self.socketio.on("disconnect")
        def disconnect_handle():
            sid = request.sid
            fid = self.sid_fid.fid_for.get(sid)
            if fid:
                self.logger.info("Edge[{}] Close Connect.".format(fid))
                if fid in self.ready_edge_fids:
                    self.ready_edge_fids.remove(fid)
                if fid in self.running_edge_fids:
                    self.running_edge_fids.remove(fid)
                if fid in self.edge_update_datas.keys():
                    self.edge_update_datas.pop(fid)
                if fid in self.edge_eval_datas.keys():
                    self.edge_eval_datas.pop(fid)
                if fid in self.fid2dataset_name.keys():
                    self.fid2dataset_name.pop(fid)
                emit("ui_edge_disconnect", {C.SID: sid}, broadcast=True, namespace="/ui")  # for ui

        @self.socketio.on("disconnect", namespace="/ui")
        def ui_disconnect_handle():
            self.logger.info("ui [{}] Close Connect.".format(request.sid))
            emit("ui_disconnect", namespace="/ui")
            disconnect(request.sid, namespace="/ui")

        @self.socketio.on("heartbeat")
        def heartbeat_handle():
            sid = request.sid
            self.logger.debug("Receive HeartBeat from [{}] , Still Alive".format(sid))
            emit("re_heartbeat", room=sid)

        @self.socketio.on_error()
        def error_handle(e):
            self.logger.error(e)

        @self.socketio.on_error(namespace="/ui")
        def ui_error_handle(e):
            self.logger.error("ui:{}".format(e))

        @self.socketio.on("get_cloud_pubkey")
        def send_cloud_pubkey():
            emit_data = {"cloud_pubkey": {"n": str(self.pubkey.n), "e": str(self.pubkey.e)}, C.FID: self.id.fid}
            emit("cloud_pubkey", emit_data)

        @self.socketio.on("edge_pubkey")
        def get_edge_pubkey(data):
            sid = request.sid
            fid = data[C.FID]
            assert self.sid_fid.fid_for.get(sid) == fid
            self.edge_pubkeys[fid] = endecrypt.toPubkey(int(data["edge_pubkey"]["n"]), int(data["edge_pubkey"]["e"]))

        @self.socketio.on("edge_wakeup")
        def client_wakeup_handle(data):
            sid = request.sid
            fid = data[C.FID]
            if self.check_child_id_by_fid(fid):
                self.sid_fid[sid] = fid
                self.edge_pubkeys[fid] = endecrypt.toPubkey(int(data["edge_pubkey"]["n"]), int(data["edge_pubkey"]["e"]))
                self.logger.info("Edge[{}] Wake Up".format(fid))
                emit("ui_edge_wakeup", {C.SID: sid, C.FID: fid}, broadcast=True, namespace="/ui")  # for ui
                emit("edge_init", room=sid)

        @self.socketio.on("edge_ready")
        def edge_init_handle(data):
            sid = request.sid
            fid = self.sid_fid.fid_for.get(sid)
            if self.check_child_id_by_fid(fid):
                self.logger.info("Edge[{}] Init".format(fid))
                self.ready_edge_fids.add(fid)
                self.fid2dataset_name[fid] = data[C.NAME_DATASET]
                emit("ui_edge_init", {C.SID: sid, C.FID: fid}, broadcast=True, namespace="/ui")  # for ui
                if self.ep.cloud_epoch == 0:
                    if len(self.ready_edge_fids) >= self.num_edges:
                        self.logger.info("Now Ready Edge(s)_Num:{} >= {}(num_edges), Federated Train Start ~".format(len(self.ready_edge_fids), self.num_edges))
                        self.edges_check_resource()
                    else:
                        self.logger.info("Now Ready Edge(s)_Num:{} < {}(num_edges) , Waiting Enough Edges To Run...".format(len(self.ready_edge_fids), self.num_edges))
                else:
                    if len(self.ready_edge_fids) <= self.num_edges:
                        self.logger.info("Now CloudEpoch:{} , A New Edge joining ... , Ready Edge(s)_Num:{} <= {}(num_edges) .".format(self.ep.c_to_str(), len(self.ready_edge_fids), self.num_edges))
                    self.halfway_edge_check_resource(fid)
            else:
                emit("fin", room=sid)

        @self.socketio.on("edge_check_resource_complete")
        def edge_check_resource_complete_handle(data):
            sid = request.sid
            if self.check_child_id_by_sid(sid):
                fid = self.sid_fid.fid_for[sid]
                data = self.rsaDecrypt(data, enable=ENABLE_ENCRYPT_EMIT)
                emit("ui_edge_check_resource_complete", {C.SID: sid, C.FID: fid}, broadcast=True, namespace="/ui")  # for ui
                self.socketio.sleep(self.sleep_time)
                self.edge_resource[fid] = data["loadavg"]
                # up to NUM_CLIENTS , begin next step
                if len(self.edge_resource) == self.num_edges:
                    runnable_edge_fids = []
                    for e_fid, loadavg in self.edge_resource.items():
                        self.logger.debug("Edge[{}] , Loadavg : {}".format(e_fid, loadavg))
                        if float(loadavg) < self.SINGLE_EDGE_MAX_LOADAVG:
                            runnable_edge_fids.append(e_fid)
                            self.logger.debug("Edge[{}] Runnable".format(e_fid))
                        else:
                            self.logger.warning("Edge[{}] Over-loadavg".format(e_fid))

                    # over half edges runnable
                    if len(runnable_edge_fids) / self.num_edges > 0.5:
                        self.wait_time = min(self.wait_time, 3)
                        self.cloud_train_next_epoch(runnable_edge_fids)
                    else:
                        self.wait_time += 1 if self.wait_time < 10 else 0
                        self.socketio.sleep(self.wait_time)
                        self.edges_check_resource()

        @self.socketio.on("halfway_edge_check_resource_complete")
        def halfway_edge_check_resource_complete_handle(data):
            sid = request.sid
            if self.check_child_id_by_sid(sid):
                fid = self.sid_fid.fid_for[sid]
                data = self.rsaDecrypt(data, enable=ENABLE_ENCRYPT_EMIT)
                emit("ui_edge_check_resource_complete", {C.SID: sid, C.FID: fid}, broadcast=True, namespace="/ui")  # for ui
                self.socketio.sleep(self.sleep_time)
                loadavg = data["loadavg"]
                self.edge_resource[fid] = loadavg
                self.logger.debug("Edge[{}] , Loadavg : {}".format(fid, loadavg))
                if float(loadavg) < self.SINGLE_EDGE_MAX_LOADAVG:
                    self.logger.info("Edge[{}] Runnable".format(fid))
                    self.wait_time = min(self.wait_time, 3)
                    self.halfway_train(fid)
                else:
                    self.logger.warning("Edge[{}] Over-loadavg".format(fid))
                    self.wait_time += 1 if self.wait_time < 10 else 0
                    self.socketio.sleep(self.wait_time)
                    self.halfway_edge_check_resource(fid)

        @self.socketio.on("edge_update_complete")
        def edge_update_complete_handle(data):
            sid = request.sid
            fid = self.sid_fid.fid_for[sid]
            data = self.rsaDecrypt(data, enable=ENABLE_ENCRYPT_EMIT)
            self.logger.debug("Received Edge[{}] decrypted-Update-Data:{} ".format(fid, data))
            emit(
                "ui_edge_train_complete", {"sid": request.sid, "ep": self.ep.cloud_epoch}, broadcast=True,
                namespace="/ui"
            )  # for ui

            edge_ep = Epoch(**data["ep"])

            if self.ep.cloud_epoch == edge_ep.cloud_epoch:
                # data["now_weights"] = copy.deepcopy(gutil.pickle2obj(data["now_weights"]))
                self.edge_update_datas[fid] = data
                # all edges upload complete
                if len(self.edge_update_datas.keys()) == len(self.running_edge_fids):
                    emit("ui_cloud_train_aggre", {"ep": self.ep.cloud_epoch}, broadcast=True, namespace="/ui")  # for ui

                    edges_weights = []
                    edges_train_loss = []
                    edges_train_contrib = []
                    edges_fids = []

                    for edge_fid, edge_data in self.edge_update_datas.items():
                        edges_fids.append(edge_fid)
                        edges_weights.append(copy.deepcopy(gutil.pickle2obj(edge_data["weights"])))
                        edges_train_loss.append(edge_data[C.TRAIN_LOSS])
                        edges_train_contrib.append(torch.tensor(edge_data[C.TRAIN_CONTRIB]))

                    cloud_loss, _ = self.get_cloud_loss_acc(C.TRAIN, edges_train_loss, None, edges_train_contrib, is_record=False)
                    self.logger.info("CloudEpoch:{} | [Train] | Contrib:{}".format(self.ep.c_to_str(), edges_train_contrib))
                    self.logger.info("CloudEpoch:{} | [Train] | Loss:{:.4f}".format(self.ep.c_to_str(), cloud_loss))
                    self.tbX.add_scalars("cloud-train", {C.LOSS: cloud_loss}, self.ep.total_cloud_ep())

                    if self.fed_mode == C.alphaFed and self.multi_dataset:
                        alpha = self.fed_params.get("alpha", 0.9)
                        cloud_fed_w_list = alphaFedAvg(alpha, edges_weights, edges_train_contrib)
                        cloud_fed_w = {fid: fed_w for fid, fed_w in zip(edges_fids, cloud_fed_w_list)}
                        self.logger.info("CloudEpoch:{} | [alphaFed-Aggre:{}] | Completed.".format(self.ep.c_to_str(), alpha))
                    else:
                        cloud_fed_w = FedAvg(edges_weights, edges_train_contrib)
                        self.logger.info("CloudEpoch:{} | [Aggre] | Completed.".format(self.ep.c_to_str()))

                    self.update_weights(cloud_fed_w)
                    self.save_ckpt(self.config.get("save_ckpt_epoch"))
                    self.logger.info("CloudEpoch:{} | [Train] | Done.".format(self.ep.c_to_str()))

                    now_weights_pickle = gutil.obj2pickle(self.weights, self.tmp_weights_path)  # weights path
                    emit_data = {
                        "ep": self.ep.serialize(),
                        "weights": now_weights_pickle,
                        "multi_weights": self.multi_dataset,
                    }

                    emit("ui_cloud_train_aggre_complete", {"ep": self.ep.cloud_epoch}, broadcast=True, namespace="/ui")  # for ui
                    self.socketio.sleep(self.sleep_time)
                    self.edge_eval_datas = dict()  # empty eval datas for next eval epoch

                    if isinstance(self.dataset_name, list):
                        vt_eval_edge_num = len(self.dataset_name)
                        choice_edge_idx = []
                        for d_n in self.dataset_name:
                            child_idxs = []
                            for idx, fid in enumerate(list(self.running_edge_fids)):
                                if self.fid2dataset_name[fid] == d_n:
                                    child_idxs.append(idx)
                            choice_edge_idx += np.random.choice(child_idxs, 1, replace=False).tolist()
                        assert len(choice_edge_idx) == vt_eval_edge_num
                    else:
                        choice_edge_idx = np.random.choice(list(range(len(self.running_edge_fids))), self.vt_eval_edge_num, replace=False)
                    for idx, fid in enumerate(list(self.running_edge_fids)):
                        _emit_data = copy.deepcopy(emit_data)
                        sid = self.sid_fid.sid_for[fid]
                        for eval_type in self.eval_types:
                            if self.eval_cfg[eval_type][C.NUM] > 0 and self.ep.cloud_epoch % self.eval_cfg[eval_type][C.NUM] == 0:
                                _emit_data[eval_type] = (idx in choice_edge_idx) or eval_type == C.TRAIN
                        emit("ui_edge_eval", {C.SID: sid, C.FID: fid, "ep": self.ep.cloud_epoch}, broadcast=True, namespace="/ui")  # for ui
                        self.socketio.sleep(self.sleep_time)
                        emit("eval_with_cloud_weights", self.rsaEncrypt(fid, _emit_data, enable=ENABLE_ENCRYPT_EMIT), room=sid)
                        self.logger.debug("Cloud Send Federated Aggregation Weights To Edge[{}]".format(fid))

        @self.socketio.on("client_train_process")
        def client_train_process_handle(data):
            sid = request.sid
            fid = data[C.FID]
            ep = Epoch(**data["ep"])
            process = data["process"]
            # self.logger.debug("Received Edge-sid:[{}] Train-process-data:{} ".format(request.sid, data))
            emit("ui_client_train_process", {C.SID: sid, C.FID: fid, "ep": ep.cec_to_str(), "process": process}, broadcast=True, namespace="/ui")  # for ui

        @self.socketio.on("eval_with_cloud_weights_complete")
        def eval_with_global_weights_complete_handle(data):
            sid = request.sid
            fid = self.sid_fid.fid_for[sid]
            data = self.rsaDecrypt(data, enable=ENABLE_ENCRYPT_EMIT)
            self.logger.debug("Receive Edge[{}] Eval Datas:{}".format(fid, data))
            emit("ui_edge_eval_complete", {C.SID: sid, C.FID: fid, "ep": self.ep.cloud_epoch}, broadcast=True, namespace="/ui")  # for ui
            self.socketio.sleep(self.sleep_time)

            self.edge_eval_datas[fid] = data
            if len(self.edge_eval_datas.keys()) == len(self.running_edge_fids):
                emit("ui_cloud_eval_aggre", {"ep": self.ep.cloud_epoch}, broadcast=True, namespace="/ui")  # for ui
                cloud_eval_types = list(list(self.edge_eval_datas.values())[0].keys())
                for eval_type in self.eval_types:
                    if eval_type in cloud_eval_types:
                        cloud_loss, cloud_acc, contrib = self.aggre_eval(eval_type, self.edge_eval_datas, is_record=True)

                        self.logger.info("CloudEpoch:{} | [Cloud-Eval-{}] | Contrib:{}".format(self.ep.c_to_str(), eval_type, contrib))
                        self.logger.info("CloudEpoch:{} | [Cloud-Eval-{}] | Loss:{:.4f}".format(self.ep.c_to_str(), eval_type, cloud_loss))
                        gutil.log_acc(logger=self.logger, acc=cloud_acc, classes=self.classes)

                        if self.tbX is not None:
                            self.tbX.add_scalars("cloud-eval/loss", {eval_type: cloud_loss}, self.ep.total_cloud_ep())
                            for k, v in cloud_acc.items():
                                if k == "mean_type":
                                    continue
                                if self.task in [C.IMG_SEGMENTATION]:
                                    for name, value in v.items():
                                        self.tbX.add_scalars("cloud-eval/{}/{}".format(k, name), {eval_type: value}, self.ep.total_cloud_ep())
                                else:
                                    self.tbX.add_scalars("cloud-eval/m{}".format(k), {eval_type: v["mean"]}, self.ep.total_cloud_ep())

                        self.logger.info("CloudEpoch:{} | [Cloud-Eval-{}] | Done.".format(self.ep.c_to_str(), eval_type))
                        self.update_best(eval_type)
                        tolerate_res = self.update_tolerate(eval_type)
                        if isinstance(tolerate_res, bool):
                            self.fin = tolerate_res

                emit("ui_cloud_eval_aggre_complete", {"ep": self.ep.cloud_epoch}, broadcast=True, namespace="/ui")  # for ui
                self.socketio.sleep(self.sleep_time)

                self.check_summary()
                self.edges_check_resource()

        @self.socketio.on("edge_fin")
        def handle_edge_fin(data):
            data = self.rsaDecrypt(data, enable=ENABLE_ENCRYPT_EMIT)
            sid = request.sid
            assert data[C.FID] == self.sid_fid.fid_for[sid]
            fid = data[C.FID]

            self.logger.info("Federated Learning Edge[{}] Fin.".format(fid))
            disconnect(sid)
            if fid in self.ready_edge_fids:
                self.ready_edge_fids.remove(fid)
            if fid in self.running_edge_fids:
                self.running_edge_fids.remove(fid)
            if fid in self.edge_update_datas.keys():
                self.edge_update_datas.pop(fid)
            if fid in self.edge_eval_datas.keys():
                self.edge_eval_datas.pop(fid)
            if fid in self.fid2dataset_name.keys():
                self.fid2dataset_name.pop(fid)
            emit("ui_edge_fin", {C.SID: sid}, broadcast=True, namespace="/ui")  # for ui
            if len(self.ready_edge_fids) == 0:
                self.tbX.close()
                self.logger.info("All Edges Fin. Federated Learning Cloud Fin.")
                os.remove(self.tmp_weights_path)
                emit("ui_cloud_fin", broadcast=True, namespace="/ui")  # for ui
                self.socketio.sleep(5)
                try:
                    if self.tbx_proc is not None:
                        self.tbx_proc.terminate()
                        self.tbx_proc.wait()
                    self.bar.close()
                    self.socketio.stop()
                finally:
                    del self.app
                    del self.socketio
                    gutil.write_complete_log(self.config_dir)
                    pid = self.config[C.PID] if os.getpid() == self.config[C.PID] else os.getpid()
                    gutil.kill(pid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cloud_config_path", type=str, dest="cloud_config_path", required=True, help="path of cloud config")
    parser.add_argument("-e", "--edge_config_paths", type=str, dest="edge_config_paths", nargs="+", required=True, help="paths of cloud config")
    parser.add_argument("-g", "--gpu", default="", dest="gpu", type=str, help="optional,specified gpu to run", required=False)
    parser.add_argument("--host", type=str, dest="host", help="optional cloud host , 'configs/base_config.yaml' has inited host")
    parser.add_argument("--port", type=int, dest="port", help="optional cloud port , 'configs/base_config.yaml' has inited port")

    args = parser.parse_args()
    cloud_config_path = args.cloud_config_path
    edge_config_paths = args.edge_config_paths
    host, port = args.host, args.port
    logger = Logger()

    assert Path(cloud_config_path).exists(), "{} not exist".format(cloud_config_path)
    for edge_config_path in edge_config_paths:
        assert Path(edge_config_path).exists(), "{} not exist".format(edge_config_path)

    config_dir = Path(cloud_config_path).parent.resolve()
    cloud_cfg = gutil.load_json(cloud_config_path)
    seed = cloud_cfg.get(C.SEED, C.DEFAULT_SEED)
    init_seed = cloud_cfg.get(C.INIT_SEED, seed)
    logger.info("init_seed:{}".format(init_seed))
    gutil.set_all_seed(init_seed)
    gpu = args.gpu if len(args.gpu) != 0 else str(cloud_cfg.get("gpu", "0"))
    cloud_cfg[C.PID] = os.getpid()
    gutil.write_json(cloud_config_path, cloud_cfg, mode="w+", indent=4)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    cloud_id, _ = gutil.build_id_connect(gutil.load_json(cloud_config_path), [gutil.load_json(edge_config_path) for edge_config_path in edge_config_paths])
    cloud = CloudDevice(cloud_cfg, cloud_id, config_dir, host, port)
    cloud.logger.info("gpu-id:{}".format(gpu))
    cloud.start()
