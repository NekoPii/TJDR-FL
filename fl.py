#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: maojingxin
"""
import argparse
import os
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent  # ..TJDR-FL/
if str(root_dir) != sys.path[0]:
    sys.path[0] = str(root_dir)

from fl_client import Client
from fl_cloud import Cloud
from fl_edge import Edge
from gutils import Logger
from gutils import gutil, constants as C


def build_HFL(cloud_cfg, edges_cfg: list, clients_cfg: list, logger=Logger()):
    cloud_id, edges_id = gutil.build_id_connect(cloud_cfg, edges_cfg)
    clients_id = None
    for edge_cfg, edge_id in zip(edge_cfgs, edges_id):
        edge_id, clients_id = gutil.build_id_connect(edge_cfg, clients_cfg, parent_id=edge_id, children_id=clients_id)

    # build env connection
    logger.info("start cloud ...")
    cloud = Cloud(cloud_cfg, cloud_id)
    logger.info("start cloud done.")
    logger.info("start edge(s) ...")
    edges = [Edge(edge_cfg, edge_id) for edge_cfg, edge_id in zip(edge_cfgs, edges_id)]
    logger.info("start {} edge(s) done.".format(len(edges)))
    logger.info("start client(s) ...")
    clients = [Client(client_cfg, client_id) for client_cfg, client_id in zip(client_cfgs, clients_id)]
    logger.info("start {} client(s) done.".format(len(clients)))

    logger.info("build connection ...")
    for edge in edges:
        assert edge.id.parent_id == cloud.id and edge.id in cloud.id.children_id
        for client in clients:
            if client.id.parent_id == edge.id and client.id in edge.id.children_id:
                edge.add_child(client)
        cloud.add_child(edge)
    logger.info("build connection done.")
    return cloud, edges, clients


if __name__ == "__main__":
    logger = Logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--config_dir", dest="config_dir", type=str, help="directory of configs", required=True)
    parser.add_argument("-g", "--gpu", default="", dest="gpu", type=str, help="optional,specified gpu to run", required=False)
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    assert config_dir.exists() and config_dir.is_dir(), "{} not exists or it is not a directory".format(config_dir)

    cloud_configs_path = sorted([str(cloud_config_path) for cloud_config_path in config_dir.glob("cloud*config.json")])
    edge_configs_path = sorted([str(edge_config_path) for edge_config_path in config_dir.glob("edge*config.json")])
    client_configs_path = sorted([str(client_config_path) for client_config_path in config_dir.glob("client*config.json")])
    gpu = args.gpu.strip()

    assert len(cloud_configs_path) == 1, "the number of cloud must 1"
    cloud_config_path = cloud_configs_path[0]

    cloud_cfg = gutil.load_json(cloud_config_path)
    seed = cloud_cfg.get(C.SEED, C.DEFAULT_SEED)
    run_seed = cloud_cfg.get(C.RUN_SEED, seed)
    logger.info("run_seed:{}".format(run_seed))
    gutil.set_all_seed(run_seed)
    gpu = gpu if len(gpu) != 0 else str(cloud_cfg.get("gpu", "0"))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    num_edges = cloud_cfg[C.NUM_EDGES]
    assert len(edge_configs_path) == num_edges

    edge_cfgs = [gutil.load_json(edge_config_path) for edge_config_path in edge_configs_path]
    num_clients = sum([edge_cfg[C.NUM_CLIENTS] for edge_cfg in edge_cfgs])
    assert len(client_configs_path) == num_clients

    client_cfgs = [gutil.load_json(client_config_path) for client_config_path in client_configs_path]

    cloud, _, _ = build_HFL(cloud_cfg, edge_cfgs, client_cfgs)

    pid = os.getpid()
    cloud_cfg[C.PID] = pid
    gutil.write_json(cloud_config_path, cloud_cfg, mode="w+", indent=4)
    for edge_cfg, edge_config_path in zip(edge_cfgs, edge_configs_path):
        edge_cfg[C.PID] = pid
        gutil.write_json(edge_config_path, edge_cfg, mode="w+", indent=4)
    for client_cfg, client_config_path in zip(client_cfgs, client_configs_path):
        client_cfg[C.PID] = pid
        gutil.write_json(client_config_path, client_cfg, mode="w+", indent=4)

    cloud.start()

    gutil.write_complete_log(config_dir)
