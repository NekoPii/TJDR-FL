#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: maojingxin
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch

cur_dir = Path(__file__).resolve().parent  # ..task/
root_dir = cur_dir.parent  # ..TJDR-FL/
if str(root_dir) != sys.path[0]:
    sys.path[0] = str(root_dir)

from task import config_generate as img_cfg_gen
from gutils import Logger
from gutils import gutil, constants as C

if __name__ == "__main__":
    logger = Logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base_config_path", type=str, dest="base_config_path", help="Specified the path of base config", default=str(img_cfg_gen.base_cfg_path.resolve()))
    parser.add_argument("-g", "--gpu", type=str, dest="gpu", help="Specified gpu to run, default is specified by config files", default="")
    parser.add_argument("-n", "--network", dest="network", action="store_true", help="Enable Network to parallel computing", default=False)
    parser.add_argument("--all_gpu", dest="all_gpu", action="store_true", help="Enable all gpu to parallel computing", default=False)
    parser.add_argument("--host", type=str, dest="host", help="optional cloud host, 'configs/base_config.yaml' has inited host", required=False)
    parser.add_argument("--port", type=int, dest="port", help="optional cloud port, 'configs/base_config.yaml' has inited port", required=False)

    args = parser.parse_args()

    base_config_path = args.base_config_path
    gpu = args.gpu
    network = args.network
    all_gpu = args.all_gpu
    host, port = args.host, args.port

    gpu_count = torch.cuda.device_count()

    base_cfg = gutil.load_yaml(base_config_path)
    seed = base_cfg[C.FED].get(C.SEED, C.DEFAULT_SEED)
    gutil.set_all_seed(seed)
    logger.info("seed: {}".format(seed))

    logger.info("generate configs from {}".format(base_config_path))
    dir_dict, parent2children_config = img_cfg_gen.generate(base_config_path=base_config_path)
    logger.info("generate config path done")
    config_dir = dir_dict["config_dir"]
    log_dir = dir_dict["logfile_dir"]
    logger.info("configs has generated and saved in :{}".format(config_dir))

    now_time = gutil.get_now_day() + "-" + gutil.get_now_time()
    temp_log_dir = Path(log_dir) / "temp_logs" / now_time
    temp_log_dir.mkdir(parents=True, exist_ok=True)
    logger.info("HFL start ...")
    gutil.write_start_log(cur_dir / "started.log", config_dir)
    if not network:
        temp_log_file = temp_log_dir / "hfl.log"
        os.system(
            # "python fl.py --config_dir {} {}".format(config_dir, "--gpu {}".format(gpu) if len(gpu) != 0 else "")
            "python ../fl.py --config_dir {} {} >>{} 2>&1 &".format(config_dir, "--gpu {}".format(gpu) if len(gpu) != 0 else "", temp_log_file)
        )
    else:
        cloud_config_path = list(parent2children_config[C.CLOUD].keys())[0]
        edge_config_paths = parent2children_config[C.CLOUD][cloud_config_path]
        temp_log_file = temp_log_dir / "hfl-cloud.log"
        logger.info("Cloud start...")
        os.system(
            # "python fl.py --config_dir {} {}".format(config_dir, "--gpu {}".format(gpu) if len(gpu) != 0 else "")
            "python ../fl_cloud.py -c {} -e {} {} {} {} >>{} 2>&1 &".format(
                cloud_config_path, " ".join(edge_config_paths),
                "--gpu {}".format(gpu) if len(gpu) != 0 else "",
                "--host {}".format(host) if host is not None and len(host) != 0 else "",
                "--port {}".format(port) if port is not None else "",
                temp_log_file
            )
        )
        time.sleep(10)
        for i, edge_config_path in enumerate(edge_config_paths, start=1):
            if all_gpu:
                gpu = str((i - 1) % gpu_count)
            logger.info("Edge#{} start...".format(i))
            temp_log_file = temp_log_dir / "hfl-edge#{}.log".format(i)
            os.system(
                # "python fl.py --config_dir {} {}".format(config_dir, "--gpu {}".format(gpu) if len(gpu) != 0 else "")
                "python ../fl_edge.py -e {} -c {} {} {} {} >>{} 2>&1 &".format(
                    edge_config_path, " ".join(parent2children_config[C.EDGE][edge_config_path]),
                    "--gpu {}".format(gpu) if len(gpu) != 0 else "",
                    "--host {}".format(host) if host is not None and len(host) != 0 else "",
                    "--port {}".format(port) if port is not None else "",
                    temp_log_file
                )
            )
    logger.info("log saved in {}".format(temp_log_dir))
