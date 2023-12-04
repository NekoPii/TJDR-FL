#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:maojingxin
"""
import sys
from pathlib import Path

cur_dir = Path(__file__).resolve().parent  # ..task/
root_dir = cur_dir.parent  # ..TJDR-FL/
if str(root_dir) != sys.path[0]:
    sys.path[0] = str(root_dir)

from gutils import constants as C, logger, gutil


def kill_config_process(config_path: Path, logger):
    if config_path.exists():
        try:
            config = gutil.load_json(config_path)
            pid = config.get(C.PID)
            gutil.kill(pid, logger)
        except Exception as e:
            logger.error(e)
    else:
        logger.error("config_path:{} not exist".format(config_path))


if __name__ == "__main__":
    logger = logger.Logger()
    stop_configs_from_dir = input("stop process from configs' directory ? [y]/n ").strip()
    if stop_configs_from_dir in ["y", "Y", ""]:
        configs_dir = Path(input("input configs' directory to stop : ").strip())
        assert configs_dir.exists(), "{} not exists".format(configs_dir)
        edges_configs_path = list()
        for cloud_config_path in configs_dir.glob("cloud*.json"):
            kill_config_process(cloud_config_path, logger)
        for edge_config_path in configs_dir.glob("edge*.json"):
            kill_config_process(edge_config_path, logger)
    else:
        config_path = Path(input("input config's path to stop : ").strip())
        kill_config_process(config_path, logger)
