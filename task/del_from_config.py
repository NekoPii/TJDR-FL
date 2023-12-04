#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author:maojingxin
"""
import shutil
import sys
from pathlib import Path

cur_dir = Path(__file__).resolve().parent  # ..task/
root_dir = cur_dir.parent  # ..TJDR-FL/
if str(root_dir) != sys.path[0]:
    sys.path[0] = str(root_dir)

from gutils import Logger
from gutils import gutil, constants as C

logger = Logger()


def del_dir(_dir: Path):
    try:
        shutil.rmtree(_dir)
        logger.info("del [{}] success.".format(_dir))
        if len(list(_dir.parent.iterdir())) == 0:
            del_empty_dir = input("! del [empty-dir] : [{}] ? : [y]/n ".format(_dir.parent)).strip() in ["y", "Y", ""]
            if del_empty_dir:
                del_dir(_dir.parent)
    except Exception as e:
        logger.error(e)


def del_from_config(config_dir: Path):
    assert config_dir.exists(), "{} not exists".format(config_dir)

    base_config_path = config_dir / "base_config.json"
    if base_config_path.exists():
        base_cfg = gutil.load_json(base_config_path)
        now_log_dir = Path(base_cfg[C.DIR]["logfile_dir"]).resolve()
        now_weights_dir = Path(base_cfg[C.DIR]["weights_dir"]).resolve()
        now_datasets_dir = Path(base_cfg[C.DIR]["generate_dataset_txt_dir"]).resolve()
        now_configs_dir = Path(base_cfg[C.DIR]["config_dir"]).resolve()
    else:
        key_word = [p.stem for p in [config_dir] + list(config_dir.parents)][:4]
        key_word.reverse()
        log_dir = cur_dir / "logs"
        weights_dir = cur_dir / "saves" / "weights"
        datasets_dir = cur_dir / "generates" / "datasets"
        configs_dir = cur_dir / "generates" / "configs"
        now_log_dir = log_dir / Path(*key_word)
        now_weights_dir = weights_dir / Path(*key_word)
        now_datasets_dir = datasets_dir / Path(key_word[0], key_word[2], key_word[3])
        now_configs_dir = configs_dir / Path(*key_word)

    del_now_log_dir = input("! del [{}] ? : [y]/n ".format(now_log_dir)).strip() in ["y", "Y", ""]
    if del_now_log_dir:
        del_dir(now_log_dir)
    del_now_weights_dir = input("! del [{}] ? : [y]/n ".format(now_weights_dir)).strip() in ["y", "Y", ""]
    if del_now_weights_dir:
        del_dir(now_weights_dir)
    del_now_datasets_dir = input("! del [{}] ? : [y]/n ".format(now_datasets_dir)).strip() in ["y", "Y", ""]
    if del_now_datasets_dir:
        del_dir(now_datasets_dir)
    del_now_configs_dir = input("! del [{}] ? : [y]/n ".format(now_configs_dir)).strip() in ["y", "Y", ""]
    if del_now_configs_dir:
        del_dir(now_configs_dir)


if __name__ == "__main__":
    config_path = input("input the config_dir to delete related files : ").strip()
    if len(config_path) != 0:
        config_path = Path(config_path)
        del_from_config(config_path)
