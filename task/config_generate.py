#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: maojingxin
"""
import shutil
import sys
import uuid
from pathlib import Path

cur_dir = Path(__file__).resolve().parent  # ..task/
root_dir = cur_dir.parent  # ..TJDR-FL/
if str(root_dir) != sys.path[0]:
    sys.path[0] = str(root_dir)

from task.utils import datasets
from gutils import Logger, ID, gutil, constants as C
from components import IMGDrawer

base_cfg_path = cur_dir / "configs" / "base_config.yaml"


def generate_dataset_txt(
        dataset_name: str,
        dataset_dir: Path,
        dataset_split: int,
        dataset_txt_path,
        dataset_type: str,
        non_iid_cfg: dict = None,
        is_augment: bool = False,
        drop_split: float = 0,
        **kwargs
):
    gt_dir_name = None
    gt_suffix = None
    # Todo: add dataset , modify below
    if dataset_name in [C.DDR_GRADING]:
        img_dir_name = ""
        img_suffix = ".jpg"
    elif dataset_name in [C.APTOS2019]:
        img_dir_name = ""
        img_suffix = ".png"
    elif dataset_name in [C.DDR_SEG, C.IDRiD]:
        img_dir_name = "image"
        img_suffix = ".jpg"
        gt_dir_name = "annotation"
        gt_suffix = ".png"

        # datasets.labels2annotations(
        #     dataset_dir / "image",
        #     dataset_dir / "label",
        #     dataset_dir / "annotation",
        #     ".jpg", ".tif", C.DATASET_CLASSES[dataset_name], dataset_type, force=False
        # )
    elif dataset_name in [C.TJDR]:
        img_dir_name = "image"
        img_suffix = ".png"
        gt_dir_name = "annotation"
        gt_suffix = ".png"

        # datasets.labels2annotations(
        #     dataset_dir / "image",
        #     dataset_dir / "label",
        #     dataset_dir / "annotation",
        #     ".png", ".png", C.DATASET_CLASSES[dataset_name], dataset_type, force=False
        # )
    elif isinstance(dataset_name, list):
        img_dir, img_suffix, gt_dir, gt_suffix = {}, {}, {}, {}
        for d_n in dataset_name:
            if isinstance(dataset_dir, dict) and dataset_dir.get(d_n) is None:
                img_dir[d_n], img_suffix[d_n], gt_dir[d_n], gt_suffix[d_n] = None, None, None, None
                continue
            d_s = dataset_split[d_n] if isinstance(dataset_split, dict) else dataset_split
            i_aug = is_augment[d_n] if isinstance(is_augment, dict) else is_augment
            img_dir[d_n], img_suffix[d_n], gt_dir[d_n], gt_suffix[d_n] = generate_dataset_txt(
                d_n, dataset_dir[d_n], d_s, dataset_txt_path[d_n],
                dataset_type, non_iid_cfg, i_aug, drop_split, **kwargs
            )
        set_vals = list(set(list(img_dir.values())))
        if len(set_vals) == 1 and set_vals[0] is None:
            return None
        return img_dir, img_suffix, gt_dir, gt_suffix
    else:
        raise ValueError(dataset_name)

    img_dir = dataset_dir / img_dir_name
    gt_dir = (dataset_dir / gt_dir_name) if gt_dir_name is not None else None
    if is_augment and dataset_type == C.TRAIN:
        datasets.dataset_augment(dataset_name, img_dir, img_suffix, gt_dir, gt_suffix, force=False, **kwargs)
    if isinstance(dataset_txt_path, list):
        if isinstance(non_iid_cfg, dict) and non_iid_cfg[C.ENABLE] is True and dataset_split > 1:
            gamma = non_iid_cfg["gamma"]
            datasets.non_iid_dataset_txt_generate(img_dir, img_suffix, gt_dir, gt_suffix, dataset_split, dataset_txt_path, is_augment, C.DATASET_CLASSES[dataset_name], gamma, **kwargs)
        else:
            datasets.iid_dataset_txt_generate(img_dir, img_suffix, gt_dir, gt_suffix, dataset_split, dataset_txt_path, is_augment, drop_split, **kwargs)
    else:
        datasets.dataset_txt_generate(img_dir, img_suffix, gt_dir, gt_suffix, dataset_txt_path, is_augment, **kwargs)

    return str(img_dir), img_suffix, str(gt_dir), gt_suffix


def generate_from_yaml(yaml_path: Path, logger: Logger, draw_distribu: bool = True):
    base_config = gutil.load_yaml(yaml_path)
    model_name = base_config[C.MODEL][C.NAME_MODEL]
    dataset_name = base_config[C.DATASET][C.NAME_DATASET]
    non_iid_cfg = base_config[C.DATASET].get(C.NON_IID)
    is_augment = base_config[C.DATASET].get("data_augment", False)
    drop_split = float(base_config[C.DATASET].get("drop_split", 0))
    enable_dr_aug = bool(base_config[C.DATASET].get("enable_dr_aug", False))

    num_edges = base_config[C.CLOUD][C.NUM_EDGES]
    assert isinstance(num_edges, int)
    num_clients_per_edge = base_config[C.EDGE][C.NUM_CLIENTS]
    if isinstance(num_clients_per_edge, (list, tuple)):
        assert len(num_clients_per_edge) == num_edges
    else:
        num_clients_per_edge = [num_clients_per_edge] * num_edges
    num_clients = sum(num_clients_per_edge)

    if isinstance(dataset_name, list):
        assert num_edges % len(dataset_name) == 0
        assert num_clients % len(dataset_name) == 0
        dataset_name = sorted(dataset_name)
        dataset_split = {}
        for d_n in dataset_name:
            dataset_split[d_n] = num_clients // len(dataset_name)
    else:
        dataset_split = base_config[C.DATASET].get(C.DATASET_SPLIT, num_clients)

    logger.info("Generating configs ...")

    now_day = gutil.get_now_day()
    now_time = gutil.get_now_time()

    if isinstance(dataset_name, list):
        dataset_model_dir = Path("+".join(dataset_name), model_name)
    else:
        dataset_model_dir = Path(dataset_name, model_name)
    datetime_dir = Path(now_day, now_time)

    generate_dir = cur_dir / "generates"
    config_dir = generate_dir / "configs" / dataset_model_dir / datetime_dir

    if isinstance(dataset_name, list):
        generate_dataset_txt_dir = generate_dir / "datasets" / "+".join(dataset_name) / datetime_dir
    else:
        generate_dataset_txt_dir = generate_dir / "datasets" / dataset_name / datetime_dir

    logfile_dir = cur_dir / "logs" / dataset_model_dir / datetime_dir
    tbX_logfile_dir = logfile_dir / "tbX"
    log_record_dir = logfile_dir / "record"

    save_dir = cur_dir / "saves"
    weights_dir = save_dir / "weights" / dataset_model_dir / datetime_dir
    best_weights_dir = weights_dir / "best"
    predict_dir = save_dir / "predict" / dataset_model_dir

    config_dir = config_dir if base_config.get(C.DIR) is None else base_config[C.DIR].get("config_dir", config_dir)
    logfile_dir = logfile_dir if base_config.get(C.DIR) is None else base_config[C.DIR].get("logfile_dir", logfile_dir)
    tbX_logfile_dir = tbX_logfile_dir if base_config.get(C.DIR) is None else base_config[C.DIR].get("tbX_logfile_dir", tbX_logfile_dir)
    log_record_dir = log_record_dir if base_config.get(C.DIR) is None else base_config[C.DIR].get("log_record_dir", log_record_dir)
    weights_dir = weights_dir if base_config.get(C.DIR) is None else base_config[C.DIR].get("weights_dir", weights_dir)
    best_weights_dir = best_weights_dir if base_config.get(C.DIR) is None else base_config[C.DIR].get("best_weights_dir", best_weights_dir)
    generate_dataset_txt_dir = generate_dataset_txt_dir if base_config.get(C.DIR) is None else base_config[C.DIR].get("generate_dataset_txt_dir", generate_dataset_txt_dir)
    predict_dir = predict_dir if base_config.get(C.DIR) is None else base_config[C.DIR].get("predict_dir", predict_dir)

    base_config[C.DIR] = dict(
        config_dir=str(config_dir),
        logfile_dir=str(logfile_dir),
        tbX_logfile_dir=str(tbX_logfile_dir),
        log_record_dir=str(log_record_dir),
        weights_dir=str(weights_dir),
        best_weights_dir=str(best_weights_dir),
        generate_dataset_txt_dir=str(generate_dataset_txt_dir),
        predict_dir=str(predict_dir)
    )

    dataset_dir = base_config[C.DATASET][C.DIR_DATASET]
    if isinstance(dataset_name, list):
        assert len(dataset_name) == len(dataset_dir), "{}!={}".format(len(dataset_name), len(dataset_dir))
        assert sorted(dataset_name) == sorted(list(dataset_dir.keys())), "{}!={}".format(sorted(dataset_name), sorted(list(dataset_dir.keys())))
        train_dataset_dir, val_dataset_dir, test_dataset_dir = {}, {}, {}
        for d_n in dataset_name:
            train_dataset_dir[d_n] = cur_dir / Path(dataset_dir[d_n][C.TRAIN])
            val_dataset_dir[d_n] = (cur_dir / Path(dataset_dir[d_n][C.VALIDATION])) if C.VALIDATION in dataset_dir[d_n] else None
            test_dataset_dir[d_n] = (cur_dir / Path(dataset_dir[d_n][C.TEST])) if C.TEST in dataset_dir[d_n] else None

            if not train_dataset_dir[d_n].exists():
                logger.error("please put train dataset in {}".format(train_dataset_dir[d_n]))
                exit(-1)
            if val_dataset_dir[d_n] and not val_dataset_dir[d_n].exists():
                logger.error("please put val dataset in {}".format(val_dataset_dir[d_n]))
                exit(-1)
            if test_dataset_dir[d_n] and not test_dataset_dir[d_n].exists():
                logger.error("please put test dataset in {}".format(test_dataset_dir[d_n]))
                exit(-1)
    else:
        train_dataset_dir = cur_dir / Path(dataset_dir[C.TRAIN])
        val_dataset_dir = (cur_dir / Path(dataset_dir[C.VALIDATION])) if C.VALIDATION in dataset_dir else None
        test_dataset_dir = (cur_dir / Path(dataset_dir[C.TEST])) if C.TEST in dataset_dir else None

        if not train_dataset_dir.exists():
            logger.error("please put train dataset in {}".format(train_dataset_dir))
            exit(-1)
        if val_dataset_dir and not val_dataset_dir.exists():
            logger.error("please put val dataset in {}".format(val_dataset_dir))
            exit(-1)
        if test_dataset_dir and not test_dataset_dir.exists():
            logger.error("please put test dataset in {}".format(test_dataset_dir))
            exit(-1)

    digit = len(str(num_clients))
    cloud_config_path = config_dir / "cloud-config.json"
    edge_configs_path = [config_dir / "edge#{:0>{}}-config.json".format(i, digit) for i in range(1, 1 + num_edges)]
    client_configs_path = [config_dir / "client#{:0>{}}-config.json".format(i, digit) for i in range(1, 1 + num_clients)]

    parent2children_config = {
        C.CLOUD: dict(),
        C.EDGE: dict(),
    }

    cloud_id = ID(0, uuid.uuid3(namespace=uuid.NAMESPACE_OID, name=str(cloud_config_path)).hex)
    edge_ids = [ID(i, uuid.uuid3(namespace=uuid.NAMESPACE_OID, name=str(edge_config_path)).hex) for i, edge_config_path in enumerate(edge_configs_path, start=1)]
    client_ids = [ID(i, uuid.uuid3(namespace=uuid.NAMESPACE_OID, name=str(client_config_path)).hex) for i, client_config_path in enumerate(client_configs_path, start=1)]

    for i, n in enumerate(num_clients_per_edge):
        for j in range(n):
            cur_client_id = client_ids[sum(num_clients_per_edge[:i]) + j]
            cur_client_id.set_parent_id(edge_ids[i])
            edge_ids[i].add_child_id(cur_client_id)
            if str(edge_configs_path[i]) in parent2children_config[C.EDGE]:
                parent2children_config[C.EDGE][str(edge_configs_path[i])].append(str(client_configs_path[sum(num_clients_per_edge[:i]) + j]))
            else:
                parent2children_config[C.EDGE][str(edge_configs_path[i])] = [str(client_configs_path[sum(num_clients_per_edge[:i]) + j])]
        edge_ids[i].set_parent_id(cloud_id)
        cloud_id.add_child_id(edge_ids[i])
        if str(cloud_config_path) in parent2children_config[C.CLOUD]:
            parent2children_config[C.CLOUD][str(cloud_config_path)].append(str(edge_configs_path[i]))
        else:
            parent2children_config[C.CLOUD][str(cloud_config_path)] = [str(edge_configs_path[i])]

    if isinstance(dataset_name, list):
        train_dataset_txt_path_list, val_dataset_txt_path, test_dataset_txt_path = {}, {}, {}
        for i, d_n in enumerate(dataset_name):
            cur_generate_dataset_txt_dir = generate_dataset_txt_dir / d_n
            cur_generate_dataset_txt_dir.mkdir(exist_ok=True, parents=True)
            train_dataset_txt_path_list[d_n] = [cur_generate_dataset_txt_dir / "client#{:0>{}}-train.txt".format(i, digit) for i in range(1 + i * dataset_split[d_n], 1 + (i + 1) * dataset_split[d_n])]
            val_dataset_txt_path[d_n] = (cur_generate_dataset_txt_dir / "{}.txt".format(C.VALIDATION)) if val_dataset_dir[d_n] else None
            test_dataset_txt_path[d_n] = (cur_generate_dataset_txt_dir / "{}.txt".format(C.TEST)) if test_dataset_dir[d_n] else None
    else:
        train_dataset_txt_path_list = [generate_dataset_txt_dir / "client#{:0>{}}-train.txt".format(i, digit) for i in range(1, 1 + num_clients)]
        val_dataset_txt_path = (generate_dataset_txt_dir / "{}.txt".format(C.VALIDATION)) if val_dataset_dir else None
        test_dataset_txt_path = (generate_dataset_txt_dir / "{}.txt".format(C.TEST)) if test_dataset_dir else None

    config_dir.mkdir(parents=True, exist_ok=True)
    tbX_logfile_dir.mkdir(parents=True, exist_ok=True)
    log_record_dir.mkdir(parents=True, exist_ok=True)
    best_weights_dir.mkdir(parents=True, exist_ok=True)
    generate_dataset_txt_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(yaml_path, config_dir / yaml_path.name)
    # gutil.write_json(config_dir / (yaml_path.stem + ".json"), base_config, mode="w+", indent=4)

    predict_info = dict()
    train_dataset_info = generate_dataset_txt(dataset_name, train_dataset_dir, dataset_split, train_dataset_txt_path_list, C.TRAIN, non_iid_cfg, is_augment, drop_split, enable_dr_aug=enable_dr_aug)
    if draw_distribu:
        drawer = IMGDrawer(logger)
        drawer.draw_train_dataset_distribu_hist(train_dataset_txt_path_list, None, C.TRAIN, dataset_name, base_config[C.MODEL][C.NUM_CLASSES], save_path=generate_dataset_txt_dir / "train_hist.png")
        drawer.draw_dataset_distribu_heatmap(train_dataset_txt_path_list, None, C.TRAIN, dataset_name, base_config[C.MODEL][C.NUM_CLASSES], save_path=generate_dataset_txt_dir / "train_heatmap.png")
    predict_info[C.TRAIN] = {
        "image_dir": train_dataset_info[0],
        "image_suffix": train_dataset_info[1]
    }
    if train_dataset_info[2] is not None:
        predict_info[C.TRAIN].update(
            {
                "gt_dir": train_dataset_info[2],
                "gt_suffix": train_dataset_info[3]
            }
        )
    if val_dataset_txt_path:
        val_dataset_info = generate_dataset_txt(dataset_name, val_dataset_dir, 1, val_dataset_txt_path, C.VALIDATION)
        if val_dataset_info is not None:
            if draw_distribu:
                drawer.draw_train_dataset_distribu_hist(
                    [val_dataset_txt_path], None, C.VALIDATION, dataset_name, base_config[C.MODEL][C.NUM_CLASSES], save_path=generate_dataset_txt_dir / "val_hist.png"
                )
                drawer.draw_dataset_distribu_heatmap(
                    [val_dataset_txt_path], None, C.VALIDATION, dataset_name, base_config[C.MODEL][C.NUM_CLASSES], save_path=generate_dataset_txt_dir / "val_heatmap.png"
                )
            predict_info[C.VALIDATION] = {
                "image_dir": val_dataset_info[0],
                "image_suffix": val_dataset_info[1]
            }
            if val_dataset_info[2] is not None:
                predict_info[C.VALIDATION].update(
                    {
                        "gt_dir": val_dataset_info[2],
                        "gt_suffix": val_dataset_info[3]
                    }
                )
    if test_dataset_txt_path:
        test_dataset_info = generate_dataset_txt(dataset_name, test_dataset_dir, 1, test_dataset_txt_path, C.TEST)
        if test_dataset_info is not None:
            if draw_distribu:
                drawer.draw_train_dataset_distribu_hist(
                    [test_dataset_txt_path], None, C.TEST, dataset_name, base_config[C.MODEL][C.NUM_CLASSES], save_path=generate_dataset_txt_dir / "test_hist.png"
                )
                drawer.draw_dataset_distribu_heatmap(
                    [test_dataset_txt_path], None, C.TEST, dataset_name, base_config[C.MODEL][C.NUM_CLASSES], save_path=generate_dataset_txt_dir / "test_heatmap.png"
                )
            predict_info[C.TEST] = {
                "image_dir": test_dataset_info[0],
                "image_suffix": test_dataset_info[1]
            }
            if test_dataset_info[2] is not None:
                predict_info[C.TEST].update(
                    {
                        "gt_dir": test_dataset_info[2],
                        "gt_suffix": test_dataset_info[3]
                    }
                )

    # generate cloud config file
    cloud_cfg = base_config[C.CLOUD]
    cloud_cfg.update(base_config[C.FED])
    cloud_cfg.update(base_config[C.MODEL])
    cloud_cfg.update(base_config[C.DATASET])
    del cloud_cfg[C.DIR_DATASET]
    cloud_cfg[C.DIR_TBX_LOGFILE] = str(tbX_logfile_dir)
    cloud_cfg[C.DIR_RECORD_FILE] = str(log_record_dir)
    cloud_cfg[C.DIR_PREDICT] = str(predict_dir)
    cloud_cfg[C.DIR_WEIGHTS] = str(weights_dir)
    cloud_cfg[C.PATH_LOGFILE] = str(logfile_dir / "cloud.log")
    cloud_cfg[C.MULTI_DATASET] = isinstance(dataset_name, list)
    cloud_cfg[C.ID] = {
        C.NID: cloud_id.nid,
        C.FID: cloud_id.fid,
        C.SID: cloud_id.sid,
        "children_id": [child_id.fid for child_id in cloud_id.children_id],
    }
    cloud_cfg["predict_info"] = predict_info
    cloud_eval_types = list(cloud_cfg[C.CLOUD_EVAL].keys())
    cloud_cfg[C.PATH_BEST_WEIGHTS] = dict()
    for _type in [C.TRAIN, C.VALIDATION, C.TEST]:
        if _type in cloud_eval_types:
            cloud_cfg[C.PATH_BEST_WEIGHTS][_type] = str(best_weights_dir / "cloud-{}-best.pt".format(_type))
    gutil.write_json(cloud_config_path, cloud_cfg, mode="w+", indent=4)

    # generate edge configs file
    edge_cfg = base_config[C.EDGE]
    edge_cfg.update(base_config[C.FED])
    edge_cfg.update(base_config[C.MODEL])
    edge_cfg[C.NAME_DATASET] = base_config[C.DATASET][C.NAME_DATASET]
    edge_cfg[C.DIR_TBX_LOGFILE] = str(tbX_logfile_dir)
    edge_cfg[C.DIR_RECORD_FILE] = str(log_record_dir)
    edge_cfg[C.DIR_PREDICT] = str(predict_dir)
    edge_cfg[C.DIR_WEIGHTS] = str(weights_dir)

    for i, (edge_id, edge_config_path) in enumerate(zip(edge_ids, edge_configs_path)):
        if isinstance(dataset_name, list):
            edge_cfg[C.NAME_DATASET] = base_config[C.DATASET][C.NAME_DATASET][i // (num_edges // len(dataset_name))]
        edge_cfg[C.PATH_LOGFILE] = str(logfile_dir / "edge#{:0>{}}.log".format(edge_id.nid, digit))
        edge_cfg[C.ID] = {
            C.NID: edge_id.nid,
            C.FID: edge_id.fid,
            C.SID: edge_id.sid,
            "parent_id": edge_id.parent_id.fid,
            "children_id": [child_id.fid for child_id in edge_id.children_id],
        }
        edge_eval_types = list(edge_cfg[C.EDGE_EVAL].keys())
        edge_cfg[C.PATH_BEST_WEIGHTS] = dict()
        for _type in [C.TRAIN, C.VALIDATION, C.TEST]:
            if _type in edge_eval_types:
                edge_cfg[C.PATH_BEST_WEIGHTS][_type] = str(best_weights_dir / "edge#{:0>{}}-{}-best.pt".format(edge_id.nid, digit, _type))
        gutil.write_json(edge_config_path, edge_cfg, mode="w+", indent=4)

    # generate client configs file
    client_cfg = base_config[C.CLIENT]
    client_cfg.update(base_config[C.FED])
    client_cfg.update(base_config[C.MODEL])
    client_cfg.update(base_config[C.DATASET])
    client_cfg[C.DIR_RECORD_FILE] = str(log_record_dir)
    client_cfg.update({C.OPTIMIZER: base_config[C.OPTIMIZER]})
    client_cfg.update({C.LR_SCHEDULE: base_config.get(C.LR_SCHEDULE)})
    del client_cfg[C.DIR_DATASET]
    client_cfg[C.TOTAL_EPOCH] = cloud_cfg[C.EPOCH] * edge_cfg[C.EPOCH] * client_cfg[C.EPOCH]

    for i, client_config_path in enumerate(client_configs_path):
        if isinstance(dataset_name, list):
            cur_idx = i // (num_clients // len(dataset_name))
            cur_dataset_name = dataset_name[cur_idx]
            client_cfg[C.NAME_DATASET] = cur_dataset_name
            client_cfg[C.TRAIN] = str(train_dataset_txt_path_list[cur_dataset_name][i % (num_clients // len(dataset_name))])
            if val_dataset_txt_path[cur_dataset_name]:
                client_cfg[C.VALIDATION] = str(val_dataset_txt_path[cur_dataset_name])
            if test_dataset_txt_path[cur_dataset_name]:
                client_cfg[C.TEST] = str(test_dataset_txt_path[cur_dataset_name])
        else:
            client_cfg[C.TRAIN] = str(train_dataset_txt_path_list[i])
            if val_dataset_txt_path:
                client_cfg[C.VALIDATION] = str(val_dataset_txt_path)
            if test_dataset_txt_path:
                client_cfg[C.TEST] = str(test_dataset_txt_path)
        client_cfg[C.PATH_LOGFILE] = str(logfile_dir / "client#{:0>{}}.log".format(client_ids[i].nid, digit))
        client_cfg[C.ID] = {
            C.NID: client_ids[i].nid,
            C.FID: client_ids[i].fid,
            C.SID: client_ids[i].sid,
            "parent_id": client_ids[i].parent_id.fid,
        }
        gutil.write_json(client_config_path, client_cfg, mode="w+", indent=4)

    logger.info("Generate completed ~")
    logger.info("num_edge_configs:{}".format(len(edge_configs_path)))
    logger.info("num_client_configs:{}".format(len(client_configs_path)))

    return base_config[C.DIR], parent2children_config


def generate(base_config_path=base_cfg_path, logger: Logger = Logger()):
    if not isinstance(base_config_path, Path):
        base_config_path = Path(base_config_path)
    if not base_config_path.exists():
        logger.error("please make sure {} exists.".format(base_config_path))
        exit(-1)

    if base_config_path.suffix == ".yaml":
        return generate_from_yaml(base_config_path, logger)
    elif base_config_path.suffix == ".json":
        pass


if __name__ == "__main__":
    gutil.set_all_seed(1234)
    generate()
    pass
