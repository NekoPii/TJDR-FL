#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: maojingxin
"""
import codecs
import copy
import inspect
import json
import logging
import os
import pickle
import random
import signal
import sys
import time
import uuid
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import List

import cv2 as cv
import math
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision.transforms.functional as tF
import yaml
from PIL import Image as PImage, ImageDraw as PImageDraw
from bidict import namedbidict
# matplotlib.use("Agg")
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm

root_dir = Path(__file__).absolute().resolve().parent.parent  # ...TJDR-FL/
if str(root_dir) != sys.path[0]:
    sys.path[0] = str(root_dir)

from gutils import Logger, constants as C
from gutils.id import ID


def generate_bidict(left_key_name: str, right_key_name: str):
    Bidict = namedbidict("Bidict", left_key_name, right_key_name)
    return Bidict()


def set_all_seed(seed, check_deterministic=False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # LSTM(cuda>10.2)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(check_deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_id_connect(parent_cfg: dict, children_cfg: List[dict], parent_id: ID = None, children_id: List[ID] = None):
    parent_id = ID(parent_cfg[C.ID][C.NID], parent_cfg[C.ID][C.FID]) if parent_id is None else parent_id
    children_id = [ID(child_cfg[C.ID][C.NID], child_cfg[C.ID][C.FID]) for child_cfg in children_cfg] if children_id is None else children_id
    assert len(children_id) == len(children_cfg)
    for i, child_cfg in enumerate(children_cfg):
        if child_cfg[C.ID][C.FID] in parent_cfg[C.ID][C.CHILDREN_ID]:
            parent_id.add_child_id(children_id[i])
            children_id[i].set_parent_id(parent_id)
    return parent_id, children_id


def write_complete_log(config_dir):
    cur_task_dir = "task" if "task" in config_dir.parts else "task_text"
    completed_f_path = root_dir / cur_task_dir / "completed.log"
    f_logger = Logger(f_path=completed_f_path)

    completed_dates = set()
    if completed_f_path.exists():
        with open(completed_f_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                completed_datetime = line.split("--", 1)[0].split(":", 1)[1]
                completed_date = completed_datetime[:8]
                completed_dates.add(completed_date)
    now_date = get_now_day()
    if now_date not in completed_dates:
        f_logger.info("{}{}".format(now_date, "-" * 100))
    f_logger.info("{}--{}".format(get_now_daytime(), config_dir))


def write_start_log(start_log_path, config_dir):
    f_logger = Logger(f_path=start_log_path)

    started_dates = set()
    if start_log_path.exists():
        with open(start_log_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                started_datetime = line.split("--", 1)[0].split(":", 1)[1]
                started_date = started_datetime[:8]
                started_dates.add(started_date)
    now_date = get_now_day()
    if now_date not in started_dates:
        f_logger.info("{}{}".format(now_date, "-" * 100))
    f_logger.info("{}--{}".format(get_now_daytime(), config_dir))


def load_f(f_path):
    f_path = Path(f_path)
    if f_path.suffix == ".yaml":
        return load_yaml(f_path)
    elif f_path.suffix == ".json":
        return load_json(f_path)
    else:
        raise ValueError("{} suffix:{} error!".format(f_path, f_path.suffix))


def load_json(f_path):
    if isinstance(f_path, str):
        f_path = Path(f_path)
    if isinstance(f_path, Path) and not f_path.exists():
        return {}
    with open(f_path, "r") as f:
        return json.load(f)


def load_yaml(f_path, encoding="UTF-8"):
    with open(f_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_pkl(f_path, encoding="bytes"):
    with open(f_path, "rb") as f:
        return pickle.load(f, encoding=encoding)


def write_json(f_path, data, mode: str, **kwargs):
    with open(f_path, mode) as f:
        json.dump(data, f, **kwargs)


def write_yaml(f_path, data, mode: str, encoding="UTF-8"):
    with open(f_path, mode, encoding=encoding) as f:
        yaml.dump(data, f)


def yaml2json(yaml_data):
    json_data = json.dumps(yaml_data, indent=5, ensure_ascii=False)
    return json_data


def json2yaml(json_data):
    yaml_datas = yaml.dump(json_data, indent=5, sort_keys=False, allow_unicode=True)
    return yaml_datas


def ordered_yaml_load(stream, Loader=yaml.SafeLoader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass

    def _construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)
    return yaml.load(stream, OrderedLoader)


def ordered_yaml_dump(data, stream=None, Dumper=yaml.SafeDumper, object_pairs_hook=OrderedDict, **kwds):
    class OrderedDumper(Dumper):
        pass

    def _dict_representer(dumper, data):
        return dumper.represent_mapping(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data.items())

    OrderedDumper.add_representer(object_pairs_hook, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)


def log_acc(logger, acc, classes, dot=2, digit_pos=">", show_full_classname=False):
    assert digit_pos in ["^", "<", ">"]
    if logger is not None and acc is not None:
        digit_bit = dot + 1 + 3 + 1
        mean_type = acc.get("mean_type", "mean")
        if mean_type == "weighted":
            mean_type = "weigh"
        classes_format = [len(str(v)) if show_full_classname and len(str(v)) > digit_bit else digit_bit for v in list(classes)]
        logger.info(
            "{:<10}:{}".format(
                "classes", "{:{}8}|".format("[{}]".format(mean_type), digit_pos)
                           + "|".join(["{:{}{}}".format(str(v)[:classes_format[i]], digit_pos, classes_format[i]) for i, v in enumerate(list(classes))])
                           + "|"
            )
        )
        for k, v in acc.items():
            if k == "mean_type":
                continue
            if k == "Acc" or k == "PA":
                log_line = "{:<10}:".format(k + "(micro)")
            else:
                log_line = "{:<10}:".format(k)
            if "mean" in v:
                log_line += "{:{}8.{}f}".format(v["mean"], digit_pos, dot)
            for i, now_class in enumerate(classes):
                if now_class in v:
                    log_line += "|{:{}{}.{}f}".format(v[now_class], digit_pos, classes_format[i], dot)
            log_line += "|"
            logger.info(log_line)


def arrf2pd(file_path: Path, is_save=False):
    with open(file_path, encoding="utf-8") as f:
        head = []
        for line in f.readlines():
            if line.startswith("@attribute"):
                head.append(line.split()[1])
            elif line.startswith("@data"):
                break
        df = pd.read_csv(f, header=None)
        df.columns = head
    if is_save:
        df.to_csv(file_path.with_suffix(".csv"))
    return df


def get_now_day():
    return time.strftime("%Y%m%d", time.localtime())


def get_now_daytime():
    return time.strftime("%Y%m%d%H%M%S", time.localtime())


def get_now_time(strf: str = None):
    return time.strftime("%H%M%S" if strf is None else strf, time.localtime())


def kill(pid: int, logger=None):
    if logger is None:
        logger = Logger()

    if pid is None:
        logger.warning("pid is None.")
        return
    try:
        os.kill(pid, 0)
        os.kill(pid, signal.SIGKILL)
        logger.info("kill pid:[{}] success.".format(pid))
    except:
        logger.info("pid:[{}] not exists.".format(pid))


def init_log(name: str, log_file_path: str, debug: bool = False):
    logger = logging.getLogger(name)
    log_formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG if debug else logging.INFO)
    fh.setFormatter(log_formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    sh = logging.StreamHandler()
    sh.setLevel(logging.WARNING)
    sh.setFormatter(log_formatter)
    logger.addHandler(sh)

    logger.propagate = False
    logger = Logger(logger)

    return logger


def nan_to_num(x):
    if x is None:
        return 0
    if isinstance(x, (list, tuple)):
        x = [nan_to_num(_x) for _x in x]
        return x
    return np.nan_to_num(x)


def list_mean(list_, weight_list=None):
    if list_ is None:
        return None
    if weight_list is None:
        weight_list = [1] * len(list_)
    assert len(list_) == len(weight_list), "{}!={}".format(len(list_), len(weight_list))
    list_weight = list(zip(list_, weight_list))
    list_weight = list(filter(lambda x: x[0] is not None and not math.isnan(x[0]) and x[1] is not None and not math.isnan(x[1]), list_weight))
    list_, weight_list = zip(*list_weight)

    _len = sum(weight_list) if weight_list else len(list_)
    mean = 0
    for i, v in enumerate(list_):
        mean += (v * weight_list[i]) if weight_list else v
    mean /= _len
    if torch.is_tensor(mean):
        mean = mean.item()
    return mean


def dict_list_mean(dict_list, weight_list=None):
    if dict_list is None:
        return None
    if len(dict_list) == 0:
        return {}
    if weight_list is None:
        weight_list = [1] * len(dict_list)
    assert len(dict_list) == len(weight_list), "{}!={}".format(len(dict_list), len(weight_list))
    dict_weight = list(zip(dict_list, weight_list))
    dict_weight = list(filter(lambda x: x[0] is not None and x[1] is not None and not math.isnan(x[1]), dict_weight))
    dict_list, weight_list = zip(*dict_weight)

    _len = sum(weight_list)
    mean_dict = copy.deepcopy(dict_list[0])
    for k in dict_list[0].keys():
        if isinstance(mean_dict[k], str):
            continue
        elif isinstance(mean_dict[k], dict):
            mean_dict[k] = dict_list_mean([dl[k] for dl in dict_list], weight_list=weight_list)
        else:
            if isinstance(mean_dict[k], (list, tuple)):
                mean_dict[k] = np.array(mean_dict[k])
            mean_dict[k] = nan_to_num(mean_dict[k]) * nan_to_num(weight_list[0])
            for i, d in enumerate(dict_list[1:], start=1):
                if isinstance(d[k], (list, tuple)):
                    d[k] = np.array(d[k])
                mean_dict[k] += (nan_to_num(d[k]) * nan_to_num(weight_list[i]))
            mean_dict[k] /= _len
            if torch.is_tensor(mean_dict[k]):
                mean_dict[k] = mean_dict[k].item()
            if isinstance(mean_dict[k], np.ndarray):
                mean_dict[k] = list(mean_dict[k])
    return mean_dict


def check_same_shape(*tensors):
    shape = None
    assert tensors is not None
    for tensor in tensors:
        assert torch.is_tensor(tensor)
        if shape is None:
            shape = tensor.shape
        if tensor.shape == shape:
            shape = tensor.shape
        else:
            return False
    return True


def check_tensors_same(t1, t2):
    assert check_same_shape(t1, t2)
    return torch.allclose(t1, t2)


def check_prob(prob, dim=1):
    assert prob is not None
    sum_prob = prob.sum(axis=dim).type(torch.float32)
    ones = torch.ones_like(sum_prob, dtype=torch.float32)
    return torch.allclose(sum_prob, ones)


def model_l2(net1: nn.Module, net2: nn.Module):
    l2 = 0
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        l2 += torch.norm(param1 - param2, p=2)
    return l2


@torch.no_grad()
def model_dist(w1, w2, ignore_bn=False):
    dist = 0
    if isinstance(w1, Iterable) and isinstance(w2, Iterable) and len(w1) == len(w2) == 2:
        dist += model_dist(w1[0], w2[0], ignore_bn)
        dist += model_dist(w1[1], w2[1], ignore_bn)
    else:
        if ignore_bn:
            w1 = remove_layer(w1, "bn")
            w2 = remove_layer(w2, "bn")
        w1 = remove_layer(w1, ["running_mean", "running_var", "num_batches_tracked"])
        w2 = remove_layer(w2, ["running_mean", "running_var", "num_batches_tracked"])
        assert w1.keys() == w2.keys(), "model weights keys should be same"
        for k in w1.keys():
            dist += torch.norm(w1[k] - w2[k], p=2)
    return dist


def remove_layer(state_dict: dict, layer_name: str or List[str]):
    keys = list(state_dict.keys())
    copy_state_dict = copy.deepcopy(state_dict)

    if isinstance(layer_name, str):
        layer_name = list([layer_name])
    for k in keys:
        for l_n in layer_name:
            if l_n in k:
                copy_state_dict.pop(k)
                break
    return copy_state_dict


def remove_requires_false(state_dict: dict, inverse=False):
    copy_state_dict = copy.deepcopy(state_dict)
    keys = list(copy_state_dict.keys())

    for k in keys:
        if not (inverse ^ copy_state_dict[k].requires_grad):
            del copy_state_dict[k]
    return copy_state_dict


def is_weights_path(path):
    if isinstance(path, str):
        path = Path(path)
        return path.suffix in [".pt", ".pth", ".pkl"]
    elif isinstance(path, Path):
        return path.suffix in [".pt", ".pth", ".pkl"]
    else:
        return False


def obj2pickle(obj, file_path=None):
    if file_path is None:
        return codecs.encode(pickle.dumps(obj), "base64").decode()
    else:
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        if isinstance(file_path, Path):
            file_path = str(file_path)
        return file_path


def pickle2obj(pickle_or_filepath):
    # filepath
    if isinstance(pickle_or_filepath, str) and Path(pickle_or_filepath).suffix == ".pkl":
        with open(pickle_or_filepath, "rb") as f:
            obj = pickle.load(f)
    # pickle file
    else:
        obj = pickle.loads(codecs.decode(pickle_or_filepath.encode(), "base64"))
    return obj


def state_dict2multigpu(weights):
    new_weights = OrderedDict()
    for k, v in weights.items():
        if "module." not in k:
            k = "module." + k
        new_weights[k] = v
    weights = new_weights
    del new_weights
    return weights


@torch.no_grad()
def zero(x):
    _x = copy.deepcopy(x)
    for w in _x:
        _x[w].zero_()
    return _x


@torch.no_grad()
def zero_(x):
    for w in x:
        x[w].zero_()


def save_weights(weights, path):
    torch.save(weights, path)


def get_func_params(frame, f_path=None, logger=None):
    args, _, _, values = inspect.getargvalues(frame)
    func_name = inspect.getframeinfo(frame)[2]
    args = list(args)
    if "self" in args:
        args.remove("self")
    if "cls" in args:
        args.remove("cls")
    values = [values[arg] for arg in args]
    if f_path is not None:
        now_logger = Logger(f_path=f_path)
        now_logger.info("{} :".format(func_name))
        for arg, value in zip(args, values):
            now_logger.info("{} :{}".format(arg, value))
    if logger is not None:
        logger.info("{} :".format(func_name))
        for arg, value in zip(args, values):
            logger.info("{} :{}".format(arg, value))
    return func_name, args, values


def get_ori_lines(f_path: str):
    assert Path(f_path).exists()
    with open(f_path, "r") as f:
        lines = f.readlines()
        return lines


def modify_file(f_path: str, content: dict):
    assert Path(f_path).exists()
    with open(f_path, "r") as f:
        lines = f.readlines()
    with open(f_path, "w+") as f:
        for l_id, line in enumerate(lines):
            for k, v in content.items():
                if k == C.FED and C.FED in line:
                    mode_line = lines[l_id + 1]
                    mode = mode_line.strip().split(": ", 1)[1].strip()
                    mode_line = mode_line.replace(mode, v)
                    lines[l_id + 1] = mode_line
                elif k == C.NON_IID and C.NON_IID in line:
                    enable_line = lines[l_id + 1]
                    if "false" in enable_line:
                        enable_line = enable_line.replace("false", v)
                    elif "true" in enable_line:
                        enable_line = enable_line.replace("true", v)
                    lines[l_id + 1] = enable_line
                elif k in line:
                    l = line
                    old_v = l.strip().split(": ", 1)[1].strip()
                    l = l.replace(old_v, v)
                    lines[l_id] = l
            f.writelines(lines[l_id])


def recovery_file(f_path: str, ori_lines: list):
    with open(f_path, "w+") as f:
        for line in ori_lines:
            f.writelines(line)


def check_list_mono(_list: list, increase=True, strict=True):
    ov = _list[0]
    for v in _list[1:]:
        if increase:
            if (strict and ov >= v) or (not strict and ov > v):
                return False
        if not increase:
            if (strict and ov <= v) or (not strict and ov < v):
                return False
        ov = v
    return True


def one_hot(labels, num_classes, ignore_label=255):
    """
    :param labels: [N] or [H,W] or [N,H,W], values in [0,num_classes)
    :param num_classes: C
    :param ignore_label: the value will be ignored
    :return: [N,C] or [C,H,W] or [N,C,H,W]
    """
    assert len(labels.shape) >= 1, "labels.shape:{} invalid".format(labels.shape)

    if isinstance(labels, np.ndarray):
        onehot = np.eye(num_classes)[labels]
        if len(onehot.shape) == 3:
            onehot = onehot.transpose((-1, 0, 1))
        if len(onehot.shape) == 4:
            onehot = onehot.transpose((0, -1, 1, 2))
        if ignore_label in labels:
            ignore = np.expand_dims(labels != ignore_label, axis=1)
            onehot = (onehot * ignore).astype(int)
    elif isinstance(labels, torch.Tensor):
        assert labels.dtype == torch.int64, "labels.dtype: {} valid, must be torch.long(torch.int64)".format(labels.dtype)
        onehot = F.one_hot(labels, num_classes)
        if len(onehot.shape) == 3:
            onehot = onehot.permute(-1, 0, 1)
        if len(onehot.shape) == 4:
            onehot = onehot.permute(0, -1, 1, 2)
        if ignore_label in labels:
            ignore = (labels != ignore_label).unsqueeze(1)
            onehot = onehot * ignore
    elif isinstance(labels, Iterable):
        onehot = []
        for label in labels:
            onehot.append(one_hot(label, num_classes, ignore_label))
        onehot = torch.stack(onehot, dim=0) if isinstance(onehot[0], torch.Tensor) else np.array(onehot)
    else:
        raise TypeError(type(labels))
    return onehot


def sharpen(preds, sharpen_coeff=1.0, logits=True):
    if logits:
        preds = F.softmax(preds, dim=1)
    if sharpen_coeff == 1.0:
        return preds
    sharpen_preds = preds ** (1 / sharpen_coeff)
    sharpen_preds = sharpen_preds / sharpen_preds.sum(dim=1, keepdim=True)
    return sharpen_preds


@torch.no_grad()
def get_pseudo_label(preds, logits=True, dim=1):
    if logits:
        preds = F.softmax(preds, dim=dim)
    pseudo_label = torch.argmax(preds, dim=dim).long()
    return pseudo_label


def clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    if img.dtype != np.uint8:
        raise TypeError("clahe supports only uint8 inputs")

    clahe_mat = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(img.shape) == 2 or img.shape[2] == 1:
        img = clahe_mat.apply(img)
    else:
        img = cv.cvtColor(img, cv.COLOR_RGB2LAB)
        img[:, :, 0] = clahe_mat.apply(img[:, :, 0])
        img = cv.cvtColor(img, cv.COLOR_LAB2RGB)

    return img


def getContours(img_path: str, contour_path: str, threshold=20, depth=0, use_clahe=False, draw_contour_only=True):
    if threshold < 0 or depth > 50:
        print("threshold:{} depth:{}".format(threshold, depth))
        return None, None

    img = cv.imread(img_path)
    if use_clahe:
        img = clahe(img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    h, w = gray.shape
    ret, binary = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)

    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # print(len(contours))
    max_index = 0
    max_size = 0

    if len(contours) == 0:
        return getContours(img_path, contour_path, threshold - 1, depth + 1, use_clahe)
    for i, contour in enumerate(contours):
        contour = np.array(contour)
        # print(contour.shape)
        c = 0
        for s in contour.shape:
            c += s
        if c > max_size:
            max_size = c
            max_index = i

    contours = [contours[max_index]]
    # print("=======")
    np_contours = np.array(contours)
    print(np_contours.shape)
    if np_contours.shape[1] <= 4:
        return getContours(img_path, contour_path, threshold + 1, depth + 1, use_clahe)
    if draw_contour_only:
        bg = np.zeros_like(img, dtype=np.uint8)
        cv.drawContours(bg, contours, 0, (0, 0, 255), 10)
        cv.imwrite(contour_path, bg)
    else:
        cv.drawContours(img, contours, 0, (0, 0, 255), 10)
        cv.imwrite(contour_path, img)

    np_contours = np.squeeze(np_contours, axis=None)
    minx = np.min(np_contours[:, 0])
    maxx = np.max(np_contours[:, 0])
    miny = np.min(np_contours[:, 1])
    maxy = np.max(np_contours[:, 1])

    x_len = 2 * (max(abs(w // 2 - minx), abs(maxx - w // 2)))
    y_len = 2 * (max(abs(h // 2 - miny), abs(maxy - h // 2)))

    size = max(x_len, y_len)
    return size, contours


def set_outer0(img_path: str, new_img_path: str, contours, test=False, mask_dir=None, **kwargs):
    img = cv.imread(img_path)
    h, w = img.shape[:2]
    mask = np.zeros((h, w)).astype(np.uint8)
    mask = cv.drawContours(mask, contours, 0, 255, -1)

    if test:
        img_p = Path(img_path)
        if mask_dir is None:
            mask_path = str(img_p.with_name("{}_mask.png".format(img_p.stem)))
        else:
            if not isinstance(mask_dir, Path):
                mask_dir = Path(mask_dir)
            mask_path = str(mask_dir / img_p.name)
        cv.imwrite(mask_path, mask)

    mask[mask > 0] = 1
    mask = mask.astype(np.bool)
    mask = np.expand_dims(mask, axis=-1)
    img = img * mask
    # for i in range(h):
    #     for j in range(w):
    #         if mask[i, j] == 0:
    #             img[i, j, :] *= 0

    cv.imwrite(new_img_path, img, [cv.IMWRITE_PNG_COMPRESSION, kwargs.get("png_compression", 3)])


def cal_dataset_norm(dataset_dir, img_suffix: str, img_size=None, dot: int = 4):
    if isinstance(dataset_dir, (list, tuple)):
        imgs_path_list = []
        for cur_dataset_dir in dataset_dir:
            if isinstance(cur_dataset_dir, str):
                cur_dataset_dir = Path(cur_dataset_dir)
            assert cur_dataset_dir.exists()
            imgs_path_list += list(cur_dataset_dir.glob("*{}".format(img_suffix)))
    else:
        if isinstance(dataset_dir, str):
            dataset_dir = Path(dataset_dir)
        assert dataset_dir.exists()
        imgs_path_list = list(dataset_dir.glob("*{}".format(img_suffix)))
    imgs_path_list = sorted(imgs_path_list)

    means, std = [], []
    img_list = []
    for path in tqdm(imgs_path_list, desc="dataset normalize"):
        img = PImage.open(path)  # C,H,W
        if img_size is not None:
            img = transforms.Resize(img_size)(img)
        img = np.asarray(img)  # H,W,C  RGB
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis, np.newaxis]
        img = img[:, :, :, np.newaxis]
        img_list.append(img)

    imgs = np.concatenate(img_list, axis=3)
    imgs = imgs.astype(np.float32) / 255.

    for i in range(imgs.shape[2]):
        pixels = imgs[:, :, i, :].ravel()  # flatten
        means.append(np.around(np.mean(pixels), dot))
        std.append(np.around(np.std(pixels), dot))

    print("mean:{}, std:{}".format(means, std))
    return means, std


def cal_pixel_counts(_dir, suffix: str, num_classes: int, ignore_label=0, fps: List[Path] = None):
    cal_fps = []
    if isinstance(_dir, list):
        for d in _dir:
            if not isinstance(d, Path):
                d = Path(d)
            cal_fps += list(d.glob("*{}".format(suffix)))
    else:
        if not isinstance(_dir, Path):
            _dir = Path(_dir)
        cal_fps += list(_dir.glob("*{}".format(suffix)))
    cal_fps = cal_fps if fps is None else fps
    print("len(cal_fps):{}".format(len(cal_fps)))

    pixel_counts = np.zeros(num_classes)
    lesion_counts = np.zeros(num_classes)

    for fp in tqdm(cal_fps):
        img = PImage.open(fp).convert("P")
        np_img = np.asarray(img)
        one_hot_img = one_hot(np_img, num_classes, -1)
        one_hot_sum = np.sum(one_hot_img, axis=(1, 2))
        pixel_counts += one_hot_sum
        lesion_counts += one_hot_sum > 0

    sum_pixel = np.sum(pixel_counts)

    if ignore_label in range(num_classes):
        pixel_counts = np.delete(pixel_counts, ignore_label)
        lesion_counts = np.delete(lesion_counts, ignore_label)

    sum_pixel_rate = np.sum(pixel_counts) / sum_pixel

    pixel_rate = pixel_counts / np.sum(pixel_counts)

    print("pixel_counts:{}".format(pixel_counts))
    print("lesion_counts:{}".format(lesion_counts))
    print("pixel_rate:{}".format(pixel_rate))
    print("sum_pixel_rate:{}".format(sum_pixel_rate))

    return pixel_counts, lesion_counts, pixel_rate, sum_pixel_rate


def get_resolution(image_dir, image_suffix: str):
    image_fps = []
    if isinstance(image_dir, list):
        for img_dir in image_dir:
            if not isinstance(image_dir, Path):
                img_dir = Path(img_dir)
            image_fps += list(img_dir.glob("*.{}".format(image_suffix)))
    else:
        if not isinstance(image_dir, Path):
            image_dir = Path(image_dir)
        image_fps = list(image_dir.glob("*.{}".format(image_suffix)))
    print("len(image):{}".format(len(image_fps)))

    resolus = set()

    min_resolu1 = np.array([np.Inf, np.inf])
    min_resolu2 = np.array([np.Inf, np.inf])
    max_resolu1 = np.array([0, 0])
    max_resolu2 = np.array([0, 0])

    for image_fp in tqdm(image_fps):
        image = PImage.open(image_fp)
        resolu = np.array([image.width, image.height])
        resolus.add(str(resolu))
        if resolu[0] < min_resolu1[0] or (resolu[0] == min_resolu1[0] and resolu[1] < min_resolu1[1]):
            min_resolu1 = resolu
        if resolu[1] < min_resolu2[1] or (resolu[1] == min_resolu2[1] and resolu[0] < min_resolu2[0]):
            min_resolu2 = resolu
        if resolu[0] > max_resolu1[0] or (resolu[0] == max_resolu1[0] and resolu[1] > max_resolu1[1]):
            max_resolu1 = resolu
        if resolu[1] > max_resolu2[1] or (resolu[1] == max_resolu2[1] and resolu[0] > max_resolu2[0]):
            max_resolu2 = resolu

    print("min_resolution:{},{}".format(min_resolu1, min_resolu2))
    print("max_resolution:{},{}".format(max_resolu1, max_resolu2))
    print("len(resolus):{}".format(len(resolus)))


def get_img_size(img):
    """Returns the size of an image as [height,width].

    Args:
        img (PIL Image or Tensor): The image to be checked.

    Returns:
        List[int]: The image size.
    """
    w, h = tF.get_image_size(img)
    return h, w


def tensor2npimg(input_image, mean: List[float] = None, std: List[float] = None, img_type=np.uint8):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        if mean is not None and std is not None:
            for i in range(len(mean)):
                image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
    else:
        image_numpy = input_image
    return image_numpy.astype(img_type)


def draw_class_predict(
        classes: list, img: PImage.Image, prob, gt: int, pred: int, save_path,
        verbose=True, figsize=(10, 10)
):
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.title(
        "gt_class:[{}] pred_class:[{}] prob:[{:.3f}]".format(classes[gt], classes[pred], prob[pred]),
        fontsize=20, pad=20
    )
    plt.axis("off")
    plt.xticks([]), plt.yticks([])
    plt.savefig(save_path)
    if verbose:
        print("[Info]:predict result saved : {}".format(save_path))
    plt.close()


@torch.no_grad()
def slide_inference(imgs, model, num_classes: int, crop_size, stride):
    """Inference by sliding-window with overlap.

    If h_crop > h_img or w_crop > w_img, the small patch will be used to
    decode without padding.
    """
    assert isinstance(crop_size, int) or (isinstance(crop_size, Iterable) and len(crop_size) == 2)
    assert isinstance(stride, int) or (isinstance(stride, Iterable) and len(stride) == 2)
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    h_crop, w_crop = crop_size
    h_stride, w_stride = stride
    batch_size, _, h_img, w_img = imgs.size()  # (N,C,H,W)
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    preds = imgs.new_zeros((batch_size, num_classes, h_img, w_img))
    count_mat = imgs.new_zeros((batch_size, 1, h_img, w_img))
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = imgs[:, :, y1:y2, x1:x2]
            crop_seg_logit = model(crop_img)
            preds += F.pad(crop_seg_logit, (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))
            count_mat[:, :, y1:y2, x1:x2] += 1

    assert (count_mat == 0).sum() == 0
    preds = preds / count_mat
    return preds


def labelme_shapes_to_label(img_path, img_shape, shapes, label_name_to_value, logger: Logger = Logger()):
    cls = np.zeros(img_shape[:2], dtype=np.int32)
    ins = np.zeros_like(cls)
    instances = []
    for shape in shapes:
        points = shape["points"]
        label = shape["label"]
        group_id = shape.get("group_id")
        if group_id is None:
            group_id = uuid.uuid1()
        shape_type = shape.get("shape_type", None)

        cls_name = label
        instance = (cls_name, group_id)

        if instance not in instances:
            instances.append(instance)
        ins_id = instances.index(instance) + 1
        cls_id = label_name_to_value[cls_name]

        mask = labelme_shape_to_mask(img_path, img_shape[:2], points, shape_type, logger=logger)
        cls[mask] = cls_id
        ins[mask] = ins_id

    return cls, ins


def labelme_shape_to_mask(
        img_path, img_shape, points, shape_type=None, line_width=10, point_size=5, logger: Logger = Logger()
):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PImage.fromarray(mask)
    draw = PImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == "circle":
        if len(xy) == 2:
            (cx, cy), (px, py) = xy
            d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
        else:
            logger.error("{}: Shape of shape_type=circle must have 2 points".format(img_path))
            return mask
    elif shape_type == "rectangle":
        assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == "line":
        assert len(xy) == 2, "Shape of shape_type=line must have 2 points"
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "linestrip":
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "point":
        assert len(xy) == 1, "Shape of shape_type=point must have 1 points"
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        if len(xy) > 2:
            draw.polygon(xy=xy, outline=1, fill=1)
        else:
            logger.error("{}: Polygon must have points more than 2".format(img_path))
            return mask
    mask = np.array(mask, dtype=bool)
    return mask


def plot_t_SNE(X: pd.DataFrame, Y: pd.DataFrame, save_path: Path):
    tsne = TSNE(n_components=2)
    X_tsne = pd.DataFrame(tsne.fit_transform(X)).rename(columns={0: "dim1", 1: "dim2"})
    data_tsne = pd.concat([X_tsne, Y], axis=1)
    plt.figure(figsize=(30, 30))
    sns.scatterplot(data_tsne, hue=Y.columns.tolist()[0], x="dim1", y="dim2", palette=sns.color_palette(palette="pastel"))
    plt.savefig(save_path)
    plt.show()
