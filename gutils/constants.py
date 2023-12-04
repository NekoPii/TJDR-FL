#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: maojingxin
"""
from pathlib import Path

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent
ROOT_PATH_STR = str(ROOT_PATH)

DEFAULT_IGNORE_LABEL = -100
DEFAULT_SEED = 3407

IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]

DDR_SEG_MEAN = [0.4211, 0.2640, 0.1104]
DDR_SEG_STD = [0.3133, 0.2094, 0.1256]

DDR_CLS_MEAN = [0.414, 0.2567, 0.1257]
DDR_CLS_STD = [0.2959, 0.2057, 0.1397]

APTOS2019_MEAN = [0.4389, 0.2326, 0.0724]
APTOS2019_STD = [0.2711, 0.1497, 0.081]

TJDR_MEAN = [0.3678, 0.2001, 0.0958]
TJDR_STD = [0.2357, 0.1463, 0.0876]

IDRiD_MEAN = [0.5469, 0.2707, 0.0769]
IDRiD_STD = [0.3053, 0.1669, 0.0953]

"""
####################
general
####################
"""
NUM_THREADS = 6
DEBUG = False
ENABLE_ENCRYPT_EMIT = False

CLASSES = "classes"
IGNORE_LABEL = "ignore_label"
SSL = "SSL"
LABELED = "labeled"
UNLABELED = "unlabeled"
NON_IID = "non_iid"

TRAIN = "train"
VALIDATION = "val"
TEST = "test"

MODEL_MODE = "model_mode"
SEG = "seg"
CNN = "cnn"

FED = "fed"
FED_MODE = "fed_mode"
FED_PARAMS = "fed_params"
FedAvg = "FedAvg"
alphaFed = "alphaFed"

IMG_SIZE = "img_size"
NAME_DATASET = "dataset_name"
NAME_MODEL = "model_name"
DATASET_SPLIT = "dataset_split"
PTR_WEIGHTS = "ptr_weights"

CLIENT = "client"
EDGE = "edge"
CLOUD = "cloud"
MODEL = "model"
DATASET = "dataset"
WEIGHTS = "weights"
DIR = "dir"

METRIC = "metric"

ENABLE = "enable"
LABEL_TYPE = "label_type"

EPOCH = "epoch"
TOTAL_EPOCH = "total_epoch"
BATCH_SIZE = "batch_size"
LOSS = "loss"

OPTIMIZER = "optimizer"
TYPE = "type"
LR = "lr"
MOMENTUM = "momentum"
WEIGHT_DECAY = "weight_decay"

TRAIN_LOSS = "train_loss"
TRAIN_ACC = "train_acc"
TRAIN_CONTRIB = "train_contrib"

LR_SCHEDULE = "lr_schedule"
MILESTONE = "milestone"
POLY = "poly"
COS = "cos"
WARMUP_POLY = "warmup_poly"
CONSTANT = "constant"
LOG_ITER = "log_iter"

RAMPUP_SCHEDULE = "rampup_schedule"
ACC = "acc"
CONTRIB = "contribution"
EVAL_BATCH_SIZE = "eval_batch_size"

GRAD_ACCUMULATE = "accumulate_grad"

NUM = "num"
NUM_WORKERS = "num_workers"
NUM_CHANNELS = "num_channels"
NUM_CLASSES = "num_classes"
NUM_CLIENTS = "num_clients"
NUM_EDGES = "num_edges"

FRAC_JOIN = "frac_join"
MULTI_DATASET = "multi_dataset"

PATH_LOGFILE = "logfile_path"
PATH_BEST_WEIGHTS = "best_weights_path"
PATH_INIT_WEIGHTS = "init_weights_path"
PATH_DATASET = "dataset_path"

DIR_DATASET = "dataset_dir"
DIR_PREDICT = "predict_dir"
DIR_WEIGHTS = "weights_dir"
DIR_RECORD_FILE = "record_file_dir"
DIR_TBX_LOGFILE = "tbX_logfile_dir"

CLIENT_EVAL = "client_eval"
EDGE_EVAL = "edge_eval"
CLOUD_EVAL = "cloud_eval"

ID = "ID"
NID = "nid"
FID = "fid"
SID = "sid"
PID = "PID"
AMP = "amp"
CHILDREN_ID = "children_id"
PARENT_ID = "parent_id"

SEED = "seed"
RUN_SEED = "run_seed"
INIT_SEED = "init_seed"
MODE = "mode"
PARAMS = "params"
GAUSS = "gauss"
LINEAR = "linear"

HOST = "host"
PORT = "port"
FIN = "fin."

SGD = "SGD"
Adam = "Adam"

TASK = {}
DATASET_CLASSES = {}

# Todo: add method and dataset , modify belows

"""
####################
image classification
####################
"""
IMG_CLASSIFICATION = "img_classification"

DDR_GRADING = "DDR_grading"
APTOS2019 = "APTOS2019"

TASK.update(
    {
        DDR_GRADING: IMG_CLASSIFICATION,
        APTOS2019: IMG_CLASSIFICATION,
    }
)

DATASET_CLASSES.update(
    {
        DDR_GRADING: ["NoDR", "Mild", "Moderate", "Severe", "Proliferative DR", "Non-gradable"],
        APTOS2019: ["NoDR", "Mild", "Moderate", "Severe", "Proliferative DR"],
    }
)

"""
####################
image segmentation
####################
"""
IMG_SEGMENTATION = "img_segmentation"
SLIDE_INFERENCE = "slide_inference"
SLIDE_CROP_SIZE = "slide_crop_size"
SLIDE_STRIDE = "slide_stride"

CUTMIX = "cutmix"

ISIC = "ISIC"
DDR_SEG = "DDR_seg"
TJDR = "TJDR"
IDRiD = "IDRiD"

TASK.update(
    {
        ISIC: IMG_SEGMENTATION,
        DDR_SEG: IMG_SEGMENTATION,
        TJDR: IMG_SEGMENTATION,
        IDRiD: IMG_SEGMENTATION,
    }
)

DATASET_CLASSES.update(
    {
        DDR_SEG: ["bg", "EX", "HE", "MA", "SE"],
        TJDR: ["bg", "EX", "HE", "MA", "SE"],
        IDRiD: ["bg", "EX", "HE", "MA", "SE"],
    }
)
