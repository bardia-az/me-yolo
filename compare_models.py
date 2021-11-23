# import torch

# # Model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5m')  # or yolov5m, yolov5l, yolov5x, custom

# # Images
# img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# # Inference
# results = model(img)

# # Results
# results.print()  # or .show(), .save(), .crop(), .pandas(), etc.


import torch
import argparse
import logging
import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, SGD, lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = ROOT.relative_to(Path.cwd())  # relative

import val  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.supplemental import AutoEncoder
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    strip_optimizer, get_latest_run, check_dataset, check_git_status, check_img_size, check_requirements, \
    check_file, check_yaml, check_suffix, print_args, print_mutation, set_logging, one_cycle, colorstr, methods, print_args_to_file
from utils.downloads import attempt_download
from utils.loss import ComputeLoss
from utils.plots import plot_labels, plot_evolve
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, select_device, \
    torch_distributed_zero_first
# from utils.loggers.wandb.wandb_utils import check_wandb_resume
from utils.metrics import fitness
from utils.loggers import Loggers
from utils.callbacks import Callbacks

def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1', type=str, default=None, help='model1 weights path')
    parser.add_argument('--model2', type=str, default=None, help='model2 weights path')
    args = parser.parse_args()

    ckpt = torch.load(args.model1)
    model = Model(cfg='models/hub/yolov5x6.yaml', ch=3)  # create
    csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
    model.load_state_dict(csd, strict=False)  # load

    ckpt2 = torch.load(args.model2)
    model2 = Model(cfg='models/hub/yolov5x6.yaml', ch=3) # create
    csd2 = ckpt2['model'].float().state_dict()  # checkpoint state_dict as FP32
    model2.load_state_dict(csd2, strict=False)  # load

    compare_models(model,model2)

    print(type(csd))
    # print(csd.size())


# if __name__ == "__main__":
#     ckpt = torch.load('runs/train/simple_autoenc_x_640_/weights/supp_best.pt')
#     model = AutoEncoder([320, 64])  # create
#     model.load_state_dict(ckpt['model'])

#     ckpt2 = torch.load('runs/train/simple_autoenc_x_640_/weights/supp_last.pt')
#     model2 = AutoEncoder([320, 64])  # create
#     model2.load_state_dict(ckpt2['model'])
#     compare_models(model,model2)

    # print(type(csd))
    # print(csd.size())