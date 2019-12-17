# Some basic setup
# Setup detectron2 logger
import torch
import torchvision
import detectron2
from detectron2.utils.logger import setup_logger

# import some common libraries
import numpy as np
import os

# import some common detectron2 utilities
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

# Packages for dataset.
from detectron2.data.datasets import register_coco_instances


# Register the dataset.
setup_logger()
register_coco_instances(
    "train_dataset", {}, "data/pascal_train.json", "data/train_images")
register_coco_instances(
    "test_dataset", {}, "data/test.json", "data/test_images")

# Configuration.
cfg = get_cfg()
# Path to the config file.
cfg.merge_from_file(
    './detectron2/configs/COCO-InstanceSegmentation/' +
    'mask_rcnn_R_50_FPN_decay_0.01_momentum_0.9.yaml')
cfg.OUTPUT_DIR = './train_logs/'

# Start training.
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
