from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet import __version__
from mmdet.apis import (get_root_logger, init_dist, set_random_seed,
                        train_detector)
from mmdet.models import build_detector
import numpy as np

cfg = Config.fromfile('./config/retinanet_r50_fpn_1x.py')
dataset = build_dataset(cfg.data.train, {'test_mode': True})
dataset.CLASSES = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '0')
dataset.flag = np.ones(len(dataset), dtype=np.uint8)
dataset.test_mode = False

# Init distributed env first, since logger depends on the dist info.
distributed = False

# init logger before other steps
logger = get_root_logger(cfg.log_level)
logger.info('Distributed training: {}'.format(distributed))

# Set random seed.
if cfg.seed is not None:
    logger.info('Set random seed to {}'.format(cfg.seed))
    set_random_seed(cfg.seed)

# Build the model.
model = build_detector(
    cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
model.CLASSES = dataset.CLASSES

if cfg.checkpoint_config is not None:
    # Save mmdet version, config file content and class names in
    # checkpoints as meta data
    cfg.checkpoint_config.meta = dict(
        mmdet_version=__version__,
        config=cfg.text,
        CLASSES=dataset.CLASSES)

# Train the model.
for epoch in range(20):
    if epoch >= 1:
        cfg.resume_from = './train_logs/latest.pth'
    cfg.total_epochs = epoch + 1
    train_detector(
        model,
        dataset,
        cfg,
        distributed=distributed,
        logger=logger)
