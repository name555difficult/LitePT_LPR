import os
import sys
import weakref
import wandb
import torch
import torch.nn as nn
import torch.utils.data
from packaging import version
from functools import partial
from pathlib import Path


if sys.version_info >= (3, 10):
    from collections.abc import Iterator
else:
    from collections import Iterator
from tensorboardX import SummaryWriter

from .base import TRAINERS
from .default import Trainer
from ..defaults import create_ddp_model, worker_init_fn
from ..hooks import HookBase, build_hooks
import utils.comm as comm
from datasets import build_dataset, point_collate_fn, collate_fn
from models import build_model
from utils.logger import get_root_logger
from utils.optimizer import build_optimizer
from utils.scheduler import build_scheduler
from utils.events import EventStorage, ExceptionWriter
from utils.registry import Registry

AMP_DTYPE = dict(
    float16=torch.float16,
    bfloat16=torch.bfloat16,
)

@TRAINERS.register_module("MultiDatasetTrainer")
class MultiDatasetTrainer(Trainer):
    def build_train_loader(self):
        from pointcept.datasets import MultiDatasetDataloader

        train_data = build_dataset(self.cfg.data.train)
        train_loader = MultiDatasetDataloader(
            train_data,
            self.cfg.batch_size_per_gpu,
            self.cfg.num_worker_per_gpu,
            self.cfg.mix_prob,
            self.cfg.seed,
        )
        self.comm_info["iter_per_epoch"] = len(train_loader)
        return train_loader
