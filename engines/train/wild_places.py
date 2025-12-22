"""
WildPlaces Trainer

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from packaging import version
from functools import partial
from pathlib import Path
from contextlib import nullcontext
from tensorboardX import SummaryWriter
import wandb


from .base import TRAINERS, TrainerBase
from .default import Trainer
from ..defaults import create_ddp_model, worker_init_fn
from ..hooks import HookBase, build_hooks
import utils.comm as comm
from models.losses.builder import build_criteria
from datasets import build_dataset, wildplaces_collate_fn, WildPlacesBatchSampler
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


@TRAINERS.register_module("WildPlacesTrainer")
class WildPlacesTrainer(Trainer):
    def __init__(self, cfg):
        super(WildPlacesTrainer, self).__init__(cfg)
        self.eval_interval_epoch = cfg.get("eval_interval_epoch", 5)
        self.build_loss_fn()

    def train(self):
        with EventStorage() as self.storage, ExceptionWriter():
            # => before train
            self.before_train()
            self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
            for self.epoch in range(self.start_epoch, self.max_epoch):
                #--BUG: Debug code
                world_size = dist.get_world_size() if dist.is_initialized() else 1
                rank = dist.get_rank() if dist.is_initialized() else 0

                print(f"[DEBUG] rank={rank} epoch={self.epoch} len(train_loader)={len(self.train_loader)}",
                    flush=True)
                #--

                # => before epoch
                if comm.get_world_size() > 1:
                    self.train_loader.batch_sampler.set_epoch(self.epoch)
                    bs = self.train_loader.batch_sampler
                    print(f"[DEBUG] rank={rank} epoch={self.epoch} "
                        f"num_batches_global={len(bs.batch_idx_all)} "
                        f"num_replicas={bs.num_replicas} drop_last={bs.drop_last}",
                        flush=True)
                
                self.model.train()
                self.data_iterator = enumerate(self.train_loader)
                self.before_epoch()
                step = 0 #--BUG: Debug code
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    #--BUG: Debug code
                    if step % 50 == 0 or step == len(self.train_loader) - 1:
                        print(f"[DEBUG] rank={rank} epoch={self.epoch} step={step}/{len(self.train_loader)}",
                            flush=True)
                    #--
                    step += 1
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def before_epoch(self):
        if self.cfg.model.head.type == "GeM":
            for p in self.model.head.parameters():
                p.requires_grad = False
        
        super().before_epoch()

    def after_epoch(self):
        if self.cfg.model.head.type == "GeM":
            if self.epoch < 3:
                for p in self.model.head.parameters():
                    p.requires_grad = False
            else:
                for p in self.model.head.parameters():
                    p.requires_grad = True
        
        super().after_epoch()

    def run_step(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            auto_cast = partial(torch.amp.autocast, device_type="cuda")
        else:
            # deprecated warning
            auto_cast = torch.cuda.amp.autocast

        input_dict = self.comm_info["input_dict"]
        batch_split_size = self.cfg.get("batch_split_size", 0)
        if batch_split_size is None or batch_split_size == 0:
            self.run_single_step(input_dict, 'train', auto_cast)
        else:
            self.run_multistaged_step(input_dict, 'train', auto_cast)

    def run_single_step(self, input_dict, phase, auto_cast=None):
        batch = input_dict['data']
        positives_mask = input_dict['positives_mask']
        negatives_mask = input_dict['negatives_mask']
        for key in batch.keys():
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].cuda(non_blocking=True)
    
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        if auto_cast is None:
            auto_cast_ctx = nullcontext()
        else:
            auto_cast_ctx = auto_cast(enabled=self.cfg.enable_amp, dtype=AMP_DTYPE[self.cfg.amp_dtype])

        with torch.set_grad_enabled(phase == 'train'):
            with auto_cast_ctx:
                y = self.model(batch)
                embeddings = y['global']
                input_dict['embeddings'] = embeddings
                loss, temp_stats = self.loss_fn.place_recognition_call(embeddings, positives_mask, negatives_mask)
                # temp_stats = self.tensors_to_numbers(temp_stats)

            self.optimizer.zero_grad()
            if self.cfg.enable_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                if self.cfg.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.clip_grad
                    )
                self.scaler.step(self.optimizer)

                # When enable amp, optimizer.step call are skipped if the loss scaling factor is too large.
                # Fix torch warning scheduler step before optimizer step.
                scaler = self.scaler.get_scale()
                self.scaler.update()
                if scaler <= self.scaler.get_scale():
                    self.scheduler.step()
            else:
                loss.backward()
                if self.cfg.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.clip_grad
                    )
                self.optimizer.step()
                self.scheduler.step()

        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
    
        self.comm_info["model_output_dict"] = temp_stats

    def run_multistaged_step(self, input_dict, phase, auto_cast=None):
        # Training step using multistaged backpropagation algorithm as per:
        # "Learning with Average Precision: Training Image Retrieval with a Listwise Loss"
        # This method will break when the model contains Dropout, as the same mini-batch will produce different embeddings.
        # Make sure mini-batches in step 1 and step 3 are the same (so that BatchNorm produces the same results)
        # See some exemplary implementation here: https://gist.github.com/ByungSun12/ad964a08eba6a7d103dab8588c9a3774
        batch = input_dict['data']
        positives_mask = input_dict['positives_mask']
        negatives_mask = input_dict['negatives_mask']

        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        if auto_cast is None:
            auto_cast_ctx = nullcontext()
        else:
            auto_cast_ctx = auto_cast(enabled=self.cfg.enable_amp, dtype=AMP_DTYPE[self.cfg.amp_dtype])

        # Stage 1 - calculate descriptors of each batch element (with gradient turned off)
        # In training phase network is in the train mode to update BatchNorm stats
        embeddings_l = []
        with torch.set_grad_enabled(False):
            with auto_cast_ctx:
                for minibatch in batch:
                    minibatch = {e: minibatch[e].cuda(non_blocking=True) for e in minibatch}
                    y = self.model(minibatch)
                    embeddings_l.append(y['global'])            

        torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

        # Stage 2 - compute gradient of the loss w.r.t embeddings
        embeddings = torch.cat(embeddings_l, dim=0)

        input_dict['embeddings'] = embeddings

        with torch.set_grad_enabled(phase == 'train'):
            with auto_cast_ctx:
                if phase == 'train':
                    embeddings.requires_grad_(True)
                loss, stats = self.loss_fn.place_recognition_call(input_dict)
                # stats = self.tensors_to_numbers(stats)

                if phase == 'train':
                    if self.cfg.enable_amp:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    embeddings_grad = embeddings.grad

        # Delete intermediary values
        embeddings_l, embeddings, y, loss = [None]*4

        # Stage 3 - recompute descriptors with gradient enabled and compute the gradient of the loss w.r.t.
        # network parameters using cached gradient of the loss w.r.t embeddings
        if phase == 'train':
            self.optimizer.zero_grad()
            i = 0
            with torch.set_grad_enabled(True):
                with auto_cast_ctx:
                    for minibatch in batch:
                        minibatch = {e: minibatch[e].cuda(non_blocking=True) for e in minibatch}
                        y = self.model(minibatch)
                        embeddings = y['global']
                        minibatch_size = len(embeddings)
                        # Compute gradients of network params w.r.t. the loss using the chain rule (using the
                        # gradient of the loss w.r.t. embeddings stored in embeddings_grad)
                        # By default gradients are accumulated
                        if self.cfg.enable_amp:
                            self.scaler.scale(embeddings).backward(gradient=embeddings_grad[i: i+minibatch_size])
                        else:
                            embeddings.backward(gradient=embeddings_grad[i: i+minibatch_size])
                        i += minibatch_size

                if self.cfg.enable_amp:
                    self.scaler.unscale_(self.optimizer)
                    if self.cfg.clip_grad is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.cfg.clip_grad
                        )
                    self.scaler.step(self.optimizer)

                    # When enable amp, optimizer.step call are skipped if the loss scaling factor is too large.
                    # Fix torch warning scheduler step before optimizer step.
                    scaler = self.scaler.get_scale()
                    self.scaler.update()
                    if scaler <= self.scaler.get_scale():
                        self.scheduler.step()
                else:
                    if self.cfg.clip_grad is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.cfg.clip_grad
                        )
                    self.optimizer.step()
                    self.scheduler.step()

        if self.cfg.empty_cache:
            torch.cuda.empty_cache() # Prevent excessive GPU memory consumption by SparseTensors
            
        self.comm_info["model_output_dict"] = stats

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()
        if self.cfg.empty_cache_per_epoch:
            torch.cuda.empty_cache()

    def build_model(self):
        model = build_model(self.cfg.model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # logger.info(f"Model: \n{self.model}")
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        return model

    def build_writer(self):
        writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
        self.logger.info(f"Tensorboard writer logging dir: {self.cfg.save_path}")
        if self.cfg.enable_wandb and comm.is_main_process():
            tag, name = Path(self.cfg.save_path).parts[-2:]
            wandb.init(
                project=self.cfg.wandb_project,
                name=f"{tag}/{name}",
                tags=[tag],
                dir=self.cfg.save_path,
                settings=wandb.Settings(api_key=self.cfg.wandb_key),
                config=self.cfg.to_dict(),
            )
        return writer

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)

        if comm.get_world_size() > 1:
            # train_sampler = WildPlacesBatchSamplerDDP(
            #     dataset=train_data,
            #     batch_size=self.cfg.batch_size_per_gpu,
            #     batch_size_limit=self.cfg.batch_size_per_gpu,
            #     batch_expansion_rate=self.cfg.get("batch_expansion_rate", None),
            #     max_batches=self.cfg.get("iter_per_epoch", None),
            #     seed=self.cfg.seed,
            # )
            assert False, "WildPlacesBatchSamplerDDP is not implemented yet."
        else:
            train_sampler = WildPlacesBatchSampler(
                dataset=train_data,
                batch_size=self.cfg.batch_size_per_gpu,
                batch_size_limit=self.cfg.batch_size_per_gpu,
                batch_expansion_rate=self.cfg.get("batch_expansion_rate", None),
                max_batches=self.cfg.get("iter_per_epoch", None),
            )

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )
        batch_split_size = self.cfg.get("batch_split_size", 0)
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=1,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            batch_sampler=train_sampler,
            collate_fn=partial(wildplaces_collate_fn, batch_split_size=batch_split_size, mix_prob=self.cfg.mix_prob),
            worker_init_fn=init_fn,
            pin_memory=False, persistent_workers=False
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                # val_sampler = WildPlacesBatchSamplerDDP(
                #     dataset=val_data,
                #     batch_size=self.cfg.batch_size_val_per_gpu,
                #     seed=self.cfg.seed,
                #     drop_last=False,
                # )
                assert False, "WildPlacesBatchSamplerDDP is not implemented yet."
            else:
                val_sampler = WildPlacesBatchSampler(
                    dataset=val_data,
                    batch_size=self.cfg.batch_size_val_per_gpu,
                )

            batch_split_size = self.cfg.get("batch_split_size", 0)
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=1,
                shuffle=False,
                batch_sampler=val_sampler,
                num_workers=self.cfg.num_worker_per_gpu,
                collate_fn=partial(wildplaces_collate_fn, batch_split_size=batch_split_size, mix_prob=self.cfg.mix_prob),
                pin_memory=False, persistent_workers=False
            )
        return val_loader

    def build_optimizer(self):
        return build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        self.cfg.scheduler.total_steps = len(self.train_loader) * self.cfg.eval_epoch
        return build_scheduler(self.cfg.scheduler, self.optimizer)
    

    def build_scaler(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            grad_scaler = partial(torch.amp.GradScaler, device="cuda")
        else:
            # deprecated warning
            grad_scaler = torch.cuda.amp.GradScaler
        scaler = grad_scaler() if self.cfg.enable_amp else None
        return scaler
    
    def build_loss_fn(self):
        self.loss_fn = build_criteria(self.cfg.model.criteria)

    def tensors_to_numbers(self, stats):
        stats = {e: stats[e].item() if torch.is_tensor(stats[e]) else stats[e] for e in stats}
        return stats