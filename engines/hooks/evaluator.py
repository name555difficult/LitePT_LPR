import os
import faiss
import time
import numpy as np
import wandb
import torch
import pickle
from pathlib import Path
import torch.distributed as dist
import pointops
from uuid import uuid4
from functools import partial

from datasets import build_dataset, collate_fn, point_collate_fn
import utils.comm as comm
from utils.misc import intersection_and_union_gpu

from .default import HookBase
from .builder import HOOKS

from datasets.transform import Compose
from metrics.semantic import ConfusionMatrix

from torch_scatter import scatter_mean
from torch.nn.functional import normalize
from tqdm import tqdm
from utils.misc import (
    AverageMeter,
    intersection_and_union,
    intersection_and_union_gpu,
    make_dirs,
)


@HOOKS.register_module()
class SemSegEvaluator(HookBase):
    def __init__(self, write_cls_iou=True):
        self.write_cls_iou = write_cls_iou

    def before_train(self):
        if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:
            wandb.define_metric("val/*", step_metric="Epoch")
        
    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            output = output_dict["seg_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            segment = input_dict["segment"]

            if "inverse" in input_dict.keys():
                assert "origin_segment" in input_dict.keys()
                pred = pred[input_dict["inverse"]]
                segment = input_dict["origin_segment"]
            intersection, union, target = intersection_and_union_gpu(
                pred,
                segment,
                self.trainer.cfg.data.num_classes,
                self.trainer.cfg.data.ignore_index,
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            # Here there is no need to sync since sync happened in dist.all_reduce
            self.trainer.storage.put_scalar("val_intersection", intersection)
            self.trainer.storage.put_scalar("val_union", union)
            self.trainer.storage.put_scalar("val_target", target)
            self.trainer.storage.put_scalar("val_loss", loss.item())
            info = "Test: [{iter}/{max_iter}] ".format(
                iter=i + 1, max_iter=len(self.trainer.val_loader)
            )
            if "origin_coord" in input_dict.keys():
                info = "Interp. " + info
            self.trainer.logger.info(
                info
                + "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )
        loss_avg = self.trainer.storage.history("val_loss").avg
        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        m_iou = np.mean(iou_class)
        m_acc = np.mean(acc_class)
        all_acc = sum(intersection) / (sum(target) + 1e-10)
        self.trainer.logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                m_iou, m_acc, all_acc
            )
        )
        for i in range(self.trainer.cfg.data.num_classes):
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.trainer.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=acc_class[i],
                )
            )
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mIoU", m_iou, current_epoch)
            self.trainer.writer.add_scalar("val/mAcc", m_acc, current_epoch)
            self.trainer.writer.add_scalar("val/allAcc", all_acc, current_epoch)
            if self.trainer.cfg.enable_wandb:
                wandb.log(
                    {
                        "Epoch": current_epoch,
                        "val/loss": loss_avg,
                        "val/mIoU": m_iou,
                        "val/mAcc": m_acc,
                        "val/allAcc": all_acc,
                    },
                    step=wandb.run.step,
                )
            if self.write_cls_iou:
                for i in range(self.trainer.cfg.data.num_classes):
                    self.trainer.writer.add_scalar(
                        f"val/cls_{i}-{self.trainer.cfg.data.names[i]} IoU",
                        iou_class[i],
                        current_epoch,
                    )
                if self.trainer.cfg.enable_wandb:
                    for i in range(self.trainer.cfg.data.num_classes):
                        wandb.log(
                            {
                                "Epoch": current_epoch,
                                f"val/cls_{i}-{self.trainer.cfg.data.names[i]} IoU": iou_class[
                                    i
                                ],
                            },
                            step=wandb.run.step,
                        )
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = m_iou  # save for saver
        self.trainer.comm_info["current_metric_name"] = "mIoU"  # save for saver

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("mIoU", self.trainer.best_metric_value)
        )

@HOOKS.register_module()
class InsSegEvaluator(HookBase):
    def __init__(self, segment_ignore_index=(-1,), instance_ignore_index=-1):
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index

        self.valid_class_names = None  # update in before train
        self.overlaps = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
        self.min_region_sizes = 100
        self.distance_threshes = float("inf")
        self.distance_confs = -float("inf")

    def before_train(self):
        self.valid_class_names = [
            self.trainer.cfg.data.names[i]
            for i in range(self.trainer.cfg.data.num_classes)
            if i not in self.segment_ignore_index
        ]

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def associate_instances(self, pred, segment, instance):
        segment = segment.cpu().numpy()
        instance = instance.cpu().numpy()
        void_mask = np.in1d(segment, self.segment_ignore_index)

        assert (
            pred["pred_classes"].shape[0]
            == pred["pred_scores"].shape[0]
            == pred["pred_masks"].shape[0]
        )
        assert pred["pred_masks"].shape[1] == segment.shape[0] == instance.shape[0]
        # get gt instances
        gt_instances = dict()
        for i in range(self.trainer.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                gt_instances[self.trainer.cfg.data.names[i]] = []
        instance_ids, idx, counts = np.unique(
            instance, return_index=True, return_counts=True
        )
        segment_ids = segment[idx]
        for i in range(len(instance_ids)):
            if instance_ids[i] == self.instance_ignore_index:
                continue
            if segment_ids[i] in self.segment_ignore_index:
                continue
            gt_inst = dict()
            gt_inst["instance_id"] = instance_ids[i]
            gt_inst["segment_id"] = segment_ids[i]
            gt_inst["dist_conf"] = 0.0
            gt_inst["med_dist"] = -1.0
            gt_inst["vert_count"] = counts[i]
            gt_inst["matched_pred"] = []
            gt_instances[self.trainer.cfg.data.names[segment_ids[i]]].append(gt_inst)

        # get pred instances and associate with gt
        pred_instances = dict()
        for i in range(self.trainer.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                pred_instances[self.trainer.cfg.data.names[i]] = []
        instance_id = 0
        for i in range(len(pred["pred_classes"])):
            if pred["pred_classes"][i] in self.segment_ignore_index:
                continue
            pred_inst = dict()
            pred_inst["uuid"] = uuid4()
            pred_inst["instance_id"] = instance_id
            pred_inst["segment_id"] = pred["pred_classes"][i]
            pred_inst["confidence"] = pred["pred_scores"][i]
            pred_inst["mask"] = np.not_equal(pred["pred_masks"][i], 0)
            pred_inst["vert_count"] = np.count_nonzero(pred_inst["mask"])
            pred_inst["void_intersection"] = np.count_nonzero(
                np.logical_and(void_mask, pred_inst["mask"])
            )
            if pred_inst["vert_count"] < self.min_region_sizes:
                continue  # skip if empty
            segment_name = self.trainer.cfg.data.names[pred_inst["segment_id"]]
            matched_gt = []
            for gt_idx, gt_inst in enumerate(gt_instances[segment_name]):
                intersection = np.count_nonzero(
                    np.logical_and(
                        instance == gt_inst["instance_id"], pred_inst["mask"]
                    )
                )
                if intersection > 0:
                    gt_inst_ = gt_inst.copy()
                    pred_inst_ = pred_inst.copy()
                    gt_inst_["intersection"] = intersection
                    pred_inst_["intersection"] = intersection
                    matched_gt.append(gt_inst_)
                    gt_inst["matched_pred"].append(pred_inst_)
            pred_inst["matched_gt"] = matched_gt
            pred_instances[segment_name].append(pred_inst)
            instance_id += 1
        return gt_instances, pred_instances

    def evaluate_matches(self, scenes):
        overlaps = self.overlaps
        min_region_sizes = [self.min_region_sizes]
        dist_threshes = [self.distance_threshes]
        dist_confs = [self.distance_confs]

        # results: class x overlap
        ap_table = np.zeros(
            (len(dist_threshes), len(self.valid_class_names), len(overlaps)), float
        )
        for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
            zip(min_region_sizes, dist_threshes, dist_confs)
        ):
            for oi, overlap_th in enumerate(overlaps):
                pred_visited = {}
                for scene in scenes:
                    for _ in scene["pred"]:
                        for label_name in self.valid_class_names:
                            for p in scene["pred"][label_name]:
                                if "uuid" in p:
                                    pred_visited[p["uuid"]] = False
                for li, label_name in enumerate(self.valid_class_names):
                    y_true = np.empty(0)
                    y_score = np.empty(0)
                    hard_false_negatives = 0
                    has_gt = False
                    has_pred = False
                    for scene in scenes:
                        pred_instances = scene["pred"][label_name]
                        gt_instances = scene["gt"][label_name]
                        # filter groups in ground truth
                        gt_instances = [
                            gt
                            for gt in gt_instances
                            if gt["vert_count"] >= min_region_size
                            and gt["med_dist"] <= distance_thresh
                            and gt["dist_conf"] >= distance_conf
                        ]
                        if gt_instances:
                            has_gt = True
                        if pred_instances:
                            has_pred = True

                        cur_true = np.ones(len(gt_instances))
                        cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                        cur_match = np.zeros(len(gt_instances), dtype=bool)
                        # collect matches
                        for gti, gt in enumerate(gt_instances):
                            found_match = False
                            for pred in gt["matched_pred"]:
                                # greedy assignments
                                if pred_visited[pred["uuid"]]:
                                    continue
                                overlap = float(pred["intersection"]) / (
                                    gt["vert_count"]
                                    + pred["vert_count"]
                                    - pred["intersection"]
                                )
                                if overlap > overlap_th:
                                    confidence = pred["confidence"]
                                    # if already have a prediction for this gt,
                                    # the prediction with the lower score is automatically a false positive
                                    if cur_match[gti]:
                                        max_score = max(cur_score[gti], confidence)
                                        min_score = min(cur_score[gti], confidence)
                                        cur_score[gti] = max_score
                                        # append false positive
                                        cur_true = np.append(cur_true, 0)
                                        cur_score = np.append(cur_score, min_score)
                                        cur_match = np.append(cur_match, True)
                                    # otherwise set score
                                    else:
                                        found_match = True
                                        cur_match[gti] = True
                                        cur_score[gti] = confidence
                                        pred_visited[pred["uuid"]] = True
                            if not found_match:
                                hard_false_negatives += 1
                        # remove non-matched ground truth instances
                        cur_true = cur_true[cur_match]
                        cur_score = cur_score[cur_match]

                        # collect non-matched predictions as false positive
                        for pred in pred_instances:
                            found_gt = False
                            for gt in pred["matched_gt"]:
                                overlap = float(gt["intersection"]) / (
                                    gt["vert_count"]
                                    + pred["vert_count"]
                                    - gt["intersection"]
                                )
                                if overlap > overlap_th:
                                    found_gt = True
                                    break
                            if not found_gt:
                                num_ignore = pred["void_intersection"]
                                for gt in pred["matched_gt"]:
                                    if gt["segment_id"] in self.segment_ignore_index:
                                        num_ignore += gt["intersection"]
                                    # small ground truth instances
                                    if (
                                        gt["vert_count"] < min_region_size
                                        or gt["med_dist"] > distance_thresh
                                        or gt["dist_conf"] < distance_conf
                                    ):
                                        num_ignore += gt["intersection"]
                                proportion_ignore = (
                                    float(num_ignore) / pred["vert_count"]
                                )
                                # if not ignored append false positive
                                if proportion_ignore <= overlap_th:
                                    cur_true = np.append(cur_true, 0)
                                    confidence = pred["confidence"]
                                    cur_score = np.append(cur_score, confidence)

                        # append to overall results
                        y_true = np.append(y_true, cur_true)
                        y_score = np.append(y_score, cur_score)

                    # compute average precision
                    if has_gt and has_pred:
                        # compute precision recall curve first

                        # sorting and cumsum
                        score_arg_sort = np.argsort(y_score)
                        y_score_sorted = y_score[score_arg_sort]
                        y_true_sorted = y_true[score_arg_sort]
                        y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                        # unique thresholds
                        (thresholds, unique_indices) = np.unique(
                            y_score_sorted, return_index=True
                        )
                        num_prec_recall = len(unique_indices) + 1

                        # prepare precision recall
                        num_examples = len(y_score_sorted)
                        # https://github.com/ScanNet/ScanNet/pull/26
                        # all predictions are non-matched but also all of them are ignored and not counted as FP
                        # y_true_sorted_cumsum is empty
                        # num_true_examples = y_true_sorted_cumsum[-1]
                        num_true_examples = (
                            y_true_sorted_cumsum[-1]
                            if len(y_true_sorted_cumsum) > 0
                            else 0
                        )
                        precision = np.zeros(num_prec_recall)
                        recall = np.zeros(num_prec_recall)

                        # deal with the first point
                        y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                        # deal with remaining
                        for idx_res, idx_scores in enumerate(unique_indices):
                            cumsum = y_true_sorted_cumsum[idx_scores - 1]
                            tp = num_true_examples - cumsum
                            fp = num_examples - idx_scores - tp
                            fn = cumsum + hard_false_negatives
                            p = float(tp) / (tp + fp)
                            r = float(tp) / (tp + fn)
                            precision[idx_res] = p
                            recall[idx_res] = r

                        # first point in curve is artificial
                        precision[-1] = 1.0
                        recall[-1] = 0.0

                        # compute average of precision-recall curve
                        recall_for_conv = np.copy(recall)
                        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                        recall_for_conv = np.append(recall_for_conv, 0.0)

                        stepWidths = np.convolve(
                            recall_for_conv, [-0.5, 0, 0.5], "valid"
                        )
                        # integrate is now simply a dot product
                        ap_current = np.dot(precision, stepWidths)

                    elif has_gt:
                        ap_current = 0.0
                    else:
                        ap_current = float("nan")
                    ap_table[di, li, oi] = ap_current
        d_inf = 0
        o50 = np.where(np.isclose(self.overlaps, 0.5))
        o25 = np.where(np.isclose(self.overlaps, 0.25))
        oAllBut25 = np.where(np.logical_not(np.isclose(self.overlaps, 0.25)))
        ap_scores = dict()
        ap_scores["all_ap"] = np.nanmean(ap_table[d_inf, :, oAllBut25])
        ap_scores["all_ap_50%"] = np.nanmean(ap_table[d_inf, :, o50])
        ap_scores["all_ap_25%"] = np.nanmean(ap_table[d_inf, :, o25])
        ap_scores["classes"] = {}
        for li, label_name in enumerate(self.valid_class_names):
            ap_scores["classes"][label_name] = {}
            ap_scores["classes"][label_name]["ap"] = np.average(
                ap_table[d_inf, li, oAllBut25]
            )
            ap_scores["classes"][label_name]["ap50%"] = np.average(
                ap_table[d_inf, li, o50]
            )
            ap_scores["classes"][label_name]["ap25%"] = np.average(
                ap_table[d_inf, li, o25]
            )
        return ap_scores

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        scenes = []
        for i, input_dict in enumerate(self.trainer.val_loader):
            assert (
                len(input_dict["offset"]) == 1
            )  # currently only support bs 1 for each GPU
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)

            loss = output_dict["loss"]

            segment = input_dict["segment"]
            instance = input_dict["instance"]
            # map to origin
            if "origin_coord" in input_dict.keys():
                idx, _ = pointops.knn_query(
                    1,
                    input_dict["coord"].float(),
                    input_dict["offset"].int(),
                    input_dict["origin_coord"].float(),
                    input_dict["origin_offset"].int(),
                )
                idx = idx.cpu().flatten().long()
                output_dict["pred_masks"] = output_dict["pred_masks"][:, idx]
                segment = input_dict["origin_segment"]
                instance = input_dict["origin_instance"]

            gt_instances, pred_instance = self.associate_instances(
                output_dict, segment, instance
            )
            scenes.append(dict(gt=gt_instances, pred=pred_instance))

            self.trainer.storage.put_scalar("val_loss", loss.item())
            self.trainer.logger.info(
                "Test: [{iter}/{max_iter}] "
                "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )

        loss_avg = self.trainer.storage.history("val_loss").avg
        comm.synchronize()
        scenes_sync = comm.gather(scenes, dst=0)
        scenes = [scene for scenes_ in scenes_sync for scene in scenes_]
        ap_scores = self.evaluate_matches(scenes)
        all_ap = ap_scores["all_ap"]
        all_ap_50 = ap_scores["all_ap_50%"]
        all_ap_25 = ap_scores["all_ap_25%"]
        self.trainer.logger.info(
            "Val result: mAP/AP50/AP25 {:.4f}/{:.4f}/{:.4f}.".format(
                all_ap, all_ap_50, all_ap_25
            )
        )
        for i, label_name in enumerate(self.valid_class_names):
            ap = ap_scores["classes"][label_name]["ap"]
            ap_50 = ap_scores["classes"][label_name]["ap50%"]
            ap_25 = ap_scores["classes"][label_name]["ap25%"]
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: AP/AP50/AP25 {AP:.4f}/{AP50:.4f}/{AP25:.4f}".format(
                    idx=i, name=label_name, AP=ap, AP50=ap_50, AP25=ap_25
                )
            )
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mAP", all_ap, current_epoch)
            self.trainer.writer.add_scalar("val/AP50", all_ap_50, current_epoch)
            self.trainer.writer.add_scalar("val/AP25", all_ap_25, current_epoch)
            if self.trainer.cfg.enable_wandb:
                wandb.log(
                    {
                        "Epoch": current_epoch,
                        "val/mAP": all_ap,
                        "val/AP50": all_ap_50,
                        "val/AP25": all_ap_25,
                    },
                    step=wandb.run.step,
                )
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = all_ap_50  # save for saver
        self.trainer.comm_info["current_metric_name"] = "AP50"  # save for saver

@HOOKS.register_module()
class WildPlacesEvaluator(HookBase):
    def __init__(self, verbose=True, skip_same_run=True, 
                 eval_no_neighbors=False, no_neighbors_sample_ratio=0.2, 
                 auto_threshold_scale=1.0):
        self.verbose = verbose
        self.skip_same_run = skip_same_run
        self.eval_no_neighbors = eval_no_neighbors
        self.no_neighbors_sample_ratio = no_neighbors_sample_ratio
        self.auto_threshold_scale = auto_threshold_scale

    def before_train(self):
        if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:
            wandb.define_metric("val/*", step_metric="Epoch")

    def after_epoch(self):
        if self.trainer.cfg.evaluate and ((self.trainer.epoch+1) % self.trainer.eval_interval_epoch == 0 or self.trainer.epoch == 0):
            self.pickle_files = self.trainer.cfg.data.test.pickle_files  # 需要在配置中设置或通过其他方式传入
            self.eval()

    def eval(self):
        from datasets import point_collate_fn
        self.trainer.model.eval()
        logger = self.trainer.logger
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        # 初始化性能记录器
        batch_time = AverageMeter()
        recall_meter = AverageMeter()
        one_percent_recall_meter = AverageMeter()
        mrr_meter = AverageMeter()
        false_positive_rate_meter = AverageMeter()

         # 初始化累积统计
        total_recall = np.zeros(25)
        total_count = 0
        all_one_percent_recall = []
        all_mrr = []
        all_false_positive_rates = []

        database_embeddings = []
        query_embeddings = []

        if self.pickle_files is None:
            raise RuntimeError("pickle_files not set. Please configure the pickle files for evaluation.")
        
        if len(self.pickle_files) == 0:
            raise RuntimeError("pickle_files is empty. Please provide at least one pickle file pair for evaluation.")

        for pickle_idx, pickles in enumerate(self.pickle_files):
            logger.info(f"Processing pickle pair {pickle_idx + 1}/{len(self.pickle_files)}")
            # 0: database, 1: query
            database_pickle_file = pickles[0]
            if not hasattr(self.trainer.cfg.data.test, 'data_root'):
                logger.error("data_root not found in config.data.test")
                continue
            p = os.path.join(self.trainer.cfg.data.test.data_root, database_pickle_file)
            try:
                with open(p, 'rb') as f:
                    database_sets = pickle.load(f)
            except Exception as e:
                logger.error(f"Failed to load database pickle file {database_pickle_file}: {e}")
                continue

            query_pickle_file = pickles[1]
            p = os.path.join(self.trainer.cfg.data.test.data_root, query_pickle_file)
            try:
                with open(p, 'rb') as f:
                    query_sets = pickle.load(f)
            except Exception as e:
                logger.error(f"Failed to load query pickle file {query_pickle_file}: {e}")
                continue
            
            logger.info(f"Database sets: {len(database_sets)}, Query sets: {len(query_sets)}")
            
            if len(database_sets) == 0 or len(query_sets) == 0:
                logger.warning(f"Skipping pickle pair {pickle_idx + 1}: empty database or query sets")
                continue
                
            database_embeddings, query_embeddings = self.get_global_embeddings(database_sets, query_sets)
            
            if not database_embeddings or not query_embeddings:
                logger.warning(f"Skipping pickle pair {pickle_idx + 1}: failed to compute embeddings")
                continue

            # 新增：每个场景/level的指标统计
            level_total_recall = np.zeros(25)
            level_total_count = 0
            level_all_one_percent_recall = []
            level_all_mrr = []
            level_all_false_positive_rates = []

            tag, name = Path(self.trainer.cfg.save_path).parts[-2:]
            model_name = f"{self.trainer.cfg.wandb_project}_{tag}_{name}"

            for i in range(len(database_sets)):
                for j in range(len(query_sets)):
                    # whether skip the intra-sequence queries
                    if (i == j and self.skip_same_run): 
                        continue
                        
                    if i >= len(database_embeddings) or j >= len(query_embeddings):
                        logger.warning(f"Skipping pair [{i}, {j}]: index out of range")
                        continue
                        
                    if database_embeddings[i] is None or query_embeddings[j] is None: 
                        logger.warning(f"Skipping pair [{i}, {j}]: embeddings are None")
                        continue
                    if hasattr(self.trainer.cfg, 'dataset_type') and 'CSCampus3D' in self.trainer.cfg.dataset_type:
                        # For CSCampus3D, we report on the aerial-only database, which is idx 1
                        if i != 1:
                            continue
                    
                    start_time = time.time()
                    
                    pair_recall, pair_opr, pair_mrr, pair_fpr = self.get_recall(i, j, database_embeddings,
                                                                    query_embeddings, query_sets,
                                                                    database_sets, log=True, model_name=model_name)
                    
                    # 更新性能记录器
                    batch_time.update(time.time() - start_time)
                    recall_meter.update(pair_recall[0])  # 记录 top-1 recall
                    one_percent_recall_meter.update(pair_opr)
                    mrr_meter.update(pair_mrr)
                    false_positive_rate_meter.update(pair_fpr)
                    
                    # 累积统计
                    total_recall += np.array(pair_recall)
                    total_count += 1
                    all_one_percent_recall.append(pair_opr)
                    all_mrr.append(pair_mrr)
                    all_false_positive_rates.append(pair_fpr)

                    # level/场景内统计
                    level_total_recall += np.array(pair_recall)
                    level_total_count += 1
                    level_all_one_percent_recall.append(pair_opr)
                    level_all_mrr.append(pair_mrr)
                    level_all_false_positive_rates.append(pair_fpr)

                    # 记录当前进度
                    logger.info(
                        "Test: Database[{}/{}] Query[{}/{}] "
                        "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                        "Top1_Recall {recall:.4f} ({avg_recall:.4f}) "
                        "1%_Recall {opr:.4f} ({avg_opr:.4f}) "
                        "MRR {mrr:.4f} ({avg_mrr:.4f})".format(
                            i + 1, len(database_sets), j + 1, len(query_sets),
                            batch_time=batch_time,
                            recall=pair_recall[0], avg_recall=recall_meter.avg,
                            opr=pair_opr, avg_opr=one_percent_recall_meter.avg,
                            mrr=pair_mrr, avg_mrr=mrr_meter.avg
                        )
                    )

            # 每个level/场景的平均指标打印
            if level_total_count > 0:
                level_ave_recall = level_total_recall / level_total_count
                level_ave_one_percent_recall = np.mean(level_all_one_percent_recall)
                level_ave_mrr = np.mean(level_all_mrr)
                level_ave_false_positive_rate = np.mean(level_all_false_positive_rates) if level_all_false_positive_rates else 0

                logger.info(
                    "Level/场景 {} 结果: "
                    "1%_Recall {:.4f}, "
                    "Top1_Recall {:.4f}, "
                    "Top5_Recall {:.4f}, "
                    "Top10_Recall {:.4f}, "
                    "MRR {:.4f}, "
                    "False_Positive_Rate {:.4f}".format(
                        pickle_idx + 1,
                        level_ave_one_percent_recall,
                        level_ave_recall[0], level_ave_recall[4], level_ave_recall[9],
                        level_ave_mrr, level_ave_false_positive_rate
                    )
                )
            else:
                logger.warning(f"Level/场景 {pickle_idx + 1} 没有有效的评估对，请检查数据。")

        # 计算最终结果
        if total_count > 0:
            ave_recall = total_recall / total_count
            ave_one_percent_recall = np.mean(all_one_percent_recall)
            ave_mrr = np.mean(all_mrr)
            ave_false_positive_rate = np.mean(all_false_positive_rates) if all_false_positive_rates else 0
            
            stats = {
                'ave_one_percent_recall': ave_one_percent_recall.item(),
                'ave_recall_1': ave_recall[0].item(),
                'ave_recall_5': ave_recall[4].item(),
                'ave_recall_10': ave_recall[9].item(),
                'ave_mrr': ave_mrr.item(),
                'ave_false_positive_rate': ave_false_positive_rate.item() if all_false_positive_rates else 0.0
            }

            # 输出最终结果
            logger.info(
                "Final Results: "
                "1%_Recall {:.4f}, "
                "Top1_Recall {:.4f}, "
                "Top5_Recall {:.4f}, "
                "Top10_Recall {:.4f}, "
                "MRR {:.4f}, "
                "False_Positive_Rate {:.4f}".format(
                    ave_one_percent_recall,
                    ave_recall[0], ave_recall[4], ave_recall[9],
                    ave_mrr, ave_false_positive_rate
                )
            )
        else:
            stats = {}
            logger.warning("No valid evaluation pairs found! Please check your configuration and data.")

        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
    

        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            for e in stats.keys():
                self.trainer.writer.add_scalar(f"val/{e}", stats[e], current_epoch)
            # self.trainer.writer.add_scalars("val/stats", avg_stats, current_epoch)

            if self.trainer.cfg.enable_wandb:
                wandb_dict = {
                    "Val/Epoch": current_epoch,
                }
                for e in stats.keys():
                    wandb_dict[f"val/{e}"] = float(stats[e])
                wandb.log(
                    wandb_dict,
                    step=wandb.run.step,
                )

        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = stats['ave_recall_1']  # save for saver
        self.trainer.comm_info["current_metric_name"] = "ave_recall_1"  # save for saver

        return stats

    def get_global_embeddings(self, database_sets, query_sets):
        database_embeddings = []
        query_embeddings = []
        
        logger = self.trainer.logger
        logger.info("Computing database embeddings...")
        for data_set in tqdm(database_sets, desc='Computing database embeddings'):
            if hasattr(self, 'test_loader') and self.test_loader is not None:
                del self.test_loader
            self.test_loader = self.build_test_loader(data_set)
            database_embeddings.append(self.model_calculation(len(data_set)))
        
        logger.info("Computing query embeddings...")
        for query_set in tqdm(query_sets, desc='Computing query embeddings'):
            if hasattr(self, 'test_loader') and self.test_loader is not None:
                del self.test_loader
            self.test_loader = self.build_test_loader(query_set)
            query_embeddings.append(self.model_calculation(len(query_set)))
    
        return database_embeddings, query_embeddings

    def model_calculation(self, data_num):
        embeddings = None
        offset = 0
        
        self.trainer.model.eval()
        with torch.no_grad():
            for idx, input_dict in enumerate(self.test_loader):
                if type(input_dict) == list:
                    embeddings_input = []
                    for j, data_part_dict in enumerate(input_dict):
                        data_part_dict = point_collate_fn(data_part_dict)
                        for key in data_part_dict.keys():
                            if isinstance(data_part_dict[key], torch.Tensor):
                                data_part_dict[key] = data_part_dict[key].cuda(non_blocking=True)
                        
                        y = self.trainer.model(data_part_dict)
                        embedding_part = y['global'].detach().cpu().numpy()
                        embedding_part = embedding_part.mean(axis=0)  # average pooling
                        
                        torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors
                        embeddings_input.append(embedding_part)

                    embedding = np.stack(embeddings_input, axis=0)
                elif isinstance(input_dict, dict):
                    for key in input_dict.keys():
                        if isinstance(input_dict[key], torch.Tensor):
                            input_dict[key] = input_dict[key].cuda(non_blocking=True)
                    
                    y = self.trainer.model(input_dict)
                    embedding = y['global'].detach().cpu().numpy()
                    torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors
                else:
                    raise TypeError(f"Unexpected input_dict type: {type(input_dict)}")

                if embeddings is None:
                    embeddings = np.zeros((data_num, embedding.shape[1]), dtype=embedding.dtype)

                embeddings[offset:offset+embedding.shape[0]] = embedding
                offset += embedding.shape[0]

        return embeddings

    def get_recall(self, m, n, database_vectors, query_vectors, query_sets, database_sets,
                log=False, model_name: str = 'model'):
        # Original PointNetVLAD code (保留)
        database_output = database_vectors[m]
        queries_output = query_vectors[n]

        # ========= 用 FAISS 替代 KDTree =========
        # 当特征已 L2-normalize 时，L2 与 Cosine 排序一致；这里仍用 L2
        xb = np.ascontiguousarray(database_output.astype(np.float32))
        xq = np.ascontiguousarray(queries_output.astype(np.float32))
        d = xb.shape[1]

        # IndexFlatL2：最基础、无训练的暴力检索
        cpu_index = faiss.IndexFlatL2(d)

        # 可选：GPU 加速（若不可用则回退 CPU）
        use_gpu = getattr(self, "faiss_use_gpu", False)
        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            except Exception:
                # 安全降级
                index = cpu_index
        else:
            index = cpu_index

        index.add(xb)

        num_neighbors = 25
        # FAISS 要求 k <= 数据库大小
        k = min(num_neighbors, xb.shape[0])

        # 一次性批量检索 -> 显著减少 Python 循环与函数调用开销
        D_all, I_all = index.search(xq, k)     # D_all 为**平方**L2距离
        # 为保持与 KDTree.query 返回的“欧氏距离”一致，开根号
        D_all = np.sqrt(np.maximum(D_all, 0.0))

        recall = [0] * num_neighbors
        recall_idx = []
        one_percent_retrieved = 0
        threshold = max(int(round(len(database_output) / 100.0)), 1)

        num_evaluated = 0
        num_false_positives = 0
        num_no_neighbor_queries = 0

        # Collect queries with and without true neighbors (保留原逻辑)
        queries_with_neighbors = []
        queries_without_neighbors = []
        for i in range(len(queries_output)):
            query_details = query_sets[n][i]
            true_neighbors = query_details[m]
            if len(true_neighbors) == 0:
                queries_without_neighbors.append(i)
            else:
                queries_with_neighbors.append(i)

        # 采样无邻居 query（保留原逻辑）
        if self.eval_no_neighbors and len(queries_without_neighbors) > 0:
            sample_size = max(1, int(len(queries_without_neighbors) * self.no_neighbors_sample_ratio))
            sampled_no_neighbor_queries = np.random.choice(
                queries_without_neighbors,
                size=min(sample_size, len(queries_without_neighbors)),
                replace=False
            )
        else:
            sampled_no_neighbor_queries = []

        queries_to_evaluate = queries_with_neighbors + list(sampled_no_neighbor_queries)

        # ======== 自动阈值标定：改用批量检索结果 ========
        auto_threshold = None
        if self.eval_no_neighbors and len(queries_with_neighbors) > 0:
            true_neighbor_distances = []
            for i in queries_with_neighbors:
                query_details = query_sets[n][i]
                true_neighbors = query_details[m]
                if len(true_neighbors) > 0:
                    distances = D_all[i]  # shape: (k,)
                    indices = I_all[i]    # shape: (k,)
                    for j in range(len(indices)):
                        if indices[j] in true_neighbors:
                            true_neighbor_distances.append(distances[j])
                            break
            if true_neighbor_distances:
                avg_true_distance = float(np.mean(true_neighbor_distances))
                auto_threshold = avg_true_distance * self.auto_threshold_scale
                if log:
                    logger = self.trainer.logger
                    logger.info(
                        f"Auto-calibrated threshold: {auto_threshold:.4f} "
                        f"(avg_true_distance: {avg_true_distance:.4f}, scale: {self.auto_threshold_scale})"
                    )

        # ======== 主循环：改为直接用 D_all/I_all 中的结果 ========
        for i in queries_to_evaluate:
            query_details = query_sets[n][i]
            true_neighbors = query_details[m]

            # 取出该 query 的前 k 个候选
            distances = D_all[i][np.newaxis, :]   # 形状对齐原代码 (1, k)
            indices = I_all[i][np.newaxis, :]

            if len(true_neighbors) == 0:
                num_no_neighbor_queries += 1
                if self.eval_no_neighbors and auto_threshold is not None:
                    # top1 距离小于阈值则视为误报
                    if distances[0][0] <= auto_threshold:
                        num_false_positives += 1
                        if log:
                            fp_ndx = indices[0][0]
                            fp = database_sets[m][fp_ndx]
                            fp_emb_dist = distances[0, 0]
                            fp_world_dist = np.sqrt(
                                (query_details['northing'] - fp['northing']) ** 2 +
                                (query_details['easting'] - fp['easting']) ** 2
                            )
                            out_fp_file_name = f"{model_name}_log_fp_no_neighbors.txt"
                            with open(out_fp_file_name, "a") as f:
                                s = "{}, {}, {:0.2f}, {:0.2f}, NO_NEIGHBORS, THRESHOLD:{:.4f}\n".format(
                                    query_details['query'], fp['query'], fp_emb_dist, fp_world_dist, auto_threshold
                                )
                                f.write(s)
                continue

                # 有真邻居的才参与召回
            num_evaluated += 1

            if log:
                # 日志部分保持不变，只是数据来源换成 distances/indices
                if indices[0][0] not in true_neighbors:
                    fp_ndx = indices[0][0]
                    fp = database_sets[m][fp_ndx]
                    fp_emb_dist = distances[0][0]
                    fp_world_dist = np.sqrt(
                        (query_details['northing'] - fp['northing']) ** 2 +
                        (query_details['easting'] - fp['easting']) ** 2
                    )
                    tp = None
                    for kpos in range(len(indices[0])):
                        if indices[0][kpos] in true_neighbors:
                            closest_pos_ndx = indices[0][kpos]
                            tp = database_sets[m][closest_pos_ndx]
                            tp_emb_dist = distances[0][kpos]
                            tp_world_dist = np.sqrt(
                                (query_details['northing'] - tp['northing']) ** 2 +
                                (query_details['easting'] - tp['easting']) ** 2
                            )
                            break

                    out_fp_file_name = f"{model_name}_log_fp.txt"
                    with open(out_fp_file_name, "a") as f:
                        s = "{}, {}, {:0.2f}, {:0.2f}".format(
                            query_details['query'], fp['query'], fp_emb_dist, fp_world_dist
                        )
                        if tp is None:
                            s += ', 0, 0, 0\n'
                        else:
                            s += ', {}, {:0.2f}, {:0.2f}\n'.format(tp['query'], tp_emb_dist, tp_world_dist)
                        f.write(s)

                s = f"{query_details['query']}, {query_details['northing']}, {query_details['easting']}"
                # 只保存前5（若 k<5 则取 k）
                top_show = min(len(indices[0]), 5)
                for kpos in range(top_show):
                    is_match = indices[0][kpos] in true_neighbors
                    e_ndx = indices[0][kpos]
                    e = database_sets[m][e_ndx]
                    e_emb_dist = distances[0][kpos]
                    world_dist = np.sqrt(
                        (query_details['northing'] - e['northing']) ** 2 +
                        (query_details['easting'] - e['easting']) ** 2
                    )
                    s += f", {e['query']}, {e_emb_dist:0.2f}, , {world_dist:0.2f}, {1 if is_match else 0}, "
                s += '\n'
                out_top5_file_name = f"{model_name}_log_search_results.txt"
                with open(out_top5_file_name, "a") as f:
                    f.write(s)

            # 计算各阶 R@K
            for j in range(len(indices[0])):   # j < = k-1
                if indices[0][j] in true_neighbors:
                    recall[j] += 1
                    recall_idx.append(j + 1)
                    break

            if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
                one_percent_retrieved += 1

        one_percent_recall = (one_percent_retrieved / float(num_evaluated)) * 100 if num_evaluated > 0 else 0
        recall = (np.cumsum(recall) / float(num_evaluated)) * 100 if num_evaluated > 0 else np.zeros(num_neighbors)
        mrr = np.mean(1 / np.array(recall_idx)) * 100 if len(recall_idx) > 0 else 0

        false_positive_rate = 0
        if self.eval_no_neighbors and num_no_neighbor_queries > 0:
            false_positive_rate = (num_false_positives / num_no_neighbor_queries) * 100
            if log:
                logger = self.trainer.logger
                logger.info(
                    f"False positive rate for queries without neighbors: "
                    f"{false_positive_rate:.2f}% ({num_false_positives}/{num_no_neighbor_queries})"
                )

        return recall, one_percent_recall, mrr, false_positive_rate

    def build_test_loader(self, data_dict):
        from datasets import build_dataset
        self.trainer.cfg.data.test['data_dict'] = data_dict
        test_dataset = build_dataset(self.trainer.cfg.data.test)
        if comm.get_world_size() > 1:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.trainer.cfg.batch_size_test_per_gpu,
            shuffle=False,
            num_workers=self.trainer.cfg.num_worker_per_gpu,
            pin_memory=True,
            sampler=test_sampler,
            # collate_fn=self.__class__.collate_fn,
            collate_fn=partial(point_collate_fn, mix_prob=self.trainer.cfg.mix_prob),
        )
        return test_loader

    @staticmethod
    def collate_fn(batch):
        return batch

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("AP", self.trainer.best_metric_value)
        )