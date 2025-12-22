import os
import time
import pickle
import numpy as np
from pathlib import Path
from collections import OrderedDict
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data
import tqdm
import faiss
from functools import partial

from .base import TesterBase, TESTERS
from ..defaults import create_ddp_model
import utils.comm as comm
from datasets import build_dataset, collate_fn, point_collate_fn
from models import build_model
from utils.logger import get_root_logger
from utils.registry import Registry
from utils.misc import (
    AverageMeter,
    intersection_and_union,
    intersection_and_union_gpu,
    make_dirs,
)

import wandb

@TESTERS.register_module()
class WildPlacesTester(TesterBase):
    def __init__(self, cfg, model=None, test_loader=None, verbose=False, 
        skip_same_run=False, eval_no_neighbors=False, 
        no_neighbors_sample_ratio=0.1, auto_threshold_scale=1.0
    ) -> None:
        test_loader = 1
        super().__init__(cfg, model, test_loader, verbose)
        self.skip_same_run = skip_same_run
        self.eval_no_neighbors = eval_no_neighbors
        self.no_neighbors_sample_ratio = no_neighbors_sample_ratio
        self.auto_threshold_scale = auto_threshold_scale
        self.pickle_files = self.cfg.data.test.pickle_files  # 需要在配置中设置或通过其他方式传入
        
    def test(self):
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

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
            if not hasattr(self.cfg.data.test, 'data_root'):
                logger.error("data_root not found in config.data.test")
                continue
            p = os.path.join(self.cfg.data.test.data_root, database_pickle_file)
            try:
                with open(p, 'rb') as f:
                    database_sets = pickle.load(f)
            except Exception as e:
                logger.error(f"Failed to load database pickle file {database_pickle_file}: {e}")
                continue

            query_pickle_file = pickles[1]
            p = os.path.join(self.cfg.data.test.data_root, query_pickle_file)
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

            tag, name = Path(self.cfg.save_path).parts[-2:]
            model_name = f"{self.cfg.wandb_project}_{tag}_{name}"

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
                    if hasattr(self.cfg, 'dataset_type') and 'CSCampus3D' in self.cfg.dataset_type:
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
                'ave_one_percent_recall': ave_one_percent_recall,
                'ave_recall': ave_recall,
                'ave_mrr': ave_mrr,
                'ave_false_positive_rate': ave_false_positive_rate
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
        return stats

    def get_global_embeddings(self, database_sets, query_sets):
        database_embeddings = []
        query_embeddings = []
        
        logger = get_root_logger()
        logger.info("Computing database embeddings...")
        for data_set in tqdm.tqdm(database_sets, desc='Computing database embeddings'):
            if hasattr(self, 'test_loader') and self.test_loader is not None:
                del self.test_loader
            self.test_loader = self.build_test_loader(data_set)
            database_embeddings.append(self.model_calculation(len(data_set)))
        
        logger.info("Computing query embeddings...")
        for query_set in tqdm.tqdm(query_sets, desc='Computing query embeddings'):
            if hasattr(self, 'test_loader') and self.test_loader is not None:
                del self.test_loader
            self.test_loader = self.build_test_loader(query_set)
            query_embeddings.append(self.model_calculation(len(query_set)))
    
        return database_embeddings, query_embeddings

    def model_calculation(self, data_num):
        self.model.eval()
        embeddings = None
        offset = 0
        
        with torch.no_grad():
            for idx, input_dict in enumerate(self.test_loader):
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                
                y = self.model(input_dict)
                embedding = y['global'].detach().cpu().numpy()
                torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

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
                    logger = get_root_logger()
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
                logger = get_root_logger()
                logger.info(
                    f"False positive rate for queries without neighbors: "
                    f"{false_positive_rate:.2f}% ({num_false_positives}/{num_no_neighbor_queries})"
                )

        return recall, one_percent_recall, mrr, false_positive_rate

    def build_test_loader(self, data_dict):
        self.cfg.data.test['data_dict'] = data_dict
        test_dataset = build_dataset(self.cfg.data.test)
        if comm.get_world_size() > 1:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size_test_per_gpu,
            shuffle=False,
            num_workers=self.cfg.num_worker_per_gpu,
            pin_memory=True,
            sampler=test_sampler,
            # collate_fn=self.__class__.collate_fn,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
        )
        return test_loader

    @staticmethod
    def collate_fn(batch):
        return batch

    # def get_recall(self, m, n, database_vectors, query_vectors, query_sets, database_sets,
    #            log=False, model_name: str = 'model'):
    #     # Original PointNetVLAD code
    #     database_output = database_vectors[m]
    #     queries_output = query_vectors[n]

    #     # When embeddings are normalized, using Euclidean distance gives the same
    #     # nearest neighbour search results as using cosine distance
    #     database_nbrs = KDTree(database_output)

    #     num_neighbors = 25
    #     recall = [0] * num_neighbors
    #     recall_idx = []

    #     one_percent_retrieved = 0
    #     threshold = max(int(round(len(database_output)/100.0)), 1)

    #     num_evaluated = 0
    #     num_false_positives = 0
    #     num_no_neighbor_queries = 0
        
    #     # Collect queries with and without true neighbors
    #     queries_with_neighbors = []
    #     queries_without_neighbors = []
        
    #     for i in range(len(queries_output)):
    #         query_details = query_sets[n][i]
    #         true_neighbors = query_details[m]
    #         if len(true_neighbors) == 0:
    #             queries_without_neighbors.append(i)
    #         else:
    #             queries_with_neighbors.append(i)
        
    #     # Sample queries without neighbors if evaluation mode is enabled
    #     if self.eval_no_neighbors and len(queries_without_neighbors) > 0:
    #         sample_size = max(1, int(len(queries_without_neighbors) * self.no_neighbors_sample_ratio))
    #         sampled_no_neighbor_queries = np.random.choice(queries_without_neighbors, 
    #                                                     size=min(sample_size, len(queries_without_neighbors)), 
    #                                                     replace=False)
    #     else:
    #         sampled_no_neighbor_queries = []
        
    #     # Process all queries with neighbors and sampled queries without neighbors
    #     queries_to_evaluate = queries_with_neighbors + list(sampled_no_neighbor_queries)
        
    #     # Auto-calibrate threshold based on average distance to true neighbors
    #     auto_threshold = None
    #     if self.eval_no_neighbors and len(queries_with_neighbors) > 0:
    #         # First pass: calculate average distance to true neighbors
    #         true_neighbor_distances = []
    #         for i in queries_with_neighbors:
    #             query_details = query_sets[n][i]
    #             true_neighbors = query_details[m]
    #             if len(true_neighbors) > 0:
    #                 # Find nearest neighbors
    #                 distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)
    #                 # Find the closest true neighbor
    #                 for j in range(len(indices[0])):
    #                     if indices[0][j] in true_neighbors:
    #                         true_neighbor_distances.append(distances[0][j])
    #                         break
            
    #         if true_neighbor_distances:
    #             avg_true_distance = np.mean(true_neighbor_distances)
    #             auto_threshold = avg_true_distance * self.auto_threshold_scale
    #             if log:
    #                 logger = get_root_logger()
    #                 logger.info(f"Auto-calibrated threshold: {auto_threshold:.4f} (avg_true_distance: {avg_true_distance:.4f}, scale: {self.auto_threshold_scale})")

    #     for i in queries_to_evaluate:
    #         # i is query element ndx
    #         query_details = query_sets[n][i]    # {'query': path, 'northing': , 'easting': }
    #         true_neighbors = query_details[m]
            
    #         # Find nearest neighbours
    #         distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)
            
    #         if len(true_neighbors) == 0:
    #             # Handle queries without true neighbors
    #             num_no_neighbor_queries += 1
    #             if self.eval_no_neighbors and auto_threshold is not None:
    #                 # Check if top1 distance is below auto-calibrated threshold (false positive)
    #                 if distances[0][0] <= auto_threshold:
    #                     num_false_positives += 1
    #                     if log:
    #                         fp_ndx = indices[0][0]
    #                         fp = database_sets[m][fp_ndx]
    #                         fp_emb_dist = distances[0, 0]
    #                         fp_world_dist = np.sqrt((query_details['northing'] - fp['northing']) ** 2 +
    #                                                 (query_details['easting'] - fp['easting']) ** 2)
                            
    #                         out_fp_file_name = f"{model_name}_log_fp_no_neighbors.txt"
    #                         with open(out_fp_file_name, "a") as f:
    #                             s = "{}, {}, {:0.2f}, {:0.2f}, NO_NEIGHBORS, THRESHOLD:{:.4f}\n".format(
    #                                 query_details['query'], fp['query'], fp_emb_dist, fp_world_dist, auto_threshold)
    #                             f.write(s)
    #             continue  # Skip recall calculation for queries without neighbors
                
    #         num_evaluated += 1

    #         if log:
    #             # Log false positives (returned as the first element)
    #             # Check if there's a false positive returned as the first element
    #             if indices[0][0] not in true_neighbors:
    #                 fp_ndx = indices[0][0]
    #                 fp = database_sets[m][fp_ndx]  # Database element: {'query': path, 'northing': , 'easting': }
    #                 fp_emb_dist = distances[0][0]  # Distance in embedding space
    #                 fp_world_dist = np.sqrt((query_details['northing'] - fp['northing']) ** 2 +
    #                                         (query_details['easting'] - fp['easting']) ** 2)
    #                 # Find the first true positive
    #                 tp = None
    #                 for k in range(len(indices[0])):
    #                     if indices[0][k] in true_neighbors:
    #                         closest_pos_ndx = indices[0][k]
    #                         tp = database_sets[m][closest_pos_ndx]  # Database element: {'query': path, 'northing': , 'easting': }
    #                         tp_emb_dist = distances[0][k]
    #                         tp_world_dist = np.sqrt((query_details['northing'] - tp['northing']) ** 2 +
    #                                                 (query_details['easting'] - tp['easting']) ** 2)
    #                         break
                                
    #                 out_fp_file_name = f"{model_name}_log_fp.txt"
    #                 with open(out_fp_file_name, "a") as f:
    #                     s = "{}, {}, {:0.2f}, {:0.2f}".format(query_details['query'], fp['query'], fp_emb_dist, fp_world_dist)
    #                     if tp is None:
    #                         s += ', 0, 0, 0\n'
    #                     else:
    #                         s += ', {}, {:0.2f}, {:0.2f}\n'.format(tp['query'], tp_emb_dist, tp_world_dist)
    #                     f.write(s)

    #             # Save details of 5 best matches for later visualization for 1% of queries
    #             s = f"{query_details['query']}, {query_details['northing']}, {query_details['easting']}"
    #             for k in range(min(len(indices[0]), 5)):
    #                 is_match = indices[0][k] in true_neighbors
    #                 e_ndx = indices[0][k]
    #                 e = database_sets[m][e_ndx]     # Database element: {'query': path, 'northing': , 'easting': }
    #                 e_emb_dist = distances[0][k]
    #                 world_dist = np.sqrt((query_details['northing'] - e['northing']) ** 2 +
    #                                         (query_details['easting'] - e['easting']) ** 2)
    #                 s += f", {e['query']}, {e_emb_dist:0.2f}, , {world_dist:0.2f}, {1 if is_match else 0}, "
    #             s += '\n'
    #             out_top5_file_name = f"{model_name}_log_search_results.txt"
    #             with open(out_top5_file_name, "a") as f:
    #                 f.write(s)

    #         for j in range(len(indices[0])):
    #             if indices[0][j] in true_neighbors:
    #                 recall[j] += 1
    #                 recall_idx.append(j+1)
    #                 break

    #         if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
    #             one_percent_retrieved += 1

    #     one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100 if num_evaluated > 0 else 0
    #     recall = (np.cumsum(recall)/float(num_evaluated))*100 if num_evaluated > 0 else np.zeros(25)
    #     mrr = np.mean(1/np.array(recall_idx))*100 if len(recall_idx) > 0 else 0
        
    #     # Calculate false positive rate for queries without neighbors
    #     false_positive_rate = 0
    #     if self.eval_no_neighbors and num_no_neighbor_queries > 0:
    #         false_positive_rate = (num_false_positives / num_no_neighbor_queries) * 100
    #         if log:
    #             logger = get_root_logger()
    #             logger.info(f"False positive rate for queries without neighbors: {false_positive_rate:.2f}% ({num_false_positives}/{num_no_neighbor_queries})")
        
    #     return recall, one_percent_recall, mrr, false_positive_rate