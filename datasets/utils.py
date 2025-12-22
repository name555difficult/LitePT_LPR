import random
from collections.abc import Mapping, Sequence
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

from models.utils import offset2batch


def collate_fn(batch):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        batch = {
            key: (
                collate_fn([d[key] for d in batch])
                if "offset" not in key
                # offset -> bincount -> concat bincount-> concat offset
                else torch.cumsum(
                    collate_fn([d[key].diff(prepend=torch.tensor([0])) for d in batch]),
                    dim=0,
                )
            )
            for key in batch[0]
        }
        return batch
    else:
        return default_collate(batch)


def point_collate_fn(batch, mix_prob=0):
    assert isinstance(
        batch[0], Mapping
    )  # currently, only support input_dict, rather than input_list
    batch = collate_fn(batch)
    if random.random() < mix_prob:
        if "instance" in batch.keys():
            offset = batch["offset"]
            start = 0
            num_instance = 0
            for i in range(len(offset)):
                if i % 2 == 0:
                    num_instance = max(batch["instance"][start : offset[i]])
                if i % 2 != 0:
                    mask = batch["instance"][start : offset[i]] != -1
                    batch["instance"][start : offset[i]] += num_instance * mask
                start = offset[i]
        if "offset" in batch.keys():
            batch["offset"] = torch.cat(
                [batch["offset"][1:-1:2], batch["offset"][-1].unsqueeze(0)], dim=0
            )

        ### fix bug ###
        # recompute grid coord !!
        grid_coord_new = []
        batch_size = len(batch["offset"])

        batch_mask = offset2batch(batch["offset"])
        for bs_id in range(batch_size):
            sample_mask = batch_mask == bs_id
            coord_sample = batch['coord'][sample_mask]
            scaled_coord_sample = coord_sample / batch['grid_size'][0]  # hack here! 
            grid_coord_sample = torch.floor(scaled_coord_sample).to(torch.int64)
            min_coord_sample= grid_coord_sample.min(0)[0]
            grid_coord_sample -= min_coord_sample

            grid_coord_new.append(grid_coord_sample)

        grid_coord_new = torch.cat(grid_coord_new, dim=0)
        batch["grid_coord"] = grid_coord_new
        ### fix bug ###

    return batch


def gaussian_kernel(dist2: np.array, a: float = 1, c: float = 5):
    return a * np.exp(-dist2 / (2 * c**2))

def in_sorted_array(e: int, array: np.ndarray) -> bool:
    pos = np.searchsorted(array, e)
    if pos == len(array) or pos == -1:
        return False
    else:
        return array[pos] == e

def wildplaces_collate_fn(batch, batch_split_size, mix_prob=0):
    # Compute positives and negatives mask
    # dataset.queries[label]['positives'] is bitarray
    labels = [e['label'].item() for e in batch]
    positives = [e['positives'].numpy() for e in batch]
    non_negatives = [e['non_negatives'].numpy() for e in batch]
    # print(f"type of positives: {type(positives[0])}")
    # print(f"Time taken for labels, positives and non_negatives: {time.time() - start_time}")
    positives_mask = [[in_sorted_array(e, d) for e in labels] for d in positives]
    negatives_mask = [[not in_sorted_array(e, d) for e in labels] for d in non_negatives]
    positives_mask = torch.tensor(positives_mask)
    negatives_mask = torch.tensor(negatives_mask)
    # print(f"Time taken for positives and negatives mask: {time.time() - start_time}")
    # Generate batches in correct format

    if batch_split_size is None or batch_split_size == 0:
        data = point_collate_fn(batch, mix_prob)
    else:
        # 优化：先处理整个batch，再分割
        # 而不是先分割再处理每个minibatch，避免重复调用point_collate_fn
        full_batch = point_collate_fn(batch, mix_prob)
        # print(f"Time taken for point_collate_fn: {time.time() - start_time}")
        full_batch.pop("positives")
        full_batch.pop("non_negatives")
        
        # 基于offset信息分割数据
        data = []
        offset = full_batch["offset"]
        
        # 计算每个样本的起始和结束索引
        start_indices = [0] + offset[:-1].tolist()
        end_indices = offset.tolist()
        
        # 按batch_split_size分组分割
        for i in range(0, len(batch), batch_split_size):
            end_idx = min(i + batch_split_size, len(batch))
            
            # 计算当前minibatch的起始和结束点云索引
            minibatch_start = start_indices[i]
            minibatch_end = end_indices[end_idx - 1]
            
            # 创建minibatch数据
            minibatch = {}
            for key, value in full_batch.items():
                if key == "offset":
                    # 重新计算offset：当前minibatch中每个样本的点数
                    minibatch_offset = []
                    current_offset = 0
                    for j in range(i, end_idx):
                        sample_points = end_indices[j] - start_indices[j]
                        current_offset += sample_points
                        minibatch_offset.append(current_offset)
                    minibatch[key] = torch.tensor(minibatch_offset, dtype=offset.dtype, device=offset.device)
                elif isinstance(value, torch.Tensor):
                    if key in ["coord", "feat", "grid_coord"]:
                        # 点云数据，需要按索引分割
                        minibatch[key] = value[minibatch_start:minibatch_end]
                    else:
                        # 其他张量数据，按batch索引分割
                        minibatch[key] = value[i:end_idx]
                else:
                    # 非张量数据，按batch索引分割
                    minibatch[key] = value[i:end_idx]
            
            data.append(minibatch)

    # print(f"Time taken for wildplaces_collate_fn: {time.time() - start_time}")

    return {
        "data": data,
        "positives_mask": positives_mask,
        "negatives_mask": negatives_mask,
    }