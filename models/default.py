import torch
import torch.nn as nn
import torch_scatter
import torch_cluster
import torch.nn.functional as F

from models.losses import build_criteria
from models.utils.structure import Point
from models.utils import offset2batch
from models.utils import NetVladWrapper, GeMWrapper, SOPWrapper, SALADWrapper
from .builder import MODELS, build_model

from models.modules import PointModel, PointSequential
import spconv.pytorch as spconv

import torch.distributed as dist
from tqdm import tqdm
import pointops

@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                # parent.feat = point.feat[inverse]
                point = parent
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict
    
@MODELS.register_module()
class DefaultPlaceRecognitioner(nn.Module):
    def __init__(
        self,
        backbone=None,
        head=None,
        criteria=None,
        freeze_backbone=False,
        default_grid_size = 0.05,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria) if criteria is not None else None
        if head.type == "NetVLAD":
            self.head = NetVladWrapper(head)
        elif head.type == "GeM":
            self.head = GeMWrapper(head)
        elif head.type == "SOP":
            self.head = SOPWrapper(head)
        elif head.type == "SALAD":
            self.head = SALADWrapper(head)
        else:
            #error
            raise ValueError(f"Unsupported head type: {head.type}")
        self.freeze_backbone = freeze_backbone
        self.default_grid_size = default_grid_size
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        if 'grid_coord' not in input_dict.keys():
            input_dict['grid_size'] = self.default_grid_size
        point = Point(input_dict)
        points = self.backbone(point)
        if type(points) == tuple:
            enc_point, dec_point = points
            tmp_PTv3 = hasattr(self.backbone, "cls_mode") and self.backbone.cls_mode
            tmp_sonata = hasattr(self.backbone, "enc_mode") and self.backbone.enc_mode
            if tmp_PTv3 or tmp_sonata:
                point = enc_point
            else:
                point = dec_point
        else:
            enc_point = points
            point = points
    
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            # 优化：使用更高效的pooling_parent合并方式，减少中间变量
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                # 优化：直接使用indexing，避免不必要的中间张量
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                # 释放不再需要的point的feat以节省内存
                del point
                point = parent
            feat = point.feat
        else:
            feat = point

        # 检查是否使用新的成batch方式
        if 'offset' in point and 'batch' in point:
            offset = point['offset']
            if isinstance(self.head, SALADWrapper):
                global_feat = self.head.forward_with_batch_info(
                    x=feat,  # (total_points, feature_dim)
                    batch=point['batch'],  # (total_points,)
                    offset=offset,  # (batch_size,)
                    coarse_point=enc_point,
                )
            else:
                # 使用新的成batch方式，直接传入拼接后的特征和batch信息
                global_feat = self.head.forward_with_batch_info(
                    x=feat,  # (total_points, feature_dim)
                    batch=point['batch'],  # (total_points,)
                    offset=offset  # (batch_size,)
                )
        else:
            # 使用原有的方式
            global_feat = self.head(feat)
            # 对于原有方式，需要从point中获取offset（如果有）
            offset = point.get('offset', None)
        
        # 优化：延迟计算local特征，仅在需要时计算
        # 按照offset将feat拆分为多个点云的列表
        # 优化：使用normalize（注意normalize不是inplace的，但内存开销较小）
        feat = F.normalize(feat, p=2, dim=-1)
        
        # 优化：仅在offset存在时创建feat_list
        if offset is not None:
            feat_list = []
            start = 0
            for end in offset:
                feat_list.append(feat[start:end])
                start = end
        else:
            # 如果没有offset信息，返回空列表
            feat_list = []
            
        return {
            'global': global_feat,
            'local': feat_list,
        }