import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .builder import LOSSES

from .loss_utils import compute_aff, sigmoid

@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
        label_smoothing=0.0,
        loss_weight=1.0,
        ignore_index=-1,
    ):
        super(CrossEntropyLoss, self).__init__()
        weight = torch.tensor(weight).cuda() if weight is not None else None
        self.loss_weight = loss_weight
        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, pred, target):
        return self.loss(pred, target) * self.loss_weight


@LOSSES.register_module()
class SmoothCELoss(nn.Module):
    def __init__(self, smoothing_ratio=0.1):
        super(SmoothCELoss, self).__init__()
        self.smoothing_ratio = smoothing_ratio

    def forward(self, pred, target):
        eps = self.smoothing_ratio
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).total(dim=1)
        loss = loss[torch.isfinite(loss)].mean()
        return loss


@LOSSES.register_module()
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5, logits=True, reduce=True, loss_weight=1.0):
        """Binary Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(BinaryFocalLoss, self).__init__()
        assert 0 < alpha < 1
        self.gamma = gamma
        self.alpha = alpha
        self.logits = logits
        self.reduce = reduce
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N)
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        if self.logits:
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        else:
            bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-bce)
        alpha = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha * (1 - pt) ** self.gamma * bce

        if self.reduce:
            focal_loss = torch.mean(focal_loss)
        return focal_loss * self.loss_weight


@LOSSES.register_module()
class FocalLoss(nn.Module):
    def __init__(
        self, gamma=2.0, alpha=0.5, reduction="mean", loss_weight=1.0, ignore_index=-1
    ):
        """Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(FocalLoss, self).__init__()
        assert reduction in (
            "mean",
            "sum",
        ), "AssertionError: reduction should be 'mean' or 'sum'"
        assert isinstance(
            alpha, (float, list)
        ), "AssertionError: alpha should be of type float"
        assert isinstance(gamma, float), "AssertionError: gamma should be of type float"
        assert isinstance(
            loss_weight, float
        ), "AssertionError: loss_weight should be of type float"
        assert isinstance(ignore_index, int), "ignore_index must be of type int"
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N, C) where C = number of classes.
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤C−1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        if len(target) == 0:
            return 0.0

        num_classes = pred.size(1)
        target = F.one_hot(target, num_classes=num_classes)

        alpha = self.alpha
        if isinstance(alpha, list):
            alpha = pred.new_tensor(alpha)
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        one_minus_pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * one_minus_pt.pow(
            self.gamma
        )

        loss = (
            F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            * focal_weight
        )
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.total()
        return self.loss_weight * loss


@LOSSES.register_module()
class DiceLoss(nn.Module):
    def __init__(self, smooth=1, exponent=2, loss_weight=1.0, ignore_index=-1):
        """DiceLoss.
        This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
        Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1), num_classes=num_classes
        )

        total_loss = 0
        for i in range(num_classes):
            if i != self.ignore_index:
                num = torch.sum(torch.mul(pred[:, i], target[:, i])) * 2 + self.smooth
                den = (
                    torch.sum(
                        pred[:, i].pow(self.exponent) + target[:, i].pow(self.exponent)
                    )
                    + self.smooth
                )
                dice_loss = 1 - num / den
                total_loss += dice_loss
        loss = total_loss / num_classes
        return self.loss_weight * loss

@LOSSES.register_module()
class TruncatedSmoothAP(nn.Module):
    def __init__(self, tau1: float = 0.01, similarity: str = 'cosine', positives_per_query: int = 4):
        # We reversed the notation compared to the paper (tau1 is sigmoid on similarity differences)
        # tau1: sigmoid temperature applied on similarity differences
        # positives_per_query: number of positives per query to consider
        # negatives_only: if True in denominator we consider positives and negatives; if False we consider all elements
        #                 (with except to the anchor itself)

        self.tau1 = tau1
        self.similarity = similarity
        self.positives_per_query = positives_per_query

    def __call__(self, input_dict):
        assert 'embeddings' in input_dict.keys()
        assert 'positives_mask' in input_dict.keys()
        assert 'negatives_mask' in input_dict.keys()
        
        embeddings = input_dict['embeddings']
        positives_mask = input_dict['positives_mask']
        negatives_mask = input_dict['negatives_mask']
        device = embeddings.device

        positives_mask = positives_mask.to(device)
        negatives_mask = negatives_mask.to(device)

        # Ranking of the retrieval set
        # For each element we ignore elements that are neither positives nor negatives

        # Compute cosine similarity scores
        # 1st dimension corresponds to q, 2nd dimension to z
        s_qz = compute_aff(embeddings, similarity=self.similarity)

        # Find the positives_per_query closest positives for each query
        s_positives = s_qz.detach().clone()
        s_positives.masked_fill_(torch.logical_not(positives_mask), -np.inf)
        #closest_positives_ndx = torch.argmax(s_positives, dim=1).view(-1, 1)  # Indices of closests positives for each query
        closest_positives_ndx = torch.topk(s_positives, k=self.positives_per_query, dim=1, largest=True, sorted=True)[1]
        # closest_positives_ndx is (batch_size, positives_per_query)  with positives_per_query closest positives
        # per each batch element

        n_positives = positives_mask.sum(dim=1)     # Number of positives for each anchor

        # Compute the rank of each example x with respect to query element q as per Eq. (2)
        s_diff = s_qz.unsqueeze(1) - s_qz.gather(1, closest_positives_ndx).unsqueeze(2)
        s_sigmoid = sigmoid(s_diff, temp=self.tau1)

        # Compute the nominator in Eq. 2 and 5 - for q compute the ranking of each of its positives with respect to other positives of q
        # Filter out z not in Positives
        pos_mask = positives_mask.unsqueeze(1)
        pos_s_sigmoid = s_sigmoid * pos_mask

        # Filter out z on the same position as the positive (they have value = 0.5, as the similarity difference is zero)
        mask = torch.ones_like(pos_s_sigmoid).scatter(2, closest_positives_ndx.unsqueeze(2), 0.)
        pos_s_sigmoid = pos_s_sigmoid * mask

        # Compute the rank for each query and its positives_per_query closest positive examples with respect to other positives
        r_p = torch.sum(pos_s_sigmoid, dim=2) + 1.
        # r_p is (batch_size, positives_per_query) matrix

        # Consider only positives and negatives in the denominator
        # Compute the denominator in Eq. 5 - add sum of Indicator function for negatives (or non-positives)
        neg_mask = negatives_mask.unsqueeze(1)
        neg_s_sigmoid = s_sigmoid * neg_mask
        r_omega = r_p + torch.sum(neg_s_sigmoid, dim=2)

        # Compute R(i, S_p) / R(i, S_omega) ration in Eq. 2
        r = r_p / r_omega

        # Compute metrics              mean ranking of the positive example, recall@1
        stats = {}
        # Mean number of positives per query
        stats['positives_per_query'] = n_positives.float().mean(dim=0)
        # Mean ranking of selected positive examples (closests positives)
        temp = s_diff.detach() > 0
        temp = torch.logical_and(temp[:, 0], negatives_mask)        # Take the best positive
        hard_ranking = temp.sum(dim=1)
        stats['best_positive_ranking'] = hard_ranking.float().mean(dim=0)
        # Recall at 1
        stats['recall_AT_1'] = (hard_ranking <= 1).float().mean(dim=0)

        # r is (N, positives_per_query) tensor
        # Zero entries not corresponding to real positives - this happens when the number of true positives is lower than positives_per_query
        valid_positives_mask = torch.gather(positives_mask, 1, closest_positives_ndx)   # () tensor
        masked_r = r * valid_positives_mask
        n_valid_positives = valid_positives_mask.sum(dim=1)

        # Filter out rows (queries) without any positive to avoid division by zero
        valid_q_mask = n_valid_positives > 0
        masked_r = masked_r[valid_q_mask]

        ap = (masked_r.sum(dim=1) / n_valid_positives[valid_q_mask]).mean()
        loss = 1. - ap
        stats['loss'] = loss
        stats['ap'] = ap
        stats['avg_embedding_norm'] = embeddings.norm(dim=1).mean()
        return loss, stats