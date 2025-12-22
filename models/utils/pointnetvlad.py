# PointNet code taken from PointNetVLAD Pytorch implementation: https://github.com/cattaneod/PointNetVlad-Pytorch
# Adapted by Jacek Komorowski

import pdb
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import math
import torch_scatter

# NOTE: The toolbox can only pool lists of features of the same length. It was specifically optimized to efficiently
# o so. One way to handle multiple lists of features of variable length is to create, via a data augmentation
# technique, a tensor of shape: 'batch_size'x'max_samples'x'feature_size'. Where 'max_samples' would be the maximum
# number of feature per list. Then for each list, you would fill the tensor with 0 values.


class NetVLADLoupe(nn.Module):
    def __init__(self, feature_size, cluster_size, output_dim, gating=True, add_batch_norm=True):
        super().__init__()
        self.feature_size = feature_size
        self.output_dim = output_dim
        self.gating = gating
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size
        self.softmax = nn.Softmax(dim=-1)
        self.cluster_weights = nn.Parameter(torch.randn(
            feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.cluster_weights2 = nn.Parameter(torch.randn(
            1, feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.hidden1_weights = nn.Parameter(
            torch.randn(cluster_size * feature_size, output_dim) * 1 / math.sqrt(feature_size))

        if add_batch_norm:
            self.cluster_biases = None
            self.bn1 = nn.BatchNorm1d(cluster_size)
        else:
            self.cluster_biases = nn.Parameter(torch.randn(
                cluster_size) * 1 / math.sqrt(feature_size))
            self.bn1 = None

        self.bn2 = nn.BatchNorm1d(output_dim)

        if gating:
            self.context_gating = GatingContext(
                output_dim, add_batch_norm=add_batch_norm)

    def forward(self, x):
        # Expects (batch_size, num_points, channels) tensor
        assert x.dim() == 3
        num_points = x.shape[1]
        activation = torch.matmul(x, self.cluster_weights)
        # bs, npts, cluster_size
        if self.add_batch_norm:
            # activation = activation.transpose(1,2).contiguous()
            activation = activation.view(-1, self.cluster_size)
            activation = self.bn1(activation)
            activation = activation.view(-1, num_points, self.cluster_size)
            # activation = activation.transpose(1,2).contiguous()
        else:
            activation = activation + self.cluster_biases
        activation = self.softmax(activation)
        # bs, npts, cluster_size --> alpha_k(x_i)
        activation = activation.view((-1, num_points, self.cluster_size))
        # bs, npts, cluster_size
        a_sum = activation.sum(-2, keepdim=True)
        # bs, 1, cluster_size
        a = a_sum * self.cluster_weights2
        # bs, feat_dim, cluster_size --> \sum_i alpha_k(x_i) c_k
        activation = torch.transpose(activation, 2, 1)
        # bs, cluster_size, npts
        x = x.view((-1, num_points, self.feature_size))
        # bs, npts, feat_dim
        vlad = torch.matmul(activation, x)
        # bs, cluster_size, feat_dim  --> \sum_i alpha_k(x_i) x_i
        vlad = torch.transpose(vlad, 2, 1)
        # bs, feat_dim, cluster_size
        vlad = vlad - a

        vlad = F.normalize(vlad, dim=1, p=2)
        # bs, feat_dim, cluster_size
        vlad = vlad.reshape((-1, self.cluster_size * self.feature_size))
        vlad = F.normalize(vlad, dim=1, p=2)

        vlad = torch.matmul(vlad, self.hidden1_weights)
        # bs, out_dim
        vlad = self.bn2(vlad)

        if self.gating:
            vlad = self.context_gating(vlad)

        return vlad


class GatingContext(nn.Module):
    def __init__(self, dim, add_batch_norm=True):
        super(GatingContext, self).__init__()
        self.dim = dim
        self.add_batch_norm = add_batch_norm
        self.gating_weights = nn.Parameter(
            torch.randn(dim, dim) * 1 / math.sqrt(dim))
        self.sigmoid = nn.Sigmoid()

        if add_batch_norm:
            self.gating_biases = None
            self.bn1 = nn.BatchNorm1d(dim)
        else:
            self.gating_biases = nn.Parameter(
                torch.randn(dim) * 1 / math.sqrt(dim))
            self.bn1 = None

    def forward(self, x):
        gates = torch.matmul(x, self.gating_weights)

        if self.add_batch_norm:
            gates = self.bn1(gates)
        else:
            gates = gates + self.gating_biases

        gates = self.sigmoid(gates)

        activation = x * gates

        return activation


class NetVladWrapper(torch.nn.Module):
    # Wrapper around NetVlad class to process sparse tensors from Minkowski networks
    def __init__(self, cfg):
        super().__init__()
        self.feature_size = cfg.in_channels
        self.output_dim = cfg.out_channels
        self.net_vlad = NetVLADLoupe(feature_size=self.feature_size, cluster_size=cfg.cluster_size, output_dim=self.output_dim,
                                     gating=cfg.gating, add_batch_norm=cfg.add_bn)

    def forward(self, x):
        # x is SparseTensor
        assert x.shape[-1] == self.feature_size
        x = self.net_vlad(x)
        assert x.shape[1] == self.output_dim
        return x    # Return (batch_size, output_dim) tensor
    
    def forward_with_batch_info(self, x, batch, offset):
        """
        处理您的成batch方式的点云特征
        Args:
            x: (total_points, feature_dim) - 所有点云拼接后的特征
            batch: (total_points,) - 每个点属于哪个点云的索引
            offset: (batch_size,) - 每个点云在拼接后点云中的长度偏移
        Returns:
            (batch_size, output_dim) - 每个点云的全局描述符
        """
        batch_size = offset.shape[0]
        
        # 计算每个点的cluster assignment
        activation = torch.matmul(x, self.net_vlad.cluster_weights)  # (total_points, cluster_size)
        
        if self.net_vlad.add_batch_norm:
            activation = self.net_vlad.bn1(activation)
        else:
            activation = activation + self.net_vlad.cluster_biases
        
        activation = self.net_vlad.softmax(activation)  # (total_points, cluster_size)
        
        # 使用torch_scatter.segment_csr来高效地按batch分组计算
        # 计算每个点云的alpha_k(x_i)的sum
        # 确保indptr是Long类型
        indptr = torch.cat([torch.zeros(1, device=offset.device, dtype=torch.long), offset.long()])
        a_sum = torch_scatter.segment_csr(
            src=activation,
            indptr=indptr,
            reduce="sum"
        )  # (batch_size, cluster_size)
        
        # 计算每个点云的a值
        a = a_sum.unsqueeze(1) * self.net_vlad.cluster_weights2  # (batch_size, 1, cluster_size)
        
        # 计算每个点云的VLAD
        # 使用segment_csr来计算 \sum_i alpha_k(x_i) x_i
        vlad = torch_scatter.segment_csr(
            src=activation.unsqueeze(-1) * x.unsqueeze(-2),  # (total_points, cluster_size, feature_dim)
            indptr=indptr,  # 复用上面创建的indptr
            reduce="sum"
        )  # (batch_size, cluster_size, feature_dim)
        
        # 转置并减去a值
        vlad = vlad.transpose(1, 2)  # (batch_size, feature_dim, cluster_size)
        vlad = vlad - a.squeeze(1)  # (batch_size, feature_dim, cluster_size)
        
        # 归一化
        vlad = F.normalize(vlad, dim=1, p=2)  # (batch_size, feature_dim, cluster_size)
        vlad = vlad.reshape(batch_size, -1)  # (batch_size, feature_dim * cluster_size)
        vlad = F.normalize(vlad, dim=1, p=2)  # (batch_size, feature_dim * cluster_size)
        
        # 通过最后的线性层
        vlad = torch.matmul(vlad, self.net_vlad.hidden1_weights)  # (batch_size, output_dim)
        vlad = self.net_vlad.bn2(vlad)
        
        if self.net_vlad.gating:
            vlad = self.net_vlad.context_gating(vlad)
        
        return vlad


# 使用示例：
"""
# 假设您有以下数据：
# - 48个点云拼接成一个大点云，形状为(427716, 3)
# - 每个点云的特征维度为256
# - offset记录每个点云的长度偏移

# 使用方式：
input_dict = {
    'coord': torch.randn(427716, 3),  # 拼接后的点云坐标
    'feat': torch.randn(427716, 256),  # 拼接后的点云特征
    'offset': torch.tensor([7151, 16512, 24264, ...]),  # 每个点云的偏移
    'batch': torch.arange(427716) % 48,  # 每个点属于哪个点云
}

# 在DefaultPlaceRecognitioner中，会自动检测并使用forward_with_batch_info方法
# 返回的global_feat形状为(48, output_dim)，每个点云一个全局描述符
"""

class GeM_pool(nn.Module):
    """
    Stable GeM for [B, N, C] -> [B, C]
    - p = softplus(w) + p_min  保证 p>0 且可选上界
    - 幂运算统一用 float32，避免 AMP/bf16 溢出
    - 只 squeeze 最后一维
    """
    def __init__(self, p_init=3.0, eps=1e-6, p_min=1e-3, p_max=10.0):
        super().__init__()
        # 用可学习参数 w，经 softplus 映射为正数
        w_init = math.log(math.exp(p_init - p_min) - 1.0) if p_init > p_min else 0.0
        self.w = nn.Parameter(torch.tensor([w_init], dtype=torch.float32))
        self.eps = float(eps)
        self.p_min = float(p_min)
        self.p_max = float(p_max)

    def _p(self):
        # p in (p_min, +inf)，再裁剪到 [p_min, p_max]
        p = F.softplus(self.w) + self.p_min
        if self.p_max is not None:
            p = torch.clamp(p, max=self.p_max)
        return p

    def forward(self, x):
        """
        x: [B, N, C]  (任意 dtype)
        return: [B, C]  (与 x 同 dtype)
        """
        assert x.dim() == 3
        dtype_in = x.dtype
        x32 = x.to(torch.float32)

        p = self._p()          # shape [1], float32
        p_inv = 1.0 / p

        # 正值下限，避免 log(0) 出现在 pow 的梯度里
        x32 = x32.clamp(min=self.eps)

        # 幂运算在 fp32 做，避免半精度溢出
        x32 = x32.pow(p)                      # [B, N, C]
        x32 = x32.permute(0, 2, 1)            # [B, C, N]
        x32 = F.adaptive_avg_pool1d(x32, 1)   # [B, C, 1]
        x32 = x32.squeeze(-1)                 # [B, C]
        x32 = x32.pow(p_inv)                  # [B, C]

        # 返回原始 dtype
        out = x32.to(dtype_in)

        # 保险起见，清掉非有限数（可加断言/日志）
        out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
        return out


class GeMWrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.feature_size = cfg.feature_size
        p_init = cfg.get("p", 3.0)
        eps = cfg.get("eps", 1e-6)
        p_min = cfg.get("p_min", 1e-3)
        p_max = cfg.get("p_max", 10.0)
        self.gem = GeM_pool(p_init=p_init, eps=eps, p_min=p_min, p_max=p_max)

    def forward(self, x, batch, offset):
        """
        x:      (total_points, C)
        batch:  (total_points,)  [可忽略]
        offset: (B,)  每个样本累计点数 (例如 [n1, n1+n2, ...])
        """
        B = offset.shape[0]
        device = x.device
        dtype_in = x.dtype

        indptr = torch.cat([
            torch.zeros(1, device=device, dtype=torch.long),
            offset.to(torch.long)
        ])
        # self._check_indptr(indptr, x.shape[0])  # 调试时可开启

        counts = indptr[1:] - indptr[:-1]              # [B]
        denom = torch.clamp(counts.unsqueeze(1).to(torch.float32), min=1.0)

        # GeM 的“幂→均值→开 1/p”按语义分解：
        x32 = x.to(torch.float32).clamp(min=self.gem.eps).pow(self.gem._p())
        sums = torch_scatter.segment_csr(src=x32, indptr=indptr, reduce="sum")  # [B, C]
        avg  = sums / denom                                                     # [B, C]
        out  = avg.pow(1.0 / self.gem._p())                                     # [B, C]
        out  = torch.where(torch.isfinite(out), out, torch.zeros_like(out))

        return out.to(dtype_in)

    @torch.no_grad()
    def _check_indptr(self, indptr, total_points):
        # 简单完整性检查（可在调试时开启）
        assert indptr.dim() == 1
        assert (indptr[1:] >= indptr[:-1]).all()
        assert indptr[0] == 0 and indptr[-1] == total_points

    def forward_with_batch_info(self, x, batch, offset):
        out = self.forward(x, batch, offset)
        # L2-norm
        out = F.normalize(out, p=2, dim=-1)
        return out


class SOP(nn.Module):
    def __init__(self, thresh=1e-8, is_vec=True, signed_sqrt=False, do_pe=True, do_fc=False, input_dim=16, is_tuple=False):
        super(SOP, self).__init__()
        self.thresh = thresh
        self.is_vec = is_vec
        self.do_pe = do_pe
        self.sop_dim = input_dim * input_dim
        self.signed_sqrt = signed_sqrt
        self.do_fc = do_fc
        self.is_tuple = is_tuple

        cs = [4096, 2048, 1024]  # redundant fc layers
        cr = self.sop_dim/cs[0]
        cs = [int(cr * x) for x in cs]
        self.fc1 = nn.Linear(cs[0], cs[1])
        self.fc2 = nn.Linear(cs[1], cs[2])

    def _so_maxpool(self, x):
        while len(x.data.shape) < 4:
            x = torch.unsqueeze(x, 0)
        batchSize, tupleLen, nFeat, dimFeat = x.data.shape
        x = torch.reshape(x, (-1, dimFeat))
        x = torch.unsqueeze(x, -1)
        x = x.matmul(x.transpose(1, 2))

        x = torch.reshape(x, (batchSize, tupleLen, nFeat, dimFeat, dimFeat))
        x = torch.max(x, 2).values
        x = torch.reshape(x, (-1, dimFeat, dimFeat))
        if self.do_pe:
            x = x.double()
            # u_, s_, vh_ = torch.linalg.svd(x)
            # dist = torch.dist(x, u_ @ torch.diag_embed(s_) @ vh_)
            # dist_same = torch.allclose(x, u_ @ torch.diag_embed(s_) @ vh_)
            # s_alpha = torch.pow(s_, 0.5)
            # x = u_ @ torch.diag_embed(s_alpha) @ vh_

            # For pytorch versions < 1.9
            u_, s_, v_ = torch.svd(x)
            # dist = torch.dist(x, u_ @ torch.diag_embed(s_) @  v_.transpose(-2, -1))
            # dist_same = torch.allclose(x, u_ @ torch.diag_embed(s_) @  v_.transpose(-2, -1))
            s_alpha = torch.pow(s_, 0.5)
            x = u_ @ torch.diag_embed(s_alpha) @ v_.transpose(-2, -1)

        x = torch.reshape(x, (batchSize, tupleLen, dimFeat, dimFeat))
        return x  # .float()

    def _l2norm(self, x):
        x = nn.functional.normalize(x, p=2, dim=-1)
        return x

    def forward(self, x):
        x = self._so_maxpool(x)

        if self.is_vec:
            x = torch.reshape(x, (x.size(0), x.size(1), -1))
        # if self.do_fc:
        #     x = F.relu(self.fc1(x.float()))
        #     x = F.relu(self.fc2(x))
        x = self._l2norm(x)
        return torch.squeeze(x)


class SOPWrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.feature_size = cfg.feature_size
        thresh = cfg.get("thresh", 1e-8)
        is_vec = cfg.get("is_vec", True)
        signed_sqrt = cfg.get("signed_sqrt", False)
        do_pe = cfg.get("do_pe", True)
        do_fc = cfg.get("do_fc", False)
        is_tuple = cfg.get("is_tuple", False)
        
        # 可学习的alpha参数，类似GeM的p参数
        alpha_init = cfg.get("alpha_init", 0.5)
        alpha_min = cfg.get("alpha_min", 1e-3)
        alpha_max = cfg.get("alpha_max", 2.0)
        
        # 使用softplus映射确保alpha > 0
        w_init = math.log(math.exp(alpha_init - alpha_min) - 1.0) if alpha_init > alpha_min else 0.0
        self.w_alpha = nn.Parameter(torch.tensor([w_init], dtype=torch.float32))
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        
        self.sop = SOP(
            thresh=thresh,
            is_vec=is_vec,
            signed_sqrt=signed_sqrt,
            do_pe=do_pe,
            do_fc=do_fc,
            input_dim=self.feature_size,
            is_tuple=is_tuple
        )
    
    def _alpha(self):
        """计算当前的alpha值，确保在[alpha_min, alpha_max]范围内"""
        alpha = F.softplus(self.w_alpha) + self.alpha_min
        if self.alpha_max is not None:
            alpha = torch.clamp(alpha, max=self.alpha_max)
        return alpha

    def forward(self, x):
        """
        标准前向传播，期望输入形状为 [B, N, C]
        注意：这里仍然使用原始SOP的固定0.5参数，保持向后兼容性
        如需使用可学习参数，请使用forward_with_batch_info方法
        """
        assert x.dim() == 3
        return self.sop(x)

    def forward_with_batch_info(self, x, batch, offset):
        """
        处理拼接的点云特征，使用向量化实现避免显式循环
        Args:
            x: (total_points, feature_dim) - 所有点云拼接后的特征
            batch: (total_points,) - 每个点属于哪个点云的索引 [可忽略]
            offset: (batch_size,) - 每个点云在拼接后点云中的长度偏移
        Returns:
            (batch_size, output_dim) - 每个点云的全局描述符
        """
        batch_size = offset.shape[0]
        device = x.device
        dtype_in = x.dtype
        feature_dim = x.shape[1]
        
        # 构建indptr用于segment_csr
        indptr = torch.cat([
            torch.zeros(1, device=device, dtype=torch.long),
            offset.to(torch.long)
        ])
        
        # 计算每个点云的点数
        point_counts = indptr[1:] - indptr[:-1]  # [batch_size]
        
        # 为每个点计算外积矩阵
        # x: (total_points, feature_dim) -> (total_points, feature_dim, feature_dim)
        x_expanded = x.unsqueeze(-1)  # (total_points, feature_dim, 1)
        outer_products = x_expanded.matmul(x_expanded.transpose(-2, -1))  # (total_points, feature_dim, feature_dim)
        
        # 使用segment_csr进行max pooling，按点云分组
        # 对每个点云内的所有点进行max pooling
        max_pooled = torch_scatter.segment_csr(
            src=outer_products.view(-1, feature_dim * feature_dim),  # (total_points, feature_dim^2)
            indptr=indptr,
            reduce="max"
        )  # (batch_size, feature_dim^2)
        
        # 重塑为矩阵形式
        max_pooled = max_pooled.view(batch_size, feature_dim, feature_dim)  # (batch_size, feature_dim, feature_dim)
        
        # 应用power encoding (SVD) - 使用可学习的alpha参数
        if self.sop.do_pe:
            # 获取当前的可学习alpha值
            alpha = self._alpha()
            
            # 使用torch.linalg.svd进行批量SVD计算
            try:
                max_pooled = max_pooled.double()
                # 尝试批量SVD (PyTorch >= 1.9)
                u_, s_, vh_ = torch.linalg.svd(max_pooled)
                s_alpha = torch.pow(s_, alpha)  # 使用可学习的alpha参数
                # 使用einsum进行批量矩阵乘法
                max_pooled = torch.einsum('bij,bj,bkj->bik', u_, s_alpha, vh_).float()
            except:
                print("batch SVD failed, falling back to loop")
                # 回退到循环实现SVD
                svd_results = []
                for i in range(batch_size):
                    if point_counts[i] > 0:  # 只对非空点云进行SVD
                        u_, s_, v_ = torch.svd(max_pooled[i].double())
                        s_alpha = torch.pow(s_, alpha)  # 使用可学习的alpha参数
                        result = u_ @ torch.diag_embed(s_alpha) @ v_.transpose(-2, -1)
                        svd_results.append(result.float())
                    else:
                        svd_results.append(torch.zeros_like(max_pooled[i]))
                max_pooled = torch.stack(svd_results, dim=0)
        
        # 重塑为向量形式
        if self.sop.is_vec:
            max_pooled = max_pooled.view(batch_size, -1)  # (batch_size, feature_dim^2)
        
        if self.sop.do_fc:
            max_pooled = F.relu(self.sop.fc1(max_pooled))
            max_pooled = F.relu(self.sop.fc2(max_pooled))
        
        # L2归一化
        result = F.normalize(max_pooled, p=2, dim=-1)
        
        return result


class SALADWrapper(nn.Module):
    """
    SALAD (Sinkhorn Algorithm for Locally Aggregated Descriptors) 适配点云数据
    使用GeM pooling生成global token，通过Sinkhorn算法进行特征聚合
    """
    def __init__(self, cfg):
        super().__init__()
        self.feature_size = cfg.feature_size
        self.num_clusters = cfg.get("num_clusters", 64)
        self.cluster_dim = cfg.get("cluster_dim", 128)
        self.token_dim = cfg.get("token_dim", 256)
        self.dropout = cfg.get("dropout", 0.3)
        self.use_sinkhorn = cfg.get("use_sinkhorn", True)
        
        # 使用GeM生成global token
        self.gem_pool = GeMWrapper(cfg.coarse_cfg)
        
        # 构建SALAD网络
        if self.dropout > 0:
            dropout_layer = nn.Dropout(self.dropout)
        else:
            dropout_layer = nn.Identity()
        
        # MLP for global scene token (从GeM输出进一步处理)
        self.token_features = nn.Sequential(
            nn.Linear(cfg.coarse_cfg.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.token_dim)
        )
        
        # MLP for local features (点云特征 -> 聚类特征)
        self.cluster_features = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            dropout_layer,
            nn.ReLU(),
            nn.Linear(512, self.cluster_dim)
        )
        
        # MLP for score matrix (点云特征 -> 聚类分配分数)
        self.score = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            dropout_layer,
            nn.ReLU(),
            nn.Linear(512, self.num_clusters)
        )
        
        # Dustbin parameter
        self.dust_bin = nn.Parameter(torch.tensor(1.0))
        
        # Sinkhorn算法参数
        self.num_iters = cfg.get("num_iters", 3)
        self.reg = cfg.get("reg", 1.0)

    def forward(self, x):
        """
        标准前向传播，期望输入形状为 [B, N, C]
        注意：这里需要额外的global token输入
        """
        assert x.dim() == 3
        # 使用GeM生成global token
        global_token = self.gem_pool(x)  # [B, feature_size]
        
        # 处理局部特征
        local_features = self.cluster_features(x)  # [B, N, cluster_dim]
        scores = self.score(x)  # [B, N, num_clusters]
        
        # 处理global token
        global_token = self.token_features(global_token)  # [B, token_dim]
        
        # 使用Sinkhorn算法
        batch_size = x.shape[0]
        device = x.device
        
        # 重塑为SALAD期望的格式
        local_features = local_features.transpose(1, 2)  # [B, cluster_dim, N]
        scores = scores.transpose(1, 2)  # [B, num_clusters, N]
        
        # Sinkhorn算法
        p = self._get_matching_probs(scores, self.dust_bin, self.num_iters, self.reg)
        p = torch.exp(p)
        p = p[:, :-1, :]  # 移除dustbin行
        
        # 特征聚合
        p_expanded = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)  # [B, cluster_dim, num_clusters, N]
        local_expanded = local_features.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)  # [B, cluster_dim, num_clusters, N]
        
        # 加权聚合
        aggregated = (local_expanded * p_expanded).sum(dim=-1)  # [B, cluster_dim, num_clusters]
        aggregated = aggregated.flatten(1)  # [B, cluster_dim * num_clusters]
        
        # 拼接global token和聚合特征
        result = torch.cat([
            F.normalize(global_token, p=2, dim=-1),
            F.normalize(aggregated, p=2, dim=-1)
        ], dim=-1)
        
        return F.normalize(result, p=2, dim=-1)

    def forward_with_batch_info(self, x, batch, offset, coarse_point):
        """
        处理拼接的点云特征，使用向量化实现避免显式循环
        Args:
            x: (total_points, feature_dim) - 所有点云拼接后的特征
            batch: (total_points,) - 每个点属于哪个点云的索引 [可忽略]
            offset: (batch_size,) - 每个点云在拼接后点云中的长度偏移
        Returns:
            (batch_size, output_dim) - 每个点云的全局描述符
        """
        batch_size = offset.shape[0]
        device = x.device
        dtype_in = x.dtype
        
        # 构建indptr（优化：提前计算，避免重复创建）
        indptr = torch.cat([
            torch.zeros(1, device=device, dtype=torch.long),
            offset.to(torch.long)
        ])
        
        # 使用GeM生成每个点云的global token（优先处理，可以尽早释放coarse_point）
        global_tokens = self.gem_pool.forward(
            x=coarse_point.feat,
            batch=coarse_point.batch,
            offset=coarse_point.offset,
        )
        # 处理global token（在特征处理前完成，减少峰值内存）
        global_tokens = self.token_features(global_tokens)  # [batch_size, token_dim]
        
        # 处理局部特征（优化：分离计算，便于内存管理）
        local_features = self.cluster_features(x)  # [total_points, cluster_dim]
        scores = self.score(x)  # [total_points, num_clusters]
        
        # 优化：如果不需要计算loss（测试模式），可以在此处释放x
        if not self.training:
            del x
        
        # 向量化的特征聚合实现
        # 使用两种策略：1) 简化的softmax近似 2) 完整的Sinkhorn算法
        
        use_sinkhorn = hasattr(self, 'use_sinkhorn') and self.use_sinkhorn
        
        if use_sinkhorn:
            # 完整的Sinkhorn算法实现（较慢但更准确）
            aggregated_features = self._vectorized_sinkhorn_aggregation(
                local_features, scores, indptr, batch_size
            )
        else:
            # 简化的softmax近似（快速且效果良好）
            aggregated_features = self._softmax_aggregation(
                local_features, scores, indptr, batch_size
            )
        
        # 优化：及时释放中间变量
        del local_features, scores
        
        # 拼接global token和聚合特征
        result = torch.cat([
            global_tokens,  # [batch_size, token_dim]
            aggregated_features  # [batch_size, num_clusters * cluster_dim]
        ], dim=-1)
        
        # 最终归一化
        result = F.normalize(result, p=2, dim=-1)
        
        return result

    def _softmax_aggregation(self, local_features, scores, indptr, batch_size):
        """
        使用softmax近似的快速特征聚合（优化版本：减少内存占用）
        """
        # 优化：对scores进行softmax归一化
        scores_normalized = F.softmax(scores, dim=-1)  # [total_points, num_clusters]
        
        # 优化：使用高效的矩阵运算替代repeat操作
        # 方法：利用einsum直接计算加权特征，避免创建大张量
        # 对于每个点云，我们需要计算：sum(weights_k * local_features) for each cluster k
        # 可以通过转置和矩阵乘法实现
        
        # 优化：使用转置后的特征和权重进行批处理计算
        # local_features: [total_points, cluster_dim]
        # scores_normalized: [total_points, num_clusters]
        # 我们需要计算每个聚类的加权特征
        
        # 策略：使用einsum进行高效计算，然后分组聚合
        # 计算 weighted_features: [total_points, num_clusters, cluster_dim]
        # 但避免显式创建这个大张量
        
        # 方法：分别对每个聚类计算，但使用向量化操作
        # 更高效的方式：重塑local_features并使用广播
        # local_features: [total_points, 1, cluster_dim]
        # scores_normalized: [total_points, num_clusters, 1]
        # weighted = local_features * scores_normalized: [total_points, num_clusters, cluster_dim]
        
        # 优化：使用einsum避免中间变量
        # 直接计算每个聚类的加权和
        aggregated_features_list = []
        for k in range(self.num_clusters):
            # 选择第k个聚类的权重 [total_points]
            weights_k = scores_normalized[:, k]  # [total_points]
            # 计算加权特征 [total_points, cluster_dim]
            weighted_feat_k = weights_k.unsqueeze(-1) * local_features
            # 使用segment_csr聚合
            aggregated_k = torch_scatter.segment_csr(
                src=weighted_feat_k,
                indptr=indptr,
                reduce="sum"
            )  # [batch_size, cluster_dim]
            aggregated_features_list.append(aggregated_k)
            # 优化：及时释放中间变量（在循环中）
            del weighted_feat_k
        
        # 拼接所有聚类特征
        aggregated_features = torch.stack(aggregated_features_list, dim=1)  # [batch_size, num_clusters, cluster_dim]
        del aggregated_features_list  # 释放列表
        
        # 对每个聚类进行归一化
        aggregated_features = F.normalize(aggregated_features, p=2, dim=-1)
        
        # 重塑为向量形式
        aggregated_features = aggregated_features.view(batch_size, -1)  # [batch_size, num_clusters * cluster_dim]
        
        return aggregated_features

    def _vectorized_sinkhorn_aggregation(self, local_features, scores, indptr, batch_size):
        """
        向量化的Sinkhorn算法特征聚合（较慢但更准确，优化版本：减少内存占用）
        """
        # 为每个点云分别计算Sinkhorn权重
        aggregated_features = []
        
        for i in range(batch_size):
            start_idx = indptr[i]
            end_idx = indptr[i + 1]
            
            if end_idx - start_idx == 0:
                # 空点云
                empty_feat = torch.zeros(self.num_clusters * self.cluster_dim, 
                                       device=local_features.device, dtype=local_features.dtype)
                aggregated_features.append(empty_feat)
                continue
            
            # 提取当前点云的特征
            cloud_local = local_features[start_idx:end_idx]  # [n_points, cluster_dim]
            cloud_scores = scores[start_idx:end_idx]  # [n_points, num_clusters]
            
            # 转置为Sinkhorn期望的格式
            cloud_local = cloud_local.transpose(0, 1)  # [cluster_dim, n_points]
            cloud_scores = cloud_scores.transpose(0, 1)  # [num_clusters, n_points]
            
            # 使用Sinkhorn算法
            p = self._get_matching_probs_single(cloud_scores.unsqueeze(0), self.dust_bin, self.num_iters, self.reg)
            p = torch.exp(p)
            p = p[0, :-1, :]  # 移除dustbin行，[num_clusters, n_points]
            
            # 优化：避免使用repeat，改用einsum或更高效的矩阵运算
            # 原始方法：repeat会创建大量内存
            # 新方法：使用einsum直接计算加权和
            
            # 方法：对于每个cluster，计算加权和
            # cloud_local: [cluster_dim, n_points]
            # p: [num_clusters, n_points]
            # 需要计算: sum(cloud_local * p[k, :], dim=-1) for each cluster k
            
            # 优化：使用矩阵乘法避免repeat
            # aggregated = cloud_local @ p.T: [cluster_dim, num_clusters]
            aggregated = torch.matmul(cloud_local, p.t())  # [cluster_dim, num_clusters]
            aggregated = aggregated.flatten()  # [cluster_dim * num_clusters]
            
            aggregated_features.append(aggregated)
            
            # 优化：及时释放中间变量
            del cloud_local, cloud_scores, p
        
        # 拼接所有结果
        result = torch.stack(aggregated_features, dim=0)  # [batch_size, num_clusters * cluster_dim]
        del aggregated_features  # 释放列表
        
        return result

    def _get_matching_probs(self, S, dustbin_score=1.0, num_iters=3, reg=1.0):
        """批量Sinkhorn算法"""
        batch_size, m, n = S.size()
        
        # 添加dustbin
        S_aug = torch.empty(batch_size, m + 1, n, dtype=S.dtype, device=S.device)
        S_aug[:, :m, :n] = S
        S_aug[:, m, :] = dustbin_score
        
        # 准备归一化的源和目标log权重
        norm = -torch.tensor(math.log(n + m), device=S.device)
        log_a = norm.expand(m + 1).contiguous()
        log_b = norm.expand(n).contiguous()
        log_a[-1] = log_a[-1] + math.log(n - m)
        log_a = log_a.expand(batch_size, -1)
        log_b = log_b.expand(batch_size, -1)
        
        # 使用log_otp_solver
        log_P = self._log_otp_solver(log_a, log_b, S_aug, num_iters, reg)
        return log_P - norm

    def _get_matching_probs_single(self, S, dustbin_score=1.0, num_iters=3, reg=1.0):
        """单个样本的Sinkhorn算法"""
        return self._get_matching_probs(S, dustbin_score, num_iters, reg)

    def _log_otp_solver(self, log_a, log_b, M, num_iters=20, reg=1.0):
        """Sinkhorn算法求解器"""
        M = M / reg
        
        u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)
        
        for _ in range(num_iters):
            u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2).squeeze()
            v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1).squeeze()
        
        return M + u.unsqueeze(2) + v.unsqueeze(1)


# Usage example:
"""
# Assume data:
input_dict = {
    'coord': torch.randn(427716, 3),  # concatenated coordinates
    'feat': torch.randn(427716, 256),  # concatenated features
    'offset': torch.tensor([7151, 16512, 24264, ...]),  # offsets
    'batch': torch.tensor([...]),  # batch indices
}

# NetVLAD usage:
netvlad_wrapper = NetVladWrapper(cfg)  # cfg contains in_channels, out_channels, cluster_size, etc.
global_feat_vlad = netvlad_wrapper.forward_with_batch_info(
    input_dict['feat'],
    input_dict['batch'],
    input_dict['offset']
)  # (48, output_dim)

# GeM usage:
gem_wrapper = GeMWrapper(cfg)  # cfg contains feature_size, p, eps, etc.
global_feat_gem = gem_wrapper.forward_with_batch_info(
    input_dict['feat'],
    input_dict['batch'],
    input_dict['offset']
)  # (48, feature_size)

# SOP usage:
sop_wrapper = SOPWrapper(cfg)  # cfg contains feature_size, thresh, is_vec, do_pe, alpha_init, etc.
global_feat_sop = sop_wrapper.forward_with_batch_info(
    input_dict['feat'],
    input_dict['batch'],
    input_dict['offset']
)  # (48, feature_size * feature_size)

# SALAD usage:
salad_wrapper = SALADWrapper(cfg)  # cfg contains feature_size, num_clusters, cluster_dim, etc.
global_feat_salad = salad_wrapper.forward_with_batch_info(
    input_dict['feat'],
    input_dict['batch'],
    input_dict['offset']
)  # (48, token_dim + cluster_dim * num_clusters)

# SOP配置示例：
# cfg = {
#     'feature_size': 256,
#     'alpha_init': 0.5,      # 初始alpha值
#     'alpha_min': 1e-3,      # alpha最小值
#     'alpha_max': 2.0,       # alpha最大值
#     'do_pe': True,          # 启用power encoding
#     'is_vec': True,         # 输出向量形式
# }

# SALAD配置示例：
# cfg = {
#     'feature_size': 256,
#     'num_clusters': 64,     # 聚类数量
#     'cluster_dim': 128,     # 聚类特征维度
#     'token_dim': 256,       # 全局token维度
#     'dropout': 0.3,         # dropout率
#     'num_iters': 3,         # Sinkhorn迭代次数
#     'reg': 1.0,             # 正则化参数
#     'gem_p': 3.0,           # GeM的p参数
#     'use_sinkhorn': False,  # 是否使用完整Sinkhorn算法（False使用快速softmax近似）
# }

# In your model, you can automatically detect and use the appropriate method:
# The model will automatically call forward_with_batch_info when dealing with concatenated point clouds
"""