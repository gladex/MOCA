import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
from utils import *
from torch_sparse import SparseTensor
from torch.nn import Parameter
from .gat_conv import GATConv
from .process import regularization, cosine_similarity

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveHead(nn.Module):
    def __init__(self, n_h, hidden_dim=512, m=0.999, T=0.07, num_heads=4):
        super().__init__()
        self.m = m
        self.T = T
        self.num_heads = num_heads

        # 学生编码器（投影头）
        self.proj = nn.Sequential(
            nn.Linear(n_h, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 教师编码器（动量编码器）
        self.momentum_proj = nn.Sequential(
            nn.Linear(n_h, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 多头注意力机制
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)

        self._init_momentum()

    def _init_momentum(self):
        """初始化教师编码器参数为学生的参数副本"""
        for param_q, param_k in zip(self.proj.parameters(), self.momentum_proj.parameters()):
            param_k.data.copy_(param_q.data.clone())
            param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        """使用 EMA 更新教师参数"""
        for param_q, param_k in zip(self.proj.parameters(), self.momentum_proj.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def _contrastive_loss(self, q, k):
        """标准 InfoNCE 损失函数，支持梯度传播"""
        if q.dim() == 3:
            q = q.squeeze(0)
        if k.dim() == 3:
            k = k.squeeze(0)

        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)

        logits = torch.einsum('nc,ck->nk', [q, k.T]) / self.T
        labels = torch.arange(logits.size(0), device=q.device)
        return F.cross_entropy(logits, labels)

    def forward(self, z, z_a):
        """
        z: 原始图嵌入（学生）
        z_a: 增强图嵌入（学生）
        """
        # 学生编码器生成 q, q_a
        q = self.proj(z)
        q_a = self.proj(z_a)

        # 教师编码器生成 k, k_a（无梯度）
        with torch.no_grad():
            self._momentum_update()
            k = self.momentum_proj(z)
            k_a = self.momentum_proj(z_a)

        # 多头注意力对齐（q 用 k_a 引导，q_a 用 k 引导）
        q   = self.multihead_attn(q.unsqueeze(0),   k_a.unsqueeze(0), k_a.unsqueeze(0))[0].squeeze(0)
        q_a = self.multihead_attn(q_a.unsqueeze(0), k.unsqueeze(0),   k.unsqueeze(0))  [0].squeeze(0)

        # 双向 InfoNCE 对比损失
        loss = (self._contrastive_loss(q, k_a) + self._contrastive_loss(q_a, k)) 
        #loss = self._contrastive_loss(q_a, k)
        return loss

# 修改后的 ContrastiveHead 类

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum

        return F.normalize(global_emb, p=2, dim=1)
torch.manual_seed(2020)

class MOCA(torch.nn.Module):
    def __init__(self, hidden_dims, device='cuda:0'):
        super(MOCA, self).__init__()
        self.device = device

        [in_dim, num_hidden, out_dim] = hidden_dims
        self.conv1 = GATConv(in_dim, num_hidden, heads=1, concat=False, dropout=0, add_self_loops=False, bias=False).to(device)
        self.conv2 = GATConv(num_hidden, out_dim, heads=1, concat=False, dropout=0, add_self_loops=False, bias=False).to(device)
        self.conv3 = GATConv(out_dim, num_hidden, heads=1, concat=False, dropout=0, add_self_loops=False, bias=False).to(device)
        self.conv4 = GATConv(num_hidden, in_dim, heads=1, concat=False, dropout=0, add_self_loops=False, bias=False).to(device)

        self.contrast_head = ContrastiveHead(hidden_dims[-1], num_heads=4).to(device)
        self.sigm = torch.nn.Sigmoid().to(device)
        self.read = AvgReadout().to(device)

        self.head1 = Parameter(torch.Tensor(hidden_dims[-1], hidden_dims[-1]))
        self.head2 = Parameter(torch.Tensor(hidden_dims[-1], hidden_dims[-1]))
        torch.nn.init.xavier_uniform_(self.head1)
        torch.nn.init.xavier_uniform_(self.head2)

    def forward(self, features, edge_index, graph_nei, graph_neg, feat_a=None):
        h1 = F.elu(self.conv1(features, edge_index))
        h2 = self.conv2(h1, edge_index, attention=False)
        z = F.normalize(torch.matmul(h2, self.head1), dim=-1)

        if feat_a is not None:
            h1_a = F.elu(self.conv1(feat_a, edge_index))
            h2_a = self.conv2(h1_a, edge_index, attention=False)
            z_a = F.normalize(torch.matmul(h2_a, self.head2), dim=-1)
        else:
            z_a = z

        h3 = F.elu(self.conv3(h2, edge_index, attention=True, tied_attention=self.conv1.attentions))
        h4 = self.conv4(h3, edge_index, attention=False)

        emb = self.sigm(self.read(z, edge_index.to_dense()))
        emb_a = self.sigm(self.read(z_a, edge_index.to_dense()))

        contr_loss = self.contrast_head(z, z_a)
  
        pair_loss = regularization(emb, graph_nei, graph_neg)

        return z, h4, contr_loss, pair_loss, emb