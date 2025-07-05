# -*-coding:utf-8-*-
import numpy as np
import pandas as pd
from tqdm import tqdm
import sklearn
import scipy.sparse as sp
from MOCA_model.utils import Transfer_pytorch_Data


import torch
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True

# from scipy.sparse import issparse
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from sklearn.neighbors import NearestNeighbors
import ot


def construct_interaction(adata, n_neighbors=3):
    """Constructing spot-to-spot interactive graph"""
    position = adata.obsm['spatial'] #从 AnnData 对象中提取存储在 obsm 中的空间坐标数据（ x, y 坐标）


    distance_matrix = ot.dist(position, position, metric='euclidean')
    n_spot = distance_matrix.shape[0] #生成[n_spot x n_spot]大小的距离矩阵，反应每一个细胞到其他细胞的欧几里得距离

    adata.obsm['distance_matrix'] = distance_matrix #存储到字典中


    interaction = np.zeros([n_spot, n_spot]) #创建一个全零的矩阵 interaction，它的大小是 n_spot x n_spot。
    for i in range(n_spot):
        vec = distance_matrix[i, :] #对于每个点 i，首先提取该点到所有其他点的距离（vec）
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1

    adata.obsm['graph_neigh'] = interaction


    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj > 1, 1, adj)

    adata.obsm['adj'] = adj # 选取与点 i 最近的 3 个点（不包括自身），并在 interaction 矩阵中将这些点对的交互关系设为 1

def construct_interaction_KNN(adata, n_neighbors=3):
    position = adata.obsm['spatial']
    n_spot = position.shape[0]
    
    # 兼容原参数（n_neighbors改为支持列表或整数）
    if isinstance(n_neighbors, int):
        radii = [n_neighbors]  # 默认单尺度
    else:
        radii = n_neighbors
    
    adj_list = []
    for k in radii:
        nbrs = NearestNeighbors(
    n_neighbors=k + 1,
    algorithm='auto',  # 使用确定性算法如 'kd_tree'/'ball_tree' 
    metric_params={'sort_results': True}  # 确保相同距离时按索引排序
)
        _, indices = nbrs.kneighbors(position)
        x = indices[:, 0].repeat(k)
        y = indices[:, 1:].flatten()
        interaction = np.zeros([n_spot, n_spot])
        interaction[x, y] = 1
        
        adj = interaction + interaction.T
        adj = np.where(adj > 1, 1, adj)
        adj_list.append(adj)
    
    # ▼▼▼ 改动点：存储为邻接矩阵列表 ▼▼▼
    if len(adj_list) == 1:
        adata.obsm['adj'] = adj_list[0]
    else:
        adata.obsm['adj_multi'] = adj_list
    print(f'Generated {len(adj_list)} scale graph(s)!')
# ====== 修改为特征掩码增强 ======
def mask_features(feature, mask_rate, seed=2020):
    """
    重构特征屏蔽增强，确保实验可重复性：
    使用独立的随机数生成器而非全局状态
    """
    generator = np.random.RandomState(seed)  # ▼ 关键修改
    mask = generator.rand(*feature.shape) > mask_rate
    return feature * mask

def get_feature(adata):
    if 'highly_variable' in adata.var.columns:
        adata_Vars = adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata

    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
        feat = adata_Vars.X.toarray()[:, ] #是稀疏矩阵 提取基因表达数据给到feat
    else:
        feat = adata_Vars.X[:, ]
    if np.min(feat) < 0:
        print("[WARN] Negative values detected in feature matrix! Applying Min-Max scaling...")
        feat = (feat - np.min(feat)) / (np.max(feat) - np.min(feat))
        # data augmentation
# ====== 修改后 =====
    feat_a = mask_features(feat, mask_rate=0.2)# 使用新增强方法，建议保留mask_rate在0.2~0.5之间#通过随机打乱输入特征矩阵的行 得到增强后的基因表达矩阵feat_a

    adata.obsm['feat'] = feat
    adata.obsm['feat_a'] = feat_a
#将提取的基因表达特征 feat 和增强后的特征 feat_a 存储到 adata.obsm 中，分别命名为 'feat' 和 'feat_a'


def add_contrastive_label(adata):
    # contrastive label
    n_spot = adata.n_obs #adata 对象中的细胞或观测点的数量（即行数）4015
    one_matrix = np.ones([n_spot, 1])
    zero_matrix = np.zeros([n_spot, 1])
    label_CSL = np.concatenate([one_matrix, zero_matrix], axis=1) #【4015，2】矩阵 第一列为1 第二列为0
    adata.obsm['label_CSL'] = label_CSL #label_CSL 被添加到 adata.obsm，并命名为 'label_CSL'


def regularization(emb, graph_nei, graph_neg): #emb为正样本全局特征
    mat = torch.sigmoid(cosine_similarity(emb))  # 计算样本中的余弦相似性
    neigh_loss = torch.mul(graph_nei, torch.log(mat)).mean() #将邻接矩阵 graph_nei 和对数相似度矩阵相乘，得到的是正样本对的损失
    neg_loss = torch.mul(graph_neg, torch.log(1 - mat)).mean() #
    pair_loss = -(neigh_loss + neg_loss) / 2
    return pair_loss
    #对于正样本，我们希望相似度 mat[i, j] 越接近 1。 对于负样本，我们希望相似度 mat[i, j] 越接近 0。
def cosine_similarity(emb):
    norm = torch.norm(emb, p=2, dim=1, keepdim=True) #计算每个spot的L2范数
    emb_normalized = emb / norm #将每个嵌入向量除以其对应的 L2 范数，得到 单位化后的向量
    mat = torch.mm(emb_normalized, emb_normalized.T) #归一化嵌入向量之间的相似度矩阵
    mat.fill_diagonal_(0)  # 将每个向量与其自身的相似度设置为0
    return mat

def _nan2zero(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)

def spatial_construct_graph1(adata, radius=150):

    coor = pd.DataFrame(adata.obsm['spatial']) #coor获取空间坐标
    coor.index = adata.obs.index #coor.index添加空间坐标的索引
    coor.columns = ['imagerow', 'imagecol']
    A=np.zeros((coor.shape[0],coor.shape[0])) #全0矩阵【4015，4015】


    nbrs = sklearn.neighbors.NearestNeighbors(radius=radius).fit(coor)
    distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
    #indices 是半径150内邻居的索引。

    for it in range(indices.shape[0]):
        A[[it] * indices[it].shape[0], indices[it]] = 1
    #对于每个点 it，它会将与 it 点在半径范围内的所有邻居点在邻接矩阵A中标记为 1

    graph_nei = torch.from_numpy(A) #邻接矩阵，邻居标记为1
    graph_neg = torch.ones(coor.shape[0], coor.shape[0]) - graph_nei #从全1矩阵中减去邻接矩阵
     #graph_neg中0的位置表示有链接，1表示非邻居
    sadj = sp.coo_matrix(A, dtype=np.float32) #将 A 转换为稀疏矩阵
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
     #保证图的边是双向的

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph_nei = graph_nei.to(device)
    graph_neg = graph_neg.to(device)

    return  sadj,graph_nei, graph_neg


