import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp

from MOCA_model import MOCA


from MOCA_model.utils import Transfer_pytorch_Data
from MOCA_model.MOCA import MOCA  # 类名应该大写


import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
# from loss import *

from MOCA_model.process import *

from tqdm import trange
from torch_geometric.transforms import ToSparseTensor
import  torch
import numpy as np
import pandas as pd
import sklearn
import scipy.sparse as sp

def MOCA_train(adata, hidden_dims=[512, 30], n_epochs=1000,lr=0.0009,key_added='MOCA',
                  gradient_clipping=5., weight_decay=0.0001, verbose=True,
                  random_seed=2020, save_loss=False, save_reconstrction=False,
                  device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):

    # # seed_everything()
    seed = random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    datatype = '10X'


    adata.X = sp.csr_matrix(adata.X)

    if 'highly_variable' in adata.var.columns:
        adata_Vars = adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata



    if verbose:
        print('Size of Input: ', adata_Vars.shape)
    if 'hybrid_graph' not in adata.uns.keys():
        raise ValueError("Haybrid_graph is not existed! Run Cal_Spatial_Net first!")

    if 'label_CSL' not in adata.obsm.keys():
        add_contrastive_label(adata)
    if 'feat' not in adata.obsm.keys():
        get_feature(adata)
    if 'adj' not in adata.obsm.keys(): 
        if datatype in ['Stereo', 'Slide']: # 10x
            construct_interaction_KNN(adata)
        else:
            construct_interaction(adata) 

    data = Transfer_pytorch_Data(adata_Vars)

    model = MOCA(hidden_dims=[data.x.shape[1]] + hidden_dims).to(device)


    class SimpleClass:
        def __init__(self, adata, device):
            self.adata = adata
            self.device = device
            #self.loss_CSL = nn.BCEWithLogitsLoss() #定义了一个二元交叉熵损失函数 BCEWithLogitsLoss，用于计算模型输出与目标标签之间的损失。
            self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device) #feat为原始基因表达矩阵，
            self.features_a = torch.FloatTensor(self.adata.obsm['feat_a'].copy()).to(self.device)#feat_a为打乱细胞顺序后的增强的基因表达矩阵，
            self.label_CSL = torch.FloatTensor(self.adata.obsm['label_CSL']).to(self.device)
            # self.adj = self.adata.obsm['adj_t']

    simple_instance = SimpleClass(adata, device)
    #创建一个 SimpleClass 的实例 simple_instance，用于后续访问其中存储的数据。
    data = data.to(device)
    data = ToSparseTensor()(data)  ##transfer data to sparse data which can ensure the reproducibility when seed fixed
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


    progress_bar = trange(1, n_epochs + 1, desc="Training MOCA")
    for epoch in progress_bar:
        model.train()
        optimizer.zero_grad()

        # 构造图
        sadj, graph_nei, graph_neg = spatial_construct_graph1(adata, radius=150)
        graph_nei = graph_nei.to(device)
        graph_neg = graph_neg.to(device)

        # 前向传播
        z, out, contr_loss, pair_loss, emb = model(
            data.x, data.adj_t, graph_nei, graph_neg, simple_instance.features_a
        )

        # 损失计算
        reconstruction_loss = F.mse_loss(data.x, out)
        reg_loss = regularization(emb, graph_nei, graph_neg)
        total_loss = (reconstruction_loss * 1) + (contr_loss *1)+ (reg_loss *0)

        # 反向传播
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()

        # 更新进度条
        progress_bar.set_postfix({
            "Epoch": epoch,
            "ContrLoss": f"{contr_loss.item():.4f}",
            "TotalLoss": f"{total_loss.item():.4f}"
        })

        # 在评估模式重新构造图数据
    model.eval()
    sadj, graph_nei, graph_neg = spatial_construct_graph1(adata, radius=150)
    graph_nei = graph_nei.to(device)
    graph_neg = graph_neg.to(device)

    with torch.no_grad():
        z, out, contr_loss, pair_loss, emb = model(
            data.x, data.adj_t, graph_nei, graph_neg, simple_instance.features_a
        )

    # 保存嵌入表示
    MOCA_rep = z.to('cpu').detach().numpy()
    adata.obsm[key_added] = MOCA_rep

    # 保存重构表达
    if save_reconstrction:
        ReX = out.detach().cpu().numpy()
        ReX[ReX < 0] = 0

        full_ReX = np.zeros(adata.shape)
        if 'highly_variable' in adata.var:
            hv_mask = adata.var['highly_variable'].values
            assert ReX.shape[1] == np.sum(hv_mask), \
                f"ReX shape {ReX.shape} does not match HVG count {np.sum(hv_mask)}"
            full_ReX[:, hv_mask] = ReX
        else:
            raise ValueError("Missing 'highly_variable' column in adata.var.")
        adata.layers['MOCA_ReX'] = full_ReX

    return adata
'''
    if save_reconstrction:

        ReX = out.to('cpu').detach().numpy()
        ReX[ReX < 0] = 0
        adata.layers['MOCA_ReX'] = ReX

    return adata
'''