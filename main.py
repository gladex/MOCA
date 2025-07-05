import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import contingency_matrix, adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
import umap

# Environment settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore")
os.environ['R_HOME'] = r'C:\Program Files\R\R-4.3.3'
os.environ['PATH'] += os.pathsep + r'C:\Program Files\R\R-4.3.3\bin'

# Project path
project_root = os.getcwd()
sys.path.append(os.path.join(project_root, "MOCA_model"))

from MOCA_model.utils import cal_hybrid_graph, stats_hybrid_graph, Transfer_pytorch_Data, mclust_R
from MOCA_model.Train import MOCA_train

# Parameter settings
rad_cutoff = 150
section_id = '151673'   
input_dir = os.path.join(project_root, "MOCA_model", "data", section_id)
truth_file_path = os.path.join(input_dir, f"cluster_labels_{section_id}.csv")
save_dir = os.path.join(project_root, "figures")
os.makedirs(save_dir, exist_ok=True)

save_spatial_path = os.path.join(save_dir, f"spatial_{section_id}.png")
save_umap_path    = os.path.join(save_dir, f"umap_{section_id}.png")
save_paga_path    = os.path.join(save_dir, f"paga_{section_id}.png")

# ✅ Load data
adata = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5')
adata.var_names_make_unique()
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# ✅ Load Ground Truth
Ann_df = pd.read_csv(truth_file_path, header=None, index_col=0)
Ann_df.columns = ['Ground Truth']
adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']

# ✅ Build graph
cal_hybrid_graph(adata, rad_cutoff=rad_cutoff)
stats_hybrid_graph(adata)

# ✅ Model training & clustering
adata_temp = adata.copy()
Transfer_pytorch_Data(adata_temp)
adata_temp = MOCA_train(adata_temp)
adata_temp = mclust_R(adata_temp, num_cluster=4, rad_cutoff=rad_cutoff, used_obsm='MOCA')

# ✅ Metrics
obs_df = adata_temp.obs.dropna()
ARI = adjusted_rand_score(obs_df['mclust'], obs_df['Ground Truth'])
NMI = normalized_mutual_info_score(obs_df['Ground Truth'], obs_df['mclust'])

def calculate_cluster_acc(y_true, y_pred):
    cm = contingency_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    return cm[row_ind, col_ind].sum() / np.sum(cm)

ACC = calculate_cluster_acc(obs_df['Ground Truth'], obs_df['mclust'])

# ✅ Print metrics
print(f"\n✅ Results: NMI = {NMI:.4f}, ARI = {ARI:.4f}, ACC = {ACC:.4f}")

# ✅ Save spatial plot
spatial_title = f"{section_id} ARI={ARI:.2f} NMI={NMI:.2f} ACC={ACC:.2f}"
plt.rcParams["figure.figsize"] = (3, 3)
sc.pl.spatial(adata_temp, color=["mclust", "Ground Truth"], title=[spatial_title, "Ground Truth"], show=False)
plt.savefig(save_spatial_path, dpi=300)
plt.close()

# ✅ Save UMAP plot
sc.pp.neighbors(adata_temp, use_rep='MOCA', n_neighbors=15, key_added='MOCA_neighbors')
umap_result = umap.UMAP(n_neighbors=15, min_dist=0.1).fit_transform(adata_temp.obsm['MOCA'])
adata_temp.obsm['X_umap'] = umap_result
sc.pl.umap(adata_temp, color='mclust', title='UMAP', show=False)
plt.savefig(save_umap_path, dpi=300)
plt.close()

# ✅ Save PAGA plot
sc.pp.neighbors(adata_temp, n_neighbors=15, use_rep='MOCA', key_added='MOCA')
sc.tl.paga(adata_temp, groups='mclust', neighbors_key='MOCA')
sc.pl.paga(adata_temp, color='mclust', threshold=0.02, fontsize=12, show=False)
plt.savefig(save_paga_path, dpi=300)
plt.close()

# ✅ Done
print(f"\n🎉 Spatial plot saved: {save_spatial_path}")
print(f"🎉 UMAP plot saved: {save_umap_path}")
print(f"🎉 PAGA plot saved: {save_paga_path}")
