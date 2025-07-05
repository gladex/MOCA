# -*- coding: utf-8 -*-
"""
main_MOCA_only.py ─ Run clustering analysis using MOCA model only (no other models)
"""

import os
import sys
import warnings
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment

# ========== 1. Environment configuration ==========
warnings.filterwarnings("ignore")
os.environ["R_HOME"] = r"C:\Program Files\R\R-4.3.3"
os.environ["PATH"] += os.pathsep + r"C:\Program Files\R\R-4.3.3\bin"
if sys.platform.startswith("win") and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(os.path.join(os.environ["R_HOME"], "bin"))
    os.add_dll_directory(os.path.join(os.environ["R_HOME"], "bin", "x64"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"▶ Using device: {device}")

# ========== 2. Import MOCA modules ==========
PROJECT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(PROJECT_DIR, "MOCA_model"))  # ✅ Allow Python to find utils.py

import warnings
warnings.filterwarnings("ignore")
import os, sys, logging
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt

# === Environment settings ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.environ['R_HOME'] = r'C:\Program Files\R\R-4.3.3'
os.environ['PATH'] += os.pathsep + r'C:\Program Files\R\R-4.3.3\bin'
sys.path.append(os.path.join(os.path.dirname(__file__), "MOCA_model"))

# === Import custom modules ===
from MOCA_model.utils import cal_hybrid_graph, stats_hybrid_graph, Transfer_pytorch_Data
from MOCA_model.Train import MOCA_train
from MOCA_model.utils import mclust_R
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment


# ========== 3. Paths and parameters ==========
DATA_PATH = r"D:\Users\Administrator\Desktop\mouse_anterior_posterior_brain_merged.h5ad"
LABEL_CSV = None  # If ground truth labels are available, set this path
N_CLUSTERS = 26
section_id = os.path.splitext(os.path.basename(DATA_PATH))[0]

# ========== 4. Load data ==========
print(f"▶ Loading data: {DATA_PATH}")
adata = sc.read_h5ad(DATA_PATH)
adata.var_names_make_unique()

# ========== 5. Scanpy preprocessing ==========
print("▶ HVG → Normalize → log1p")
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
# After preprocessing
sc.pp.log1p(adata)

# Build hybrid graph 
from MOCA_model.utils import cal_hybrid_graph
print("▶ Building hybrid graph")
adata = cal_hybrid_graph(adata, k_cutoff=10, model='KNN')
print("▶ Training MOCA model")
adata = MOCA_train(adata, n_epochs=1000, device=device, save_reconstrction=True)

# ========== 7. Clustering with mclust ==========
print("▶ Running mclust clustering")
adata = mclust_R(adata, num_cluster=N_CLUSTERS, rad_cutoff=150, used_obsm='MOCA')


# ========== 8. Evaluation metrics (if ground truth available) ==========
def cluster_acc(y_true, y_pred):
    cm = contingency_matrix(y_true, y_pred)
    r, c = linear_sum_assignment(-cm)
    return cm[r, c].sum() / cm.sum()

if LABEL_CSV and os.path.exists(LABEL_CSV):
    gt = pd.read_csv(LABEL_CSV, header=None, index_col=0)
    gt.columns = ["Ground Truth"]
    adata.obs["Ground Truth"] = gt.loc[adata.obs_names, "Ground Truth"]
    obs = adata.obs.dropna()
    print(f"▶ ARI={adjusted_rand_score(obs['Ground Truth'], obs['mclust']):.3f}  "
          f"NMI={normalized_mutual_info_score(obs['Ground Truth'], obs['mclust']):.3f}  "
          f"ACC={cluster_acc(obs['Ground Truth'], obs['mclust']):.3f}")
else:
    print("▶ No ground truth provided, skipping ARI/NMI/ACC calculation.")

# ========== 9. Spatial clustering visualization ==========
print("▶ Visualizing spatial clusters")
if "spatial" in adata.obsm:
    adata.obsm["spatial"][:, 1] *= -1

palette = dict(zip(
    adata.obs["mclust"].unique(),
    sns.color_palette("tab20", len(adata.obs["mclust"].unique()))
))
plt.rcParams["figure.figsize"] = (6, 6)
sc.pl.embedding(
    adata,
    basis="spatial",
    color="mclust",
    palette=palette,
    size=80,
    title="MOCA Spatial Clusters",
    show=True
)

print("✅ MOCA single-model clustering workflow finished")

# ========== 10. Spatial expression visualization (selected genes) ==========
print("▶ Visualizing spatial expression (Gpsm1, Gpr88, Cbln3, Hpca)")

target_genes = ['Gpsm1', 'Gpr88', 'Cbln3', 'Hpca']
available_genes = [g for g in target_genes if g in adata.var_names]

if not available_genes:
    print("❌ None of the selected genes found in the data")
else:
    for gene in available_genes:
        sc.pl.embedding(
            adata,
            basis="spatial",
            color=gene,            
            cmap="viridis",         
            size=80,
            title=f"{gene} expression (viridis style)",
            show=True
        )
