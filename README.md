# MOCA

**Momentum Contrastive Learning with Dual-Stage Multi-Head Attention for Precise Spatial Domain Identification in Spatially Resolved Transcriptomics**

This study presents MOCA, a momentum contrastive learning framework with multi-head attention for accurate spatial domain identification in spatial transcriptomics (ST) data.

Specifically:

Momentum contrastive learning is employed to generate stable self-supervised signals by applying exponential moving average (EMA) updates to the student encoder.

A multi-head attention mechanism is introduced to align student and teacher embeddings in the spatial dimension, improving representation consistency before contrastive optimization.

A dual-stage InfoNCE contrastive loss is designed to enforce semantic alignment across views, enhancing robustness and generalization in self-supervised learning.

Together, these components allow MOCA to learn high-quality spot representations and outperform state-of-the-art methods across diverse ST datasets.

## Environments

- torch         1.13.0+cu117
- scanpy        1.9.2
- rpy2          3.5.1
- matplotlib    3.5.3
- numpy         1.21.6
- pandas        1.3.5
- stlearn       0.4.12
- r-base        4.0.3

### 🚀 Demo Example

A quick demo on the DLPFC dataset (section `151673`) is provided in [`demo.py`](demo.py). This script performs the full workflow including graph construction, training, clustering, and visualization.

### ✅ Input Requirements

To run MOCA, prepare the following inputs for each section:
- `filtered_feature_bc_matrix.h5`: preprocessed gene expression matrix
- `cluster_labels.csv`: cluster label file
- `spatial/tissue_hires_image.png`: high-resolution tissue image
- `spatial/tissue_positions_list.csv`: spatial coordinate file

All inputs should follow the 10x Genomics Visium format.

### 🧪 Additional Experiments

For integration of mouse brain slices(mouse_anterior_posterior_brain_merged.h5ad), use Integration.py. This script performs cross-slice data alignment and visualization.

Default training uses 1000 epochs.

Results are automatically saved (spatial map, UMAP, PAGA).
