import os
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
from scipy.spatial import distance_matrix

path_to_reference_adata= "/mnt/sata4/Alex_Xenium_Data/Perturb8/SI/combined/guides_assigned.h5ad"
query_adata = sc.read(path_to_reference_adata)

import sys
sys.path.insert(0, '/home/amonell/piloting/scvi-tools/src')
print("Added to path:", sys.path[0])

import scvi.external.resolvi as RESOLVI
import scvi
print("Importing from:", scvi.external.resolvi.__file__)

query_adata.obsm['X_cellcharter_0'] = np.zeros(np.shape(query_adata.obsm['X_cellcharter']))

RESOLVI.RESOLVI.setup_anndata(
    query_adata, 
    labels_key="resolvi_predicted",
    layer="counts",
    batch_key="batch", 
    perturbation_key="guide_rnas", 
    control_perturbation="sgCd19",
    background_key="guide_rnas",
    background_category="Other cells",
    spatial_embedding_key="X_cellcharter"
)

supervised_resolvi = RESOLVI.RESOLVI(
    query_adata, semisupervised=True, mixture_k = 1, n_latent = 10, override_mixture_k_in_semisupervised=False,control_penalty_weight=10000000
)

priors = supervised_resolvi.compute_dataset_dependent_priors()
print(priors)

supervised_resolvi.module.guide.downsample_counts_mean = float(
    supervised_resolvi.module.guide.downsample_counts_mean
)
supervised_resolvi.module.guide.downsample_counts_std = float(
    supervised_resolvi.module.guide.downsample_counts_std
)
supervised_resolvi.train(
    max_epochs=300,
    check_val_every_n_epoch=1,
    log_every_n_steps=1,
    lr=3e-4,
    train_on_perturbed_only=True
)

mini_adata = query_adata[query_adata.obs['guide_rnas'] != 'Other cells']
# Get denoised expression assuming control (no perturbations)
control_expr = supervised_resolvi.get_denoised_expression_control(mini_adata, n_samples=1000)

mini_adata.layers['resolvi_expression_no_shift'] = control_expr.values

sc.tl.pca(mini_adata, layer='resolvi_expression_no_shift')
sc.pp.neighbors(mini_adata)
sc.tl.umap(mini_adata)
sc.pl.umap(mini_adata, color=['guide_rnas', 'Klf2', 'Perturbation_10', 'Perturbation_4'], layer='resolvi_expression_no_shift', ncols=2)
sc.pl.umap(mini_adata, color=['guide_rnas', 'Klf2', 'Perturbation_10', 'Perturbation_4'], ncols=2, vmax=5)

perturbed_expr = supervised_resolvi.get_denoised_expression_perturbed(mini_adata, n_samples=1000)

mini_adata.layers['resolvi_expression_with_shift'] = perturbed_expr.values

sc.tl.pca(mini_adata, layer='resolvi_expression_with_shift')
sc.pp.neighbors(mini_adata)
sc.tl.umap(mini_adata)
sc.pl.umap(mini_adata, color=['guide_rnas', 'Klf2', 'Perturbation_10', 'Perturbation_4'], layer='resolvi_expression_with_shift', ncols=2)
sc.pl.umap(mini_adata, color=['guide_rnas', 'Klf2', 'Perturbation_10', 'Perturbation_4'], ncols=2, vmax=1)

target_gene = 'Cxcr3'
cluster_cellcharter = 4

indices = (query_adata.obs['guide_rnas'] == 'sg'+target_gene) & (query_adata.obs['cluster_cellcharter'] == cluster_cellcharter)
shift_outputs = supervised_resolvi.get_shift_network_outputs(
    #control_perturbation=16,
    indices=indices
)

df_shifts = pd.DataFrame(shift_outputs.values, columns=shift_outputs.columns, index=shift_outputs.index)
adata_shifts = sc.AnnData(X= df_shifts.values, obs=pd.DataFrame(query_adata[indices].obs['guide_rnas'].values, index=df_shifts.index.values, columns=['guide_rnas']), var=pd.DataFrame(index=df_shifts.columns))
mask = (mini_adata.obs['guide_rnas'] == 'sg'+target_gene) & (mini_adata.obs['cluster_cellcharter'] == cluster_cellcharter)
adata_shifts.layers['raw_shifts'] = perturbed_expr.values[mask] - control_expr.values[mask]

adata_shifts_filtered = adata_shifts.copy()

shifts =np.mean(adata_shifts_filtered.layers['raw_shifts'], axis = 0)
counts = np.mean(query_adata[indices].layers['counts'], axis = 0)

df = pd.DataFrame(zip(shifts, np.log1p(counts)), index = adata_shifts_filtered.var_names, columns = ['shift', 'mean_counts'])

# Create volcano plot: shifts vs mean counts
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.scatter(df['mean_counts'], df['shift'], alpha=0.6, s=30)

# Annotate specific genes of interest with highlighted box
genes_to_highlight = ['Itgae', 'Gzma', 'Perturbation_4', 'Klf2', 'Ccl5']
for gene in genes_to_highlight:
    if gene in df.index:
        idx = df.index.get_loc(gene)
        plt.annotate(gene, 
                    (df['mean_counts'].iloc[idx], df['shift'].iloc[idx]),
                    fontsize=10, alpha=0.9,
                    xytext=(5, 5), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

# Annotate all other genes with normal text
for gene in df.index:
    if gene not in genes_to_highlight:
        idx = df.index.get_loc(gene)
        plt.annotate(gene, 
                    (df['mean_counts'].iloc[idx], df['shift'].iloc[idx]),
                    fontsize=8, alpha=0.6,
                    xytext=(5, 5), textcoords='offset points')

plt.xlabel('Log Mean Counts', fontsize=12)
plt.ylabel('Shift', fontsize=12)
plt.title('Volcano Plot: Shifts vs Log Mean Counts', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()