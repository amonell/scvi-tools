import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import scipy.sparse

def calculate_actual_differences_by_cluster(adata, control='sgCd19'):
    """Calculate actual expression differences between treatment and control by cluster"""
    actual_effects = {}
    clusters = adata.obs['cluster_cellcharter'].unique()
    genes_of_interest = ['Itgae', 'Klf2', 'Gzma']
    
    # Get all sgRNAs (excluding control and Other cells)
    sgRNAs = np.unique(adata[~adata.obs['guide_rnas'].isin([control, 'Other cells'])].obs['guide_rnas'])
    
    for sgRNA in sgRNAs:
        cluster_effects = {}
        for cluster in clusters:
            # Get mean expression for control and treatment groups in this cluster
            mask_control = (adata.obs['guide_rnas'] == control) & (adata.obs['cluster_cellcharter'] == cluster)
            mask_treated = (adata.obs['guide_rnas'] == sgRNA) & (adata.obs['cluster_cellcharter'] == cluster)
            
            control_cells = adata[mask_control].X
            treated_cells = adata[mask_treated].X
            
            # Convert sparse matrices to dense if needed
            if scipy.sparse.issparse(control_cells):
                control_cells = control_cells.toarray()
            if scipy.sparse.issparse(treated_cells):
                treated_cells = treated_cells.toarray()
                
            # Calculate mean expression for genes of interest
            gene_indices = [list(adata.var_names).index(gene) for gene in genes_of_interest]
            control_mean = np.mean(control_cells[:, gene_indices], axis=0)
            treated_mean = np.mean(treated_cells[:, gene_indices], axis=0)
            
            # Calculate effect as difference in means
            effect = treated_mean - control_mean
            cluster_effects[cluster] = {gene: eff for gene, eff in zip(genes_of_interest, effect)}
            
        actual_effects[sgRNA] = cluster_effects
    
    return actual_effects

def get_predicted_effects_by_cluster(adata):
    """Extract predicted treatment effects from AnnData layers by cluster with debugging"""
    predicted_effects = {}
    clusters = adata.obs['cluster_cellcharter'].unique()
    genes_of_interest = ['Itgae', 'Klf2', 'Gzma']
    gene_indices = [list(adata.var_names).index(gene) for gene in genes_of_interest]
    
    # Get all unique perturbations from layer names
    layer_names = list(adata.layers.keys())
    perturbations = set(name.split('_')[1] for name in layer_names if name.startswith('treated_'))
    
    for perturbation in perturbations:
        cluster_effects = {}
        treated_layer = f'treated_{perturbation}'
        control_layer = f'control_{perturbation}'
        
        if treated_layer in adata.layers and control_layer in adata.layers:

            for cluster in clusters:
                mask = (adata.obs['guide_rnas'].isin(['sgCd19', perturbation])) & (adata.obs['cluster_cellcharter'] == cluster)
                treated_values = adata.layers[treated_layer][mask][:, gene_indices]
                control_values = adata.layers[control_layer][mask][:, gene_indices]
                
                effect = np.mean(treated_values - control_values, axis=0)
                
                cluster_effects[cluster] = {gene: eff for gene, eff in zip(genes_of_interest, effect)}
                
        predicted_effects[perturbation] = cluster_effects
            
    return predicted_effects

def plot_comparison(actual_effects, predicted_effects, sgRNA, model_name, outfolder, adata):
    """Plot comparison of actual vs predicted effects across neighborhoods with value annotations and CATE distribution"""
    genes_of_interest = ['Itgae', 'Klf2', 'Gzma']
    clusters = list(actual_effects[sgRNA].keys())
    
    # Create a figure with 2 rows: top for scatter plots, bottom for CATE distributions
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.5])
    fig.suptitle(f'Treatment Effects by Neighborhood\n{sgRNA} ({model_name})')
    
    # Top row: scatter plots
    for idx, gene in enumerate(genes_of_interest):
        ax_scatter = fig.add_subplot(gs[0, idx])
        
        actual_values = [actual_effects[sgRNA][cluster][gene] for cluster in clusters]
        predicted_values = [predicted_effects[sgRNA][cluster][gene] for cluster in clusters]
        
        # Create scatter plot
        ax_scatter.scatter(actual_values, predicted_values, alpha=0.6)
        ax_scatter.set_xlabel('Actual Effect')
        ax_scatter.set_ylabel('Predicted Effect')
        ax_scatter.set_title(gene)
        
        # Add correlation coefficient
        corr = np.corrcoef(actual_values, predicted_values)[0,1]
        ax_scatter.text(0.05, 0.95, f'r = {corr:.3f}', 
                       transform=ax_scatter.transAxes)
        
        # Add diagonal line
        # lims = [
        #     min(ax_scatter.get_xlim()[0], ax_scatter.get_ylim()[0]),
        #     max(ax_scatter.get_xlim()[1], ax_scatter.get_ylim()[1]),
        # ]
        # ax_scatter.plot(lims, lims, 'k--', alpha=0.5)
        
        # Add annotations for each point
        for i, cluster in enumerate(clusters):
            ax_scatter.annotate(
                f'{cluster}\nTrue: {actual_values[i]:.10f}\nPred: {predicted_values[i]:.10f}',
                (actual_values[i], predicted_values[i]),
                xytext=(10, 10),
                textcoords='offset points',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
                fontsize=8
            )
    
        # Bottom row: CATE distributions
        ax_dist = fig.add_subplot(gs[1, idx])
        
        # Calculate per-cell CATEs for this gene
        if sgRNA == 'sgCxcr3':
            treated_layer = f'treated_{sgRNA}'
            control_layer = f'control_{sgRNA}'
            
            if treated_layer in adata.layers and control_layer in adata.layers:
                gene_idx = list(adata.var_names).index(gene)
                cell_cates = adata.layers[treated_layer][:, gene_idx] - adata.layers[control_layer][:, gene_idx]
                
                # Plot distribution
                sns.histplot(cell_cates, ax=ax_dist, bins=50)
                ax_dist.set_title(f'Per-cell CATE Distribution - {gene}')
                ax_dist.set_xlabel('CATE')
                ax_dist.set_ylabel('Count')
                
                # Add mean line
                mean_cate = np.mean(cell_cates)
                ax_dist.axvline(mean_cate, color='r', linestyle='--', 
                              label=f'Mean CATE: {mean_cate:.2f}')
                ax_dist.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(outfolder, f'{sgRNA}_neighborhood_comparison_{model_name}.pdf'),
                bbox_inches='tight',
                dpi=300)
    plt.close()

def main():
    # Parameters
    rf_path = '/mnt/sata2/Analysis_Alex_2/perturb4_no_baysor/CausalST/final_object_rf_predictions.h5ad'
    tarnet_path = '/mnt/sata2/Analysis_Alex_2/perturb4_no_baysor/CausalST/final_object_tarnet_predictions.h5ad'
    outfolder = '/home/amonell/Desktop/spatial_perturb/2024_10_06_perturb4_no_baysor/CausalST/scripts/figures/comparisons'
    os.makedirs(outfolder, exist_ok=True)
    
    # Load data
    print("Loading data...")
    adata_rf = sc.read(rf_path)
    adata_tarnet = sc.read(tarnet_path)
    
    # Calculate actual effects by cluster
    print("Calculating actual effects...")
    actual_effects = calculate_actual_differences_by_cluster(adata_rf)
    
    # Get predicted effects by cluster
    print("Extracting predicted effects...")
    rf_effects = get_predicted_effects_by_cluster(adata_rf)
    tarnet_effects = get_predicted_effects_by_cluster(adata_tarnet)
    
    # Generate comparisons
    print("Generating comparisons...")
    for sgRNA in actual_effects.keys():
        plot_comparison(actual_effects, rf_effects, sgRNA, 'RF', outfolder, adata_rf)
        plot_comparison(actual_effects, tarnet_effects, sgRNA, 'TARNet', outfolder, adata_tarnet)

if __name__ == "__main__":
    main() 