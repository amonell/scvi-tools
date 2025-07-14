import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

def calculate_treatment_effects(adata):
    """Calculate treatment effects from control and treated predictions"""
    treatment_effects = {}
    
    # Get all sgRNAs (excluding control and Other cells)
    sgRNAs = np.unique(adata[~adata.obs['guide_rnas'].isin(['sgCd19', 'Other cells'])].obs['guide_rnas'])
    
    for sgRNA in sgRNAs:
        control_layer = f'control_{sgRNA}'
        treated_layer = f'treated_{sgRNA}'
        
        if control_layer in adata.layers and treated_layer in adata.layers:
            # Calculate treatment effect as treated - control
            effect = adata.layers[treated_layer] - adata.layers[control_layer]
            treatment_effects[sgRNA] = effect
    
    return treatment_effects

def plot_ate_heatmap(adata, treatment_effects, outfolder='/home/amonell/Desktop/spatial_perturb/2024_10_06_perturb4_no_baysor/CausalST/scripts/figures/rf/global'):
    """Plot average treatment effects across all genes and perturbations"""
    
    # Initialize DataFrame to store ATEs
    ate_df = pd.DataFrame(index=adata.var_names)
    
    # Calculate ATEs for each sgRNA
    for sgRNA, effects in treatment_effects.items():
        # Compute mean ATEs across cells
        ATEs = pd.Series(
            np.mean(effects, axis=0),
            index=adata.var_names
        )
        ate_df[sgRNA] = ATEs
    
    # Plot global heatmap
    plt.figure(figsize=(15, 10))
    sns.clustermap(ate_df.T, 
                  cmap='coolwarm',
                  center=0,
                  xticklabels=False)
    plt.title('Global Treatment Effects Heatmap (RF)')
    plt.savefig(os.path.join(outfolder, 'global_ate_heatmap_rf.pdf'), 
                bbox_inches='tight')
    plt.close()
    
    return ate_df

def plot_top_effects(adata, sgRNA, treatment_effects, n_genes=15, outfolder='/home/amonell/Desktop/spatial_perturb/2024_10_06_perturb4_no_baysor/CausalST/scripts/figures/rf/per_perturbation'):
    """Plot top up/down regulated genes for a specific perturbation"""
    
    if sgRNA not in treatment_effects:
        print(f"No treatment effects found for {sgRNA}")
        return
    
    # Calculate mean effects
    ATEs = pd.Series(
        np.mean(treatment_effects[sgRNA][adata.obs['guide_rnas'].isin(['sgCd19', sgRNA])], axis=0),
        index=adata.var_names
    )
    
    # Sort genes by effect size
    ATEs_sorted = ATEs.sort_values(ascending=False)
    
    # Get top and bottom genes
    top_genes = ['Itgae', 'Cd69']
    bottom_genes = ['Klf2', 'Mki67']
    var_names = top_genes + bottom_genes
    
    # Create matrix plot
    var_group_labels = [f'Top {n_genes} Upregulated', f'Top {n_genes} Downregulated']
    var_group_positions = [(0, n_genes-1), (n_genes, 2*n_genes-1)]
    
    # Create temporary layer for plotting
    temp_layer_name = f'temp_effects_{sgRNA}'
    adata.layers[temp_layer_name] = treatment_effects[sgRNA]
    
    sc.pl.matrixplot(
        adata[adata.obs['guide_rnas'].isin(['sgCd19', sgRNA])],
        layer=temp_layer_name,
        var_names=var_names,
        groupby='cluster_cellcharter',
        cmap='coolwarm',
        vcenter=0,
        var_group_labels=var_group_labels,
        var_group_positions=var_group_positions,
        title=f'Treatment Effects by Cluster ({sgRNA}) - RF'
    )
    plt.savefig(os.path.join(outfolder, f'{sgRNA}_top_effects_rf.pdf'), 
                bbox_inches='tight')
    plt.close()
    
    # Remove temporary layer
    del adata.layers[temp_layer_name]

def plot_cluster_specific_effects(adata, treatment_effects, outfolder='/home/amonell/Desktop/spatial_perturb/2024_10_06_perturb4_no_baysor/CausalST/scripts/figures/rf/per_cluster'):
    """Plot treatment effects by cluster for each perturbation"""
    
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    # Get unique clusters
    clusters = adata.obs['cluster_cellcharter'].unique()
    
    for sgRNA, effects in treatment_effects.items():
        # Calculate mean effects per cluster
        cluster_effects = []
        for cluster in clusters:
            mask = adata.obs['cluster_cellcharter'] == cluster
            mean_effect = np.mean(effects[mask], axis=0)
            cluster_effects.append(mean_effect)
            
        # Create cluster effects matrix
        cluster_df = pd.DataFrame(cluster_effects, 
                                index=clusters,
                                columns=adata.var_names)
        
        # Plot clustermap
        plt.figure(figsize=(12, 8))
        sns.clustermap(cluster_df,
                      cmap='coolwarm',
                      center=0)
        plt.title(f'Cluster-specific Treatment Effects ({sgRNA}) - RF')
        plt.savefig(os.path.join(outfolder, f'{sgRNA}_cluster_effects_rf.pdf'), 
                    bbox_inches='tight')
        plt.close()

def main():
    # Parameters
    path_to_adata = '/mnt/sata2/Analysis_Alex_2/perturb4_no_baysor/CausalST/final_object_rf_predictions.h5ad'
    base_outfolder = '/home/amonell/Desktop/spatial_perturb/2024_10_06_perturb4_no_baysor/CausalST/scripts/figures/rf'
    
    # Create output directories
    os.makedirs(f'{base_outfolder}/global', exist_ok=True)
    os.makedirs(f'{base_outfolder}/per_perturbation', exist_ok=True)
    os.makedirs(f'{base_outfolder}/per_cluster', exist_ok=True)
    
    # Load data
    print("Loading data...")
    adata = sc.read(path_to_adata)
    
    # Calculate treatment effects
    print("Calculating treatment effects...")
    treatment_effects = calculate_treatment_effects(adata)
    
    if treatment_effects:
        # Generate plots
        print("Generating global heatmap...")
        ate_df = plot_ate_heatmap(adata, treatment_effects)
        
        print("Generating top effects plots...")
        for sgRNA in treatment_effects.keys():
            plot_top_effects(adata, sgRNA, treatment_effects)
        
        print("Generating cluster-specific effects...")
        plot_cluster_specific_effects(adata, treatment_effects)
    else:
        print("No treatment effects could be calculated. Check layer names in the AnnData object.")

if __name__ == "__main__":
    main() 