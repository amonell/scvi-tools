import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from tqdm import tqdm

def calculate_predicted_cates(adata, perturbation):
    """Calculate CATEs from the predicted control and treated values"""
    # Get cells that were part of the synthetic experiment
    mask = adata.obs['guide_rnas'].isin(['Ctrl', perturbation])
    adata = adata[mask]
    
    # Get predictions for these cells
    control_layer = f'control_{perturbation}'
    treated_layer = f'treated_{perturbation}'
    
    if control_layer in adata.layers and treated_layer in adata.layers:
        predicted_cates = adata.layers[treated_layer] - adata.layers[control_layer]
        return predicted_cates
    else:
        raise KeyError(f"Missing prediction layers for {perturbation}")

def evaluate_predictions(storage_dir, dataset_number):
    """Evaluate predictions against ground truth CATEs"""
    
    # Load original synthetic dataset to get perturbation info and gene names
    synthetic_path = os.path.join(storage_dir, 'synthetic_datasets', 'CATE_testing', 
                                'datasets', f'synthetic_dataset_{dataset_number}.h5ad')
    synthetic_adata = sc.read(synthetic_path)
    gene_names = synthetic_adata.var_names.tolist()
    
    # Get perturbation name (excluding Ctrl and No guide)
    perturbation = [g for g in synthetic_adata.obs['guide_rnas'].unique() 
                   if g not in ['Ctrl', 'No guide']][0]
    
    # Load ground truth CATEs
    cates_path = os.path.join(storage_dir, 'synthetic_datasets', 'CATE_testing', 
                             'ground_truth_cates', f'cates_{dataset_number}.npy')
    ground_truth_cates = np.load(cates_path)
    
    # Load RF predictions for this specific dataset
    predictions_path = os.path.join(storage_dir, 'synthetic_datasets', 'CATE_testing',
                                  'rf_dataset_predictions', f'final_object_rf_predictions_{dataset_number}.h5ad')
    adata_pred = sc.read(predictions_path)
    
    # Calculate predicted CATEs
    predicted_cates = calculate_predicted_cates(adata_pred, perturbation)
    
    # Ensure we're comparing the same cells
    n_synthetic = len(ground_truth_cates)
    predicted_cates = predicted_cates[:n_synthetic]
    
    # Calculate metrics
    correlations = []
    mse = []
    mae = []
    
    # Calculate per-cell metrics
    for i in range(len(ground_truth_cates)):
        corr, _ = stats.pearsonr(ground_truth_cates[i], predicted_cates[i])
        correlations.append(corr)
        mse.append(np.mean((ground_truth_cates[i] - predicted_cates[i])**2))
        mae.append(np.mean(np.abs(ground_truth_cates[i] - predicted_cates[i])))
    
    # Calculate average treatment effects
    ate_true = np.mean(ground_truth_cates, axis=0)
    ate_pred = np.mean(predicted_cates, axis=0)
    ate_corr, _ = stats.pearsonr(ate_true, ate_pred)
    
    # Create evaluation plots
    plt.rcParams.update({'font.size': 16})  # Set global font size
    fig, axes = plt.subplots(3, 2, figsize=(15, 20))
    
    # Original plots for first row
    sns.histplot(correlations, ax=axes[0,0])
    axes[0,0].set_title(f'Distribution of Per-cell Correlations\nMean: {np.mean(correlations):.3f}')
    axes[0,0].set_xlabel('Correlation')
    
    # ATE scatter plot
    axes[0,1].scatter(ate_true, ate_pred, alpha=0.5)
    for i, gene in enumerate(gene_names):
        axes[0,1].annotate(gene, 
                          (ate_true[i], ate_pred[i]),
                          xytext=(5, 5), 
                          textcoords='offset points',
                          fontsize=8,
                          alpha=0.7)
    axes[0,1].set_xlabel('True ATE')
    axes[0,1].set_ylabel('Predicted ATE')
    axes[0,1].set_title(f'True vs Predicted ATEs\nCorrelation: {ate_corr:.3f}')
    
    # Add identity line
    lims = [
        min(axes[0,1].get_xlim()[0], axes[0,1].get_ylim()[0]),
        max(axes[0,1].get_xlim()[1], axes[0,1].get_ylim()[1]),
    ]
    axes[0,1].plot(lims, lims, 'k--', alpha=0.5)
    
    # Add gene-specific treatment effect plots
    target_genes = ['Itgae', 'Tcf7', 'Gzma']
    for idx, gene in enumerate(target_genes):
        if gene in gene_names:
            gene_idx = gene_names.index(gene)
            
            # Get true and predicted values for this gene
            true_values = ground_truth_cates[:, gene_idx]
            pred_values = predicted_cates[:, gene_idx]
            
            # Create scatter plot
            axes[1+idx//2, idx%2].scatter(true_values, pred_values, alpha=0.5)
            axes[1+idx//2, idx%2].set_xlabel(f'True Treatment Effect - {gene}')
            axes[1+idx//2, idx%2].set_ylabel(f'Predicted Treatment Effect - {gene}')
            
            # Add correlation information
            gene_corr, _ = stats.pearsonr(true_values, pred_values)
            axes[1+idx//2, idx%2].set_title(f'{gene} Treatment Effects\nCorrelation: {gene_corr:.3f}')
            
            # Add identity line
            lims = [
                min(axes[1+idx//2, idx%2].get_xlim()[0], axes[1+idx//2, idx%2].get_ylim()[0]),
                max(axes[1+idx//2, idx%2].get_xlim()[1], axes[1+idx//2, idx%2].get_ylim()[1]),
            ]
            axes[1+idx//2, idx%2].plot(lims, lims, 'k--', alpha=0.5)
        else:
            print(f"Warning: Gene {gene} not found in dataset")
    
    plt.tight_layout()
    
    # Save plots
    plots_dir = os.path.join(storage_dir, 'synthetic_datasets', 'CATE_testing', 'evaluation')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, f'evaluation_dataset_{dataset_number}.pdf'))
    plt.close()
    
    # Save metrics
    metrics = {
        'dataset': dataset_number,
        'perturbation': perturbation,
        'mean_cell_correlation': np.mean(correlations),
        'std_cell_correlation': np.std(correlations),
        'mean_mse': np.mean(mse),
        'mean_mae': np.mean(mae),
        'ate_correlation': ate_corr
    }
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(plots_dir, f'metrics_dataset_{dataset_number}.csv'))
    
    return metrics

def main(dataset_number=2):
    storage_dir = '/mnt/sata2/alex_storage/Desktop/spatial_perturb/CausalST'
    
    try:
        metrics = evaluate_predictions(storage_dir, dataset_number)
        print(f"\nEvaluation metrics for dataset {dataset_number}:")
        for key, value in metrics.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"Error processing dataset {dataset_number}: {str(e)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, default=2)
    args = parser.parse_args()
    main(args.dataset) 