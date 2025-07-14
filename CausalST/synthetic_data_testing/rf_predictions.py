import scanpy as sc
import os
import sys
from simplified_analysis.preprocessing import *
from simplified_analysis.model import *
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def train_and_predict_rf(adata, 
                        control='Ctrl',
                        neighbors=30,
                        intermediate_dim=64,
                        latent_dim=10,
                        beta=1.2,
                        encoder_epochs=750,
                        encoder_lr=0.001,
                        n_jobs=-1):
    """
    Train RF model and generate predictions for each perturbation
    """
    # Make observation names unique at the start
    adata.obs_names_make_unique()
    
    # Prepare data - only keep cells with sgRNAs
    adata_only_sgRNAs = remove_no_guides(adata, key='guide_rnas', no_guide='No guide')
    
    # Create neighborhoods
    neighborhoods = make_neighborhood(adata_only_sgRNAs, adata, 
                                   n_neighbors=neighbors, 
                                   spatial_key='X_spatial', 
                                   no_guide='No guide')
    
    # Split background and target indices
    background_indices = np.where(adata_only_sgRNAs.obs['guide_rnas'] == control)[0]
    target_indices = np.where(adata_only_sgRNAs.obs['guide_rnas'] != control)[0]
    
    background_neighborhood = neighborhoods[background_indices]
    target_neighborhood = neighborhoods[target_indices]
    
    # Convert to torch tensors with explicit dtype
    input_dim = len(adata_only_sgRNAs.var.index.values)
    background_neighborhood = torch.tensor(np.array(background_neighborhood), dtype=torch.float32)
    target_neighborhood = torch.tensor(np.array(target_neighborhood), dtype=torch.float32)
    
    # Train encoder
    encoder_model = pretrain_contrastive_VAE(input_dim, 
                                           intermediate_dim, 
                                           latent_dim, 
                                           target_neighborhood, 
                                           background_neighborhood, 
                                           encoder_epochs, 
                                           beta, 
                                           encoder_lr)
    
    # Transform whole dataset
    neighborhoods_whole = make_neighborhood(adata, adata, 
                                         n_neighbors=neighbors, 
                                         spatial_key='X_spatial', 
                                         no_guide='No guide')
    neighborhoods_whole = torch.tensor(neighborhoods_whole, dtype=torch.float32)
    
    # Get latent space representation
    with torch.no_grad():
        mu, _ = encoder_model.s_encoder(neighborhoods_whole)
        latent_space = mu.numpy()
        print("\nLatent space statistics:")
        print("Range:", np.min(latent_space), "-", np.max(latent_space))
        print("Mean:", np.mean(latent_space))
        print("Std:", np.std(latent_space))
    
    # Get unique perturbations
    perturbations = [p for p in adata.obs['guide_rnas'].unique() if p not in [control, 'No guide']]
    
    # Store training losses
    training_losses = {}
    
    # For each perturbation
    for comparison in perturbations:
        # Instead of gene-by-gene training, train one RF for all genes
        print(f"\nTraining RF for {comparison}")
        
        # Get indices for current comparison
        indices = np.where(adata.obs['guide_rnas'].isin([control, comparison]))[0]
        current_latent = latent_space[indices]
        current_expression = adata[indices].X.toarray() if scipy.sparse.issparse(adata[indices].X) else adata[indices].X
        current_treatment = (adata[indices].obs['guide_rnas'] == comparison).astype(int)
        
        # Prepare X matrix
        X = np.column_stack([current_latent, current_treatment])
        control_x = np.column_stack([current_latent, np.zeros(len(current_latent))])
        treated_x = np.column_stack([current_latent, np.ones(len(current_latent))])
        
        # Train single RF for all genes with anti-overfitting parameters
        rf_model = RandomForestRegressor(
            n_estimators=200,          # Increased from 100 for better averaging
            random_state=42,
            n_jobs=30,
            max_depth=8,               # Reduced from 10 to prevent deep, overfit trees
            min_samples_split=5,       # Require more samples to split a node
            min_samples_leaf=3,        # Require more samples in leaf nodes
            max_features='sqrt',       # Use subset of features for each split
            bootstrap=True,            # Use bootstrapping for better generalization
            oob_score=True            # Enable out-of-bag score to monitor overfitting
        )
        
        # Train on all genes at once
        rf_model.fit(X, current_expression)
        
        # Calculate and store both training and OOB scores
        training_score = rf_model.score(X, current_expression)
        oob_score = rf_model.oob_score_
        print(f"\nComparison: {comparison}")
        print(f"Training R² score: {training_score:.3f}")
        print(f"Out-of-bag R² score: {oob_score:.3f}")
        print(f"Difference (potential overfitting): {training_score - oob_score:.3f}")
        
        # Calculate training loss (average across all genes)
        y_pred = rf_model.predict(X)
        loss = mean_squared_error(current_expression, y_pred)
        training_losses[comparison] = loss
        
        # Make predictions for all genes at once
        control_full = rf_model.predict(control_x)
        treated_full = rf_model.predict(treated_x)
        
        # Create full-size arrays initialized with zeros
        control_layer = np.zeros((len(adata), adata.shape[1]))
        treated_layer = np.zeros((len(adata), adata.shape[1]))
        
        # Assign predictions only to relevant indices
        control_layer[indices] = control_full
        treated_layer[indices] = treated_full
        
        # Store predictions in adata.layers
        adata.layers[f'control_{comparison}'] = control_layer
        adata.layers[f'treated_{comparison}'] = treated_layer
        
        # Inside the perturbation loop, after making predictions
        print(f"\nPrediction statistics for {comparison}:")
        print("Control range:", np.min(control_full), "-", np.max(control_full))
        print("Treatment range:", np.min(treated_full), "-", np.max(treated_full))
        print("Mean difference:", np.mean(treated_full - control_full))
        
        # After RF training
        feature_importances = rf_model.feature_importances_
        print(f"\nFeature importances for gene {gene}:")
        print("Latent features:", feature_importances[:-1].mean())
        print("Treatment indicator:", feature_importances[-1])
    
    # Plot training losses (now just one loss per perturbation)
    plt.figure(figsize=(10, 6))
    plt.bar(training_losses.keys(), training_losses.values())
    plt.xlabel('Perturbation')
    plt.ylabel('Mean Squared Error')
    plt.title('RF Training Loss')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the loss plot
    plots_dir = os.path.join('/mnt/sata2/alex_storage/Desktop/spatial_perturb/CausalST',
                            'synthetic_datasets', 'CATE_testing', 'rf_training_plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, f'rf_training_loss.pdf'))
    plt.close()
    
    return adata

def main(dataset_number=2):
    # Parameters
    storage_dir = '/mnt/sata2/alex_storage/Desktop/spatial_perturb/CausalST'
    
    # Create paths
    dataset_path = os.path.join(storage_dir, 'synthetic_datasets', 'CATE_testing', 
                               'datasets', f'synthetic_dataset_{dataset_number}.h5ad')
    outfolder = os.path.join(storage_dir, 'synthetic_datasets', 'CATE_testing', 
                            'rf_dataset_predictions')
    os.makedirs(outfolder, exist_ok=True)
    
    # Load data
    print("Loading data...")
    adata = sc.read(dataset_path)
    
    # After loading data, before training
    print("Data statistics:")
    if scipy.sparse.issparse(adata.X):
        print("Expression range:", np.min(adata.X.data), "-", np.max(adata.X.data))
    else:
        print("Expression range:", np.min(adata.X), "-", np.max(adata.X))
    
    # Run analysis
    print("Running RF analysis...")
    adata = train_and_predict_rf(adata)
    
    # Save results
    print("Saving results...")
    output_path = os.path.join(outfolder, f'final_object_rf_predictions_{dataset_number}.h5ad')
    adata.write(output_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, default=2)
    args = parser.parse_args()
    main(args.dataset) 