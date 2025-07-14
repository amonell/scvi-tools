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
import umap
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_predict_rf(adata, 
                        control='sgCd19',
                        neighbors=30,
                        intermediate_dim=64,
                        latent_dim=10,
                        beta=1.2,
                        encoder_epochs=750,
                        encoder_lr=0.001):
    """
    Train RF model and generate predictions for each perturbation
    """
    # Define outfolder for UMAP plots
    outfolder = '/home/amonell/Desktop/spatial_perturb/2024_10_06_perturb4_no_baysor/CausalST/scripts/figures/umap'
    os.makedirs(outfolder, exist_ok=True)
    
    # Prepare data - only keep cells with sgRNAs
    adata_only_sgRNAs = remove_no_guides(adata, key='guide_rnas', no_guide='Other cells')
    
    # Create neighborhoods
    neighborhoods = make_neighborhood(adata_only_sgRNAs, adata, 
                                   n_neighbors=neighbors, 
                                   spatial_key='X_spatial', 
                                   no_guide='Other cells')
    
    # Split background and target indices
    background_indices = np.where(adata_only_sgRNAs.obs['guide_rnas'] == control)[0]
    target_indices = np.where(adata_only_sgRNAs.obs['guide_rnas'] != control)[0]
    
    background_neighborhood = neighborhoods[background_indices]
    target_neighborhood = neighborhoods[target_indices]
    
    # Convert to torch tensors
    input_dim = len(adata_only_sgRNAs.var.index.values)
    background_neighborhood = torch.tensor(np.array(background_neighborhood).astype(np.float32))
    target_neighborhood = torch.tensor(np.array(target_neighborhood).astype(np.float32))
    
    # Train encoder
    encoder_model = pretrain_contrastive_VAE(input_dim, 
                                           intermediate_dim, 
                                           latent_dim, 
                                           target_neighborhood, 
                                           background_neighborhood, 
                                           encoder_epochs, 
                                           beta, 
                                           encoder_lr)
    
    # Process each perturbation
    for comparison in tqdm(adata.obs['guide_rnas'].cat.categories):
        if comparison not in [control, 'Other cells']:
            print(f"Processing {comparison}")
            
            # Get inputs for current comparison
            x, y, t = get_causal_inputs(adata, control, comparison, n_neighbors=neighbors)
            
            # Transform inputs using encoder
            with torch.no_grad():
                mu, logvar = encoder_model.s_encoder(x)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                x_transformed = (mu + eps * std).numpy()
            
            # Calculate class weights
            t_numpy = t.numpy().ravel()
            n_samples = len(t_numpy)
            n_control = np.sum(t_numpy == 0)
            n_treated = np.sum(t_numpy == 1)
            
            # Compute balanced weights
            weight_control = n_samples / (2 * n_control)
            weight_treated = n_samples / (2 * n_treated)
            sample_weights = np.where(t_numpy == 0, weight_control, weight_treated)
            
            print(f"\nClass weights:")
            print(f"Control weight: {weight_control:.3f}")
            print(f"Treatment weight: {weight_treated:.3f}")
            
            # Train single RF for all genes with sample weights
            rf_model = RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=30,
                max_depth=8,
                min_samples_split=5,
                min_samples_leaf=3,
                max_features='sqrt',
                bootstrap=True,
                oob_score=True
            )
            
            x_train_with_t = np.hstack((x_transformed, t.reshape(-1, 1)))
            rf_model.fit(x_train_with_t, y, sample_weight=sample_weights)
            
            # Initialize arrays for full dataset with zeros
            control_full = np.zeros((adata.shape[0], adata.shape[1]))
            treated_full = np.zeros((adata.shape[0], adata.shape[1]))
        
            # Get indices for current comparison
            mask = adata.obs['guide_rnas'].isin([control, comparison])
            indices = np.where(mask)[0]
            
            # Generate predictions for subset
            subset_transformed = encoder_model.s_encoder(x)[0].detach().numpy()
            
            # Create treatment indicators for subset
            treated_x = np.hstack((subset_transformed, np.ones((subset_transformed.shape[0], 1))))
            control_x = np.hstack((subset_transformed, np.zeros((subset_transformed.shape[0], 1))))
            
            # Generate predictions for subset
            control_pred = rf_model.predict(control_x)
            treated_pred = rf_model.predict(treated_x)
            
            # Create new arrays for this specific comparison
            control_layer = np.zeros((adata.shape[0], adata.shape[1]))
            treated_layer = np.zeros((adata.shape[0], adata.shape[1]))
            
            # Get indices for current comparison
            mask = adata.obs['guide_rnas'].isin([control, comparison])
            indices = np.where(mask)[0]
            
            # Fill in the predictions at the correct indices
            control_layer[indices] = control_pred
            treated_layer[indices] = treated_pred
            
            # Store predictions in adata.layers with unique names for this comparison
            layer_key_control = f'control_{comparison}'
            layer_key_treated = f'treated_{comparison}'
            
            print(f"\nStoring predictions for {comparison}:")
            print(f"Control layer key: {layer_key_control}")
            print(f"Treated layer key: {layer_key_treated}")
            print(f"Non-zero predictions: {np.sum(treated_layer != 0)} cells")
            
            adata.layers[layer_key_control] = control_layer
            adata.layers[layer_key_treated] = treated_layer
            
            # After getting x, y, t
            print(f"\nInputs for {comparison}:")
            print("x shape:", x.shape)
            print("y shape:", y.shape)
            print("t shape:", t.shape)
            print("x range:", torch.min(x).item(), "-", torch.max(x).item())
            print("y range:", torch.min(y).item(), "-", torch.max(y).item())
            print("Unique t values:", torch.unique(t).numpy())

            # After transformation
            print("\nAfter transformation:")
            print("x_transformed shape:", x_transformed.shape)
            print("x_transformed range:", np.min(x_transformed), "-", np.max(x_transformed))
            print("x_transformed mean:", np.mean(x_transformed))
            print("x_transformed std:", np.std(x_transformed))

            # After training RF
            print("\nRF predictions:")
            y_pred = rf_model.predict(x_train_with_t)
            print("Prediction range:", np.min(y_pred), "-", np.max(y_pred))
            print("True value range:", torch.min(y).item(), "-", torch.max(y).item())
            print("Mean absolute error:", np.mean(np.abs(y.numpy() - y_pred)))
            print("Feature importances:", rf_model.feature_importances_)

            # Inside the perturbation loop, after getting x, y, t
            print(f"\nDebugging RF for {comparison}:")
            
            # Check input data
            print("\n1. Input Data:")
            print("Treatment values (t):", torch.unique(t).numpy())
            print("Treatment counts:", np.bincount((t == 1).numpy().ravel().astype(int)))
            print("X features mean:", torch.mean(x, dim=0)[:5])  # First 5 features
            print("Y values range:", torch.min(y).item(), "-", torch.max(y).item())
            print("Y values mean per treatment:")
            print("- Control:", torch.mean(y[t.ravel() == 0], dim=0)[:5])  # First 5 genes
            print("- Treated:", torch.mean(y[t.ravel() == 1], dim=0)[:5])
            
            # After RF training
            print("\n2. RF Model:")
            print("Feature importance summary:")
            print("- Latent features (mean):", np.mean(rf_model.feature_importances_[:-1]))
            print("- Treatment indicator:", rf_model.feature_importances_[-1])
            
            # Check predictions
            print("\n3. Predictions:")
            # Make predictions for same data with different treatment
            control_x = np.column_stack([x_transformed, np.zeros(len(x_transformed))])
            treated_x = np.column_stack([x_transformed, np.ones(len(x_transformed))])
            
            control_pred = rf_model.predict(control_x)
            treated_pred = rf_model.predict(treated_x)
            cate = treated_pred - control_pred
            
            print("CATE statistics:")
            print("Mean CATE:", np.mean(cate))
            print("CATE range:", np.min(cate), "-", np.max(cate))
            print("CATE std:", np.std(cate))
            
            # Compare to actual differences - FIXED THIS PART
            mask_treated = t.ravel() == 1
            mask_control = t.ravel() == 0
            actual_diff = torch.mean(y[mask_treated], dim=0) - torch.mean(y[mask_control], dim=0)
            print("\n4. Actual vs Predicted Effects:")
            print("Mean actual difference:", actual_diff.mean().item())
            print("Mean predicted difference (CATE):", np.mean(cate))
    
    return adata

def plot_latent_space(latent_space, labels, title, save_path):
    """Plot UMAP of latent space colored by treatment"""
    # Create UMAP embedding
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(latent_space)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Convert categorical labels to numerical for coloring
    unique_labels = np.unique(labels)
    label_to_num = {label: i for i, label in enumerate(unique_labels)}
    numeric_labels = np.array([label_to_num[label] for label in labels])
    
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                         c=numeric_labels, cmap='Set2', 
                         alpha=0.6)
    plt.title(title)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    
    # Add legend
    handles = [plt.scatter([], [], c=plt.cm.Set2(i/len(unique_labels)), 
               label=str(label)) for i, label in enumerate(unique_labels)]
    plt.legend(handles=handles)
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Parameters
    path_to_adata = '/mnt/sata2/Analysis_Alex_2/perturb4_no_baysor/final_object_corrected.h5ad'
    outfolder = '/mnt/sata2/Analysis_Alex_2/perturb4_no_baysor/CausalST'
    
    # Load data
    print("Loading data...")
    adata = sc.read(path_to_adata)
    
    # Run analysis
    print("Running RF analysis...")
    adata = train_and_predict_rf(adata)
    
    # Save results
    print("Saving results...")
    adata.write(os.path.join(outfolder, 'final_object_rf_predictions.h5ad'))
    
if __name__ == "__main__":
    main() 