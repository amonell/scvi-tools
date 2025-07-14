import scanpy as sc
import os
import sys
from simplified_analysis.preprocessing import *
from simplified_analysis.model import *
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import scipy.sparse
from sklearn.neighbors import KDTree

from simplified_analysis.architectures.common import CommonModel
from simplified_analysis.architectures.treatment import TreatmentModel
from simplified_analysis.architectures.tarnet import TARNet

# Check GPU availability at script start
print("\nChecking GPU availability...")
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"GPU is available:")
    print(f"  Device: {device}")
    print(f"  GPU Model: {torch.cuda.get_device_name(0)}")
    print(f"  Current GPU memory allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
    print(f"  Current GPU memory reserved: {torch.cuda.memory_reserved(0)/1e9:.2f} GB")
    print(f"  Max GPU memory allocated: {torch.cuda.max_memory_allocated(0)/1e9:.2f} GB")
    print(f"  CUDA version: {torch.version.cuda}")
else:
    device = torch.device('cpu')
    print("No GPU available, using CPU")

def train_tarnet(x, y, t, encoder_model, epochs=100, batch_size=32, lr=0.001):
    """Train TARNet model"""
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move encoder to device first
    encoder_model = encoder_model.to(device)
    
    # Initialize models
    output_dim = y.shape[1]
    with torch.no_grad():
        test_input = torch.randn(1, x.shape[1], device=device)  # Create test input on correct device
        encoder_output = encoder_model(test_input)
        latent_output = encoder_output[0] if isinstance(encoder_output, tuple) else encoder_output
        latent_dim = latent_output.shape[1]
    
    control_model = TreatmentModel(output_dim, latent_dim, 64)
    treatment_model = TreatmentModel(output_dim, latent_dim, 64)
    
    # Create TARNet and move to GPU if available
    model = TARNet(encoder_model, control_model, treatment_model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Move data to GPU if available
    x = x.to(device)
    y = y.to(device)
    t = t.to(device)
    
    # Create dataset and dataloader
    dataset = TensorDataset(x, y, t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    model.train()
    best_loss = float('inf')
    for epoch in range(epochs):
        total_loss = 0
        num_batches = len(dataloader)
        
        for batch_x, batch_y, batch_t in dataloader:
            batch_x, batch_y, batch_t = batch_x.to(device), batch_y.to(device), batch_t.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x, batch_t)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch+1}: loss = {avg_loss:.4f}')
    
    print(f"Training completed. Best loss: {best_loss:.4f}")
    return model

def train_and_predict_tarnet(adata, 
                           control='Ctrl',
                           neighbors=30,
                           intermediate_dim=64,
                           latent_dim=10,
                           beta=1.2,
                           encoder_epochs=750,
                           encoder_lr=0.001,
                           tarnet_epochs=100,
                           tarnet_lr=0.001):
    """
    Train TARNet model and generate predictions for each perturbation
    """
    # Prepare data
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
    
    # Transform whole dataset
    neighborhoods_whole = make_neighborhood(adata, adata, 
                                         n_neighbors=neighbors, 
                                         spatial_key='X_spatial', 
                                         no_guide='No guide')
    
    # Process each perturbation
    perturbations = [p for p in adata.obs['guide_rnas'].unique() 
                    if p not in [control, 'No guide']]
    print(f"\nFound {len(perturbations)} perturbations to process")
    
    for comparison in perturbations:
        print(f"\nProcessing perturbation: {comparison}")
        
        # Get inputs for current comparison
        x, y, t = get_causal_inputs(adata, control, comparison, n_neighbors=neighbors)
        print(f"Input shapes: x={x.shape}, y={y.shape}, t={t.shape}")
        
        # Convert to tensors and move to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = torch.FloatTensor(x).to(device)
        y = torch.FloatTensor(y).to(device)
        t = torch.FloatTensor(t).to(device)
        
        # Train TARNet model
        print("\nTraining TARNet model...")
        tarnet_model = train_tarnet(x, y, t, encoder_model, 
                                  epochs=tarnet_epochs, 
                                  lr=tarnet_lr)
        
        # Generate predictions
        print("\nGenerating predictions...")
        tarnet_model.eval()
        with torch.no_grad():
            # Initialize arrays for full dataset with zeros
            control_full = np.zeros((adata.shape[0], adata.shape[1]))
            treated_full = np.zeros((adata.shape[0], adata.shape[1]))
            
            # Get indices for current comparison
            mask = adata.obs['guide_rnas'].isin([control, comparison])
            indices = np.where(mask)[0]
            
            # Generate predictions for subset
            control_t = torch.zeros_like(t).to(device)
            treat_t = torch.ones_like(t).to(device)
            
            control_pred = tarnet_model(x, control_t).cpu().numpy()
            treated_pred = tarnet_model(x, treat_t).cpu().numpy()
            
            # Fill in the predictions at the correct indices
            control_full[indices] = control_pred
            treated_full[indices] = treated_pred
            
            # Store predictions in adata.layers
            adata.layers[f'control_{comparison}'] = control_full
            adata.layers[f'treated_{comparison}'] = treated_full
            
        print(f"Predictions stored in layers: 'control_{comparison}' and 'treated_{comparison}'")
        
    print("\nAll perturbations processed successfully!")
    return adata

def main(dataset_number=2):
    # Parameters
    storage_dir = '/mnt/sata2/alex_storage/Desktop/spatial_perturb/CausalST'
    
    # Create paths
    dataset_path = os.path.join(storage_dir, 'synthetic_datasets', 'CATE_testing', 
                               'datasets', f'synthetic_dataset_{dataset_number}.h5ad')
    outfolder = os.path.join(storage_dir, 'synthetic_datasets', 'CATE_testing', 
                            'tarnet_dataset_predictions')
    os.makedirs(outfolder, exist_ok=True)
    
    # Load data
    print("Loading data...")
    adata = sc.read(dataset_path)
    
    # Run analysis
    print("Running TARNet analysis...")
    adata = train_and_predict_tarnet(adata, control='Ctrl')
    
    # Save results
    print("Saving results...")
    output_path = os.path.join(outfolder, f'final_object_tarnet_predictions_{dataset_number}.h5ad')
    adata.write(output_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, default=2)
    args = parser.parse_args()
    main(args.dataset) 