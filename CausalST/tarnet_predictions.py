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

def train_tarnet(x, y, t, encoder_model, epochs=100, batch_size=32, lr=0.001):
    """Train TARNet model"""
    # Initialize models
    output_dim = y.shape[1]
    latent_dim = encoder_model.s_encoder.mean_layer.out_features
    control_model = TreatmentModel(output_dim, latent_dim, 64)
    treatment_model = TreatmentModel(output_dim, latent_dim, 64)
    
    # Create TARNet
    model = TARNet(encoder_model.s_encoder, control_model, treatment_model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Create dataset and dataloader
    dataset = TensorDataset(x, y, t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y, batch_t in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x, batch_t)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss/len(dataloader)}')
    
    return model

def train_and_predict_tarnet(adata, 
                           control='sgCd19',
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
    
    # Transform whole dataset
    neighborhoods_whole = make_neighborhood(adata, adata, 
                                         n_neighbors=neighbors, 
                                         spatial_key='X_spatial', 
                                         no_guide='Other cells')
    
    # Process each perturbation
    for comparison in tqdm(adata.obs['guide_rnas'].cat.categories):
        if comparison not in [control, 'Other cells']:
            print(f"Processing {comparison}")
            
            # Get inputs for current comparison
            x, y, t = get_causal_inputs(adata, control, comparison, n_neighbors=neighbors)
            
            # Train TARNet model
            tarnet_model = train_tarnet(x, y, t, encoder_model, 
                                      epochs=tarnet_epochs, 
                                      lr=tarnet_lr)
            
            # Generate predictions
            tarnet_model.eval()
            with torch.no_grad():
                # Initialize arrays for full dataset with zeros
                control_full = np.zeros((adata.shape[0], adata.shape[1]))
                treated_full = np.zeros((adata.shape[0], adata.shape[1]))
                
                # Get indices for current comparison
                mask = adata.obs['guide_rnas'].isin([control, comparison])
                indices = np.where(mask)[0]
                
                # Generate predictions for subset
                control_t = torch.zeros_like(t)
                treat_t = torch.ones_like(t)
                
                control_pred = tarnet_model(x, control_t).numpy()
                treated_pred = tarnet_model(x, treat_t).numpy()
                
                # Fill in the predictions at the correct indices
                control_full[indices] = control_pred
                treated_full[indices] = treated_pred
                
                # Store predictions in adata.layers
                adata.layers[f'control_{comparison}'] = control_full
                adata.layers[f'treated_{comparison}'] = treated_full
            
    return adata

def main():
    # Parameters
    path_to_adata = '/mnt/sata2/Analysis_Alex_2/perturb4_no_baysor/final_object_corrected.h5ad'
    outfolder = '/mnt/sata2/Analysis_Alex_2/perturb4_no_baysor/CausalST'
    
    # Load data
    print("Loading data...")
    adata = sc.read(path_to_adata)
    
    # Run analysis
    print("Running TARNet analysis...")
    adata = train_and_predict_tarnet(adata)
    
    # Save results
    print("Saving results...")
    adata.write(os.path.join(outfolder, 'final_object_tarnet_predictions.h5ad'))
    
if __name__ == "__main__":
    main() 