"""Spatial encoder module."""

import torch
import torch.nn as nn
from typing import Iterable, Optional

class SpatialEncoder(nn.Module):
    """
    A simple spatial encoder that can be easily swapped out for more complex implementations.
    
    Parameters
    ----------
    n_input_spatial : int
        Number of spatial input features (typically 2 or 3 for x,y[,z] coordinates)
    n_latent : int
        Dimensionality of the latent space (should match gene expression latent space)
    n_hidden : int
        Number of nodes per hidden layer
    n_layers : int
        Number of hidden layers
    dropout_rate : float
        Dropout rate to use in training
    use_batch_norm : bool
        Whether to use batch normalization
    use_layer_norm : bool
        Whether to use layer normalization
    """
    def __init__(
        self,
        n_input_spatial: int,
        n_latent: int,
        n_hidden: int = 128,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        
        # Build a simple MLP for spatial encoding
        layers = []
        
        # Input layer
        layers.append(nn.Linear(n_input_spatial, n_hidden))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(n_hidden))
        if use_layer_norm:
            layers.append(nn.LayerNorm(n_hidden))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout_rate))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(n_hidden))
            if use_layer_norm:
                layers.append(nn.LayerNorm(n_hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
        
        # Mean and variance encoders for the latent space
        self.encoder = nn.Sequential(*layers)
        self.mean_encoder = nn.Linear(n_hidden, n_latent)
        self.var_encoder = nn.Linear(n_hidden, n_latent)
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        spatial_coords: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        *cat_covs: Iterable[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the spatial encoder.
        
        Parameters
        ----------
        spatial_coords
            Tensor of spatial coordinates
        batch_index
            Batch indices, if any
        *cat_covs
            Categorical covariates, if any
            
        Returns
        -------
        tuple of:
            - mean of the latent distribution
            - variance of the latent distribution
            - sampled latent representation
        """
        # Encode spatial coordinates
        q = self.encoder(spatial_coords)
        
        # Get mean and variance
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q))  # Ensure positive variance
        
        # Sample from the latent distribution using reparameterization trick
        eps = torch.randn_like(q_m)
        z = q_m + torch.sqrt(q_v) * eps
        
        return q_m, q_v, z 