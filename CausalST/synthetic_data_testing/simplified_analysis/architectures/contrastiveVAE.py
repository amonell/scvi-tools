import torch
import torch.nn as nn
import numpy as np

class VariationalEncoder(nn.Module):
    def __init__(self, input_dim, intermediate_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim).float(),
            nn.ReLU(),
            nn.Linear(intermediate_dim, intermediate_dim).float(),
            nn.ReLU(),
            nn.Linear(intermediate_dim, latent_dim * 2).float()
        )
        self.float()

    def forward(self, x):
        x = x.float()
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

class BetaVariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, intermediate_dim, latent_dim):
        super().__init__()
        
        # Convert all layers to float32 explicitly
        self.encoder = VariationalEncoder(input_dim, intermediate_dim, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim).float(),
            nn.ReLU(),
            nn.Linear(intermediate_dim, intermediate_dim).float(),
            nn.ReLU(),
            nn.Linear(intermediate_dim, input_dim).float()
        )
        
        # Ensure all parameters are float32
        self.float()
    
    def s_encoder(self, x):
        # Convert numpy array to torch tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        else:
            x = x.float()
        
        mu, logvar = self.encoder(x)
        return mu, logvar
    
    def forward(self, x):
        # Ensure input is float32
        x = x.float()
        mu, logvar = self.s_encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

def loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    Compute the Beta-VAE loss function.
    """
    # Reconstruction loss (MSE)
    MSE = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return MSE + beta * KLD
