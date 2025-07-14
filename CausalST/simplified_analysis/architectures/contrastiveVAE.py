import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalEncoder(nn.Module):
    def __init__(self, input_dim, intermediate_dims, latent_dim):
        super(VariationalEncoder, self).__init__()
        self.intermediate_layers = nn.ModuleList()
        for dim in intermediate_dims:
            self.intermediate_layers.append(nn.Linear(input_dim, dim))
            input_dim = dim
        self.mean_layer = nn.Linear(input_dim, latent_dim)
        self.logvar_layer = nn.Linear(input_dim, latent_dim)
        
    def forward(self, x):
        for layer in self.intermediate_layers:
            x = F.relu(layer(x))
        mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

class VariationalDecoder(nn.Module):
    def __init__(self, latent_dim, intermediate_dims, output_dim):
        super(VariationalDecoder, self).__init__()
        self.intermediate_layers = nn.ModuleList()
        input_dim = 2 * latent_dim
        for dim in intermediate_dims:
            self.intermediate_layers.append(nn.Linear(input_dim, dim))
            input_dim = dim
        self.output_layer = nn.Linear(input_dim, output_dim)
        
    def forward(self, z, s):
        x = torch.cat([z, s], dim=-1)
        for layer in self.intermediate_layers:
            x = F.relu(layer(x))
        return self.output_layer(x)

class BetaVariationalAutoencoder(nn.Module):
    def __init__(self, input_dim=347, intermediate_dim=64, latent_dim=10, beta=1.0):
        super(BetaVariationalAutoencoder, self).__init__()
        intermediate_dim = [intermediate_dim] if isinstance(intermediate_dim, int) else intermediate_dim
        
        self.z_encoder = VariationalEncoder(input_dim, intermediate_dim, latent_dim)
        self.s_encoder = VariationalEncoder(input_dim, intermediate_dim, latent_dim)
        self.decoder = VariationalDecoder(latent_dim, intermediate_dim, input_dim)
        self.beta = beta

    def forward(self, tg_inputs, bg_inputs):
        tg_mean, tg_logvar = self.z_encoder(tg_inputs)
        tg_z = self.z_encoder.reparameterize(tg_mean, tg_logvar)
        tg_s_mean, tg_s_logvar = self.s_encoder(tg_inputs)
        tg_s = self.s_encoder.reparameterize(tg_s_mean, tg_s_logvar)
        bg_s_mean, bg_s_logvar = self.s_encoder(bg_inputs)
        bg_s = self.s_encoder.reparameterize(bg_s_mean, bg_s_logvar)
        
        tg_outputs = self.decoder(tg_z, tg_s)
        bg_outputs = self.decoder(torch.zeros_like(bg_s), bg_s)
        fg_outputs = self.decoder(tg_z, torch.zeros_like(tg_s))
        
        return tg_outputs, bg_outputs, fg_outputs, tg_mean, tg_logvar, tg_s_mean, tg_s_logvar, bg_s_mean, bg_s_logvar

    def get_latent_z(self, inputs):
        with torch.no_grad():
            mean, logvar = self.z_encoder(inputs)
            return mean

    def get_latent_s(self, inputs):
        with torch.no_grad():
            mean, logvar = self.s_encoder(inputs)
            return mean

# Loss function
def loss_function(tg_inputs, bg_inputs, tg_outputs, bg_outputs, tg_mean, tg_logvar, tg_s_mean, tg_s_logvar, bg_s_mean, bg_s_logvar, beta):
    reconstruction_loss = F.mse_loss(tg_outputs, tg_inputs, reduction='sum') + F.mse_loss(bg_outputs, bg_inputs, reduction='sum')
    
    # KL divergence losses
    kl_divergence_tg = -0.5 * torch.sum(1 + tg_logvar - tg_mean.pow(2) - tg_logvar.exp())
    kl_divergence_tg_s = -0.5 * torch.sum(1 + tg_s_logvar - tg_s_mean.pow(2) - tg_s_logvar.exp())
    kl_divergence_bg_s = -0.5 * torch.sum(1 + bg_s_logvar - bg_s_mean.pow(2) - bg_s_logvar.exp())

    total_kl_divergence = kl_divergence_tg + kl_divergence_tg_s + kl_divergence_bg_s
    total_loss = reconstruction_loss + beta * total_kl_divergence

    return total_loss
