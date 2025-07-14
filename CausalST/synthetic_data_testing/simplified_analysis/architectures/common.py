import torch
import torch.nn.functional as F
import torch.nn as nn

from simplified_analysis.architectures.contrastiveVAE import VariationalEncoder

class CommonModel(nn.Module):
    def __init__(self, input_dim, intermediate_dims, latent_dim, weights_path='s_encoder_weights.pth'):
        super(CommonModel, self).__init__()
        self.encoder = VariationalEncoder(input_dim, intermediate_dims, latent_dim)
        self.load_pretrained_encoders(weights_path)

    def freeze_encoders(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def load_pretrained_encoders(self, weights_path):
        state_dict = torch.load(weights_path)
        new_state_dict = {}
        for k, v in state_dict.items():
            if 'mean_layer' in k:
                new_state_dict[k.replace('mean_layer', 'latent_layer')] = v
            elif 'logvar_layer' in k:
                continue
            else:
                new_state_dict[k] = v
        self.encoder.load_state_dict(new_state_dict, strict=False)
        self.freeze_encoders()

    def forward(self, x):
        mean, logvar = self.encoder(x)
        return mean, logvar