import torch
import torch.nn as nn
import torch.nn.functional as F

class TreatmentModel(nn.Module):
    def __init__(self, output_dim, latent_dims, intermediate_dim, dropout_prob=0.1):
        super(TreatmentModel, self).__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(latent_dims, output_dim)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        return x