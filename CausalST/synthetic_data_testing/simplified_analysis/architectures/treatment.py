import torch
import torch.nn as nn

class TreatmentModel(nn.Module):
    def __init__(self, output_length, latent_dim, hidden_dims=[64], dropout_rate=0.2):
        super().__init__()
        
        layers = []
        input_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)  # After ReLU
            ])
            input_dim = hidden_dim
        
        # Final layer without dropout
        layers.append(nn.Linear(input_dim, output_length))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        """For inference with dropout disabled"""
        self.eval()  # Sets the model to evaluation mode
        with torch.no_grad():
            prediction = self.model(x)
        self.train()  # Sets the model back to training mode
        return prediction