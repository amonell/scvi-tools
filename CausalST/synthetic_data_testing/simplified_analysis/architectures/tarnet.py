import torch
import torch.nn as nn

class TARNet(nn.Module):
    def __init__(self, common_model, control_model, treatment_model):
        super(TARNet, self).__init__()
        self.common_model = common_model
        self.control_model = control_model
        self.treatment_model = treatment_model
        self.freeze_encoders()

    def freeze_encoders(self):
        for param in self.common_model.parameters():
            param.requires_grad = False

    def forward(self, x, t):

        z = self.common_model(x)[0]
        
        # Use the latent representation for both control and treatment models
        control = self.control_model(z)
        treatment = self.treatment_model(z)
        
        # Select outputs based on treatment indicator `t`
        t = t.view(-1, 1).float()
        selected_output = control * (1 - t) + treatment * t

        return selected_output
