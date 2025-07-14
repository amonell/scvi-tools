import torch
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from simplified_analysis.architectures.contrastiveVAE import BetaVariationalAutoencoder, loss_function
from simplified_analysis.architectures.common import CommonModel
from simplified_analysis.architectures.tarnet import TARNet
from simplified_analysis.architectures.treatment import TreatmentModel

def pretrain_contrastive_VAE(input_dim, intermediate_dim, latent_dim, target_neighborhood, background_neighborhood, epochs, beta, lr):
    model = BetaVariationalAutoencoder(input_dim, intermediate_dim, latent_dim, beta)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        tg_outputs, bg_outputs, fg_outputs, tg_mean, tg_logvar, tg_s_mean, tg_s_logvar, bg_s_mean, bg_s_logvar = model(target_neighborhood, background_neighborhood)
        loss = loss_function(target_neighborhood, background_neighborhood, tg_outputs, bg_outputs, tg_mean, tg_logvar, tg_s_mean, tg_s_logvar, bg_s_mean, bg_s_logvar, beta)
        loss.backward()
        optimizer.step()
        if epoch % (epochs//10) == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    return model

def train_causal_model(x, y, t, encoder_model, intermediate_dim, latent_dim, num_epochs=30, lr=0.001, batch_size=16):
    output_length = y.shape[1]

    # Instantiate the models
    common_model = encoder_model.s_encoder
    control_model = TreatmentModel(output_length, latent_dim, [intermediate_dim])
    treatment_model = TreatmentModel(output_length, latent_dim, [intermediate_dim])

    # Create the TARNet model
    tarnet = TARNet(common_model, control_model, treatment_model)

    # Create a DataLoader
    dataset = TensorDataset(x, t, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(tarnet.parameters(), lr=lr)

    # Training loop
    loss_values = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (batch_x, batch_t, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = tarnet(batch_x, batch_t)
            loss = criterion(outputs[0], batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        loss_values.append(running_loss)

    print("Training complete")

    # Plot the training curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), loss_values, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    return tarnet