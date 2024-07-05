import lightning as L
import torch
import torch.nn.functional as F
import torch.nn as nn
from lightning.pytorch.utilities.types import STEP_OUTPUT


def loss_fn(recon_x, x, mu, log_var):
    bce = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return bce + kld


class VAE(L.LightningModule):
    def __init__(self, in_channels: int, latent_dim: int):
        super().__init__()
        self.save_hyperparameters()

        self.latent_dim = latent_dim

        self.en_fc1 = nn.Linear(in_channels, 1024)
        self.en_fc2_mu = nn.Linear(1024, latent_dim)
        self.en_fc2_sigma = nn.Linear(1024, latent_dim)
        self.de_fc3 = nn.Linear(latent_dim, 1024)
        self.de_fc4 = nn.Linear(1024, in_channels)

    def encode(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.en_fc1(x)
        x = torch.relu(x)
        return self.en_fc2_mu(x), self.en_fc2_sigma(x)

    def decode(self, z):
        z = self.de_fc3(z)
        z = torch.relu(z)
        z = self.de_fc4(z)
        z = torch.sigmoid(z)
        return z

    def reparameterize(self, mu, sigma):
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, sigma = self.encode(x)
        z = self.reparameterize(mu, sigma)
        return self.decode(z), mu, sigma

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        pred, mu, sigma = self(batch)
        loss = loss_fn(pred, batch, mu, sigma)

        self.log("train_loss", loss.item() / batch.size(0), sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        pred, mu, sigma = self(batch)
        loss = loss_fn(pred, batch, mu, sigma)

        self.log("val_loss", loss.item() / batch.size(0), sync_dist=True)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
