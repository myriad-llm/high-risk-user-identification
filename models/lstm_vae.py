import os

import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.nn import functional as F

from .vae import VAE


class LSTM_VAE(L.LightningModule):
    def __init__(self, feature: int, hidden_size: int, dim_encoded: int, num_layers: int, num_classes: int, vae_ckpt_path: str = None):
        super().__init__()

        self.save_hyperparameters()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dim_encoded = dim_encoded

        if vae_ckpt_path is not None:
            assert os.path.exists(vae_ckpt_path), "VAE checkpoint path does not exist"
        self.vae_pretrained = True if vae_ckpt_path is not None else False

        if self.vae_pretrained:
            self.vae = VAE.load_from_checkpoint(vae_ckpt_path, in_channels=feature, latent_dim=dim_encoded)
            self.vae.freeze()
        else:
            self.vae = VAE(in_channels=feature, latent_dim=dim_encoded)

        self.lstm = nn.LSTM(dim_encoded, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        b, seq_len, f_dim = x.shape

        inputs_reshaped = x.view(-1, f_dim)
        vae_pred, mu, sigma = self.vae(inputs_reshaped)

        inputs_encoded = mu.view(b, seq_len, -1)

        h0 = torch.zeros(self.num_layers, inputs_encoded.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, inputs_encoded.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(inputs_encoded, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out, vae_pred, mu, sigma

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        seqs, time_diff, labels = batch
        outputs, vae_pred, mu, sigma = self(seqs)
        labels = torch.argmax(labels, dim=1)
        loss = self.loss_fn(vae_pred, seqs, mu, sigma, outputs, labels)
        self.log('train_loss', loss.item() / batch.size(0), sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        seqs, time_diff, labels = batch
        outputs, vae_pred, mu, sigma = self(seqs)
        labels = torch.argmax(labels, dim=1)
        loss = self.loss_fn(vae_pred, seqs, mu, sigma, outputs, labels)
        self.log('val_loss', loss.item() / batch.size(0), sync_dist=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def loss_fn(self, recon_x, x, mu, log_var, outputs, labels, alpha=0.5):
        loss = F.cross_entropy(outputs, labels)
        if self.vae_pretrained:
            bce = F.binary_cross_entropy(recon_x, x.view(-1, x.shape[2]), reduction='sum')
            kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            vae_loss = (bce + kld) / x.shape[1]
            loss = alpha * vae_loss + (1 - alpha) * loss
        return loss
