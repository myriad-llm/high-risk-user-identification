import os

import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.nn import functional as F
import torchmetrics
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
from torchmetrics.collections import MetricCollection

from .vae import VAE


class LSTM_VAE(L.LightningModule):
    def __init__(
        self,
        feature: int,
        hidden_size: int,
        dim_encoded: int,
        num_layers: int,
        num_classes: int,
        vae_ckpt_path: str = None,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dim_encoded = dim_encoded

        if vae_ckpt_path is not None:
            assert os.path.exists(vae_ckpt_path), "VAE checkpoint path does not exist"
        self.vae_pretrained = True if vae_ckpt_path is not None else False

        if self.vae_pretrained:
            self.vae = VAE.load_from_checkpoint(
                vae_ckpt_path, in_channels=feature, latent_dim=dim_encoded
            )
            self.vae.freeze()
        else:
            self.vae = VAE(in_channels=feature, latent_dim=dim_encoded)

        self.lstm = nn.LSTM(dim_encoded, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        metrics = MetricCollection(
            [
                BinaryAccuracy(),
                BinaryPrecision(),
                BinaryRecall(),
                BinaryF1Score()
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="valid_")

    def forward(self, x):
        b, seq_len, f_dim = x.shape

        inputs_reshaped = x.view(-1, f_dim)
        vae_pred, mu, sigma = self.vae(inputs_reshaped)

        inputs_encoded = mu.view(b, seq_len, -1)

        h0 = torch.zeros(self.num_layers, inputs_encoded.size(0), self.hidden_size).to(
            x.device
        )
        c0 = torch.zeros(self.num_layers, inputs_encoded.size(0), self.hidden_size).to(
            x.device
        )
        out, _ = self.lstm(inputs_encoded, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out, vae_pred, mu, sigma

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        seqs, _, labels, _ = batch
        outputs, vae_pred, mu, sigma = self(seqs)
        labels = torch.argmax(labels, dim=1)
        loss = self.loss_fn(vae_pred, seqs, mu, sigma, outputs, labels)
        self.log("train_loss", loss.item(), sync_dist=True, on_step=False, on_epoch=True, batch_size=seqs.size(0))
        preds = torch.argmax(outputs, dim=1)
        metrics = self.train_metrics(preds, labels)
        self.log_dict(metrics, 
                      sync_dist=True, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        seqs, _, labels, _ = batch
        outputs, vae_pred, mu, sigma = self(seqs)
        labels = torch.argmax(labels, dim=1)
        loss = self.loss_fn(vae_pred, seqs, mu, sigma, outputs, labels)
        self.log("val_loss", loss.item(), sync_dist=True, batch_size=seqs.size(0))
        preds = torch.argmax(outputs, dim=1)
        self.valid_metrics.update(preds, labels)
        return loss

    def on_validation_epoch_end(self):
        metrics = self.valid_metrics.compute()
        self.log_dict(metrics,
                        sync_dist=True, prog_bar=False, on_step=False, on_epoch=True)
        self.valid_metrics.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def loss_fn(self, recon_x, x, mu, log_var, outputs, labels, alpha=0.5):
        loss = F.cross_entropy(outputs, labels)
        if self.vae_pretrained:
            bce = F.binary_cross_entropy(
                recon_x, x.view(-1, x.shape[2]), reduction="sum"
            )
            kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            vae_loss = (bce + kld) / x.shape[1]
            loss = alpha * vae_loss + (1 - alpha) * loss
        return loss
