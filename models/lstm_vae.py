import lightning as L
from torchmetrics import Precision, Recall, F1Score, Accuracy
import torch
from torch.nn import functional as F
import torch.nn as nn
import os

from .vae import VAE

class LSTM_VAE(L.LightningModule):
    def __init__(self, feature: int, hidden_size: int, dim_encoded: int, num_layers: int, num_classes: int, vae_ckpt_path: str = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dim_encoded = dim_encoded
        if vae_ckpt_path is not None:
            assert os.path.exists(vae_ckpt_path), "VAE checkpoint path does not exist"
        self.vae_pretrained = True if vae_ckpt_path is not None else False
        if self.vae_pretrained:
            self.vae = VAE.load_from_checkpoint(vae_ckpt_path)
            self.vae.freeze()
        else:
            self.vae = VAE(in_channels=feature, latent_dim=dim_encoded)

        self.lstm = nn.LSTM(dim_encoded, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

        # self.precision = Precision(average='macro', num_classes=num_classes)
        # self.recall = Recall(average='macro', num_classes=num_classes)
        # self.f1 = F1Score(average='macro', num_classes=num_classes)
        # self.accuracy = Accuracy()

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

    def training_step(self, batch, batch_idx):
        seqs, time_diff, labels = batch
        outputs, vae_pred, mu, sigma = self(seqs)
        labels = torch.argmax(labels, dim=1)
        loss = self.loss_fn(vae_pred, seqs, mu, sigma, outputs, labels)
        self.log('train_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        seqs, time_diff, labels = batch
        outputs, vae_pred, mu, sigma = self(seqs)
        labels = torch.argmax(labels, dim=1)
        loss = self.loss_fn(vae_pred, seqs, mu, sigma, outputs, labels)
        self.log('test_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def loss_fn(self, recon_x, x, mu, log_var, outputs, labels, alpha=0.5):
        loss = 0
        ce_loss = F.cross_entropy(outputs, labels)
        if self.vae_ckpt is not None:
            bce = F.binary_cross_entropy(recon_x, x.view(-1, x.shape[2]), reduction='sum')
            kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            vae_loss = (bce + kld) / x.shape[1]
            loss = alpha * vae_loss +  (1-alpha)* ce_loss
        loss = ce_loss
        return loss