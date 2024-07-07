from typing import Callable, Iterable

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.optim import AdamW, Optimizer

OptimizerCallable = Callable[[Iterable], Optimizer]


class Encoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int):
        super().__init__()
        self.l = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )

    def forward(self, x):
        return self.l(x)


class Decoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int):
        super().__init__()
        self.l = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, in_channels),
        )

    def forward(self, x):
        return self.l(x)


class AE(L.LightningModule):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        optimizer: OptimizerCallable = AdamW,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.optimizer = optimizer

        self.encode = Encoder(in_channels, latent_dim)
        self.decode = Decoder(in_channels, latent_dim)

    def forward(self, x):
        latent_feature = self.encode(x)
        return self.decode(latent_feature)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        pred = self(batch)
        loss = F.cross_entropy(pred, batch)

        self.log("train_loss", loss.item(), sync_dist=True, batch_size=batch.size(0))

        return loss

    def configure_optimizers(self) -> Optimizer:
        optim = self.optimizer(
            self.parameters(),
        )

        return optim
