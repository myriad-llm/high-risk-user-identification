import lightning as L
import torch
import torch.nn.functional as F
import torch.nn as nn
from lightning.pytorch.utilities.types import STEP_OUTPUT


class LSTM(L.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.save_hyperparameters()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        seq, _, labels = batch
        labels = torch.argmax(labels, dim=1)

        outputs = self(seq)
        loss = F.cross_entropy(outputs, labels)

        self.log("train_loss", loss.item() / batch.size(0), sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        seq, _, labels = batch
        labels = torch.argmax(labels, dim=1)

        outputs = self(seq)
        loss = F.cross_entropy(outputs, labels)

        self.log("val_loss", loss.item() / batch.size(0), sync_dist=True)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        seq, _, _ = batch
        return self(seq)
