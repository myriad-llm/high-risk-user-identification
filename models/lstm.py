import lightning as L
import torch
import torch.nn.functional as F
from torch import nn


class LSTM(L.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
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

    def training_step(self, batch, batch_idx):
        seq, _, labels = batch
        labels = torch.argmax(labels, dim=1)

        outputs = self(seq)
        loss = F.cross_entropy(outputs, labels)

        self.log_dict({"train_loss": loss.item()})

        return loss

    def validation_step(self, batch, batch_idx):
        seq, _, labels = batch
        labels = torch.argmax(labels, dim=1)

        outputs = self(seq)
        loss = F.cross_entropy(outputs, labels)

        self.log_dict({"val_loss": loss.item()})

        return loss

    def test_step(self, batch, batch_idx):
        seq, _, labels = batch
        labels = torch.argmax(labels, dim=1)

        outputs = self(seq)
        loss = F.cross_entropy(outputs, labels)

        self.log_dict({"test_loss": loss.item()})

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        seq, _, _ = batch
        return self(seq)
