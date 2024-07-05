import lightning as L
import torch
from torch import nn


class LitLSTM(L.LightningModule):
    def __init__(self, lstm):
        super(LitLSTM, self).__init__()
        self.lstm = lstm
        self.loss_fn = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        seq, _, labels = batch
        labels = torch.argmax(labels, dim=1)

        outputs = self.lstm(seq)
        loss = self.loss_fn(outputs, labels)

        self.log_dict({"train_loss": loss.item()})

        return loss

    def validation_step(self, batch, batch_idx):
        seq, _, labels = batch
        labels = torch.argmax(labels, dim=1)

        outputs = self.lstm(seq)
        loss = self.loss_fn(outputs, labels)

        self.log_dict({"val_loss": loss.item()})

        return loss

    def test_step(self, batch, batch_idx):
        seq, _, labels = batch
        labels = torch.argmax(labels, dim=1)

        outputs = self.lstm(seq)
        loss = self.loss_fn(outputs, labels)

        self.log_dict({"test_loss": loss.item()})

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        seq, _, _ = batch
        return self.lstm(seq)
