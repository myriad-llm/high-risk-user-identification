import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
from torchmetrics.collections import MetricCollection
from models.common import CallRecordsEmbeddings
from torch.optim import AdamW
from typing import Callable, Iterable
from torch.optim import Optimizer

OptimizerCallable = Callable[[Iterable], Optimizer]

class LSTM(L.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, embedding_items_path, optimizer: OptimizerCallable=AdamW):
        super().__init__()
        self.save_hyperparameters()

        self.optimizer = optimizer
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = CallRecordsEmbeddings(embedding_items_path, input_size=input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
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
        self.callbacks = None

    def forward(self, x, seq_lens):
        # inputs: [b, seq, embed]
        # timestamps: [b, seq]
        # h: [b, hid]
        # c: [b, hid]
        b, seq, embed = x.shape
        
       
        x = self.embedding(x) 

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))

        # batch_indices = torch.arange(b).to(out.device)
        # last_output = out[batch_indices, seq_lens - 1, :]

        out = self.fc(out[:, -1, :])
        return out

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        seqs, _, labels, _, seq_lens = batch
        labels = torch.argmax(labels, dim=1)

        outputs = self(seqs, seq_lens)
        loss = F.cross_entropy(outputs, labels)

        self.log("train_loss", loss.item(), sync_dist=True)

        preds = torch.argmax(outputs, dim=1)
        metrics = self.train_metrics(preds, labels)
        self.log_dict(metrics, sync_dist=True, prog_bar=False, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        seqs, _, labels, _, seq_lens = batch
        labels = torch.argmax(labels, dim=1)

        outputs = self(seqs, seq_lens)
        loss = F.cross_entropy(outputs, labels)

        self.log("val_loss", loss.item(), sync_dist=True)

        preds = torch.argmax(outputs, dim=1)

        self.valid_metrics.update(preds, labels)

        return loss
    
    def on_validation_epoch_end(self):
        metrics = self.valid_metrics.compute()
        self.log_dict(metrics, sync_dist=True, prog_bar=False, on_step=False, on_epoch=True)
        self.valid_metrics.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optim = self.optimizer(
            self.parameters(),
        )
        return optim

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        seqs, _, _, msisdns, seq_lens = batch
        outputs = torch.argmax(
            torch.softmax(self(seqs, seq_lens), dim=1),
            dim=1,
        )
        return outputs, msisdns

    def configure_callbacks(self):
        if self.callbacks:
            return self.callbacks
        return []
    
    def set_callbacks(self, callbacks) -> None:
        self.callbacks = callbacks 
