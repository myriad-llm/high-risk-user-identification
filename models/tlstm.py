import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
from torchmetrics.collections import MetricCollection

class TimeLSTM(L.LightningModule):
    def __init__(self, feature, hidden_size, num_classes, bidirectional=False):
        # assumes that batch_first is always true
        super().__init__()
        self.save_hyperparameters()

        self.hidden_size = hidden_size
        self.input_size = feature
        self.W_all = nn.Linear(hidden_size, hidden_size * 4)
        self.U_all = nn.Linear(feature, hidden_size * 4)
        self.W_d = nn.Linear(hidden_size, hidden_size)
        self.bidirectional = bidirectional
        self.num_classes = num_classes
        self.fc = nn.Linear(hidden_size, num_classes)

        metrics = MetricCollection(
            [
                BinaryAccuracy(),
                BinaryPrecision(),
                BinaryRecall(),
                BinaryF1Score(),
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="valid_")
            

    def forward(self, inputs, timestamps, reverse=False):
        # inputs: [b, seq, embed]
        # h: [b, hid]
        # c: [b, hid]
        b, seq, embed = inputs.shape
        h = torch.zeros(b, self.hidden_size, requires_grad=False).to(inputs.device)
        c = torch.zeros(b, self.hidden_size, requires_grad=False).to(inputs.device)
        outputs = []
        for s in range(seq):
            c_s1 = torch.tanh(self.W_d(c))
            c_s2 = c_s1 * timestamps[:, s:s + 1].expand_as(c_s1)
            c_l = c - c_s1
            c_adj = c_l + c_s2
            outs = self.W_all(h) + self.U_all(inputs[:, s])
            f, i, o, c_tmp = torch.chunk(outs, 4, 1)
            f = torch.sigmoid(f)
            i = torch.sigmoid(i)
            o = torch.sigmoid(o)
            c_tmp = torch.sigmoid(c_tmp)
            c = f * c_adj + i * c_tmp
            h = o * torch.tanh(c)
            outputs.append(h)
        if reverse:
            outputs.reverse()
        outputs = torch.stack(outputs, 1)
        output = self.fc(outputs[:, -1, :])
        return output
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        seqs, time_diffs, labels, _ = batch
        labels = torch.argmax(labels, dim=1)

        outputs = self(seqs, time_diffs)
        loss = self.loss_fn(outputs, labels)

        self.log("train_loss", loss.item(), sync_dist=True)

        preds = torch.argmax(outputs, dim=1)
        metrics = self.train_metrics(preds, labels)
        self.log_dict(metrics, sync_dist=True, prog_bar=False, on_step=False, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        seqs, time_diffs, labels, _ = batch
        labels = torch.argmax(labels, dim=1)

        outputs = self(seqs, time_diffs)
        loss = self.loss_fn(outputs, labels)

        self.log("val_loss", loss.item(), sync_dist=True)

        preds = torch.argmax(outputs, dim=1)

        self.valid_metrics.update(preds, labels)

        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        seqs, time_diffs, _, msisdns = batch
        outputs = torch.argmax(
            torch.softmax(self(seqs, time_diffs), dim=1),
            dim=1,
        )
        return outputs, msisdns
    
    def on_validation_epoch_end(self):
        metrics = self.valid_metrics.compute()
        self.log_dict(metrics, sync_dist=True, prog_bar=False, on_step=False, on_epoch=True)
        self.valid_metrics.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def loss_fn(self, outputs, labels):
        return F.cross_entropy(outputs, labels)
