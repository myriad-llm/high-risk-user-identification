import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
)
from torchmetrics.collections import MetricCollection
from torch.optim import Optimizer
from typing import Callable, Iterable

OptimizerCallable = Callable[[Iterable], Optimizer]


class TLSTM_Unit(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.W_all = nn.Linear(input_dim, hidden_dim * 4)
        self.U_all = nn.Linear(hidden_dim, hidden_dim * 4)
        self.W_d = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h, c, x, t):
        c_st = torch.tanh(self.W_d(c))
        c_st_dis = c_st * t
        c = c - c_st + c_st_dis

        outs = self.W_all(x) + self.U_all(h)
        f, i, o, c_tmp = torch.chunk(outs, 4, 1)
        f, i, o, c_tmp = (
            torch.sigmoid(f),
            torch.sigmoid(i),
            torch.sigmoid(o),
            torch.tanh(c_tmp),
        )
        c_t = f * c + i * c_tmp

        new_h = o * torch.tanh(c_t)
        return new_h, c_t


class TimeLSTM(L.LightningModule):
    def __init__(
        self,
        input_size,
        hidden_size: int,
        num_classes: int,
        dropout_rate: float,
        num_heads: int,
        optimizer: OptimizerCallable,
        bidirectional: bool=False,
    ):
        # assumes that batch_first is always true
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.bidirectional = bidirectional
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.tlstm_unit = TLSTM_Unit(self.input_size, self.hidden_size)
        # 注意力机制
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        # self.dropout = nn.Dropout(dropout_rate)
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

    def forward(self, inputs, timestamps, seq_lens, reverse=False):
        # inputs: [b, seq, embed]
        # timestamps: [b, seq]
        # h: [b, hid]
        # c: [b, hid]
        b, seq, embed = inputs.shape
        h = torch.zeros(b, self.hidden_size, requires_grad=False).to(inputs.device)
        c = torch.zeros(b, self.hidden_size, requires_grad=False).to(inputs.device)
        outputs = []
        for s in range(seq):
            input = inputs[:, s, :]
            time = timestamps[:, s].unsqueeze(1)
            h, c = self.tlstm_unit(h, c, input, time)
            outputs.append(h)
        if reverse:
            outputs.reverse()
        outputs = torch.stack(outputs, dim=1)
        # gen padding_mask
        padding_mask = self.gen_padding_mask(seq, seq_lens)

        attention_output, _ = self.multi_head_attention(
            query=outputs, key=outputs, value=outputs, key_padding_mask=padding_mask
        )
        batch_indices = torch.arange(b).to(outputs.device)
        last_output = attention_output[batch_indices, seq_lens - 1, :]
        # last_output = torch.stack([outputs[i, seq_lens[i] - 1, :] for i in range(b)])
        # last_output = self.dropout(outputs[-1])
        output = self.fc(last_output)
        return output

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        seqs, time_diffs, labels, _, seq_lens = batch
        labels = torch.argmax(labels, dim=1)

        outputs = self(seqs, time_diffs, seq_lens)
        loss = self.loss_fn(outputs, labels)

        self.log("train_loss", loss.item(), sync_dist=True)

        preds = torch.argmax(outputs, dim=1)
        metrics = self.train_metrics(preds, labels)
        self.log_dict(
            metrics, sync_dist=True, prog_bar=False, on_step=False, on_epoch=True
        )

        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        seqs, time_diffs, labels, _, seq_lens = batch
        labels = torch.argmax(labels, dim=1)

        outputs = self(seqs, time_diffs, seq_lens)
        loss = self.loss_fn(outputs, labels)

        self.log("val_loss", loss.item(), sync_dist=True)

        preds = torch.argmax(outputs, dim=1)

        self.valid_metrics.update(preds, labels)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        seqs, time_diffs, _, msisdns, seq_lens = batch
        outputs = torch.argmax(
            torch.softmax(self(seqs, time_diffs, seq_lens), dim=1),
            dim=1,
        )
        return outputs, msisdns

    def on_validation_epoch_end(self):
        metrics = self.valid_metrics.compute()
        self.log_dict(
            metrics, sync_dist=True, prog_bar=False, on_step=False, on_epoch=True
        )
        self.valid_metrics.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optim = self.optimizer(
            self.parameters(),
        )
        return optim

    def loss_fn(self, outputs, labels):
        return F.cross_entropy(outputs, labels)

    def gen_padding_mask(self, max_seq_len, seq_lens):
        seq_range = torch.arange(max_seq_len).unsqueeze(0).to(self.device)  # [1, seq]
        seq_lens_expanded = seq_lens.unsqueeze(1).to(self.device)  # [b, 1]
        padding_mask = seq_range >= seq_lens_expanded  # [b, seq]
        return padding_mask