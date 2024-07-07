from typing import Callable, Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
)
from torchmetrics.collections import MetricCollection
from torch.optim import Optimizer


OptimizerCallable = Callable[[Iterable], Optimizer]


class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads):
        super().__init__()
        self.save_hyperparameters()

        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.query = nn.Linear(input_size, hidden_size * num_heads)
        self.key = nn.Linear(input_size, hidden_size * num_heads)
        self.value = nn.Linear(input_size, hidden_size * num_heads)

        self.fc_out = nn.Linear(hidden_size * num_heads, hidden_size * 8)
        metrics = MetricCollection(
            [BinaryAccuracy(), BinaryPrecision(), BinaryRecall(), BinaryF1Score()]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="valid_")

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 分割头
        Q = Q.view(batch_size, -1, self.num_heads, self.hidden_size).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.hidden_size).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.hidden_size).transpose(1, 2)

        # 计算注意力得分
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (
            self.hidden_size**0.5
        )
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 加权求和
        attention_output = torch.matmul(attention_weights, V)
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.hidden_size * self.num_heads)
        )

        output = self.fc_out(attention_output)

        return output


class BiLSTMWithImprovedAttention(L.LightningModule):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_classes,
        attention_size,
        num_layer,
        dropout_rate,
        num_heads,
        optimizer: OptimizerCallable = torch.optim.AdamW,
    ):
        super(BiLSTMWithImprovedAttention, self).__init__()

        # 双向LSTM层
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layer,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate,
        )

        # 多头注意力机制
        self.attention = MultiHeadAttention(hidden_size * 2, attention_size, num_heads)

        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)

        # 全连接层
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # LSTM层
        lstm_out, _ = self.lstm(x)

        # Dropout
        lstm_out = self.dropout(lstm_out)

        # 多头注意力机制
        attention_output = self.attention(lstm_out)
        attention_output = attention_output[:, -1, :]
        output = self.fc(attention_output)
        return output

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        seqs, _, labels, _ = batch
        labels = torch.argmax(labels, dim=1)

        outputs = self(seqs)
        loss = self.loss_fn(outputs, labels)

        self.log("train_loss", loss.item(), sync_dist=True)

        preds = torch.argmax(outputs, dim=1)
        metrics = self.train_metrics(preds, labels)
        self.log_dict(
            metrics, sync_dist=True, prog_bar=False, on_step=False, on_epoch=True
        )

        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        seqs, _, labels, _ = batch
        labels = torch.argmax(labels, dim=1)

        outputs = self(seqs)
        loss = self.loss_fn(outputs, labels)

        self.log("val_loss", loss.item(), sync_dist=True)

        preds = torch.argmax(outputs, dim=1)

        self.valid_metrics.update(preds, labels)

        return loss

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

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        seqs, _, _, msisdns = batch
        outputs = torch.argmax(
            torch.softmax(self(seqs), dim=1),
            dim=1,
        )
        return outputs, msisdns

    def loss_fn(self, outputs, labels):
        return F.cross_entropy(outputs, labels)
