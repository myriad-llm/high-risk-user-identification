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

class BiLSTMWithImprovedAttention(L.LightningModule):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_classes,
        attention_size,
        num_layers,
        dropout_rate,
        num_heads,
        optimizer: OptimizerCallable = torch.optim.AdamW,
    ):
        super(BiLSTMWithImprovedAttention, self).__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer


        # 双向LSTM层
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_rate,
        )

        # 多头注意力机制
        # self.attention = MultiHeadAttention(hidden_size * 2, attention_size, num_heads)
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
            

        # Dropout层
        # self.dropout = nn.Dropout(dropout_rate)

        # 全连接层
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
        metrics = MetricCollection(
            [BinaryAccuracy(), BinaryPrecision(), BinaryRecall(), BinaryF1Score()]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="valid_")

    def forward(self, x, seq_lens):
        b, max_seq_len, dim = x.size()
        # LSTM层
        lstm_out, _ = self.lstm(x) # (batch_size, seq_len, hidden_size * 2)

        padding_mask_forward = self.gen_padding_mask(max_seq_len=max_seq_len, seq_lens=seq_lens)

        # Dropout
        # lstm_out = self.dropout(lstm_out)

        # 多头注意力机制
        attention_output, _ = self.multi_head_attention(
            query=lstm_out,
            key=lstm_out,
            value=lstm_out,
            key_padding_mask=padding_mask_forward,
        )
        batch_indices = torch.arange(b).to(lstm_out.device)
        last_attention_output = attention_output[batch_indices, seq_lens - 1, :]
        output = self.fc(last_attention_output)
        return output

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        seqs, _, labels, _, seq_lens  = batch
        labels = torch.argmax(labels, dim=1)

        outputs = self(seqs, seq_lens)
        loss = self.loss_fn(outputs, labels)

        self.log("train_loss", loss.item(), sync_dist=True)

        preds = torch.argmax(outputs, dim=1)
        metrics = self.train_metrics(preds, labels)
        self.log_dict(
            metrics, sync_dist=True, prog_bar=False, on_step=False, on_epoch=True
        )

        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        seqs, _, labels, _, seq_lens= batch
        labels = torch.argmax(labels, dim=1)

        outputs = self(seqs, seq_lens)
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
        seqs, _, _, msisdns, seq_lens = batch
        outputs = torch.argmax(
            torch.softmax(self(seqs, seq_lens), dim=1),
            dim=1,
        )
        return outputs, msisdns

    def loss_fn(self, outputs, labels):
        return F.cross_entropy(outputs, labels)

    def gen_padding_mask(self, max_seq_len, seq_lens):
        seq_range = torch.arange(max_seq_len).unsqueeze(0).to(self.device)  # [1, seq]
        seq_lens_expanded = seq_lens.unsqueeze(1).to(self.device)  # [b, 1]
        padding_mask = seq_range >= seq_lens_expanded  # [b, seq]
        return padding_mask