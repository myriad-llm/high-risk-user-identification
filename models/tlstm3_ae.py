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

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class TLSTM3_Unit(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        # 输入x->W(Wx_)   上一个时间步的h->U(Wh_)
        self.W_all = nn.Linear(input_dim, hidden_dim * 3)
        self.U_all = nn.Linear(hidden_dim, hidden_dim * 3)
        self.W_t1 = nn.Linear(1, hidden_dim)
        self.W_t2 = nn.Linear(1, hidden_dim)
        self.W_to = nn.Linear(hidden_dim, hidden_dim)

        self.b_t1 = nn.Parameter(torch.zeros(hidden_dim))
        self.b_t2 = nn.Parameter(torch.zeros(hidden_dim))
        self.b_i = nn.Parameter(torch.zeros(hidden_dim))
        self.b_o = nn.Parameter(torch.zeros(hidden_dim))
        self.b_c = nn.Parameter(torch.zeros(hidden_dim))

        # 确保 W_t1 <= 0 by initializing it as 负值
        self.W_t1.weight.data.uniform_(-1, 0)

    # h ：上一个时间步的隐藏状态
    # c ：上一个时间步的细胞状态
    # x ：当前时间步的输入特征向量
    # t ：当前时间步与上一个时间步之间的时间间隔
    def forward(self, h, c, x, t):
        # t = t.view(-1, 1)  # 将 t 视图为形状为 [32, 1]
        # print("Shapes before T1 calculation:")
        # print("t shape:", t.shape)
        # print("self.W_t1(t) shape:", self.W_t1(t).shape)
        # print("self.W_t1.weight.data * t shape:", (self.W_t1.weight.data * t).shape)
        # print("self.b_t1 shape:", self.b_t1.shape)
        # Calculate time gates
        T1 = torch.sigmoid(self.W_t1(t) + self.W_t1.weight.data * t + self.b_t1)
        T2 = torch.sigmoid(self.W_t2(t) + self.W_t2.weight.data * t + self.b_t2)


        # Calculate gates
        # Wx + Uh
        outs = self.W_all(x) + self.U_all(h)
        i, o, c_tmp = torch.chunk(outs, 3, 1)
        i, o, c_tmp = (
            torch.sigmoid(i + self.b_i),
            torch.sigmoid(o + t * self.W_to.weight.data + self.b_o),
            torch.tanh(c_tmp + self.b_c),
        )

        # Update cell state
        cell = (1 - i * T1) * c + i * T1 * c_tmp
        c_tilde = (1 - i) * c + i * T2 * c_tmp

        # Calculate new hidden state
        # o = torch.sigmoid(o + t * self.W_t1.weight.data + h @ self.U_all.weight[:, self.hidden_dim:self.hidden_dim * 2].t())
        new_h = o * torch.tanh(cell)

        return new_h, c_tilde


class TimeLSTM3_AE(L.LightningModule):
    def __init__(
            self,
            input_size,
            hidden_size: int,
            num_classes: int,
            dropout_rate: float,
            num_heads: int,
            optimizer: OptimizerCallable,
            ae_encoding_dim: int,
            bidirectional: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.bidirectional = bidirectional
        self.num_classes = num_classes
        self.num_heads = num_heads

        # TLSTM unit
        self.tlstm_unit = TLSTM3_Unit(self.input_size, self.hidden_size)

        # Autoencoder
        self.autoencoder = Autoencoder(self.input_size, ae_encoding_dim)

        # Global pooling layer
        self.global_pooling = nn.AdaptiveAvgPool1d(1)

        # Attention mechanism
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size * 2, num_classes)

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
        b, seq, embed = inputs.shape
        h = torch.zeros(b, self.hidden_size, requires_grad=False).to(inputs.device)
        c = torch.zeros(b, self.hidden_size, requires_grad=False).to(inputs.device)
        outputs = []
        # 将序列编码为batch级别的特征
        encoded_seqs, decoded_seqs = self.autoencoder(inputs.view(-1, embed))
        # print("encoded_seqs shape: ", encoded_seqs.shape)
        encoded_seqs = encoded_seqs.view(b, seq, -1)
        # print("after view encoded_seqs shape: ", encoded_seqs.shape)
        for s in range(seq):
            input = encoded_seqs[:, s, :]
            time = timestamps[:, s].unsqueeze(1)
            h, c = self.tlstm_unit(h, c, input, time)
            outputs.append(h)

        if reverse:
            outputs.reverse()
        outputs = torch.stack(outputs, dim=1)

        padding_mask = self.gen_padding_mask(seq, seq_lens)

        global_features = self.global_pooling(outputs.permute(0, 2, 1)).squeeze(-1)

        attention_output, _ = self.multi_head_attention(
            query=outputs, key=outputs, value=outputs, key_padding_mask=padding_mask
        )

        batch_indices = torch.arange(b).to(outputs.device)
        last_output = attention_output[batch_indices, seq_lens - 1, :]

        last_output = torch.cat([last_output, global_features], dim=1)

        output = self.fc(last_output)

        return output, decoded_seqs.view(b, seq, embed)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        seqs, time_diffs, labels, _, seq_lens = batch
        labels = torch.argmax(labels, dim=1)

        outputs, decoded = self(seqs, time_diffs, seq_lens)
        loss = self.loss_fn(outputs, labels) + self.ae_loss_fn(seqs, decoded)

        self.log("train_loss", loss.item(), sync_dist=True, batch_size=32)

        preds = torch.argmax(outputs, dim=1)
        metrics = self.train_metrics(preds, labels)
        self.log_dict(
            metrics, sync_dist=True, prog_bar=False, on_step=False, on_epoch=True
        )

        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        seqs, time_diffs, labels, _, seq_lens = batch
        labels = torch.argmax(labels, dim=1)

        outputs, decoded = self(seqs, time_diffs, seq_lens)
        loss = self.loss_fn(outputs, labels) + self.ae_loss_fn(seqs, decoded)

        self.log("val_loss", loss.item(), sync_dist=True, batch_size=32)

        preds = torch.argmax(outputs, dim=1)

        self.valid_metrics.update(preds, labels)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        seqs, time_diffs, _, msisdns, seq_lens = batch
        outputs, _ = torch.argmax(
            torch.softmax(self(seqs, time_diffs, seq_lens)[0], dim=1),
            dim=1,
        )
        return outputs, msisdns

    def on_validation_epoch_end(self):
        metrics = self.valid_metrics.compute()
        self.log_dict(
            metrics, sync_dist=True, prog_bar=False, on_step=False, on_epoch=True
        )
        self.valid_metrics.reset()

    def configure_optimizers(self):
        optim = self.optimizer(
            self.parameters(),
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optim,
            mode="max",
            patience=5,
            factor=0.5,
        )
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "valid_BinaryF1Score",
            }
        }

    def loss_fn(self, outputs, labels):
        return F.cross_entropy(outputs, labels)

    def ae_loss_fn(self, inputs, decoded):
        return F.mse_loss(decoded, inputs)

    def gen_padding_mask(self, max_seq_len, seq_lens):
        seq_range = torch.arange(max_seq_len).unsqueeze(0).to(self.device)  # [1, seq]
        seq_lens_expanded = seq_lens.unsqueeze(1).to(self.device)  # [b, 1]
        padding_mask = seq_range >= seq_lens_expanded  # [b, seq]
        return padding_mask

