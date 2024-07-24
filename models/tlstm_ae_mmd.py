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
from .common import CallRecordsEmbeddings

OptimizerCallable = Callable[[Iterable], Optimizer]

class TLSTM_Unit(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.W_all = nn.Linear(input_dim, hidden_dim * 4)
        self.U_all = nn.Linear(hidden_dim, hidden_dim * 4)
        self.W_d = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h, c, x, t):
        c_st = torch.sigmoid(self.W_d(c))
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

class TLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.tlstm_unit = TLSTM_Unit(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, inputs, time_diffs, seq_lens, h_init = None, c_init = None, all_time_step_outputs = True):
        b, seq, embed = inputs.shape
        h, c = h_init, c_init
        if h is None:
            h = torch.zeros(b, self.hidden_dim, requires_grad=False).to(inputs.device)
        if c is None:
            c = torch.zeros(b, self.hidden_dim, requires_grad=False).to(inputs.device)
        assert h.shape[1] == self.hidden_dim, f"{h.shape[1]} != {self.hidden_dim}"
        assert c.shape[1] == self.hidden_dim, f"{c.shape[1]} != {self.hidden_dim}"
        hs, cs = [], []
        for s in range(seq):
            input = inputs[:, s, :]
            time_diff = time_diffs[:, s].unsqueeze(1)
            h, c = self.tlstm_unit(h, c, input, time_diff)
            hs.append(h)
            cs.append(c)
        hs = torch.stack(hs, dim=1)
        cs = torch.stack(cs, dim=1)

        # TODO: 如果 反向，无法只取有效的最后一个输出
        # batch_indices = torch.arange(b).to(inputs.device)
        # output = self.output_layer(hs[batch_indices, seq_lens - 1, :])

        if all_time_step_outputs:
            # hs: batch * seq * hidden_dim, can't directly use output_layer
            hs_reshaped = hs.reshape(-1, self.hidden_dim)
            outputs = self.output_layer(hs_reshaped)
            outputs = outputs.reshape(b, seq, -1)
        else:
            output = self.output_layer(hs[:, -1, :]) 
        
        return outputs if all_time_step_outputs else output, (hs, cs)

class TLSTM_Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim1: int,
        encoded_dim: int,
        output_dim1: int,
        output_dim2: int,
    ):
        super().__init__()
        self.tlstm_encoder1 = TLSTM(in_channels, hidden_dim1, output_dim1)
        self.tlstm_encoder2 = TLSTM(output_dim1, encoded_dim, output_dim2)
    
    def forward(self, x, time_diffs, seq_lens, h_init = None, c_init = None):
        b, seq, f_dim = x.shape

        outputs1, (hs1, cs1) = self.tlstm_encoder1(x, time_diffs, seq_lens, h_init = None, c_init= None, all_time_step_outputs = True) # b * seq * hidden_dim

        output2, (hs2, cs2) = self.tlstm_encoder2(outputs1, time_diffs, seq_lens, h_init = None, c_init = None, all_time_step_outputs = True) # b * seq * hidden_dim

        # batch_indices = torch.arange(b).to(x.device)
        # representation = hs2[batch_indices, seq_lens - 1, :] # b * hidden_dim
        # decoder_cs = cs2[batch_indices, seq_lens - 1, :] # b * hidden_dim
        representation = hs2[:, -1, :] # b * encoded_dim
        decoder_cs = cs2[:, -1, :] # b * encoded_dim

        return representation, decoder_cs
    
class TLSTM_Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim1: int,
        hidden_dim2: int,
        output_dim1: int,
        decoded_dim: int,
    ):
        super().__init__()
        self.tlstm_decoder1 = TLSTM(in_channels, hidden_dim1, output_dim1)
        self.tlstm_decoder2 = TLSTM(output_dim1, hidden_dim2, decoded_dim)
    
    def forward(self, x, time_diffs, seq_lens, encoded_h, encoded_c):
        b, seq, f_dim = x.shape
        time_diffs = torch.zeros_like(time_diffs)

        inputs1 = torch.flip(x, [1])

        outputs1, (hs1, cs1) = self.tlstm_decoder1(inputs1, time_diffs, seq_lens, h_init = encoded_h, c_init = encoded_c, all_time_step_outputs = True)

        outputs2, (hs2, cs2) = self.tlstm_decoder2(outputs1, time_diffs, seq_lens, h_init = None, c_init = None, all_time_step_outputs = True)

        return outputs2

class MMD_loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss


class TimeLSTMAutoEncoder_MMD(L.LightningModule):
    def __init__(
        self,
        input_size,
        embedding_items_path,
        hidden_dim1_e: int,
        encoded_dim: int,
        output_dim1_e: int,
        output_dim2_e: int,
        hidden_dim2_d: int,
        output_dim1_d: int,
        decoded_dim: int,
        mask_rate: float = 0.1,
        optimizer: OptimizerCallable = torch.optim.Adam,
        mmd_weight: float = 0.05  # Weight for the MMD loss
    ):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer

        self.mask_rate = mask_rate
        self.embeddings = CallRecordsEmbeddings(input_size=input_size, embedding_items_path=embedding_items_path)
        self.encoder = TLSTM_Encoder(
            in_channels=input_size,
            hidden_dim1=hidden_dim1_e,
            encoded_dim=encoded_dim,
            output_dim1=output_dim1_e,
            output_dim2=output_dim2_e,
        )
        self.decoder = TLSTM_Decoder(
            in_channels=input_size,
            hidden_dim1=encoded_dim,
            hidden_dim2=hidden_dim2_d,
            output_dim1=output_dim1_d,
            decoded_dim=decoded_dim,
        )

        self.mmd_loss = MMD_loss()
        self.mmd_weight = mmd_weight

    def forward(self, x, time_diffs, seq_lens, mask=False):
        original_x = self.embeddings(x)
        if mask:
            x = self.mask_input(original_x, seq_lens, self.mask_rate)
        else:
            x = original_x
        representation, decoder_cs = self.encoder(x, time_diffs, seq_lens)
        outputs = self.decoder(x, time_diffs, seq_lens, representation, decoder_cs)
        return representation, outputs, original_x

    def training_step(self, batch, batch_idx):
        x, time_diffs, labels, _, seq_lens = batch  # 目标标签现在是 `labels`
        representation, outputs, x = self(x, time_diffs, seq_lens, mask=True)

        # Generate Gaussian samples for MMD calculation
        true_samples = torch.randn(representation.size()).to(self.device)
        
        # Compute the losses
        recon_loss = F.mse_loss(outputs, x)
        mmd_loss = self.mmd_loss(true_samples, representation)
        
        # Total loss
        total_loss = recon_loss + self.mmd_weight * mmd_loss
        self.log("train_loss", total_loss, batch_size=x.size(0), prog_bar=True)
        self.log("train_recon_loss", recon_loss, batch_size=x.size(0), prog_bar=True)
        self.log("train_mmd_loss", mmd_loss, batch_size=x.size(0))
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, time_diffs, labels, _, seq_lens = batch
        representation, outputs, x = self(x, time_diffs, seq_lens, mask=False)

        # Generate Gaussian samples for MMD calculation
        true_samples = torch.randn(representation.size()).to(self.device)
        
        # Compute the losses
        recon_loss = F.mse_loss(outputs, x)
        mmd_loss = self.mmd_loss(true_samples, representation)
        
        # Total loss
        total_loss = recon_loss + self.mmd_weight * mmd_loss
        self.log("val_loss", total_loss, batch_size=x.size(0), prog_bar=True)
        self.log("val_recon_loss", recon_loss, batch_size=x.size(0), prog_bar=True)
        self.log("val_mmd_loss", mmd_loss, batch_size=x.size(0))
        
        return total_loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, time_diffs, _, msisdns, seq_lens = batch
        representation, outputs, x = self(x, time_diffs, seq_lens, mask=False)
        return representation, msisdns

    def configure_optimizers(self):
        return self.optimizer(self.parameters())
   
    def mask_input(self, x, seq_lens, mask_rate):
        b, seq, f_dim = x.shape
        mask = torch.zeros(b, seq, dtype=torch.bool).to(x.device)
       
        mask_lengths = (seq_lens * mask_rate).long().to(x.device)
        for i in range(b):
            mask_len = mask_lengths[i]
            seq_len = seq_lens[i]
            if mask_len < 1:
                continue
            # mask_rate * seq_len will be masked, and not continuous
            mask[i, :seq_len] = torch.randperm(seq_len).to(x.device) < mask_len
        mask = mask.unsqueeze(2).expand(b, seq, f_dim)
        x = x * ~mask
        return x