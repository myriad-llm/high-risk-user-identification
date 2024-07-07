from itertools import repeat
from math import sqrt
from typing import List, Tuple

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor, exp, sigmoid, tanh
from torch.nn.functional import gelu, silu
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)
from torchmetrics.collections import MetricCollection


class CausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
    ):
        self._padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self._padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input: Tensor) -> Tensor:
        if input.dim() == 2:
            input = torch.unsqueeze(input, dim=1)

        result = super(CausalConv1d, self).forward(input)
        if self._padding != 0:
            return result[..., : -self._padding]
        return result


class BlockLinear(nn.Module):
    def __init__(
        self,
        block_dims: List[int | List[int]],
        bias: bool = False,
    ):
        super(BlockLinear, self).__init__()

        self._blocks = nn.ParameterList(
            [nn.Parameter(torch.randn(size, requires_grad=True)) for size in block_dims]
        )

        self._bias = nn.Parameter(torch.zeros(sum(block_dims))) if bias else None

    def forward(self, inp: Tensor) -> Tensor:
        # Assemble the blocks into a block-diagonal matrix
        full = torch.block_diag(*self._blocks)

        out = torch.matmul(inp, full)

        if self._bias is not None:
            out = out + self._bias

        return out


def enlarge_as(src: Tensor, other: Tensor) -> Tensor:
    """
    Add sufficient number of singleton dimensions
    to tensor a **to the right** so to match the
    shape of tensor b. NOTE that simple broadcasting
    works in the opposite direction.
    """
    return torch.reshape(
        src,
        src.shape + (1,) * (other.dim() - src.dim()),
    ).contiguous()


class sLSTM(nn.Module):
    """The scalar-Long Short Term Memory (sLSTM) module as
    originally introduced in Beck et al. (2024)] see:
    (https://arxiv.org/abs/2405.04517).

    This model is a variant of the standard LSTM model and
    offers two major improvements:
    - Exponential gating with appropriate state normalization
        to avoid overflows induced by the exponential function.
    - A new memory mixing within heads but not across heads.
    """

    def __init__(
        self,
        input_size: int,
        head_size: int,
        head_num: int,
        ker_size: int = 4,
        p_factor: float = 4 / 3,
    ) -> None:
        super().__init__()

        self.inp_dim = input_size
        self.head_dim = head_size
        self.head_num = head_num

        self.inp_norm = nn.LayerNorm(input_size)
        self.hid_norm = nn.GroupNorm(head_num, head_size * head_num)

        self.causal_conv = CausalConv1d(1, 1, kernel_size=ker_size)

        self.W_z = nn.Linear(input_size, head_num * head_size)
        self.W_i = nn.Linear(input_size, head_num * head_size)
        self.W_o = nn.Linear(input_size, head_num * head_size)
        self.W_f = nn.Linear(input_size, head_num * head_size)

        self.R_z = BlockLinear([(head_size, head_size)] * head_num)
        self.R_i = BlockLinear([(head_size, head_size)] * head_num)
        self.R_o = BlockLinear([(head_size, head_size)] * head_num)
        self.R_f = BlockLinear([(head_size, head_size)] * head_num)

        # NOTE: The factor of two in the output dimension of the up_proj
        # is due to the fact that the output needs to branch into two
        # separate outputs to account for the the gated GeLU connection.
        # See Fig. 9 in the paper.
        proj_dim = int(p_factor * head_num * head_size)
        self.up_proj = nn.Linear(head_num * head_size, 2 * proj_dim)
        self.down_proj = nn.Linear(proj_dim, input_size)

    @property
    def device(self) -> str:
        """Get the device of the model.

        Returns:
            str: The device of the model.
        """
        return next(self.parameters()).device

    def init_hidden(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Initialize the hidden state of the sLSTM model.

        Args:
            batch_size (int): The batch size of the input sequence.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: The hidden state tuple containing the cell state,
                normalizer state, hidden state, and stabilizer state.
        """

        n_0 = torch.ones(self.head_num * self.head_dim, device=self.device)
        c_0 = torch.zeros(self.head_num * self.head_dim, device=self.device)
        h_0 = torch.zeros(self.head_num * self.head_dim, device=self.device)
        m_0 = torch.zeros(self.head_num * self.head_dim, device=self.device)

        return c_0, n_0, h_0, m_0

    def forward(
        self,
        seq: Tensor,
        hid: Tuple[Tensor, Tensor, Tensor, Tensor],
        use_conv: bool = False,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]:
        """Forward pass of the sLSTM model.

        Args:
            seq (Tensor): The input sequence tensor of shape (batch_size, input_dim).
            hid (Tuple[Tensor, Tensor, Tensor, Tensor]): The hidden state tuple containing the cell state,
                normalizer state, hidden state, and stabilizer state.

        Returns:
            Tuple[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]: The output tensor with the residual
                connection and the newly updated hidden state tuple.
        """

        b, d = seq.shape

        # Separate the hidden (previous) state into the cell state,
        # the normalizer state, the hidden state, and the stabilizer state.
        c_tm1, n_tm1, h_tm1, m_tm1 = hid

        x_t: Tensor = self.inp_norm(seq)

        # Optional causal convolution block for the input
        # and forget gates. See Fig. 9 in the paper.
        if use_conv:
            # FIXME: The causal conv branch is broken.
            x_c = self.causal_conv(x_t)
            x_c = silu(x_c).squeeze()
        else:
            x_c = x_t

        # Project the input to the different heads for all
        # the gates.
        # NOTE: For input (i) and forget (f) inputs we use
        # the output of the causal conv. See Fig. 9 in the paper.
        i_t: Tensor = self.W_i(x_c) + self.R_i(h_tm1)
        f_t: Tensor = self.W_f(x_c) + self.R_f(h_tm1)
        z_t: Tensor = self.W_z(x_t) + self.R_z(h_tm1)
        o_t: Tensor = self.W_o(x_t) + self.R_o(h_tm1)

        # Compute the gated outputs for the newly computed inputs
        m_t = torch.max(f_t + m_tm1, i_t)

        i_t = exp(i_t - m_t)  # Eq. (16) in ref. paper | or Eq. (38) in supp. mat.
        f_t = exp(
            f_t - m_t + m_tm1
        )  # Eq. (17) in ref. paper | or Eq. (39) in supp. mat.

        z_t = tanh(z_t)  # Eq. (11) in ref. paper
        o_t = sigmoid(o_t)  # Eq. (14) in ref. paper

        # Update the internal states of the model
        c_t = f_t * c_tm1 + i_t * z_t  # Eq. (8) in ref. paper
        n_t = f_t * n_tm1 + i_t  # Eq. (9) in ref. paper
        h_t = o_t * (c_t / n_t)  # Eq. (10) in ref. paper

        # Compute the output of the LSTM block
        out = self.hid_norm(h_t)

        # Perform up-and-down projection of the output with
        # projection factor 4/3. See Fig. (9) in supp. mat.
        out1, out2 = self.up_proj(out).chunk(2, dim=-1)

        out = out1 + gelu(out2)
        out = self.down_proj(out)

        # Return output with the residual connection and the
        # newly updated hidden state.
        return out + seq, (c_t, n_t, h_t, m_t)


class mLSTM(nn.Module):
    """The matrix-Long Short Term Memory (mLSTM) module as
    originally introduced in Beck et al. (2024)] see:
    (https://arxiv.org/abs/2405.04517).

    This model is a variant of the standard LSTM model and
    offers superior memory due to its storing values in a
    matrix instead of a scalar. It is fully parallelizable
    and updates internal memory with the covariance rule.
    """

    def __init__(
        self,
        input_size: int,
        head_size: int,
        head_num: int,
        p_factor: int = 2,
        ker_size: int = 4,
    ) -> None:
        super().__init__()

        self.inp_dim = input_size
        self.head_num = head_num
        self.head_dim = head_size

        hid_dim = head_num * head_size

        self.inp_norm = nn.LayerNorm(input_size)
        self.hid_norm = nn.GroupNorm(head_num, hid_dim)

        # NOTE: The factor of two in the output dimension of the up_proj
        # is due to the fact that the output needs to branch into two
        self.up_l_proj = nn.Linear(input_size, int(p_factor * input_size))
        self.up_r_proj = nn.Linear(input_size, hid_dim)
        self.down_proj = nn.Linear(hid_dim, input_size)

        self.causal_conv = CausalConv1d(1, 1, kernel_size=ker_size)

        self.skip = nn.Conv1d(
            int(p_factor * input_size), hid_dim, kernel_size=1, bias=False
        )

        self.W_i = nn.Linear(int(p_factor * input_size), head_num)
        self.W_f = nn.Linear(int(p_factor * input_size), head_num)
        self.W_o = nn.Linear(int(p_factor * input_size), hid_dim)

        self.W_q = nn.Linear(int(p_factor * input_size), hid_dim)
        self.W_k = nn.Linear(int(p_factor * input_size), hid_dim)
        self.W_v = nn.Linear(int(p_factor * input_size), hid_dim)

    @property
    def device(self) -> str:
        """Get the device of the model.

        Returns:
            str: The device of the model.
        """
        return next(self.parameters()).device

    def init_hidden(self, bs: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Initialize the hidden state of the sLSTM model.

        Args:
            batch_size (int): The batch size of the input sequence.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: The hidden state tuple containing the cell state,
                normalizer state, hidden state, and stabilizer state.
        """

        c_0 = torch.zeros(
            bs, self.head_num, self.head_dim, self.head_dim, device=self.device
        )
        n_0 = torch.ones(bs, self.head_num, self.head_dim, device=self.device)
        m_0 = torch.zeros(bs, self.head_num, device=self.device)

        return c_0, n_0, m_0

    def forward(
        self,
        seq: Tensor,
        hid: Tuple[Tensor, Tensor],
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """_summary_

        Args:
            seq (Tensor): _description_
            hid (Tuple[Tensor, Tensor]): _description_

        Returns:
            Tuple[Tensor, Tuple[Tensor, Tensor]]: _description_
        """

        # Separate the hidden (previous) state into the cell state,
        # the normalizer state, the hidden state, and the stabilizer state.
        c_tm1, n_tm1, m_tm1 = hid

        x_n: Tensor = self.inp_norm(seq)  # shape: b i

        x_t = self.up_l_proj(x_n)  # shape: b (i * p_factor)
        r_t = self.up_r_proj(x_n)  # shape: b (h d)

        # Compute the causal convolutional input (to be
        # used for the query and key gates)
        x_c = self.causal_conv(x_t)  # shape: b 1 (i * p_factor)
        x_c = silu(x_c).squeeze()  # shape: b (i * p_factor)

        q_t = self.W_q(x_c).view(-1, self.head_num, self.head_dim)
        k_t = self.W_k(x_c).view(-1, self.head_num, self.head_dim) / sqrt(self.head_dim)
        v_t = self.W_v(x_t).view(-1, self.head_num, self.head_dim)

        i_t: Tensor = self.W_i(x_c)  # shape: b h
        f_t: Tensor = self.W_f(x_c)  # shape: b h
        o_t: Tensor = self.W_o(x_t)  # shape: b (h d)

        # Compute the gated outputs for the newly computed inputs
        m_t = torch.max(f_t + m_tm1, i_t)

        i_t = exp(i_t - m_t)  # Eq. (25) in ref. paper
        f_t = exp(f_t - m_t + m_tm1)  # Eq. (26) in ref. paper
        o_t = sigmoid(o_t)  # Eq. (27) in ref. paper

        # Update the internal states of the model
        c_t = enlarge_as(f_t, c_tm1) * c_tm1 + enlarge_as(i_t, c_tm1) * einsum(
            v_t, k_t, "b h d, b h p -> b h d p"
        )
        n_t = enlarge_as(f_t, n_tm1) * n_tm1 + enlarge_as(i_t, k_t) * k_t
        h_t = o_t * rearrange(
            einsum(c_t, q_t, "b h d p, b h p -> b h d")
            / einsum(n_t, q_t, "b h d, b h d -> b h").clamp(min=1).unsqueeze(-1),
            "b h d -> b (h d)",
        )  # Eq. (21) in ref. paper

        x_c = rearrange(x_c, "b i -> b i 1")
        out = self.hid_norm(h_t) + self.skip(x_c).squeeze()  # shape: b (h d)
        out = out * silu(r_t)  # shape: b (h d)
        out = self.down_proj(out)  # shape: h i

        # Return output with the residual connection and the
        # newly updated hidden state.
        return out + seq, (c_t, n_t, m_t)


class xLSTM(L.LightningModule):
    """The extended Long Short Term Memory (xLSTM) module as
    originally introduced in Beck et al. (2024)] see:
    (https://arxiv.org/abs/2405.04517).

    This model stacks sLSTM and mLSTM modules with residual
    connections and offers superior memory and performance
    compared to the standard LSTM model, achieving competitive
    or better performance and scaling than Transformer models
    or State-Space models.
    """

    def __init__(
        self,
        input_size: int,
        head_size: int,
        head_num: int,
        num_layers: int,
        signature: Tuple[int, int],
        num_classes: int = 2,
        p_factor: Tuple[float, float] = (2, 4 / 3),
        ker_size: int = 4,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        m_factor, s_factor = p_factor
        mlstm_parameter = {
            "input_size": input_size,
            "head_size": head_size,
            "head_num": head_num,
            "p_factor": m_factor,
            "ker_size": ker_size,
        }
        slstm_parameter = {
            "input_size": input_size,
            "head_size": head_size,
            "head_num": head_num,
            "p_factor": s_factor,
            "ker_size": ker_size,
        }

        m_num, s_num = signature
        which = [True] * m_num + [False] * s_num
        self.xlstm: List[mLSTM | sLSTM] = nn.ModuleList(
            [
                mLSTM(**mlstm_parameter) if v else sLSTM(**slstm_parameter)
                for w in repeat(which, num_layers)
                for v in w
            ]
        )
        self.fc = nn.Linear(input_size, num_classes)

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

    def forward(
        self,
        seq: Tensor,
        hid: List[Tuple[Tensor, ...]] | None = None,
    ) -> Tuple[Tensor, List[Tuple[Tensor, ...]]]:
        seq = torch.transpose(seq, 0, 1)
        if hid is None:
            hid = [
                l.init_hidden(seq.shape[1]) if isinstance(l, mLSTM) else l.init_hidden()
                for l in self.xlstm
            ]

        out = []
        for inp in seq:
            # Compute model output and update the hidden states
            for i, lstm in enumerate(self.xlstm):
                inp, hid[i] = lstm(inp, hid[i])
            out.append(inp)

        out = torch.stack(out, dim=1)
        out = self.fc(out[:, -1, :])

        return out, hid

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        seqs, _, labels, _ = batch
        labels = torch.argmax(labels, dim=1)

        outputs, _ = self(seqs)
        loss = F.cross_entropy(outputs, labels)

        self.log(
            "train_loss",
            loss.item(),
            sync_dist=True,
            batch_size=seqs.size(0),
        )

        preds = torch.argmax(outputs, dim=1)
        metrics = self.train_metrics(preds, labels)
        self.log_dict(
            metrics,
            sync_dist=True,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )

        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        seqs, _, labels, _ = batch
        labels = torch.argmax(labels, dim=1)

        outputs, _ = self(seqs)
        loss = F.cross_entropy(outputs, labels)

        self.log("val_loss", loss.item(), sync_dist=True)

        preds = torch.argmax(outputs, dim=1)
        self.valid_metrics.update(preds, labels)

        return loss

    def on_validation_epoch_end(self):
        metrics = self.valid_metrics.compute()
        self.log_dict(
            metrics,
            sync_dist=True,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.valid_metrics.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        seqs, _, _, msisdns = batch
        outputs, _ = self(seqs)
        outputs = torch.argmax(
            torch.softmax(self(outputs), dim=1),
            dim=1,
        )
        return outputs, msisdns
