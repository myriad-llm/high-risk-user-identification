from typing import Any

import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)
from torchmetrics.collections import MetricCollection
from transformers import BertConfig, BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

from typing import Callable, Iterable
from torch.optim import Optimizer

OptimizerCallable = Callable[[Iterable], Optimizer]

class BertClassification(L.LightningModule):
    def __init__(
        self,
        input_size,
        vocab_size: int,
        hidden_size: int,
        num_hidden_layers: int = 12,
        num_classes: int = 2,
        dropout_rate: float = 0.1,
        optimizer: OptimizerCallable = torch.optim.AdamW, 
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.optimizer = optimizer
        self.num_classes = num_classes
        self.model = BertForSequenceClassification(
            BertConfig(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=input_size,
                num_labels=num_classes,
                classifier_dropout=dropout_rate,
            )
        )

        metrics = MetricCollection(
            [
                BinaryAccuracy(),
                BinaryPrecision(),
                BinaryRecall(),
                BinaryF1Score(),
            ]
        )
        self.train_metrics = metrics.clone(prefix="train-")
        self.valid_metrics = metrics.clone(prefix="val-")
        self.callbacks = None

    def training_step(self, batch, batch_idx) -> Tensor:
        labels = (batch["labels"][:, 0] == 1).long().to(self.device)

        outputs: SequenceClassifierOutput = self.model(
            input_ids=batch["input_ids"],
            labels=labels,
            return_dict=True,
        )

        self.log("train-loss", outputs.loss, sync_dist=True)

        metrics = self.train_metrics(
            torch.argmax(outputs.logits, 1),
            labels,
        )
        self.log_dict(
            metrics,
            prog_bar=True,
            sync_dist=True,
        )

        return outputs.loss

    def validation_step(self, batch, batch_idx) -> Tensor:
        labels = (batch["labels"][:, 0] == 1).long().to(self.device)

        outputs: SequenceClassifierOutput = self.model(
            input_ids=batch["input_ids"],
            labels=labels,
            return_dict=True,
        )
        self.log("val-loss", outputs.loss, sync_dist=True)

        self.valid_metrics.update(
            torch.argmax(outputs.logits, 1),
            labels,
        )

        return outputs.loss

    def on_validation_epoch_end(self):
        metrics = self.valid_metrics.compute()
        self.log_dict(metrics, sync_dist=True, on_step=False, on_epoch=True)
        self.valid_metrics.reset()

    def predict_step(self, batch, batch_idx):
        msisdns = batch["msisdns"]
        outputs: SequenceClassifierOutput = self.model(
            input_ids=batch["input_ids"],
            return_dict=True,
        )

        predicted_labels = torch.argmax(
            torch.softmax(outputs.logits, dim=1),
            dim=1,
        )

        return predicted_labels, msisdns

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = self.optimizer(self.parameters())
        return optimizer
    
    def configure_callbacks(self):
        if self.callbacks:
            return self.callbacks
        return []
    
    def set_callbacks(self, callbacks) -> None:
        self.callbacks = callbacks
