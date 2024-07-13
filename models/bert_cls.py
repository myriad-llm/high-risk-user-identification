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


class BertClassification(L.LightningModule):
    def __init__(
        self,
        input_size,
        vocab_size: int,
        hidden_size: int,
        num_hidden_layers: int = 12,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.model = BertForSequenceClassification(
            BertConfig(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=input_size,
                num_labels=num_classes,
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
        self.valid_metrics = metrics.clone(prefix="valid-")

    def training_step(self, batch, batch_idx) -> Tensor:
        labels = (batch["labels"][:, 0] == 1).long().to(self.device)

        outputs: SequenceClassifierOutput = self.model(
            input_ids=batch["input_ids"],
            labels=labels,
            return_dict=True,
        )

        self.log("train-loss", outputs.loss, batch_size=len(batch))

        metrics = self.train_metrics(
            torch.argmax(outputs.logits, 1),
            labels,
        )
        self.log_dict(
            metrics,
            prog_bar=True,
            batch_size=len(batch),
        )

        return outputs.loss

    def validation_step(self, batch, batch_idx) -> None:
        labels = (batch["labels"][:, 0] == 1).long().to(self.device)

        outputs: SequenceClassifierOutput = self.model(
            input_ids=batch["input_ids"],
            labels=labels,
            return_dict=True,
        )
        self.log("valid-loss", outputs.loss, batch_size=len(batch))

        metrics = self.valid_metrics(
            torch.argmax(outputs.logits, 1),
            labels,
        )
        self.log_dict(
            metrics,
            prog_bar=True,
            batch_size=len(batch),
        )

    def predict_step(self, batch, batch_idx):
        labels = (batch["labels"][:, 0] == 1).long().to(self.device)

        outputs: SequenceClassifierOutput = self.model(
            input_ids=batch["input_ids"],
            labels=labels,
            return_dict=True,
        )

        predicted_labels = torch.argmax(
            torch.softmax(outputs.logits, dim=1),
            dim=1,
        )
        return predicted_labels, None

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
