from typing import Any

import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor
from torchmetrics.functional import accuracy
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
        self.accuracy = accuracy
        self.model = BertForSequenceClassification(
            BertConfig(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=input_size,
                num_labels=num_classes,
            )
        )

    def training_step(self, batch, batch_idx) -> Tensor:
        outputs: SequenceClassifierOutput = self.model(
            input_ids=batch["input_ids"],
            labels=batch["labels"],
            return_dict=True,
        )
        self.log("train-loss", outputs.loss, batch_size=len(batch))
        return outputs.loss

    def validation_step(self, batch, batch_idx) -> None:
        outputs: SequenceClassifierOutput = self.model(
            input_ids=batch["input_ids"],
            labels=batch["labels"],
            return_dict=True,
        )
        self.log("val-loss", outputs.loss, batch_size=len(batch))

        logits = outputs.logits
        predicted_labels = torch.argmax(logits, 1)
        acc = self.accuracy(
            predicted_labels,
            batch["label"],
            num_classes=self.num_classes,
            task="multiclass",
        )
        self.log("val-acc", acc, batch_size=len(batch), prog_bar=True)

    def predict_step(self, batch, batch_idx):
        outputs: SequenceClassifierOutput = self.model(
            input_ids=batch["input_ids"],
            labels=batch["labels"],
            return_dict=True,
        )

        predicted_labels = torch.argmax(
            torch.softmax(outputs.logits, dim=1),
            dim=1,
        )
        return predicted_labels, None

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
