import os
import time
from typing import Any

import pandas as pd
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter


class AEPredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        predictions: Any,
        batch_indices: Any,
    ) -> None:
        train_predictions, val_predictions = predictions

        train_all_features, train_all_msisdns = [], []
        val_all_features, val_all_msisdns = [], []

        for features, msisdns in train_predictions:
            train_all_features.extend(features.cpu().numpy())
            train_all_msisdns.extend(msisdns)

        for features, msisdns in val_predictions:
            val_all_features.extend(features.cpu().numpy())
            val_all_msisdns.extend(msisdns)
        
        train_res = pd.DataFrame(train_all_features)
        print(train_res.shape)
        train_res["msisdn"] = pd.Series(train_all_msisdns).astype(str)
        train_res = train_res[['msisdn'] + [col for col in train_res.columns if col != 'msisdn']]

        val_res = pd.DataFrame(val_all_features)
        print(val_res.shape)
        val_res["msisdn"] = pd.Series(val_all_msisdns).astype(str)
        val_res = val_res[['msisdn'] + [col for col in val_res.columns if col != 'msisdn']]

        os.makedirs(self.output_dir, exist_ok=True)
        train_res.to_csv(
            os.path.join(
                self.output_dir,
                # f"{time.strftime(r'%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))}_train.csv",
                "train.csv",
            ),
            index=False,
        )
        print(train_res.describe())

        val_res.to_csv(
            os.path.join(
                self.output_dir,
                # f"{time.strftime(r'%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))}_val.csv",
                "val.csv",
            ),
            index=False,
        )
        print(val_res.describe())
