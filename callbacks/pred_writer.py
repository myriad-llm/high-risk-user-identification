import os
import time
from typing import Any

import pandas as pd
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter


class LSTMPredictionWriter(BasePredictionWriter):
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
        all_predictions = []
        all_msisdns = []
        for outputs, msisdns in predictions:
            all_predictions.extend(outputs.cpu().numpy())
            all_msisdns.extend(msisdns)

        res = pd.DataFrame()
        res["msisdn"] = pd.Series(all_msisdns).astype(str)
        res["is_sa"] = pd.Series(all_predictions).astype(int)

        os.makedirs(self.output_dir, exist_ok=True)
        res.to_csv(
            os.path.join(
                self.output_dir,
                f"{time.strftime(r'%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))}.csv",
            ),
            index=False,
        )
        print(res.describe())


class BertPredictionWriter(BasePredictionWriter):
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
        all_predictions = []
        all_msisdns = []
        for outputs, msisdns in predictions:
            all_predictions.extend(outputs.cpu().numpy())
            all_msisdns.extend(msisdns)

        res = pd.DataFrame()
        res["msisdn"] = pd.Series(all_msisdns).astype(str)
        res["is_sa"] = pd.Series(all_predictions).astype(int)

        def determine_is_sa(group):
            most_common_is_sa = group["is_sa"].mode()
            return most_common_is_sa[0]

        res = res.groupby("msisdn").apply(determine_is_sa).reset_index()
        res.columns = ["msisdn", "is_sa"]

        os.makedirs(self.output_dir, exist_ok=True)
        res.to_csv(
            os.path.join(
                self.output_dir,
                f"{time.strftime(r'%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))}.csv",
            ),
            index=False,
        )
        print(res.describe())
