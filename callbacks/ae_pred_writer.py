import os
import time
from typing import Any

import pandas as pd
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
import numpy as np

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

        # HACK： should use pl_module to get the embedding layer to embed the input, not from the forward function
        train_all_features, train_all_msisdns, train_embedded_x, train_seq_lens = [], [], [], []
        val_all_features, val_all_msisdns, val_embedded_x, val_seq_lens = [], [], [], []

        for features, msisdns, (embedded_x, seq_lens) in train_predictions:
            train_all_features.extend(features.cpu().numpy())
            train_all_msisdns.extend(msisdns)
            # embedded_x: batch * seq_len * feature_dim -> (batch * seq_len) * feature_dim
            embedded_x = embedded_x.reshape(-1, embedded_x.shape[-1])
            # remove padding
            non_zero_mask = embedded_x.any(dim=-1)
            # assert mask 取出的长度和 seq_lens 一致
            assert non_zero_mask.sum() == sum(seq_lens)
            embedded_x = embedded_x[non_zero_mask]
            train_seq_lens.extend(seq_lens)
            train_embedded_x.extend(embedded_x.cpu().numpy())

        for features, msisdns, (embedded_x, seq_lens) in val_predictions:
            val_all_features.extend(features.cpu().numpy())
            val_all_msisdns.extend(msisdns)
            # embedded_x: batch * seq_len * feature_dim -> (batch * seq_len) * feature_dim
            embedded_x = embedded_x.reshape(-1, embedded_x.shape[-1])
            # remove padding
            non_zero_mask = embedded_x.any(dim=-1)
            # assert mask 取出的长度和 seq_lens 一致
            assert non_zero_mask.sum() == sum(seq_lens)
            embedded_x = embedded_x[non_zero_mask]
            val_seq_lens.extend(seq_lens)
            val_embedded_x.extend(embedded_x.cpu().numpy())
        
        train_res = pd.DataFrame(train_all_features)
        print(train_res.shape)
        train_res["msisdn"] = pd.Series(train_all_msisdns).astype(str)
        train_res = train_res[['msisdn'] + [col for col in train_res.columns if col != 'msisdn']]

        val_res = pd.DataFrame(val_all_features)
        print(val_res.shape)
        val_res["msisdn"] = pd.Series(val_all_msisdns).astype(str)
        val_res = val_res[['msisdn'] + [col for col in val_res.columns if col != 'msisdn']]

        train_embedded_x = np.vstack(train_embedded_x)
        train_embedded_x = pd.DataFrame(train_embedded_x)
        # msisdn will be repeated by seq_lens
        train_all_msisdns = [msisdn for msisdn, seq_len in zip(train_all_msisdns, train_seq_lens) for _ in range(seq_len)]
        train_embedded_x["msisdn"] = pd.Series(train_all_msisdns).astype(str)
        train_embedded_x = train_embedded_x[['msisdn'] + [col for col in train_embedded_x.columns if col != 'msisdn']]

        val_embedded_x = np.vstack(val_embedded_x)
        val_embedded_x = pd.DataFrame(val_embedded_x)
        # msisdn will be repeated by seq_lens
        val_all_msisdns = [msisdn for msisdn, seq_len in zip(val_all_msisdns, val_seq_lens) for _ in range(seq_len)]
        val_embedded_x["msisdn"] = pd.Series(val_all_msisdns).astype(str)
        val_embedded_x = val_embedded_x[['msisdn'] + [col for col in val_embedded_x.columns if col != 'msisdn']]

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

        train_embedded_x.to_csv(
            os.path.join(
                self.output_dir,
                # f"{time.strftime(r'%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))}_train_embedded_x.csv",
                "train_embedded_x.csv",
            ),
            index=False,
        )
        print(train_embedded_x.describe())

        val_embedded_x.to_csv(
            os.path.join(
                self.output_dir,
                # f"{time.strftime(r'%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))}_val_embedded_x.csv",
                "val_embedded_x.csv",
            ),
            index=False,
        )
        print(val_embedded_x.describe())

        torch.save(pl_module.embeddings.state_dict(), os.path.join(self.output_dir, "embeddings_weight.pth"))
