import itertools
import math
import warnings
from typing import Sequence, Union, cast

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import BertTokenizer

from .call_records_v2 import CallRecordsV2
from .data_collator import *


def split(
    dataset: Dataset,
    lengths: Sequence[Union[int, float]],
) -> List[Subset]:
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(
                    f"Length of split at index {i} is 0. "
                    f"This might result in an empty dataset."
                )

        # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = torch.arange(sum(lengths))
    lengths = cast(Sequence[int], lengths)
    return [
        Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(itertools.accumulate(lengths), lengths)
    ]


class CallRecordsV2DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        seed: int,
        num_workers: int,
        mlm: bool,
        mlm_probability: float,
        seq_len: int,
        num_bins: int,
        flatten: bool,
        stride: int,
        adap_thres: int,
        return_labels: bool,
        collator_fn: str,
    ) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers

        self.full = CallRecordsV2(
            root=data_dir,
            mlm=mlm,
            seq_len=seq_len,
            num_bins=num_bins,
            flatten=flatten,
            stride=stride,
            adap_thres=adap_thres,
            return_labels=return_labels,
        )
        self.vocab = self.full.vocab

        self.pred = CallRecordsV2(
            root=data_dir,
            mlm=mlm,
            seq_len=seq_len,
            num_bins=num_bins,
            flatten=flatten,
            stride=stride,
            adap_thres=adap_thres,
            predict=True,
        )
        assert len(self.pred.vocab) == len(self.vocab)

        self.tokenizer = BertTokenizer(
            self.vocab.filename,
            do_lower_case=False,
            **self.vocab.get_special_tokens(),
        )

        self.collator_fn = eval(collator_fn)(
            tokenizer=self.tokenizer,
            mlm=mlm,
            mlm_probability=mlm_probability,
        )

        self.train, self.val = split(self.full, [0.9, 0.1])

    @property
    def feature_dim(self):
        return self.full.ncols - 1

    def dataloader(self, dataset: Dataset, shuffle: bool = False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self.collator_fn,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self.dataloader(self.train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self.dataloader(self.val)

    def predict_dataloader(self) -> DataLoader:
        return self.dataloader(self.pred)
