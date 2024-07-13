import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer

from .call_records_v2 import CallRecordsV2
from .data_collator import CallRecordsDataCollatorForLanguageModeling


class CallRecordsV2DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        seed: int,
        num_workers: int,
        mlm: bool,
        mlm_probability: float = 0.15,
        seq_len: int = 16,
        num_bins: int = 10,
        flatten: int = False,
        stride: int = 5,
        adap_thres: int = 10**8,
        return_labels: int = False,
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

        self.tokenizer = BertTokenizer(
            self.vocab.filename,
            do_lower_case=False,
            **self.vocab.get_special_tokens(),
        )

        self.collator_fn = CallRecordsDataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=mlm,
            mlm_probability=mlm_probability,
        )

        self.train, self.val = random_split(
            self.full, [0.9, 0.1], generator=torch.Generator().manual_seed(self.seed)
        )

    @property
    def feature_dim(self):
        return 18

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
