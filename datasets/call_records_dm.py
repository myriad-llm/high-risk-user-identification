import lightning as L
import torch
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from datasets import CallRecords
from utils import pad_collate


class CallRecordsDataModuleBase(L.LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            batch_size: int,
            seed: int,
            non_seq: bool,
            num_workers: int,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seed = seed
        self.non_seq = non_seq
        self.num_workers = num_workers

        self.full = CallRecords(root=self.data_dir, predict=False, non_seq=non_seq)
        self.pred = CallRecords(root=self.data_dir, predict=True, non_seq=non_seq)

    @property
    def feature_dim(self):
        return self.full.features_num

    def dataloader(self, dataset: Dataset, shuffle: bool = False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=None if self.non_seq else pad_collate,
            num_workers=self.num_workers,
            persistent_workers=True,
        )


class CallRecordsDataModule(CallRecordsDataModuleBase):
    def __init__(
            self,
            data_dir: str,
            batch_size: int,
            seed: int,
            non_seq: bool = False,
            num_workers: int = 2,
    ):
        super().__init__(data_dir, batch_size, seed, non_seq, num_workers)

        self.train, self.val = random_split(
            self.full, [0.9, 0.1], generator=torch.Generator().manual_seed(self.seed)
        )

    def train_dataloader(self):
        return self.dataloader(self.train, shuffle=True)

    def val_dataloader(self):
        return self.dataloader(self.val)

    def predict_dataloader(self):
        return self.dataloader(self.pred)


class CallRecords4VAEDataModule(CallRecordsDataModuleBase):
    def __init__(
            self,
            data_dir: str,
            batch_size: int,
            seed: int,
            num_workers: int = 2,
    ):
        super().__init__(data_dir, batch_size, seed, True, num_workers)

        self.train = ConcatDataset([self.full, self.pred])

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.dataloader(self.train, shuffle=True)
