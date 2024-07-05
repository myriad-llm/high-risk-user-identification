import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from datasets import CallRecords
from utils import pad_collate


class CallRecordsDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, seed: int, non_seq: bool = False, num_workers: int = 2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seed = seed
        self.non_seq = non_seq
        self.num_workers = num_workers

        self.full = CallRecords(root=self.data_dir, predict=False, non_seq=non_seq)
        self.train, self.val, self.test = random_split(
            self.full, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(self.seed)
        )
        self.pred = CallRecords(root=self.data_dir, predict=True)

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

    def train_dataloader(self):
        return self.dataloader(self.train, shuffle=True)

    def val_dataloader(self):
        return self.dataloader(self.val)

    def test_dataloader(self):
        return self.dataloader(self.test)

    def predict_dataloader(self):
        return self.dataloader(self.pred)
