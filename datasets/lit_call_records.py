import lightning as L
import torch
from torch.utils.data import DataLoader, random_split

from datasets import CallRecords
from utils import pad_collate


class CallRecordsDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, seed: int, ratio: float = 0.9):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seed = seed

        self.full = CallRecords(root=self.data_dir, train=True, ratio=ratio)
        self.train, self.val = random_split(
            self.full, [0.9, 0.1], generator=torch.Generator().manual_seed(self.seed)
        )
        self.test = CallRecords(root=self.data_dir, train=False, ratio=ratio)
        self.pred = CallRecords(root=self.data_dir, valid=True, ratio=ratio)

    @property
    def feature_dim(self):
        return self.full.features_num

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, collate_fn=pad_collate, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, collate_fn=pad_collate, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, collate_fn=pad_collate)

    def predict_dataloader(self):
        return DataLoader(self.pred, batch_size=self.batch_size, shuffle=False, collate_fn=pad_collate)
