import lightning as L
import torch
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from datasets.aug import CallRecordsAug
from utils import pad_collate


class CallRecordsDataModuleBase(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        seed: int,
        non_seq: bool,
        num_workers: int,
        time_div: int = 3600,
        mask_rate: float = 0.0,
        aug_ratio: float = 0.0,
        aug_times: int = 0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seed = seed
        self.non_seq = non_seq
        self.num_workers = num_workers
        self.time_div = time_div
        self.mask_rate = mask_rate

        self.full = CallRecordsAug(
            root=self.data_dir,
            predict=False,
            non_seq=non_seq,
            time_div=time_div,
            mask_rate=mask_rate,
            aug_ratio=aug_ratio,
            aug_times=aug_times,
        )
        self.pred = CallRecordsAug(
            root=self.data_dir,
            predict=True,
            non_seq=non_seq,
            time_div=time_div,
            mask_rate=0.0,
        )

    @property
    def feature_dim(self):
        return self.full.features_num

    @property
    def embedding_items_path(self):
        return self.full.embedding_items_path

    def dataloader(self, dataset: Dataset, shuffle: bool = False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=None if self.non_seq else pad_collate,
            num_workers=self.num_workers,
            persistent_workers=True,
        )


class CallRecordsAugDataModule(CallRecordsDataModuleBase):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        seed: int,
        non_seq: bool = False,
        num_workers: int = 2,
        time_div: int = 3600,
        mask_rate: float = 0.0,
        aug_ratio: float = 0.0,
        aug_times: int = 0,
        for_ae: bool = False,
    ):
        super().__init__(
            data_dir,
            batch_size,
            seed,
            non_seq,
            num_workers,
            time_div,
            mask_rate,
            aug_ratio,
            aug_times,
        )

        self.for_ae = for_ae

        full_length = len(self.full)
        val_rate = 0.3
        val_size = int(full_length * val_rate)
        start_index = torch.randint(high=full_length - val_size, size=(1,)).item()
        end_index = start_index + val_size
        if end_index <= full_length:
            indices_val = list(range(start_index, end_index))
        else:
            overflow = end_index - full_length
            indices_val = list(range(start_index, full_length)) + list(range(overflow))

        indices_train = [i for i in range(full_length) if i not in indices_val]
        self.train = torch.utils.data.Subset(self.full, indices_train)
        self.val = torch.utils.data.Subset(self.full, indices_val)

    def train_dataloader(self):
        if self.for_ae:
            return self.dataloader(self.full, shuffle=True)
        return self.dataloader(self.train, shuffle=True)

    def val_dataloader(self):
        if self.for_ae:
            return self.dataloader(self.pred)
        return self.dataloader(self.val)

    def predict_dataloader(self):
        if self.for_ae:
            return [self.dataloader(self.full), self.dataloader(self.pred)]
        return self.dataloader(self.pred)
