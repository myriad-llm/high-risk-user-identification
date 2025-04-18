import os
import pickle as pkl
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import *
from utils.augmentation import Augmentation
from utils.dataclass import EmbeddingItem
import json


@dataclass
class Item:
    msisdn: str
    records: torch.Tensor
    records_len: int
    time_diff: Optional[torch.Tensor]
    labels: Optional[torch.Tensor]


class CallRecordsAug(Dataset):
    resources = [
        ("trainSet_res_with_distances.csv", "trainSet_ans.csv"),
        ("validationSet_res_with_distances.csv", None),
    ]

    manifests = [
        "data_records.pt",
        "data_labels.pt",
        "data_seq_index_with_time_diff.pkl",
        "predict_records.pt",
        "predict_seq_index_with_time_diff.pkl",
        "embedding_items.pkl",
    ]

    time_format: Dict[str, str] = {
        "start_time": "%Y%m%d%H%M%S",
        "open_datetime": "%Y%m%d%H%M%S",
    }
    datetime_columns_type: Dict[str, str] = {key: "str" for key in time_format}
    datetime_columns: List[str] = list(time_format.keys())

    numeric_columns_type: Dict[str, str] = {
        "call_duration": "int32",
        "cfee": "int32",
        "lfee": "int32",
        "hour": "int8",
    }
    numeric_columns: List[str] = list(numeric_columns_type.keys())

    area_code_columns: List[str] = [
        "home_area_code",
        "visit_area_code",
        "called_home_code",
        "called_code",
    ]
    area_code_columns_type: Dict[str, str] = {key: "str" for key in area_code_columns}

    city_columns: List[str] = ["phone1_loc_city", "phone2_loc_city"]
    city_columns_type: Dict[str, str] = {key: "str" for key in city_columns}

    province_columns: List[str] = ["phone1_loc_province", "phone2_loc_province"]
    province_columns_type: Dict[str, str] = {key: "str" for key in province_columns}

    a_product_id_columns: List[str] = ["a_product_id"]
    a_product_id_columns_type: Dict[str, str] = {
        key: "str" for key in a_product_id_columns
    }

    categorical_columns: List[str] = [
        "a_serv_type",
        "long_type1",
        "roam_type",
        "dayofweek",
    ]
    categorical_columns_type: Dict[str, str] = {
        key: "category" for key in categorical_columns
    }

    phone_type_columns: List[str] = ["phone1_type", "phone2_type"]
    phone_type_columns_type: Dict[str, str] = {
        key: "category" for key in phone_type_columns
    }

    distance_columns: List[str] = ["distance"]
    distance_columns_type: Dict[str, str] = {key: "float32" for key in distance_columns}

    def __init__(
        self,
        root: Union[str, Path],
        predict: bool = False,
        non_seq: bool = False,
        time_div: int = 3600,
        mask_rate: float = 0.0,
        aug_ratio: float = 0.0,
        aug_times: int = 0,
    ) -> None:
        if isinstance(root, str):
            root = os.path.expanduser(root)
        self.root = root
        self.predict = predict
        self.non_seq = non_seq
        self.time_div = time_div
        self.mask_rate = mask_rate
        self.aug_ratio = aug_ratio
        self.aug_times = aug_times

        def time_map(x):
            x_div = x / self.time_div
            return 1 / torch.log(x_div + torch.e)

        if self._check_legacy_exists():
            self.records, self.labels, self.seq_index_with_time_diff, self.embedding_items = (
                self._load_legacy_data()
            )

            self.seq_index_with_time_diff = [
                (seq, msisdn, time_map(time_diff))
                for seq, msisdn, time_diff in self.seq_index_with_time_diff
            ]
            return

        if not self._check_exists():
            raise RuntimeError("Dataset not found.")

        data, pred, embedding_items = self._load_data()

        self._save_legacy_data(data, pred, embedding_items)

        self.records, self.labels, self.seq_index_with_time_diff = (
            pred if self.predict else data
        )
        self.seq_index_with_time_diff = [
            (seq, msisdn, time_map(time_diff))
            for seq, msisdn, time_diff in self.seq_index_with_time_diff
        ]
        self.embedding_items = embedding_items

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[Item, torch.Tensor]:
        if self.non_seq:
            return self.records[index]

        seq_index, msisdn, time_diff = self.seq_index_with_time_diff[index]
        seq, label = self.records[seq_index], (
            self.labels[index] if self.labels is not None else None
        )
        seq_len = len(seq_index)

        if self.mask_rate > 0:
            mask_num = int(seq_len * self.mask_rate)
            mask_idx = torch.randperm(seq_len)[:mask_num]
            seq[mask_idx] = 0  # Assuming the goal is to mask with zeros

        return Item(msisdn, seq, seq_len, time_diff, label)

    def __len__(self) -> int:
        if self.non_seq:
            return len(self.records)

        return len(self.seq_index_with_time_diff)

    @property
    def embedding_items_path(self) -> str:
        for item in self.embedding_items:
            print(item)
        embedding_items_path = os.path.join(
            self.processed_folder, "embedding_items.pkl"
        )
        assert os.path.exists(embedding_items_path), "embedding_items.pkl not found"
        return embedding_items_path

    @property
    def features_num(self) -> int:
        # HACK: According to embedding_items to calculate feature_num
        feature_num = self.records.shape[1]
        for item in self.embedding_items:
            feature_num -= len(item.x_col_index)
            feature_num += item.embedding_dim * len(item.x_col_index)
        return feature_num

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "processed")

    def _check_legacy_exists(self) -> bool:
        if not os.path.exists(self.processed_folder):
            return False

        return all(
            os.path.isfile(os.path.join(self.processed_folder, file))
            for file in self.manifests
        )

    def _load_legacy_data(
        self,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        List[Tuple[List[int], torch.Tensor]],
        List[EmbeddingItem],
    ]:
        records_file = "predict_records.pt" if self.predict else "data_records.pt"
        labels_file = None if self.predict else "data_labels.pt"
        seq_file = (
            "predict_seq_index_with_time_diff.pkl"
            if self.predict
            else "data_seq_index_with_time_diff.pkl"
        )
        embedding_items_file = "embedding_items.pkl"

        records = torch.load(os.path.join(self.processed_folder, records_file))

        labels = None
        if labels_file:
            labels = torch.load(os.path.join(self.processed_folder, labels_file))

        with open(os.path.join(self.processed_folder, seq_file), "rb") as f:
            seq_index_with_time_diff = pkl.load(f)

        with open(os.path.join(self.processed_folder, embedding_items_file), "rb") as f:
            embedding_items = pkl.load(f)

        return (
            records.to_dense(),
            labels.to_dense() if labels is not None else None,
            seq_index_with_time_diff,
            embedding_items,
        )

    def _save_legacy_data(
        self,
        data: Tuple[torch.Tensor, torch.Tensor, List[Tuple[List[int], torch.Tensor]]],
        pred: Tuple[torch.Tensor, None, List[Tuple[List[int], torch.Tensor]]],
        embedding_items: List[EmbeddingItem],
    ) -> None:
        if not os.path.exists(self.processed_folder):
            os.makedirs(self.processed_folder)

        def save_tensor(t: torch.Tensor, filename: str) -> None:
            path = os.path.join(self.processed_folder, filename)
            torch.save(t.to_sparse_coo(), path)

        def save_pickle(d: object, filename: str) -> None:
            with open(os.path.join(self.processed_folder, filename), "wb") as f:
                pkl.dump(d, f, pkl.HIGHEST_PROTOCOL)

        (data_records, data_labels, data_seq_index_with_time_diff) = data
        (pred_records, _, pred_seq_index_with_time_diff) = pred

        save_tensor(data_records, "data_records.pt")
        save_tensor(data_labels, "data_labels.pt")
        save_pickle(data_seq_index_with_time_diff, "data_seq_index_with_time_diff.pkl")
        save_tensor(pred_records, "predict_records.pt")
        save_pickle(
            pred_seq_index_with_time_diff, "predict_seq_index_with_time_diff.pkl"
        )
        save_pickle(embedding_items, "embedding_items.pkl")

    def _check_exists(self) -> bool:
        return all(
            os.path.isfile(os.path.join(self.raw_folder, file))
            for data_pair in self.resources
            for file in data_pair
            if file is not None
        )

    def _load_data(self) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor, List[Tuple[List[int], torch.Tensor]]],
        Tuple[torch.Tensor, None, List[Tuple[List[int], torch.Tensor]]],
        List[EmbeddingItem],
    ]:
        (train_records_df, train_labels_df), (val_records_df, _) = (
            self._load_dataframes()
        )

        train_records_df, train_labels_df = self.augment(
            train_records_df, train_labels_df, self.aug_ratio, self.aug_times
        )

        remap_column_group = {
            "area_code": self.area_code_columns,
            # 'city': self.city_columns,
            "province": self.province_columns,
            "a_product_id": self.a_product_id_columns,
            "phone_type": self.phone_type_columns,
        }
        remap_column_group.update({col: [col] for col in self.categorical_columns})

        value_dicts = {
            group: generate_value_dict(columns, train_records_df, val_records_df)
            for group, columns in remap_column_group.items()
        }
        value_dicts.update(
            {
                col: generate_value_dict([col], train_records_df, val_records_df)
                for col in self.categorical_columns
            }
        )

        os.makedirs(self.processed_folder, exist_ok=True)
        with open(os.path.join(self.processed_folder, "value_dicts.json"), "w") as f:
            json.dump(value_dicts, f)

        train_records_df = remap_data(train_records_df, remap_column_group, value_dicts)
        val_records_df = remap_data(val_records_df, remap_column_group, value_dicts)

        # TODO
        train_records_df.drop(columns=self.city_columns, axis=1, inplace=True)
        val_records_df.drop(columns=self.city_columns, axis=1, inplace=True)

        apply_cols = {
            col: len(value_dicts[group])
            for group, columns in remap_column_group.items()
            for col in columns
        }
        print(apply_cols)
        # train_records_df = apply_onehot(train_records_df, apply_cols)
        # val_records_df = apply_onehot(val_records_df, apply_cols)

        train_records_df = add_open_count(train_records_df)
        val_records_df = add_open_count(val_records_df)

        train_records_seq_ids = gen_seq_ids(train_records_df)
        val_records_seq_ids = gen_seq_ids(val_records_df)

        train_seq_index_with_time_diff, train_seq_msisdns = split_seq_and_time_diff(
            train_records_df, train_records_seq_ids
        )
        val_seq_index_with_time_diff, val_seq_msisdns = split_seq_and_time_diff(
            val_records_df, val_records_seq_ids
        )

        train_records_df.drop(columns=["msisdn", "start_time"], axis=1, inplace=True)
        val_records_df.drop(columns=["msisdn", "start_time"], axis=1, inplace=True)

        # HACK: Make sure numeric columns are in the front, it's not necessary to sort the columns, but it's easier to debug.
        numeric_columns = [
            "call_duration",
            "cfee",
            "lfee",
            "hour",
            "open_count",
            "distance",
        ]
        embedding_columns = list(train_records_df.columns.difference(numeric_columns))

        train_records_df = train_records_df[numeric_columns + embedding_columns]
        val_records_df = val_records_df[numeric_columns + embedding_columns]

        # categorical values embedding
        # if sqrt(vocab_size) < 3, embedding_dim = 3
        min_embedding_dim = 3
        embedding_names = [
            "area_code",
            "province",
            "phone_type",
            "a_product_id",
            "a_serv_type",
            "dayofweek",
            "long_type1",
            "roam_type",
        ]
        embedding_items = []
        for embedding_name in embedding_names:
            if embedding_name == "area_code":
                vocab_size = len(value_dicts["area_code"])
                col_idx = [
                    train_records_df.columns.get_loc(col)
                    for col in self.area_code_columns
                ]
            elif embedding_name == "province":
                vocab_size = len(value_dicts["province"])
                col_idx = [
                    train_records_df.columns.get_loc(col)
                    for col in self.province_columns
                ]
            elif embedding_name == "phone_type":
                vocab_size = len(value_dicts["phone_type"])
                col_idx = [
                    train_records_df.columns.get_loc(col)
                    for col in self.phone_type_columns
                ]
            else:
                vocab_size = len(value_dicts[embedding_name])
                col_idx = [train_records_df.columns.get_loc(embedding_name)]
            vocab_size += 1  # for padding
            col_idx_tensor = torch.tensor(col_idx, dtype=torch.int32)
            embedding_dim = max(min_embedding_dim, int(vocab_size**0.5))
            embedding_items.append(
                EmbeddingItem(embedding_name, vocab_size, embedding_dim, col_idx_tensor)
            )

        # numeric values scaling
        train_records_df, val_records_df = apply_scaler(
            [train_records_df, val_records_df],
            numeric_columns,
        )

        train_labels_df = (
            train_labels_df.set_index("msisdn").loc[train_seq_msisdns].reset_index()
        )
        train_labels_df = pd.get_dummies(
            train_labels_df["is_sa"], columns=["is_sa"], dtype="int8"
        )

        return (
            (
                torch.tensor(train_records_df.values, dtype=torch.float32),
                torch.tensor(train_labels_df.values, dtype=torch.float32),
                train_seq_index_with_time_diff,
            ),
            (
                torch.tensor(val_records_df.values, dtype=torch.float32),
                None,
                val_seq_index_with_time_diff,
            ),
            embedding_items,
        )

    @staticmethod
    def load_records(path: str) -> pd.DataFrame:
        dtypes = {
            "msisdn": "str",
            **CallRecordsAug.datetime_columns_type,
            **CallRecordsAug.numeric_columns_type,
            **CallRecordsAug.area_code_columns_type,
            **CallRecordsAug.city_columns_type,
            **CallRecordsAug.province_columns_type,
            **CallRecordsAug.a_product_id_columns_type,
            **CallRecordsAug.categorical_columns_type,
            **CallRecordsAug.phone_type_columns_type,
            **CallRecordsAug.distance_columns_type,
        }
        usecols = (
            ["msisdn"]
            + CallRecordsAug.datetime_columns
            + CallRecordsAug.numeric_columns
            + CallRecordsAug.area_code_columns
            + CallRecordsAug.city_columns
            + CallRecordsAug.province_columns
            + CallRecordsAug.a_product_id_columns
            + CallRecordsAug.categorical_columns
            + CallRecordsAug.phone_type_columns
            + CallRecordsAug.distance_columns
        )

        df = pd.read_csv(
            path,
            sep=",",
            usecols=usecols,
            dtype=dtypes,
        )

        for col in CallRecordsAug.datetime_columns:
            df[col] = pd.to_datetime(
                df[col], format=CallRecordsAug.time_format[col], errors="coerce"
            )

        df["start_time"] = df["start_time"].apply(lambda x: x.timestamp()).astype("int")

        return df

    @staticmethod
    def load_labels(path: str) -> pd.DataFrame:
        return pd.read_csv(
            path,
            sep=",",
            dtype={"msisdn": "str", "is_sa": "bool"},
        )

    def _load_dataframes(
        self,
    ) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, None]]:
        # load all training and validation data
        dataframes = [
            (
                self.load_records(os.path.join(self.raw_folder, record)),
                (
                    self.load_labels(os.path.join(self.raw_folder, label))
                    if label is not None
                    else None
                ),
            )
            for record, label in self.resources
        ]

        train_dataframes = [
            (records, labels) for records, labels in dataframes if labels is not None
        ]
        val_dataframes = [records for records, labels in dataframes if labels is None]

        # combine all training/validation data into a single dataframe and return
        return (
            (
                pd.concat([df for df, _ in train_dataframes], ignore_index=True),
                pd.concat([df for _, df in train_dataframes], ignore_index=True),
            ),
            (
                pd.concat(val_dataframes, ignore_index=True),
                None,
            ),
        )

    def augment(self, train_data, train_labels, ratio_range, times):
        # 3 + 4 * times 因为正负样本比例为 1:4 可以均衡至 1:1
        if times == 0:
            return train_data, train_labels

        addition_train_data = []
        addition_train_labels = []

        pbar = tqdm(train_data.groupby("msisdn"))
        for msisdn, group in pbar:
            if msisdn == 0:
                continue
            pbar.set_description(f"Augmenting msisdn {msisdn}")
            label = train_labels[train_labels["msisdn"] == msisdn].iloc[0]["is_sa"]
            aug = Augmentation(group, label, "msisdn", "is_sa")
            # 对正负样本进行平衡 样本比 1:4
            if label == 1:
                res_df, res_labels = aug.times(
                    ratio=ratio_range, times=3 + times * 4, method="mask"
                )
            else:
                res_df, res_labels = aug.times(
                    ratio=ratio_range, times=times, method="mask"
                )
            addition_train_data.append(res_df)
            addition_train_labels.append(res_labels)
        addition_train_data = pd.concat(addition_train_data)
        addition_train_labels = pd.concat(addition_train_labels)
        addition_train_data.shape
        train_data = pd.concat(
            [train_data, addition_train_data], ignore_index=True
        ).reset_index(drop=True)
        train_labels = pd.concat(
            [train_labels, addition_train_labels], ignore_index=True
        ).reset_index(drop=True)
        # 按照 msisdn, start_time 排序
        train_data.sort_values(by=["msisdn", "start_time"]).reset_index(drop=True)
        train_labels.sort_values(by=["msisdn"]).reset_index(drop=True)

        return train_data, train_labels
