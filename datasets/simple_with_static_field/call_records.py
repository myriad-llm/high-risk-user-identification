import os
import pickle as pkl
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import Dataset

from utils import *


class CallRecords(Dataset):
    resources = [
        ("cleaned_trainSet_res.csv", "trainSet_ans.csv"),
        ("cleaned_validationSet_res.csv", None),
    ]

    manifests = [
        "data_records.pt",
        "data_labels.pt",
        "data_seq_index_with_time_diff.pkl",
        "predict_records.pt",
        "predict_seq_index_with_time_diff.pkl",
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

    other_party_columns: List[str] = ["other_party"]
    other_party_columns_type: Dict[str, str] = {
        "other_party": "str",
    }

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

    dayofweek_columns: List[str] = ["dayofweek"]
    dayofweek_columns_type: Dict[str, str] = {key: "int" for key in dayofweek_columns}

    categorical_columns: List[str] = [
        "a_serv_type",
        "long_type1",
        "roam_type",
        # "dayofweek",
        "phone1_type",
        "phone2_type",
    ]
    categorical_columns_type: Dict[str, str] = {
        key: "category" for key in categorical_columns
    }

    def __init__(
        self,
        root: Union[str, Path],
        predict: bool = False,
        non_seq: bool = False,
        time_div: int = 3600,
        mask_rate: float = 0.0,
    ) -> None:
        if isinstance(root, str):
            root = os.path.expanduser(root)
        self.root = root
        self.predict = predict
        self.non_seq = non_seq
        self.time_div = time_div
        self.mask_rate = mask_rate

        def time_map(x):
            x_div = x / self.time_div
            return 1 / torch.log(x_div + torch.e)

        if self._check_legacy_exists():
            self.records, self.labels, self.seq_index_with_time_diff = (
                self._load_legacy_data()
            )

            self.seq_index_with_time_diff = [
                (seq, msisdn, time_map(time_diff))
                for seq, msisdn, time_diff in self.seq_index_with_time_diff
            ]
            return

        if not self._check_exists():
            raise RuntimeError("Dataset not found.")

        data, pred = self._load_data()

        self._save_legacy_data(data, pred)

        self.records, self.labels, self.seq_index_with_time_diff = (
            pred if self.predict else data
        )
        self.seq_index_with_time_diff = [
            (seq, msisdn, time_map(time_diff))
            for seq, msisdn, time_diff in self.seq_index_with_time_diff
        ]

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]] | torch.Tensor:
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

        return seq, time_diff, label, msisdn, seq_len

    def __len__(self) -> int:
        if self.non_seq:
            return len(self.records)

        return len(self.seq_index_with_time_diff)

    @property
    def features_num(self) -> int:
        return self.records.shape[1]

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
        torch.Tensor, Optional[torch.Tensor], List[Tuple[List[int], torch.Tensor]]
    ]:
        records_file = "predict_records.pt" if self.predict else "data_records.pt"
        labels_file = None if self.predict else "data_labels.pt"
        seq_file = (
            "predict_seq_index_with_time_diff.pkl"
            if self.predict
            else "data_seq_index_with_time_diff.pkl"
        )

        records = torch.load(os.path.join(self.processed_folder, records_file))

        labels = None
        if labels_file:
            labels = torch.load(os.path.join(self.processed_folder, labels_file))

        with open(os.path.join(self.processed_folder, seq_file), "rb") as f:
            seq_index_with_time_diff = pkl.load(f)

        return (
            records.to_dense(),
            labels.to_dense() if labels is not None else None,
            seq_index_with_time_diff,
        )

    def _save_legacy_data(
        self,
        data: Tuple[torch.Tensor, torch.Tensor, List[Tuple[List[int], torch.Tensor]]],
        pred: Tuple[torch.Tensor, None, List[Tuple[List[int], torch.Tensor]]],
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
    ]:
        (train_records_df, train_labels_df), (val_records_df, _) = (
            self._load_dataframes()
        )

        remap_column_group = {
            "area_code": ["visit_area_code", "called_code"],
            # "area_code": self.area_code_columns,
            # 'city': self.city_columns,
            # "province": self.province_columns,
            "a_product_id": self.a_product_id_columns,
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

        train_records_df, train_static_columns = add_static_features(train_records_df)
        val_records_df, val_static_columns = add_static_features(val_records_df)

        train_records_df = remap_data(train_records_df, remap_column_group, value_dicts)
        val_records_df = remap_data(val_records_df, remap_column_group, value_dicts)

        train_records_df["is_workday"] = train_records_df["dayofweek"].apply(
            lambda x: 0 if x in ["6", "7"] else 1
        )
        val_records_df["is_workday"] = val_records_df["dayofweek"].apply(
            lambda x: 0 if x in ["6", "7"] else 1
        )

        # TODO
        train_records_df.drop(columns=self.city_columns + self.dayofweek_columns + ['home_area_code', 'called_home_code'] + self.province_columns, axis=1, inplace=True)
        val_records_df.drop(columns=self.city_columns + self.dayofweek_columns + ['home_area_code', 'called_home_code'] + self.province_columns, axis=1, inplace=True)

        apply_cols = {
            col: len(value_dicts[group])
            for group, columns in remap_column_group.items()
            for col in columns
        }
        print(apply_cols)
        train_records_df = apply_onehot(train_records_df, apply_cols)
        val_records_df = apply_onehot(val_records_df, apply_cols)

        train_records_seq_ids = gen_seq_ids(train_records_df)
        val_records_seq_ids = gen_seq_ids(val_records_df)

        train_seq_index_with_time_diff, train_seq_msisdns = split_seq_and_time_diff(
            train_records_df, train_records_seq_ids
        )
        val_seq_index_with_time_diff, val_seq_msisdns = split_seq_and_time_diff(
            val_records_df, val_records_seq_ids
        )


        train_records_df.drop(columns=["msisdn", "start_time", "other_party", "open_datetime"], axis=1, inplace=True)
        val_records_df.drop(columns=["msisdn", "start_time", "other_party", "open_datetime"], axis=1, inplace=True)

        for i in range(len(train_static_columns)):
            assert train_static_columns[i] == val_static_columns[i]

        train_records_df, val_records_df = apply_scaler(
            [train_records_df, val_records_df],
            ["call_duration", "cfee", "lfee", "hour", "open_count"] + train_static_columns,
        )

        train_labels_df = (
            train_labels_df.set_index("msisdn").loc[train_seq_msisdns].reset_index()
        )
        train_labels_df = pd.get_dummies(
            train_labels_df["is_sa"], columns=["is_sa"]
        ).astype("int")

        examine_data(train_records_df)
        examine_data(val_records_df)

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
        )

    @staticmethod
    def load_records(path: str) -> pd.DataFrame:
        dtypes = {
            "msisdn": "str",
            **CallRecords.datetime_columns_type,
            **CallRecords.numeric_columns_type,
            **CallRecords.area_code_columns_type,
            **CallRecords.city_columns_type,
            **CallRecords.province_columns_type,
            **CallRecords.a_product_id_columns_type,
            **CallRecords.categorical_columns_type,
            **CallRecords.dayofweek_columns_type,
        }
        usecols = (
            ["msisdn"]
            + CallRecords.datetime_columns
            + CallRecords.numeric_columns
            + CallRecords.area_code_columns
            + CallRecords.city_columns
            + CallRecords.province_columns
            + CallRecords.a_product_id_columns
            + CallRecords.categorical_columns
            + CallRecords.dayofweek_columns
            + CallRecords.other_party_columns
        )

        df = pd.read_csv(
            path,
            sep=",",
            usecols=usecols,
            dtype=dtypes,
        )

        for col in CallRecords.datetime_columns:
            df[col] = pd.to_datetime(
                df[col], format=CallRecords.time_format[col], errors="coerce"
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
   