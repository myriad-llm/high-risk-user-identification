import os
import pickle as pkl
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from tqdm import tqdm


def generate_value_dict(columns: list[str], data: pd.DataFrame, valid: pd.DataFrame) -> Dict[str, int]:
    return {
        v: k
        for k, v in enumerate(pd.unique(
            pd.concat([data[columns], valid[columns]]).values.ravel()
        ))
    }


def remap_data(
        df: pd.DataFrame,
        column_group: Dict[str, List[str]],
        value_dicts: Dict[str, Dict[str, int]],
) -> pd.DataFrame:
    for group, columns in column_group.items():
        for col in columns:
            df[col] = df[col].apply(lambda x: value_dicts[group][x])

    return df


def apply_onehot(df: pd.DataFrame, apply_cols: Dict[str, int]) -> pd.DataFrame:
    for apply_col, onehot_len in tqdm(apply_cols.items(), desc='Applying One-Hot'):
        temp_one_hot = pd.DataFrame(
            columns=[f'{apply_col}_{j}' for j in range(onehot_len)],
            data=np.eye(onehot_len)[df[apply_col].values],
        )
        temp_one_hot = temp_one_hot.astype(pd.SparseDtype('int', 0))
        df.drop([apply_col], axis=1, inplace=True)
        df = pd.concat([df, temp_one_hot], axis=1)
    return df


def add_open_count(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(
        df,
        df.groupby('msisdn')['open_datetime'].nunique().reset_index(name='open_count'),
        on='msisdn',
    )
    df['open_count'] = df['open_count'].astype('int32')
    df.drop(columns=['open_datetime'], axis=1, inplace=True)
    return df


def gen_seq_ids(df: pd.DataFrame) -> Dict[str, List[int]]:
    seq_ids = defaultdict(list)
    for idx, msisdn in enumerate(df['msisdn']):
        seq_ids[msisdn].append(idx)
    seq_ids = dict(sorted(seq_ids.items(), key=lambda x: len(x[1]), reverse=True))

    for msisdn, ids in seq_ids.items():
        assert len(ids) == ids[-1] - ids[0] + 1, f'not sorted by msisdn'
    temp_start_time = df['start_time']
    for msisdn, ids in seq_ids.items():
        assert temp_start_time.iloc[ids].is_monotonic_increasing, f'{msisdn} is not sorted by start_time'
    return seq_ids


def split_seq_and_time_diff(
        df: pd.DataFrame,
        seq_ids: Dict[str, List[int]],
) -> Tuple[
    List[Tuple[List[int], torch.Tensor]],
    List[str],
]:
    seq_msisdn = [df['msisdn'].iloc[indices[0]] for indices in seq_ids.values()]
    seq_index_with_time_diff = [
        (indices, torch.tensor([0] + [
            df['start_time'].iloc[indices[j]] - df['start_time'].iloc[indices[j - 1]]
            for j in range(1, len(indices))
        ], dtype=torch.float32))
        for indices in tqdm(seq_ids.values(), desc='Splitting Sequence and Time Diff')
    ]
    for seq, time_diff in seq_index_with_time_diff:
        assert all(i >= 0 for i in time_diff), 'time_diff should be positive'
        assert len(seq) == len(time_diff), 'seq and time_diff length mismatch'

    return seq_index_with_time_diff, seq_msisdn


def apply_scaler(dataframes: List[pd.DataFrame], columns: List[str]) -> List[pd.DataFrame]:
    df = pd.concat(dataframes)
    for col in columns:
        scaler = MinMaxScaler()
        df[col] = scaler.fit(df[col].values.reshape(-1, 1))
        for dataframe in dataframes:
            dataframe[col] = scaler.transform(dataframe[col].values.reshape(-1, 1))
    return dataframes


def split_data(
        labels: torch.Tensor,
        seq_index_with_time_diff: List[Tuple[List[int], torch.Tensor]],
        ratio: float,
) -> Tuple[
    Tuple[torch.Tensor, List[Tuple[List[int], torch.Tensor]]],
    Tuple[torch.Tensor, List[Tuple[List[int], torch.Tensor]]],
]:
    paired = list(zip(seq_index_with_time_diff, labels))
    np.random.shuffle(paired)
    train_size = int(len(paired) * ratio)
    train_paired, test_paired = paired[:train_size], paired[train_size:]
    # 解包配对以分离数据和标签
    train_seq_index_with_time_diff, train_labels = zip(*train_paired)
    test_seq_index_with_time_diff, test_labels = zip(*test_paired)
    return (
        (torch.stack(train_labels), train_seq_index_with_time_diff),
        (torch.stack(test_labels), test_seq_index_with_time_diff),
    )


class CallRecords(Dataset):
    resources = [
        ("trainSet_res.csv", "trainSet_ans.csv"),
        ("validationSet_res.csv", None),
    ]

    manifests = [
        "records.pt",
        "train_labels.pt",
        "train_seq_index_with_time_diff.pkl",
        "test_labels.pt",
        "test_seq_index_with_time_diff.pkl",
        "val_records.pt",
        "val_seq_index_with_time_diff.pkl",
    ]

    time_format: Dict[str, str] = {
        'start_time': '%Y%m%d%H%M%S',
        'open_datetime': '%Y%m%d%H%M%S',
    }
    datetime_columns_type: Dict[str, str] = {key: 'str' for key in time_format}
    datetime_columns: List[str] = list(time_format.keys())

    numeric_columns_type: Dict[str, str] = {
        'call_duration': 'int32',
        'cfee': 'int32',
        'lfee': 'int32',
        'hour': 'int8',
    }
    numeric_columns: List[str] = list(numeric_columns_type.keys())

    area_code_columns: List[str] = ['home_area_code', 'visit_area_code', 'called_home_code', 'called_code']
    area_code_columns_type: Dict[str, str] = {key: 'str' for key in area_code_columns}

    city_columns: List[str] = ['phone1_loc_city', 'phone2_loc_city']
    city_columns_type: Dict[str, str] = {key: 'str' for key in city_columns}

    province_columns: List[str] = ['phone1_loc_province', 'phone2_loc_province']
    province_columns_type: Dict[str, str] = {key: 'str' for key in province_columns}

    a_product_id_columns: List[str] = ['a_product_id']
    a_product_id_columns_type: Dict[str, str] = {key: 'str' for key in a_product_id_columns}

    categorical_columns: List[str] = [
        'a_serv_type', 'long_type1', 'roam_type',
        'dayofweek', 'phone1_type', 'phone2_type',
    ]
    categorical_columns_type: Dict[str, str] = {key: 'category' for key in categorical_columns}

    def __init__(
            self,
            root: Union[str, Path],
            train: bool = True,
            valid: bool = False,
            non_seq: bool = False,
    ) -> None:
        if isinstance(root, str):
            root = os.path.expanduser(root)
        self.root = root

        self.train = train
        self.valid = valid
        self.non_seq = non_seq

        def time_map(x):
            x / 60
            return 1 / torch.log(x + torch.e)

        if self._check_legacy_exists():
            self.records, self.labels, self.seq_index_with_time_diff = self._load_legacy_data()

            self.seq_index_with_time_diff = [(seq, time_map(time_diff)) for seq, time_diff in self.seq_index_with_time_diff]
            return

        if not self._check_exists():
            raise RuntimeError("Dataset not found.")

        data, val = self._load_data()
        data_records, data_labels, data_seq_index_with_time_diff = data
        (
            train_labels_and_seq_index_with_time_diff,
            test_labels_and_seq_index_with_time_diff,
        ) = split_data(data_labels, data_seq_index_with_time_diff, 0.8)

        self._save_legacy_data(
            (data_records, train_labels_and_seq_index_with_time_diff, test_labels_and_seq_index_with_time_diff),
            val,
        )

        self.records, self.labels, self.seq_index_with_time_diff = data if self.train else val
        self.seq_index_with_time_diff = [(seq, time_map(time_diff)) for seq, time_diff in self.seq_index_with_time_diff]

    def __getitem__(
            self,
            index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]] | torch.Tensor:
        if self.non_seq:
            return self.records[index]

        seq_index, time_diff = self.seq_index_with_time_diff[index]
        seq, label = self.records[seq_index], self.labels[index] if self.labels is not None else None

        return seq, time_diff, label

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

    def _load_legacy_data(self) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Tuple[List[int], torch.Tensor]]]:
        records_file = 'val_records.pt' if self.valid else 'records.pt'
        labels_file = None if self.valid else 'train_labels.pt' if self.train else 'test_labels.pt'
        seq_file = 'val_seq_index_with_time_diff.pkl' if self.valid \
            else 'train_seq_index_with_time_diff.pkl' if self.train else 'test_seq_index_with_time_diff.pkl'

        records = torch.load(os.path.join(self.processed_folder, records_file))

        labels = None
        if labels_file:
            labels = torch.load(os.path.join(self.processed_folder, labels_file))

        with open(os.path.join(self.processed_folder, seq_file), 'rb') as f:
            seq_index_with_time_diff = pkl.load(f)

        return records.to_dense(), labels.to_dense() if labels is not None else None, seq_index_with_time_diff

    def _save_legacy_data(
            self,
            data: Tuple[
                torch.Tensor,
                Tuple[torch.Tensor, List[Tuple[List[int], torch.Tensor]]],
                Tuple[torch.Tensor, List[Tuple[List[int], torch.Tensor]]],
            ],
            val: Tuple[torch.Tensor, None, List[Tuple[List[int], torch.Tensor]]],
    ) -> None:
        if not os.path.exists(self.processed_folder):
            os.makedirs(self.processed_folder)

        def save_tensor(t: torch.Tensor, filename: str) -> None:
            path = os.path.join(self.processed_folder, filename)
            torch.save(t.to_sparse_coo(), path)

        def save_pickle(d: object, filename: str) -> None:
            with open(os.path.join(self.processed_folder, filename), 'wb') as f:
                pkl.dump(d, f, pkl.HIGHEST_PROTOCOL)

        (
            data_records,
            (train_labels, train_seq_index_with_time_diff),
            (test_labels, test_seq_index_with_time_diff),
        ) = data
        (val_records, _, val_seq_index_with_time_diff) = val

        save_tensor(data_records, 'records.pt')
        save_tensor(train_labels, 'train_labels.pt')
        save_pickle(train_seq_index_with_time_diff, 'train_seq_index_with_time_diff.pkl')
        save_tensor(test_labels, 'test_labels.pt')
        save_pickle(test_seq_index_with_time_diff, 'test_seq_index_with_time_diff.pkl')
        save_tensor(val_records, 'val_records.pt')
        save_pickle(val_seq_index_with_time_diff, 'val_seq_index_with_time_diff.pkl')

    def _check_exists(self) -> bool:
        return all(
            os.path.isfile(os.path.join(self.raw_folder, file))
            for data_pair in self.resources
            for file in data_pair if file is not None
        )

    def _load_data(self) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor, List[Tuple[List[int], torch.Tensor]]],
        Tuple[torch.Tensor, None, List[Tuple[List[int], torch.Tensor]]],
    ]:
        (train_records_df, train_labels_df), (val_records_df, _) = self._load_dataframes()

        remap_column_group = {
            'area_code': self.area_code_columns,
            # 'city': self.city_columns,
            'province': self.province_columns,
            'a_product_id': self.a_product_id_columns,
        }
        remap_column_group.update({
            col: [col]
            for col in self.categorical_columns
        })

        value_dicts = {
            group: generate_value_dict(columns, train_records_df, val_records_df)
            for group, columns in remap_column_group.items()
        }
        value_dicts.update({
            col: generate_value_dict([col], train_records_df, val_records_df)
            for col in self.categorical_columns
        })

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
        train_records_df = apply_onehot(train_records_df, apply_cols)
        val_records_df = apply_onehot(val_records_df, apply_cols)

        train_records_df = add_open_count(train_records_df)
        val_records_df = add_open_count(val_records_df)

        train_records_seq_ids = gen_seq_ids(train_records_df)
        val_records_seq_ids = gen_seq_ids(val_records_df)

        train_seq_index_with_time_diff, train_seq_msisdns = split_seq_and_time_diff(train_records_df, train_records_seq_ids)
        val_seq_index_with_time_diff, val_seq_msisdns = split_seq_and_time_diff(val_records_df, val_records_seq_ids)

        train_records_df.drop(columns=['msisdn', 'start_time'], axis=1, inplace=True)
        val_records_df.drop(columns=['msisdn', 'start_time'], axis=1, inplace=True)

        train_records_df, val_records_df = apply_scaler(
            [train_records_df, val_records_df],
            ['call_duration', 'cfee', 'lfee', 'hour', 'open_count'],
        )

        train_labels_df = train_labels_df.set_index('msisdn').loc[train_seq_msisdns].reset_index()
        train_labels_df = pd.get_dummies(train_labels_df['is_sa'], columns=['is_sa']).astype('int')

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
            'msisdn': 'str',
            **CallRecords.datetime_columns_type,
            **CallRecords.numeric_columns_type,
            **CallRecords.area_code_columns_type,
            **CallRecords.city_columns_type,
            **CallRecords.province_columns_type,
            **CallRecords.a_product_id_columns_type,
            **CallRecords.categorical_columns_type,
        }
        usecols = (
                ['msisdn'] + CallRecords.datetime_columns + CallRecords.numeric_columns
                + CallRecords.area_code_columns + CallRecords.city_columns + CallRecords.province_columns
                + CallRecords.a_product_id_columns + CallRecords.categorical_columns
        )

        df = pd.read_csv(
            path,
            sep=',',
            usecols=usecols,
            dtype=dtypes,
        )

        for col in CallRecords.datetime_columns:
            df[col] = pd.to_datetime(df[col], format=CallRecords.time_format[col], errors='coerce')

        df['start_time'] = df['start_time'].apply(lambda x: x.timestamp()).astype('int')

        return df

    @staticmethod
    def load_labels(path: str) -> pd.DataFrame:
        return pd.read_csv(
            path,
            sep=',',
            dtype={'msisdn': 'str', 'is_sa': 'bool'},
        )

    def _load_dataframes(self) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, None]]:
        # load all training and validation data
        dataframes = [
            (
                self.load_records(os.path.join(self.raw_folder, record)),
                self.load_labels(os.path.join(self.raw_folder, label)) if label is not None else None,
            )
            for record, label in self.resources
        ]

        train_dataframes = [
            (records, labels)
            for records, labels in dataframes
            if labels is not None
        ]
        val_dataframes = [
            records
            for records, labels in dataframes
            if labels is None
        ]

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


if __name__ == '__main__':
    CallRecords('../data')
