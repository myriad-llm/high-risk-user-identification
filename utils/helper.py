from collections import defaultdict
from typing import Dict, List, Tuple
from dataclasses import is_dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from .augmentation import Augmentation
from jsonargparse import Namespace

def generate_value_dict(
    columns: list[str], data: pd.DataFrame, valid: pd.DataFrame
) -> Dict[str, int]:
    return {
        # for embedding, 0 is reserved for padding(padding_idx=0)
        v: k+1
        for k, v in enumerate(
            pd.unique(pd.concat([data[columns], valid[columns]]).values.ravel())
        )
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


def pad_collate(batch):
    # HACK: all dataset getitem should return dataclass. temp type check for compatibility.
    if isinstance(batch[0], tuple):
        seqs, time_diffs, labels, msisdns, seq_lens = zip(*batch)
    elif is_dataclass(batch[0]):
        seqs, time_diffs, labels, msisdns, seq_lens = zip(
            *[
                (item.records, item.time_diff, item.labels, item.msisdn, item.records_len)
                for item in batch
            ]
        )


    seq_padded = pad_sequence(seqs, batch_first=True)
    time_diff_padded = pad_sequence(time_diffs, batch_first=True)
    if labels[0] is not None:
        labels = torch.stack(labels)
    seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
    return seq_padded, time_diff_padded, labels, msisdns, seq_lens


def apply_onehot(df: pd.DataFrame, apply_cols: Dict[str, int]) -> pd.DataFrame:
    for apply_col, onehot_len in tqdm(apply_cols.items(), desc="Applying One-Hot"):
        temp_one_hot = pd.DataFrame(
            columns=[f"{apply_col}_{j}" for j in range(onehot_len)],
            data=np.eye(onehot_len)[df[apply_col].values],
        )
        temp_one_hot = temp_one_hot.astype(pd.SparseDtype("int8", 0))
        df.drop([apply_col], axis=1, inplace=True)
        df = pd.concat([df, temp_one_hot], axis=1)
    return df


def add_open_count(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(
        df,
        df.groupby("msisdn")["open_datetime"].nunique().reset_index(name="open_count"),
        on="msisdn",
    )
    df["open_count"] = df["open_count"].astype("int32")
    df.drop(columns=["open_datetime"], axis=1, inplace=True)
    return df


def gen_seq_ids(df: pd.DataFrame) -> Dict[str, List[int]]:
    seq_ids = defaultdict(list)
    for idx, msisdn in enumerate(df["msisdn"]):
        seq_ids[msisdn].append(idx)
    seq_ids = dict(sorted(seq_ids.items(), key=lambda x: len(x[1]), reverse=True))

    for msisdn, ids in seq_ids.items():
        assert len(ids) == ids[-1] - ids[0] + 1, f"not sorted by msisdn"
    temp_start_time = df["start_time"]
    for msisdn, ids in seq_ids.items():
        assert temp_start_time.iloc[
            ids
        ].is_monotonic_increasing, f"{msisdn} is not sorted by start_time"
    return seq_ids


def split_seq_and_time_diff(
    df: pd.DataFrame,
    seq_ids: Dict[str, List[int]],
) -> Tuple[
    List[Tuple[List[int], str, torch.Tensor]],
    List[str],
]:
    seq_msisdn = [df["msisdn"].iloc[indices[0]] for indices in seq_ids.values()]
    seq_index_with_time_diff = [
        (
            indices,
            df["msisdn"].iloc[indices[0]],
            torch.tensor(
                [0]
                + [
                    df["start_time"].iloc[indices[j]]
                    - df["start_time"].iloc[indices[j - 1]]
                    for j in range(1, len(indices))
                ],
                dtype=torch.float32,
            ),
        )
        for indices in tqdm(seq_ids.values(), desc="Splitting Sequence and Time Diff")
    ]
    for seq, msisdn, time_diff in seq_index_with_time_diff:
        assert all(i >= 0 for i in time_diff), "time_diff should be positive"
        assert len(seq) == len(time_diff), "seq and time_diff length mismatch"

    return seq_index_with_time_diff, seq_msisdn


def apply_scaler(
    dataframes: List[pd.DataFrame], columns: List[str]
) -> List[pd.DataFrame]:
    df = pd.concat(dataframes)
    for col in columns:
        scaler = MinMaxScaler()
        df[col] = scaler.fit(df[col].values.reshape(-1, 1))
        for dataframe in dataframes:
            dataframe[col] = scaler.transform(dataframe[col].values.reshape(-1, 1))
    return dataframes


def examine_data(df: pd.DataFrame) -> None:
    for col in df.columns:
        assert not df[col].isnull().values.any(), f"{col} has null values"
        assert not df[col].isna().values.any(), f"{col} has na values"
        assert df[col].dtype != "object", f"{col} is object type"
        assert df[col].max() <= 1 and df[col].min() >= 0, f"{col} is not scaled"


def add_static_features(df: pd.DataFrame, groupby_column: str="msisdn") -> pd.DataFrame:
    # static
    static_data = df.groupby(groupby_column).agg(
        account_person_num=('other_party', 'nunique'),
        called_area_num=('called_code', 'nunique'),
        call_num=('a_serv_type', lambda x: x[x.isin(['01', '03'])].count()),
        called_num=('a_serv_type', lambda x: x[x == '02'].count()),
        # phone1_combined_area_code_num=('phone1_combined_area_code', 'nunique'),
        magic_dayofweek=('dayofweek', lambda x: x.value_counts().mean()),
        call_duration_quantile=('call_duration', lambda x: x.quantile(0.60)),
        work_day_num=('dayofweek', lambda x: x[x.isin(['1', '2', '3', '4', '5'])].count()),
        weekend_num=('dayofweek', lambda x: x[x.isin(['6', '7'])].count()),
        open_count = ('open_datetime', 'nunique'),
    )

    temp = df.groupby(['msisdn', 'other_party']).size().reset_index(name='call_count')
    static_call_count_quantile_60 = temp.groupby('msisdn')['call_count'].quantile(0.6).reset_index(name='call_count_quantile_60')
    static_data = static_data.merge(static_call_count_quantile_60, on='msisdn', how='left')
    static_columns = static_data.columns.to_list()
    static_columns.remove('msisdn')
    return df.merge(static_data, on=groupby_column, how='left'), static_columns

def augment(train_data, train_labels, ratio_range, times):
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
    train_data = pd.concat([train_data, addition_train_data], ignore_index=True).reset_index(drop=True)
    train_labels = pd.concat([train_labels, addition_train_labels], ignore_index=True).reset_index(drop=True)
    # 按照 msisdn, start_time 排序
    train_data.sort_values(by=["msisdn", "start_time"]).reset_index(drop=True)
    train_labels.sort_values(by=["msisdn"]).reset_index(drop=True)

    return train_data, train_labels


def convert_to_namespace(d):
        if isinstance(d, dict):
            return Namespace(**{k: convert_to_namespace(v) for k, v in d.items()})
        return d