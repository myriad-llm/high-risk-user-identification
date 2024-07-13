from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

def generate_value_dict(
    columns: list[str], data: pd.DataFrame, valid: pd.DataFrame
) -> Dict[str, int]:
    return {
        v: k
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
    seqs, time_diffs, labels, msisdns, seq_lens = zip(*batch)

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
        temp_one_hot = temp_one_hot.astype(pd.SparseDtype("int", 0))
        df.drop([apply_col], axis=1, inplace=True)
        df = pd.concat([df, temp_one_hot], axis=1)
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
        df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
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
        # 地区总数
        # phone1_combined_area_code_num=('phone1_combined_area_code', 'nunique'),
        # 我不知道含义
        magic_dayofweek=('dayofweek', lambda x: x.value_counts().mean()),
        # 前 60% 通话时长
        call_duration_quantile=('call_duration', lambda x: x.quantile(0.60)),
        # 最长通话时间
        # 工作日数量
        work_day_num=('dayofweek', lambda x: x[x.isin(['1', '2', '3', '4', '5'])].count()),
        # 周末数量
        weekend_num=('dayofweek', lambda x: x[x.isin(['6', '7'])].count()),
        # 开户次数
        open_count = ('open_datetime', 'nunique'),
    )

    temp = df.groupby(['msisdn', 'other_party']).size().reset_index(name='call_count')
    static_call_count_quantile_60 = temp.groupby('msisdn')['call_count'].quantile(0.6).reset_index(name='call_count_quantile_60')
    static_data = static_data.merge(static_call_count_quantile_60, on='msisdn', how='left')
    static_columns = static_data.columns.to_list()
    static_columns.remove('msisdn')
    return df.merge(static_data, on=groupby_column, how='left'), static_columns