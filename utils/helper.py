from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


def pad_collate(batch):
    seq, time_diff, labels = zip(*batch)

    seq_padded = pad_sequence(seq, batch_first=True)
    time_diff_padded = pad_sequence(time_diff, batch_first=True)
    labels = torch.stack(labels)
    return seq_padded, time_diff_padded, labels


def apply_onehot(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    onehot_df = pd.get_dummies(df, columns=columns, dummy_na=False)
    for col in onehot_df.columns:
        if onehot_df[col].dtype == 'bool':
            onehot_df[col] = onehot_df[col].astype('int8')
    return onehot_df


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
    List[Tuple[List[int], torch.Tensor]],
    List[str],
]:
    seq_msisdn = [df["msisdn"].iloc[indices[0]] for indices in seq_ids.values()]
    seq_index_with_time_diff = [
        (
            indices,
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
    for seq, time_diff in seq_index_with_time_diff:
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
