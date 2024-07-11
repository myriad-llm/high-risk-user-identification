import os
import pickle as pkl
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import Dataset

from utils import *
from utils import logger

from .col_def import *
from .vocab import Vocabulary

log = logger.get_logger(
    name=__name__,
    level=logger.matching_logger_level("INFO"),
)


def read_records(path: str) -> pd.DataFrame:
    dtypes = {
        "msisdn": "str",
        **DTAETIME_COLUMNS_TYPE,
        **NUMERIC_COLUMNS_TYPE,
        **AREA_CODE_COLUMNS_TYPE,
        **A_PRODUCT_ID_COLUMNS_TYPE,
        **CATEGORICAL_COLUMNS_TYPE,
    }
    usecols = (
        ["msisdn"]
        + DTAETIME_COLUMNS
        + NUMERIC_COLUMNS
        + AREA_CODE_COLUMNS
        + A_PRODUCT_ID_COLUMNS
        + CATEGORICAL_COLUMNS
    )

    df = pd.read_csv(path, sep=",", usecols=usecols, dtype=dtypes)
    log.info(f"{path} is read.")

    return df


def read_labels(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=",",
        dtype={"msisdn": "str", "is_sa": "int"},
    )


def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


class CallRecordsFata(Dataset):
    resources = [
        ("trainSet_res.csv", "trainSet_ans.csv"),
        ("validationSet_res.csv", None),
    ]

    manifests = [
        "vocab.nb",
        "train.pkl",
    ]

    def __init__(
        self,
        root: Union[str, Path],
        mlm,
        seq_len=23,
        num_bins=10,
        flatten=False,
        stride=5,
        adap_thres=10**8,
        return_labels=False,
    ) -> None:
        if isinstance(root, str):
            root = os.path.expanduser(root)
        self.root = root

        self.mlm = mlm
        self.seq_len = seq_len
        self.num_bins = num_bins
        self.flatten = flatten
        self.records_stride = stride
        self.vocab = Vocabulary(adap_thres)

        self.data = None
        self.labels = None
        self.window_label = None
        self.ncols = None

        self.return_labels = return_labels

        # if self._check_legacy_exists():
        #     self._load_legacy_data()
        #     return

        if not self._check_exists():
            raise RuntimeError("Dataset not found.")

        self._load_data()
        # self._save_legacy_data()

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "processed")

    def __getitem__(self, index):
        if self.flatten:
            return_data = torch.tensor(self.data[index], dtype=torch.long)
        else:
            return_data = torch.tensor(self.data[index], dtype=torch.long).reshape(
                self.seq_len, -1
            )

        if self.return_labels:
            return_data = (
                return_data,
                torch.tensor(self.labels[index], dtype=torch.long),
            )

        return return_data

    def __len__(self):
        return len(self.data)

    def _check_exists(self) -> bool:
        return all(
            os.path.isfile(os.path.join(self.raw_folder, file))
            for data_pair in self.resources
            for file in data_pair
            if file is not None
        )

    def _check_legacy_exists(self) -> bool:
        if not os.path.exists(self.processed_folder):
            return False

        return all(
            os.path.isfile(os.path.join(self.processed_folder, file))
            for file in self.manifests
        )

    def _load_legacy_data(self) -> None:
        return

    def _save_legacy_data(self) -> None:
        path = os.path.join(self.processed_folder, "vocab.nb")
        log.info(f"saving vocab at {path}")
        self.vocab.save_vocab(path)

        path = os.path.join(self.processed_folder, "train.pkl")
        with open(path, "wb") as f:
            pkl.dump(
                {
                    "data": self.data,
                    "labels": self.labels,
                    "window_label": self.window_label,
                },
                f,
                pkl.HIGHEST_PROTOCOL,
            )

    def _load_data(self) -> None:
        train_records_df = read_records(
            os.path.join(self.raw_folder, "cleaned_trainSet_res.csv")
        )
        train_labels_df = read_labels(
            os.path.join(self.raw_folder, "trainSet_ans.csv"),
        )
        val_records_df = read_records(
            os.path.join(self.raw_folder, "cleaned_validationSet_res.csv")
        )
        train_len = len(train_records_df)

        df = pd.concat([train_records_df, val_records_df], ignore_index=True)
        msisdn2ids = {msisdn: idx for idx, msisdn in enumerate(df["msisdn"].unique())}

        df["msisdn"] = df["msisdn"].map(msisdn2ids)
        train_labels_df["msisdn"] = train_labels_df["msisdn"].map(msisdn2ids)

        df = self._encode_data(df)

        self._init_vocab(df)

        # split data
        train_records_df, val_records_df = (
            df[:train_len].copy(),
            df[train_len:].copy(),
        )
        del df
        # TODO： reindex 
        train_records_df.reset_index(drop=True, inplace=True)
        val_records_df.reset_index(drop=True, inplace=True)

        self.data, self.labels, self.window_label = self._prepare_samples(
            train_records_df, train_labels_df
        )

    def _encode_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: before fitting, generate static fields
        static_data = df.groupby("msisdn").agg(
            total_open_count=("open_datetime", "nunique"), # open_count
            contact_person_count=("other_party", "nunique"), # contact_person_count
            avg_call_duration=("call_duration", "mean"), # avg_call_duration
        )


        sub_columns = [
            AREA_CODE_COLUMNS,
            *[[col] for col in A_PRODUCT_ID_COLUMNS],
            *[[col] for col in CATEGORICAL_COLUMNS],
        ]

        log.info("label fit transform")
        for cols in tqdm.tqdm(sub_columns):
            # enc = LabelEncoder()
            # for col in tqdm.tqdm(cols, desc="fit", leave=True):
            #     enc.fit(df[col])
            # for col in tqdm.tqdm(cols, desc="transform", leave=True):
            #     df[col] = enc.transform(df[col])
            for col in cols:
                LabelEncoder().fit_transform(df[col])

        log.info("timestamp fit transform")
        for col in tqdm.tqdm(DTAETIME_COLUMNS):
            timestamp = pd.DataFrame(
                pd.to_datetime(
                    df[col],
                    format=TIME_FORMAT[col],
                    errors="coerce",
                ).astype(np.int64)
            )
            df[col] = MinMaxScaler().fit_transform(timestamp, timestamp)

        log.info("timestamp quant transform")
        for col in tqdm.tqdm(DTAETIME_COLUMNS):
            bin_edges, _, _ = self._quantization_binning(df[col].to_numpy())
            df[col] = self._quantize(df[col], bin_edges)

        log.info("numeric quant transform")
        for col in tqdm.tqdm(NUMERIC_COLUMNS):
            bin_edges, _, _ = self._quantization_binning(df[col].to_numpy())
            df[col] = self._quantize(df[col], bin_edges)

        return df

    def _init_vocab(self, df: pd.DataFrame) -> None:
        column_names = list(df.columns)

        self.vocab.set_field_keys(column_names)

        for column in column_names:
            unique_values = (
                df[column].value_counts(sort=True).to_dict()
            )  # returns sorted
            for val in unique_values:
                self.vocab.set_id(val, column)

        log.info(f"total columns: {list(column_names)}")
        log.info(f"total vocabulary size: {len(self.vocab.id2token)}")

        for column in self.vocab.field_keys:
            vocab_size = len(self.vocab.token2id[column])
            log.info(f"column : {column}, vocab size : {vocab_size}")

            if vocab_size > self.vocab.adap_thres:
                log.info(f"\tsetting {column} for adaptive softmax")
                self.vocab.adap_sm_cols.add(column)

    def _prepare_samples(
        self,
        train_records_df: pd.DataFrame,
        train_labels_df: pd.DataFrame,
    ) -> None:
        log.info("preparing user level data...")
        data, labels, window_label = [], [], []

        records_data, records_labels, columns_names = self._user_level_data(
            train_records_df, train_labels_df
        )

        log.info("creating call record samples with vocab")
        for user_idx in tqdm.tqdm(range(len(records_data))):
            user_row = records_data[user_idx]
            user_row_ids = self._format_records(user_row, columns_names)

            user_labels = records_labels[user_idx]

            bos_token = self.vocab.get_id(
                self.vocab.bos_token, special_token=True
            )  # will be used for GPT2
            eos_token = self.vocab.get_id(
                self.vocab.eos_token, special_token=True
            )  # will be used for GPT2
            for jdx in range(
                0, len(user_row_ids) - self.seq_len + 1, self.records_stride
            ):
                ids = user_row_ids[jdx : (jdx + self.seq_len)]
                ids = [idx for ids_lst in ids for idx in ids_lst]  # flattening
                if (
                    not self.mlm and self.flatten
                ):  # for GPT2, need to add [BOS] and [EOS] tokens
                    ids = [bos_token] + ids + [eos_token]
                data.append(ids)

            for jdx in range(
                0, len(user_labels) - self.seq_len + 1, self.records_stride
            ):
                ids = user_labels[jdx : (jdx + self.seq_len)]
                labels.append(ids)

                is_sa = 0
                if len(np.nonzero(ids)[0]) > 0:
                    is_sa = 1
                window_label.append(is_sa)

        assert len(data) == len(labels)
        """
            ncols = total fields - 1 (special tokens) - 1 (label)
            if bert:
                ncols += 1 (for sep)
        """
        self.ncols = len(self.vocab.field_keys) - 2 + (1 if self.mlm else 0)
        log.info(f"ncols: {self.ncols}")
        log.info(f"total samples: {len(data)}")

        return data, labels, window_label

    @staticmethod
    def label_fit_transform(column, enc_type="label"):
        if enc_type == "label":
            mfit = LabelEncoder()
        else:
            mfit = MinMaxScaler()
        mfit.fit(column)

        return mfit, mfit.transform(column)

    def _quantization_binning(self, data):
        qtls = np.arange(0.0, 1.0 + 1 / self.num_bins, 1 / self.num_bins)
        bin_edges = np.quantile(data, qtls, axis=0)  # (num_bins + 1, num_features)
        bin_widths = np.diff(bin_edges, axis=0)
        bin_centers = bin_edges[:-1] + bin_widths / 2  # ()
        return bin_edges, bin_centers, bin_widths

    def _quantize(self, inputs, bin_edges):
        quant_inputs = np.zeros(inputs.shape[0])
        for i, x in enumerate(inputs):
            quant_inputs[i] = np.digitize(x, bin_edges)
        quant_inputs = quant_inputs.clip(1, self.num_bins) - 1  # Clip edges
        return quant_inputs

    def _user_level_data(
        self,
        train_records: pd.DataFrame,
        train_labels: pd.DataFrame,
    ) -> Tuple[
        List[List[Union[int, float]]],
        List[List[int]],
        List[str],
    ]:
        train_labels_indexed = train_labels.set_index("msisdn")

        records_data, records_labels = [], []

        unique_users = train_records["msisdn"].unique()
        columns_names = list(train_records.columns)

        for user in tqdm.tqdm(unique_users):
            user_data = train_records.loc[train_records["msisdn"] == user]
            user_records, user_labels = [], []

            for _, row in user_data.iterrows():
                row = list(row)

                user_records.extend(row)
                user_labels.append(train_labels_indexed.loc[row[0], "is_sa"])

            records_data.append(user_records)
            records_labels.append(user_labels)

        return records_data, records_labels, columns_names

    def _format_records(self, records_lst, column_names):
        records_lst = list(
            divide_chunks(records_lst, len(self.vocab.field_keys) - 1)
        )  # 1 to ignore SPECIAL
        user_vocab_ids = []

        sep_id = self.vocab.get_id(self.vocab.sep_token, special_token=True)

        for records in records_lst:
            vocab_ids = []
            for jdx, field in enumerate(records):
                vocab_id = self.vocab.get_id(field, column_names[jdx])
                vocab_ids.append(vocab_id)

            # TODO : need to handle ncols when sep is not added
            if (
                self.mlm
            ):  # and self.flatten:  # only add [SEP] for BERT + flatten scenario
                vocab_ids.append(sep_id)

            user_vocab_ids.append(vocab_ids)

        return user_vocab_ids
