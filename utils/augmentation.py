import pandas as pd
import numpy as np
from typing import List, Tuple, Callable, Union


class Augmentation:
    """
    Data augmentation is performed by grouping the data by msisdn, for example, randomly masking a part of the data for msisdn A, and then adding it to the training set
    """

    def __init__(self, df: pd.DataFrame, label: int, id_column_name: str="msisdn", label_column_name: str = "is_sa"):
        """

        Args:
            df (pd.DataFrame): A sequence's all data, with has a id column named `id_column`, and without the label column.
            label (int): The sequence's label
            id_column (str): The id column name
            label_column (str): The label column name, which is used to store the label of the new sequences
        """
        self.df = df.drop(columns=[id_column_name])
        self.label = label
        self.call_count = 0
        self.id = df[id_column_name].iloc[0]
        self.id_column_name = id_column_name
        self.label_column_name = label_column_name

    def count_calls(func: Callable):
        def wrapper(self, *args, **kwargs):
            self.call_count += 1
            return func(self, *args, **kwargs)

        return wrapper

    def times(
        self,
        times: int,
        method: str,
        **method_kwargs,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """

        Args:
            times (int): The new sequence's number
            method (str): The method name, must be same as the method name in the class
            method_kwargs: The method's kwargs

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: res_dfs, res_labels. res_dfs is all the new sequences, res_labels is the labels of the new sequences: `res_labels[[id_column, label_name]]`, n * 2
        """
        
        res_dfs = []
        for i in range(times):
            method_func = getattr(self, method)
            assert method_func is not None, f"Method {method} not found"
            res_df, label = method_func(**method_kwargs)
            if res_df is None and label is None:
                continue
            else:
                res_dfs.append(res_df)

        if len(res_dfs) == 0:
            return None, None

        res_dfs = pd.concat(res_dfs)
        unique_ids = res_dfs[self.id_column_name].unique()
        res_labels = pd.DataFrame([self.label] * len(unique_ids), columns=[self.label_column_name])
        res_labels = pd.concat(
            [pd.DataFrame(unique_ids, columns=[self.id_column_name]), res_labels], axis=1
        )
        return res_dfs, res_labels

    @count_calls
    def mask(self, ratio)-> Tuple[pd.DataFrame, int]:
        """

        Args:
            ratio (Union[float, List[float], Tuple[float, float]]): if `ratio` is a list, then randomly choose a ratio from the list. if `ratio` is a number, then the ratio is the same every time. if `ratio` is a tuple, then randomly choose a ratio between ratio[0] and ratio[1]


        Raises:
            ValueError: if ratio is not a valid type of float, list, tuple

        Returns:
            Tuple[pd.DataFrame, str]: res_df, label. res_df is the new sequence, id is the first col of res_df, label is the label of the new sequence.
        """
        if type(ratio) == list:
            ratio = np.random.choice(ratio)
        elif type(ratio) == tuple:
            ratio = np.random.uniform(ratio[0], ratio[1])
        elif type(ratio) == int or type(ratio) == float:
            ratio = ratio
        else:
            raise ValueError(f"Invalid ratio type: {type(ratio)}")

        if int(ratio * self.df.shape[0]) < 1:
            return None, None

        num_rows_to_mask = int(ratio * self.df.shape[0])
        mask_indices = np.random.choice(
            self.df.index, size=num_rows_to_mask, replace=False
        )
        mask = self.df.index.isin(mask_indices)
        new_df = self.df[~mask].reset_index(drop=True)
        new_id = self.id + f"_{self.call_count}"
        ids = pd.DataFrame([new_id] * new_df.shape[0], columns=[self.id_column_name])
        res_df = pd.concat([ids, new_df], axis=1, ignore_index=False)
        return res_df, self.label

    @count_calls
    def interpolation(self, label, ratio):
        """
        Randomly insert rows with a ratio of ratio
        """
        pass

    @count_calls
    def noise(self, label, ratio):
        """
        """
        # for col in self.df.columns:
        #     if col in self.numeric_columns:
        #         self.df[col] *= np.random.uniform(1-ratio, 1+ratio)
        # return self.df, self.label, self.id
        pass

    @count_calls
    def time_smoothing(self, label, ratio):
        """ """
        pass

    @count_calls
    def sliding_window(self, window_size: int, step_size: int):
        """
        Create new samples using a sliding window approach.

        Args:
            window_size (int): The size of the sliding window.
            step_size (int): The step size for sliding the window.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: res_dfs, res_labels. res_dfs is all the new sequences, res_labels is the labels of the new sequences: `res_labels[[id_column, label_name]]`, n * 2
        """
        res_dfs = []
        total_length = len(self.df)

        # Apply sliding window
        for start in range(0, total_length - window_size + 1, step_size):
            end = start + window_size
            window_df = self.df.iloc[start:end].reset_index(drop=True)
            if window_df.empty:
                continue
            new_id = self.id + f"_{self.call_count}_{start//step_size+1}"
            ids = pd.DataFrame([new_id] * window_df.shape[0], columns=[self.id_column_name])
            res_df = pd.concat([ids, window_df], axis=1, ignore_index=False)
            res_dfs.append(res_df)

        if len(res_dfs) == 0:
            return None, None

        res_dfs = pd.concat(res_dfs)
        unique_ids = res_dfs[self.id_column_name].unique()
        res_labels = pd.DataFrame([self.label] * len(unique_ids), columns=[self.label_column_name])
        res_labels = pd.concat(
            [pd.DataFrame(unique_ids, columns=[self.id_column_name]), res_labels], axis=1
        )
        return res_dfs, res_labels

    @count_calls
    def max_pooling(self, window_size: int, step_size: int):
        """
        Apply max pooling to downsample the time series data by selecting the most frequent value in each column.

        Args:
            window_size (int): The size of the pooling window.
            step_size (int): The step size for sliding the window.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: res_dfs, res_labels. res_dfs is all the new sequences, res_labels is the labels of the new sequences: `res_labels[[id_column, label_name]]`, n * 2
        """
        res_dfs = []
        total_length = len(self.df)

        for start in range(0, total_length - window_size + 1, step_size):
            end = start + window_size
            window_df = self.df.iloc[start:end]
            if window_df.empty:
                continue

            # Find the most frequent value in each column
            pooled_df = window_df.mode().iloc[0].to_frame().T
            new_id = f"{self.id}_{self.call_count}"
            ids = pd.DataFrame([new_id] * pooled_df.shape[0], columns=[self.id_column_name])
            res_df = pd.concat([ids, pooled_df], axis=1, ignore_index=False)
            res_dfs.append(res_df)

        if len(res_dfs) == 0:
            return None, None

        res_dfs = pd.concat(res_dfs)
        unique_ids = res_dfs[self.id_column_name].unique()
        res_labels = pd.DataFrame([self.label] * len(unique_ids), columns=[self.label_column_name])
        res_labels = pd.concat(
            [pd.DataFrame(unique_ids, columns=[self.id_column_name]), res_labels], axis=1
        )
        return res_dfs, res_labels


if __name__ == "__main__":
    from tqdm import tqdm

    # 读取CSV文件
    train_data = pd.read_csv(
        "../data/raw/trainSet_res_with_distances.csv", dtype={"msisdn": "str"}
    )
    train_labels = pd.read_csv("../data/raw/trainSet_ans.csv", dtype={"msisdn": "str"})

    # 读取验证集
    validation_data = pd.read_csv(
        "../data/raw/validationSet_res_with_distances.csv", dtype={"msisdn": "str"}
    )

    # 转换时间格式
    train_data["start_time"] = pd.to_datetime(
        train_data["start_time"], format="%Y%m%d%H%M%S"
    )
    train_data["end_time"] = pd.to_datetime(
        train_data["end_time"], format="%Y%m%d%H%M%S"
    )
    train_data["open_datetime"] = pd.to_datetime(
        train_data["open_datetime"], format="%Y%m%d%H%M%S"
    )
    train_data["update_time"] = pd.to_datetime(train_data["update_time"])
    train_data["date"] = pd.to_datetime(train_data["date"])

    validation_data["start_time"] = pd.to_datetime(
        validation_data["start_time"], format="%Y%m%d%H%M%S"
    )
    validation_data["end_time"] = pd.to_datetime(
        validation_data["end_time"], format="%Y%m%d%H%M%S"
    )
    validation_data["open_datetime"] = pd.to_datetime(
        validation_data["open_datetime"], format="%Y%m%d%H%M%S", errors="coerce"
    )
    validation_data["update_time"] = pd.to_datetime(validation_data["update_time"])
    validation_data["date"] = pd.to_datetime(validation_data["date"])

    addition_train_data = []
    addition_train_labels = []

    times = 1
    ratio_range = 0.1
    window_size = 10  # 设置滑动窗口的大小
    step_size = 5  # 设置滑动窗口的步长
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
                # ratio=ratio_range,
                times=times * 4,
                window_size=window_size, step_size=step_size,
                method="max_pooling"
            )
        else:
            res_df, res_labels = aug.times(
                # ratio=ratio_range,
                times=times,
                window_size=window_size, step_size=step_size,
                method="max_pooling"
            )
        if res_df is not None and res_labels is not None:
            addition_train_data.append(res_df)
            addition_train_labels.append(res_labels)



            
    addition_train_data = pd.concat(addition_train_data)
    addition_train_labels = pd.concat(addition_train_labels)
    print("addition_train_data.shape is: ", addition_train_data.shape)
    addition_train_data.to_csv("addition_train_data.csv", index=False)  
