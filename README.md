## high-risk-user-identification

### Datasets

```shell
mkdir -p data/CallRecords/raw
```

Put `trainSet_ans.csv`, `testSet_res.csv` and `validationSet_res.csv` files
at `data/CallRecords/raw/` directory.

At the first time to run this code, the raw data will be preprocessed and saved
at `data/CallRecords/processed/` directory.
After that, no longer need to process the raw data again.

dataset `call_records` return:

seq: BATCH * SEQ_LEN * FEATURE
time_diff: BATCH * SEQ_LEN
labels: BATCH * CLASS_NUM  

Train:

```shell
CUDA_VISIBLE_DEVICES=2 python main.py fit --config ./configs/<name>.yaml
```

The logs and checkpoint will be saved at `./wandb/`

Predict:

Before predict, you need to modify the `ckpt_path` in the config file to determine which checkpoint to use.

```shell
CUDA_VISIBLE_DEVICES=2 python main.py predict --config ./configs/<name>.yaml
```