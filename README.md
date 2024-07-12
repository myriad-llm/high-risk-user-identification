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

if you want to make the `feature_num` in the config file self-adapting to the dataset, you must define the `feature_num` attribute in the datamodule to return the feature number.

```python
class CallRecordsDataModule(pl.LightningDataModule):
    @property
    def feature_num(self):
        ...
        return feature_num
```

Correspondingly, you must add the `input_size` parameter in the model.

```python
class MyModel(LightningModule):
    def __init__(self, input_size, ...):
        super().__init__()
        self.input_size = input_size
        ...
```

If `feature_num` not exist in the config file, it will return an error like this:

```shell
 "C:\Users\shiwenbo\mambaforge\envs\pytorch\lib\site-packages\torch\nn\modules\rnn.py", line 117, in __init__
w_ih = Parameter(torch.empty((gate_size, layer_input_size), **factory_kwargs))
TypeError: empty(): argument 'size' failed to unpack the object at pos 2 with error "type must be tuple of ints,but got NoneType"
```