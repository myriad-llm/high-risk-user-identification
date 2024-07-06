## tlstm-impl

Put `trainSet_ans.csv`, `testSet_res.csv` and `validationSet_res.csv` files
at `data/CallRecords/raw/` directory.

At the first time to run this code, the raw data will be preprocessed and saved
at `data/CallRecords/processed/` directory.
After that, no longer need to process the raw data again.

Train:

```shell
CUDA_VISIBLE_DEVICES=2 python main.py fit --config ./configs/<name>.yaml
```

The logs and checkpoint will be saved at `./wandb/`