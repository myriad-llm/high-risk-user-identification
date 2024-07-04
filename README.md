## tlstm-impl

Put `trainSet_ans.csv`, `testSet_res.csv` and `validationSet_res.csv` files
at `data/CallRecords/raw/` directory.

At the first time to run this code, the raw data will be preprocessed and saved
at `data/CallRecords/processed/` directory.
After that, no longer need to process the raw data again.

```shell
CUDA_VISIBLE_DEVICES=2 python main.py --lr=0.001 --data="./data" --batch_size=128 --epoch=100
```

The checkpoint will be saved at `./checkpoint/` directory.

```shell
python out.py --ckpt="./checkpoint/ckpt.pth" --data="./data" --batch_size=128
```

The output will be saved at `./res/` directory.
