import os

from lightning.pytorch.cli import LightningCLI

from callbacks import LSTM_VAE_PredictionWriter, LSTMPredictionWriter
from datasets import CallRecords4VAEDataModule, CallRecordsDataModule
from models import LSTM, LSTM_VAE, VAE


class MyLightningCLI(LightningCLI):
    def before_instantiate_classes(self) -> None:
        path = self.config_dump["trainer"]["logger"]["init_args"]["save_dir"]
        os.makedirs(path, exist_ok=True)
        return super().before_instantiate_classes()


def cli_main():
    cli = MyLightningCLI()


if __name__ == "__main__":
    cli_main()
