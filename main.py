from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import WandbLogger

from models import LSTM, VAE, LSTM_VAE
from datasets import CallRecordsDataModule, CallRecords4VAEDataModule


def cli_main():
    WandbLogger()
    cli = LightningCLI()


if __name__ == '__main__':
    cli_main()
