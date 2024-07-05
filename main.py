from lightning.pytorch.cli import LightningCLI

from models import LSTM, VAE, LSTM_VAE
from datasets import CallRecordsDataModule


def cli_main():
    cli = LightningCLI()


if __name__ == '__main__':
    cli_main()
