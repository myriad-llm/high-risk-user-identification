import os

from lightning.pytorch.cli import LightningCLI


class MyLightningCLI(LightningCLI):
    def before_instantiate_classes(self) -> None:
        path = self.config_dump["trainer"]["logger"]["init_args"]["save_dir"]
        os.makedirs(path, exist_ok=True)
        return super().before_instantiate_classes()


def cli_main():
    cli = MyLightningCLI(save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    cli_main()
