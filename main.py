import os

from lightning.pytorch.cli import LightningCLI


class MyLightningCLI(LightningCLI):
    def before_instantiate_classes(self) -> None:
        try:
            path = self.config.trainer.logger.init_args.save_dir
            os.makedirs(path, exist_ok=True)
        except (AttributeError, KeyError, TypeError):
            pass
        return super().before_instantiate_classes()

    def instantiate_classes(self) -> None:
        self.config[self.subcommand].model.init_args.input_size = 0 # just need to add a arbitrary int value here
        config_init = self.parser.instantiate_classes(self.config)
        fake_datamodule = config_init.get(str(self.subcommand), config_init).get('data')
        if hasattr(fake_datamodule, 'feature_dim'):
            self.config[self.subcommand].model.init_args.input_size = fake_datamodule.feature_dim
        else:
            raise KeyError("Attribution 'feature_dim' not found in datamodule, you must implement it in datamodule")
        return super().instantiate_classes()


def cli_main():
    cli = MyLightningCLI(save_config_kwargs={"overwrite": True}, parser_kwargs={"parser_mode": "omegaconf"})


if __name__ == "__main__":
    cli_main()
