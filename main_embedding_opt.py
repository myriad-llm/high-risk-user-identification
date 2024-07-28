import os

import torch
from lightning.pytorch.cli import LightningCLI
import optuna
from jsonargparse import Namespace
import wandb

torch.set_float32_matmul_precision("medium")


class MyLightningCLI(LightningCLI):
    def __init__(self, update_config=None, *args, **kwargs):
        # update config
        if update_config is not None:
            self.update_config = update_config
        else:
            self.update_config = {}
        super().__init__(*args, **kwargs)

    def before_instantiate_classes(self) -> None:
        try:
            path = self.config.trainer.logger.init_args.save_dir
            os.makedirs(path, exist_ok=True)
        except (AttributeError, KeyError, TypeError):
            pass
        # update config
        if self.update_config:
            self.config.update(self.update_config)

        return super().before_instantiate_classes()

    def instantiate_classes(self) -> None:
        self.config.model.init_args.input_size = (
            1  # just need to add a arbitrary int value here
        )
        self.config.model.init_args.embedding_items_path = "fake"

        config_init = self.parser.instantiate_classes(self.config)
        # fake_datamodule = config_init.get(str(self.subcommand), config_init).get("data")
        fake_datamodule = config_init.get("data")

        if hasattr(fake_datamodule, "feature_dim"):
            self.config.model.init_args.input_size = fake_datamodule.feature_dim
        else:
            raise KeyError(
                "Attribution 'feature_dim' not found in datamodule, you must implement it in datamodule"
            )
        if hasattr(fake_datamodule, "embedding_items_path"):
            self.config.model.init_args.embedding_items_path = (
                fake_datamodule.embedding_items_path
            )
        else:
            raise KeyError(
                "Attribution 'embedding_items_path' not found in datamodule, you must implement it in datamodule"
            )
        return super().instantiate_classes()


def objective(trial: optuna.trial.Trial) -> float:
    hidden_dim = trial.suggest_categorical("hidden_dim", [2, 4, 8, 16, 32, 64])
    num_layers = trial.suggest_categorical("num_layers", [1, 2, 3])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    max_epochs = 150

    cli_update = {
        "data": {
            "init_args": {
                "batch_size": batch_size,
            }
        },
        "model": {
            "init_args": {
                "hidden_size": hidden_dim,
                "num_layers": num_layers,
                "optimizer": {"init_args": {"lr": lr}},
            }
        },
        "trainer": {"max_epochs": max_epochs},
    }

    def _convert_to_namespace(d):
        if isinstance(d, dict):
            return Namespace(**{k: _convert_to_namespace(v) for k, v in d.items()})
        return d

    cli_update = _convert_to_namespace(cli_update)

    # HACK: not valid for all logger
    # this is must when you use wandb logger，otherwise the optuna will use the same wandb run during the optimization.
    wandb.finish()

    # HACK: there is a problem, datamodule will be instantiated per execution. 
    # And the cache function in datamodule will cause the result that hyperparameter updates to the datamodule have no effect.
    cli = MyLightningCLI(
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "omegaconf"},
        run=False,
        update_config=cli_update,
    )
    # after instantiation, the config.data will be instantiated as cli.datamodule

    trainer, model, datamodule = cli.trainer, cli.model, cli.datamodule
    trainer.fit(model, datamodule=datamodule)

    monitor = "valid_BinaryF1Score"

    return trainer.callback_metrics[monitor].item()


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=2)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
