import os

import torch
from lightning.pytorch.cli import LightningCLI
import optuna
from jsonargparse import Namespace
import wandb
import yaml
from callbacks import PyTorchLightningPruningCallback

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

        config_init = self.parser.instantiate_classes(self.config)
        # fake_datamodule = config_init.get(str(self.subcommand), config_init).get("data")
        fake_datamodule = config_init.get("data")

        if hasattr(fake_datamodule, "feature_dim"):
            self.config.model.init_args.input_size = fake_datamodule.feature_dim
        else:
            raise KeyError(
                "Attribution 'feature_dim' not found in datamodule, you must implement it in datamodule"
            )
        return super().instantiate_classes()


def objective(trial: optuna.trial.Trial) -> float:
    monitor = "val-BinaryF1Score"

    try:
        # model
        hidden_size = trial.suggest_int("hidden_size", 18, 1260, step=18)
        num_hidden_layers = trial.suggest_int("num_layers", 1, 12, step=1)
        dropout_rate = trial.suggest_float("dropout_rate", 0, 0.5)
        lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
        weight_decay = trial.suggest_categorical("weight_decay", [1e-1, 1e-2, 1e-3, 1e-4, 1e-5])

        # data
        mlm_probability = trial.suggest_float("mlm_probability", 0, 0.5)
        seq_len = 24 # bert just allow the seq_len <= 512, 24 is the longest seq_len that can be used
        stride = seq_len - int(seq_len * trial.suggest_float("overlap", 0, 1))
        trial.set_user_attr("stride", stride)

        # trainer
        max_epochs = 5

        cli_update = {
            "data": {
                "init_args": {
                    "mlm_probability": mlm_probability,
                    "seq_len": seq_len,
                    "stride": stride,
                }
            },
            "model": {
                "init_args": {
                    "hidden_size": hidden_size,
                    "num_hidden_layers": num_hidden_layers,
                    "dropout_rate": dropout_rate,
                    "optimizer": {
                        "init_args": {
                            "lr": lr,
                            "weight_decay": weight_decay
                            }
                        },
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

        # add pruning callback
        model.set_callbacks([PyTorchLightningPruningCallback(trial, monitor=monitor)])

        trainer.fit(model, datamodule=datamodule)

        return trainer.callback_metrics[monitor].item()
    except Exception as e:
        print(f"Trial failed with exception: {e}")
        return float("-inf")


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    with open("best_params.yaml", "w") as f:
        yaml.dump(trial.params, f)
