from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.cli import LightningCLI

# Needed for LightningCLI to find
from src.models.heterolgn import HeteroLGN
from src.models.loader import DataModule
from src.models.reclcn import RecLGN


class CustomCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "data.num_users", "model.init_args.num_users", apply_on="instantiate"
        )
        parser.link_arguments(
            "data.num_recipes", "model.init_args.num_recipes", apply_on="instantiate"
        )
        # parser.link_arguments(
        #     "data.recipe_feats", "model.init_args.recipe_feats", apply_on="instantiate"
        # )


if __name__ == "__main__":
    CustomCLI(
        datamodule_class=DataModule,
        trainer_defaults={
            "callbacks": [
                EarlyStopping(monitor="val_recall", mode="max", patience=5),
            ]
        },
    )
