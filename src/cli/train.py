from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from src.models.lightgcn.loader import DataModule

# Needed for LightningCLI to find
from src.models.lightgcn.model import HeteroLGN


class CustomCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            'data.num_users',
            'model.init_args.num_users',
            apply_on="instantiate"
        )
        parser.link_arguments(
            'data.num_recipes',
            'model.init_args.num_recipes',
            apply_on="instantiate"
        )


if __name__ == '__main__':
    CustomCLI(
        datamodule_class=DataModule,
        trainer_defaults={
            'callbacks': [
                EarlyStopping(monitor='val_loss', mode='min', patience=5),
                ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, save_weights_only=True)
            ]
        }
    )
