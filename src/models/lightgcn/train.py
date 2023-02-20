from pytorch_lightning.cli import LightningCLI

from src.models.lightgcn.loader import DataModule
from src.models.lightgcn.model import Model


class CustomCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            'data.num_users',
            'model.num_users',
            apply_on="instantiate"
        )
        parser.link_arguments(
            'data.num_recipes',
            'model.num_recipes',
            apply_on="instantiate"
        )


if __name__ == '__main__':
    CustomCLI(Model, DataModule)
