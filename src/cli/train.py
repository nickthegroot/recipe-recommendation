from pytorch_lightning.cli import LightningCLI

from src.models.lightgcn.loader import DataModule

# Needed for LightningCLI to find
import src.models.lightgcn.model


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
    CustomCLI(datamodule_class=DataModule)
