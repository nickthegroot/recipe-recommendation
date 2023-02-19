from pathlib import Path

import click
import pytorch_lightning as pl
from torch_geometric.data import LightningLinkData

import src.config as Config
from src.data.get_data import get_data
from src.models.lightgcn import Model as LightGCN


@click.command()
@click.argument('processed_path', type=click.Path(exists=True), default=Config.PROCESSED_DATA_DIR)
@click.option('--model', type=click.Choice(['lightgcn']), required=True)
@click.option('--max-epochs', type=click.INT, default=10)
def main(processed_path: str, model: str, max_epochs: int):
    processed_path = Path(processed_path)
    data = get_data(processed_path)
    loader = LightningLinkData(
        data,
        input_train_edges=('user', 'reviews', 'recipe'),
        num_neighbors=[8, 4],
        neg_sampling='binary',
        batch_size=4096,
        drop_last=True,
    )

    if model == 'lightgcn':
        model = LightGCN(
            num_users=data['user'].x.size(0),
            num_recipes=data['recipe'].x.size(0),
            embedding_dim=64,
            num_layers=3,
        )
    else:
        raise NotImplementedError

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=pl.loggers.TensorBoardLogger('logs'),
    )
    trainer.fit(model, loader)
    trainer.save_checkpoint(
        Path(Config.MODEL_DIR) / f'{model}.ckpt',
        weights_only=True
    )


if __name__ == '__main__':
    main()
