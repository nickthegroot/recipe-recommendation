from pathlib import Path

import pytorch_lightning as pl
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader

from src import config as Config
from src.data.get_data import get_data


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = Config.PROCESSED_DATA_DIR,
        batch_size: int = 4096,
    ):
        super().__init__()
        self.batch_size = batch_size

        data = get_data(Path(data_dir))
        # LightGCN works only with user - item graphs, ignore higher order nodes
        del data['ingredient']
        del data['uses']
        del data['rev_uses']

        transform = T.RandomLinkSplit(
            num_val=0.1,
            num_test=0.1,
            edge_types=("user", "reviews", "recipe"),
            rev_edge_types=("recipe", "rev_reviews", "user"),
        )
        self.train_data, self.val_data, self.test_data = transform(data)

        self.num_users = self.train_data['user'].x.size(0)
        self.num_recipes = self.train_data['recipe'].x.size(0)

    def train_dataloader(self):
        edge_label_index = self.train_data['user',
                                           'reviews', 'recipe'].edge_label_index

        # Possible improvement: use a temporal sampling strat to ensure later data doesn't affect earlier data
        return LinkNeighborLoader(
            self.train_data,
            edge_label_index=(('user', 'reviews', 'recipe'), edge_label_index),
            shuffle=True,
            num_neighbors=[8],
            neg_sampling='triplet',
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        edge_label_index = self.val_data['user', 'reviews', 'recipe'].edge_label_index
        return LinkNeighborLoader(
            self.val_data,
            edge_label_index=(('user', 'reviews', 'recipe'), edge_label_index),
            shuffle=True,
            num_neighbors=[8],
            neg_sampling='triplet',
            batch_size=self.batch_size,
        )
