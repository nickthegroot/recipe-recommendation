from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader

from src import config as Config


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = Config.PROCESSED_DATA_DIR,
        batch_size: int = 2048,
    ):
        super().__init__()
        self.batch_size = batch_size

        data = HeteroData()
        data_dir = Path(data_dir)

        # Add nodes for each type.
        df_rec = pd.read_parquet(
            data_dir / "recipes.parquet",
            columns=[
                "minutes",
                "n_steps",
                "calories",
                "total_fat_pdv",
                "sugar_pdv",
                "sodium_pdv",
                "protein_pdv",
                "saturated_fat_pdv",
                "carbohydrates_pdv",
            ],
        )
        rec_id_idx = pd.Series({id: i for i, id in enumerate(df_rec.index)})
        data["recipe"].x = torch.tensor(df_rec.values)

        df_usr = pd.read_parquet(data_dir / "users.parquet")
        df_usr.set_index("user_id", inplace=True)
        usr_id_idx = pd.Series({id: i for i, id in enumerate(df_usr.index)})
        # Users have no features
        data["user"].num_nodes = df_usr.shape[0]

        # Add edges for each type.
        df_rev_edgelist = pd.read_parquet(data_dir / "review_edgelist.parquet")
        df_rev_edgelist["user_id"] = usr_id_idx[df_rev_edgelist["user_id"]].reset_index(
            drop=True
        )
        df_rev_edgelist["recipe_id"] = rec_id_idx[
            df_rev_edgelist["recipe_id"]
        ].reset_index(drop=True)
        data["user", "reviews", "recipe"].edge_index = torch.tensor(
            df_rev_edgelist[["user_id", "recipe_id"]].values
        ).T
        data["user", "reviews", "recipe"].edge_attr = torch.tensor(
            df_rev_edgelist["rating"].values
        )

        # Add IDs
        data["user"].id = torch.arange(data["user"].num_nodes)
        data["recipe"].id = torch.arange(data["recipe"].num_nodes)

        data = T.ToUndirected()(data)

        transform = T.RandomLinkSplit(
            num_val=0.1,
            num_test=0.1,
            neg_sampling_ratio=0,
            edge_types=("user", "reviews", "recipe"),
            rev_edge_types=("recipe", "rev_reviews", "user"),
        )
        self.train_data, self.val_data, self.test_data = transform(data)

        self.num_users = data["user"].num_nodes
        self.num_recipes = data["recipe"].num_nodes

    def train_dataloader(self):
        return LinkNeighborLoader(
            self.train_data,
            edge_label_index=(
                ("user", "reviews", "recipe"),
                self.train_data["user", "reviews", "recipe"].edge_label_index,
            ),
            shuffle=True,
            # Pick a user (batch), get all recipes (-1), then get all users of that recipe (-1)
            num_neighbors=[-1, -1],
            neg_sampling="triplet",
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return LinkNeighborLoader(
            self.val_data,
            edge_label_index=(
                ("user", "reviews", "recipe"),
                self.val_data["user", "reviews", "recipe"].edge_label_index,
            ),
            # Pick a user (batch), get all recipes (-1), then get all users of that recipe (-1)
            num_neighbors=[-1, -1],
            batch_size=self.batch_size,
        )
