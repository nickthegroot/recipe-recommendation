from copy import copy
from pathlib import Path
from typing import Any

import pandas as pd
import pytorch_lightning as pl
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

from src import config as Config


class IdentityEncoder(object):
    def __init__(
        self,
        one_dim: bool = False,
        dtype=None,
    ):
        self.one_dim = one_dim
        self.dtype = dtype

    def __call__(self, df):
        x = torch.from_numpy(df.values).to(self.dtype)
        if not self.one_dim:
            x = x.view(-1, 1)
        return x


class DataModule(object):
    def __load_node_file(self, path: Path, encoders=None):
        encoder_keys = []
        if encoders is not None:
            encoder_keys = encoders.keys()

        df = pd.read_parquet(
            path,
            columns=encoder_keys,
        )
        mapping = {index: i for i, index in enumerate(df.index.unique())}

        x = None
        if encoders is not None:
            xs = [encoder(df[col]) for col, encoder in encoders.items()]
            x = torch.cat(xs, dim=-1)

        return x, mapping

    def __load_edge_file(
        self,
        path: Path,
        src_index_col: str,
        src_mapping: dict[Any, int],
        dst_index_col: str,
        dst_mapping: dict[Any, int],
        encoders=None,
        extra_encoders=None,
    ):
        encoder_keys = []
        if encoders is not None:
            encoder_keys = encoders.keys()

        extra_encoder_keys = []
        if extra_encoders is not None:
            extra_encoder_keys = extra_encoders.keys()

        df = pd.read_parquet(
            path,
            columns=[
                src_index_col,
                dst_index_col,
                *encoder_keys,
                *extra_encoder_keys,
            ],
        )

        src = [src_mapping[index] for index in df[src_index_col]]
        dst = [dst_mapping[index] for index in df[dst_index_col]]
        edge_index = torch.tensor([src, dst])

        edge_attr = None
        if encoders is not None:
            edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
            edge_attr = torch.cat(edge_attrs, dim=-1)

        extra_edge_attr = None
        if extra_encoders is not None:
            extra_edge_attr = {
                col: encoder(df[col]) for col, encoder in extra_encoders.items()
            }

        return edge_index, edge_attr, extra_edge_attr

    def __init__(
        self,
        device: torch.device,
        data_dir: str = Config.PROCESSED_DATA_DIR,
    ):
        super().__init__()
        self.device = device

        data_dir = Path(data_dir)

        _, user_map = self.__load_node_file(
            data_dir / "user_nodes.parquet",
        )
        self.num_users = len(user_map)

        recipe_x, recipe_map = self.__load_node_file(
            data_dir / "recipe_nodes.parquet",
            encoders={
                "minutes": IdentityEncoder(torch.int64),
                "n_steps": IdentityEncoder(torch.int64),
                "calories": IdentityEncoder(torch.float),
                "total_fat_pdv": IdentityEncoder(torch.float),
                "sugar_pdv": IdentityEncoder(torch.float),
                "sodium_pdv": IdentityEncoder(torch.float),
                "protein_pdv": IdentityEncoder(torch.float),
                "saturated_fat_pdv": IdentityEncoder(torch.float),
                "carbohydrates_pdv": IdentityEncoder(torch.float),
            },
        )
        self.num_recipes = len(recipe_map)

        edge_index, edge_label, extra_edge_attr = self.__load_edge_file(
            data_dir / "review_edges.parquet",
            src_index_col="user_id",
            src_mapping=user_map,
            dst_index_col="recipe_id",
            dst_mapping=recipe_map,
            encoders={"rating": IdentityEncoder(dtype=torch.float64)},
            extra_encoders={
                "is_train": IdentityEncoder(dtype=torch.bool, one_dim=True),
                "is_val": IdentityEncoder(dtype=torch.bool, one_dim=True),
                "is_test": IdentityEncoder(dtype=torch.bool, one_dim=True),
            },
        )

        data = HeteroData()

        data["user"].id = torch.tensor(list(user_map.keys()))
        data["recipe"].x = recipe_x
        data["recipe"].id = torch.tensor(list(recipe_map.keys()))

        train_data, val_data, test_data = copy(data), copy(data), copy(data)
        train_mask = extra_edge_attr["is_train"]
        val_mask = extra_edge_attr["is_val"]  # | extra_edge_attr["is_train"]
        test_mask = (
            extra_edge_attr["is_test"]
            # | extra_edge_attr["is_val"]
            # | extra_edge_attr["is_train"]
        )
        train_data["user", "reviews", "recipe"].edge_index = edge_index[:, train_mask]
        train_data["user", "reviews", "recipe"].edge_attr = edge_label[train_mask]

        val_data["user", "reviews", "recipe"].edge_index = edge_index[:, val_mask]
        val_data["user", "reviews", "recipe"].edge_attr = edge_label[val_mask]

        test_data["user", "reviews", "recipe"].edge_index = edge_index[:, test_mask]
        test_data["user", "reviews", "recipe"].edge_attr = edge_label[test_mask]

        transform = T.ToUndirected()
        self.train_data, self.val_data, self.test_data = (
            transform(train_data),
            transform(val_data),
            transform(test_data),
        )

        self.train_data = self.train_data.to(device)
        self.val_data = self.val_data.to(device)
        self.test_data = self.test_data.to(device)
