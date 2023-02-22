import pytorch_lightning as pl
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import LightGCN as TGLightGCN
from torch_geometric.typing import Adj, OptTensor


class LightGCN(pl.LightningModule):
    def __init__(
        self,
        num_users: int,
        num_recipes: int,
        embedding_dim: int = 64,
        num_layers: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_users = num_users
        num_nodes = num_users + num_recipes
        self.model = TGLightGCN(num_nodes, embedding_dim, num_layers)

    def forward(self, edge_index: Adj, edge_label_index: OptTensor = None):
        return self.model.forward(edge_index, edge_label_index)

    def recommend(self, x: HeteroData, user_idx: torch.Tensor, top_k: int):
        edge_index = x['reviews'].edge_index
        edge_index[1] += self.num_users

        return self.model.recommend(edge_index, user_idx, k=top_k)

    def common_step(self, x: HeteroData, batch_idx):
        # Convert the bi-adjacency matrix to the adjacency matrix of a unipartite graph.
        # Items are indexed after users. This step is needed because LightGCN only supports unipartite graph.
        edge_index = x['reviews'].edge_index + self.num_users

        src_idx = x['user'].src_index
        pos_idx = x['recipe'].dst_pos_index + self.num_users
        pos_edges = torch.vstack([src_idx, pos_idx])
        pos_ranking = self(edge_index, pos_edges)

        neg_idx = x['recipe'].dst_neg_index + self.num_users
        neg_edges = torch.vstack([src_idx, neg_idx])
        neg_ranking = self(edge_index, neg_edges)

        # Calc loss using BPR (same user, pos/neg links)
        loss = self.model.recommendation_loss(pos_ranking, neg_ranking)
        return loss

    def training_step(self, x: HeteroData, batch_idx):
        loss = self.common_step(x, batch_idx)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)
