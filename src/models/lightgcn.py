import pytorch_lightning as pl
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import LightGCN


class Model(pl.LightningModule):
    def __init__(
        self,
        num_users: int,
        num_recipes: int,
        embedding_dim: int,
        num_layers: int,
    ):
        super().__init__()
        num_nodes = num_users + num_recipes
        self.model = LightGCN(num_nodes, embedding_dim, num_layers)

    def forward(self, review_edge_index: HeteroData):
        return self.model(review_edge_index)

    def recommend(self, edge_index: torch.Tensor, user_idx: torch.Tensor, top_k: int):
        return self.model.recommend(edge_index, user_idx, k=top_k)

    def calculate_loss(self, ranking: torch.Tensor, labels: torch.Tensor):
        pos_rankings = ranking[labels == 1]
        neg_rankings = ranking[labels == 0]
        return self.model.recommendation_loss(pos_rankings, neg_rankings)

    def training_step(self, x, batch_idx):
        edges = x['user', 'reviews', 'recipe']
        ranking = self(edges.edge_label_index)
        loss = self.calculate_loss(ranking, edges.edge_label)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)
