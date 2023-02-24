import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.models.lightgcn import BPRLoss
from torch_geometric.typing import Adj, OptTensor
from torch_scatter import scatter_add


class HeteroLGConv(MessagePassing):
    """
    Heterogeneous version of the LightGCN Convolutional Layer

    Takes in a bipartite graph and computes the output features of the nodes
    based on a weighted sum of its neighbors' features.
    """
    def __init__(self, normalize: bool = True):
        super().__init__(aggr='add')
        self.normalize = normalize

    def forward(
        self,
        x: tuple[torch.Tensor, torch.Tensor],
        edge_index: Adj,
        edge_weight: OptTensor = None
    ):
        in_x, out_x = x
        if self.normalize:
            if edge_weight is None:
                edge_weight = torch.ones(
                    (edge_index.size(1), ),
                    dtype=in_x.dtype,
                    device=edge_index.device
                )

            num_in_nodes = in_x.size(0)
            num_out_nodes = out_x.size(0)
            edge_weight = torch.ones(
                (edge_index.size(1), ),
                dtype=in_x.dtype,
                device=edge_index.device
            )

            in_idx, out_idx = edge_index[0], edge_index[1]
            in_deg = scatter_add(edge_weight, in_idx, dim=0, dim_size=num_in_nodes)
            in_deg_inv_sqrt = in_deg.pow_(-0.5)
            in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float('inf'), 0)

            out_deg = scatter_add(edge_weight, out_idx, dim=0, dim_size=num_out_nodes)
            out_deg_inv_sqrt = out_deg.pow_(-0.5)
            out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float('inf'), 0)

            edge_weight = in_deg_inv_sqrt[in_idx] * edge_weight * out_deg_inv_sqrt[out_idx]

        return self.propagate(
            edge_index,
            x=out_x,
            edge_weight=edge_weight,
            size=None
        )

    def message(self, x_j: torch.Tensor, edge_weight: OptTensor):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

class HeteroLGN(pl.LightningModule):
    def __init__(
        self,
        num_users: int,
        num_recipes: int,
        embedding_dim: int = 64,
        num_layers: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_layers = num_layers
        self.usr_embedding = nn.Embedding(num_users, embedding_dim)
        self.rec_embedding = nn.Embedding(num_recipes, embedding_dim)
        
        self.alpha = 1 / (num_layers + 1)
        self.conv = HeteroLGConv()

    def forward(
        self,
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        usr_x = self.usr_embedding.weight
        usr_out = usr_x * self.alpha
        rec_x = self.rec_embedding.weight
        rec_out = rec_x * self.alpha

        for _ in range(self.num_layers):
            # Calculate new recipe embeddings
            usr_rec_edges = edge_index_dict[('user', 'reviews', 'recipe')]
            rec_x = self.conv((usr_x, rec_x), usr_rec_edges)
            rec_out += rec_x * self.alpha
            
            # Calculate new user embeddings
            rec_usr_edges = edge_index_dict[('recipe', 'rev_reviews', 'user')]
            usr_x = self.conv((rec_x, usr_x), rec_usr_edges)
            usr_out += usr_x * self.alpha

        return usr_out, rec_out

    # def recommend(self, x: HeteroData, user_idx: torch.Tensor, top_k: int):
    #     # edge_index = x['reviews'].edge_index
    #     # edge_index[1] += self.num_users

    #     # return self.model.recommend(edge_index, user_idx, k=top_k)

    def common_step(self, x: HeteroData, batch_idx):
        # Convert the bi-adjacency matrix to the adjacency matrix of a unipartite graph.
        # Items are indexed after users. This step is needed because LightGCN only supports unipartite graph.
        _, rec_out = self(x.edge_index_dict)

        pos_idx = x['recipe'].dst_pos_index
        neg_idx = x['recipe'].dst_neg_index
        pos_x = rec_out[pos_idx]
        neg_x = rec_out[neg_idx]

        log_prob = F.logsigmoid(pos_x - neg_x).mean()
        loss = -log_prob
        return loss

    def training_step(self, x: HeteroData, batch_idx):
        loss = self.common_step(x, batch_idx)
        self.log('train_loss', loss, batch_size=x['user'].src_index.size(0))
        return loss

    def validation_step(self, x: HeteroData, batch_idx):
        loss = self.common_step(x, batch_idx)
        self.log('val_loss', loss, batch_size=x['user'].src_index.size(0))
        return loss

    # def test_step(self, x: HeteroData, batch_idx):
    #     loss = self.common_step(x, batch_idx)
    #     self.log('test_loss', loss)
    #     return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
