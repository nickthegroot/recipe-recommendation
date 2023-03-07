import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_torch_coo_tensor

from src.models.conv import HeteroLGConv


class HeteroLGN(pl.LightningModule):
    def __init__(
        self,
        num_users: int,
        num_recipes: int,
        embedding_dim: int = 64,
        num_layers: int = 2,
        k: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.k = k

        self.num_layers = num_layers
        self.num_users = num_users
        self.num_recipes = num_recipes
        self.usr_embedding = nn.Embedding(num_users, embedding_dim)
        self.rec_embedding = nn.Embedding(num_recipes, embedding_dim)

        self.alpha = 1 / (num_layers + 1)
        self.conv = HeteroLGConv()

    def forward(
        self,
        id_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        usr_x = self.usr_embedding(id_dict["user"])
        usr_out = usr_x * self.alpha
        rec_x = self.rec_embedding(id_dict["recipe"])
        rec_out = rec_x * self.alpha

        for _ in range(self.num_layers):
            # Calculate new recipe embeddings
            usr_rec_edges = edge_index_dict[("user", "reviews", "recipe")]
            rec_x = self.conv((usr_x, rec_x), usr_rec_edges)
            rec_out += rec_x * self.alpha

            # Calculate new user embeddings
            rec_usr_edges = edge_index_dict[("recipe", "rev_reviews", "user")]
            usr_x = self.conv((rec_x, usr_x), rec_usr_edges)
            usr_out += usr_x * self.alpha

        return usr_out, rec_out

    def _loss(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor):
        """BPR Loss"""
        log_prob = F.logsigmoid(pos_scores - neg_scores).mean()
        return -log_prob

    def training_step(self, x: HeteroData, batch_idx):
        usr_out, rec_out = self(x.id_dict, x.edge_index_dict)

        src_usr_idx = x["user"].src_index
        pos_idx = x["recipe"].dst_pos_index
        neg_idx = x["recipe"].dst_neg_index
        pos_x = (rec_out[pos_idx] * usr_out[src_usr_idx]).sum(dim=-1)
        neg_x = (rec_out[neg_idx] * usr_out[src_usr_idx]).sum(dim=-1)

        loss = self._loss(pos_x, neg_x)
        self.log("train_loss", loss, batch_size=x["user"].src_index.size(0))
        return loss

    def validation_step(self, x: HeteroData, batch_idx):
        input_usr_idx = x["reviews"].edge_label_index[0]
        batch_size = input_usr_idx.size(0)
        target = to_torch_coo_tensor(
            x["reviews"].edge_label_index, size=[self.num_users, self.num_recipes]
        )

        usr_out, rec_out = self(x.id_dict, x.edge_index_dict)
        scores = usr_out[input_usr_idx] @ rec_out.T
        recs = scores.topk(self.k, dim=-1).indices
        recs = torch.sparse_coo_tensor(
            indices=torch.vstack(
                [
                    input_usr_idx[:, None].expand_as(recs).flatten(),
                    recs.flatten(),
                ]
            ),
            values=torch.ones(recs.numel()),
            size=[self.num_users, self.num_recipes],
            device=recs.device,
        )

        retrieved_and_relevant = torch.sparse.sum(recs * target, dim=1).to_dense()[
            input_usr_idx
        ]
        retrieved = torch.sparse.sum(recs, dim=1).to_dense()[input_usr_idx]
        relevant = torch.sparse.sum(target, dim=1).to_dense()[input_usr_idx]

        precision = retrieved_and_relevant / retrieved
        recall = retrieved_and_relevant / relevant
        self.log("val_precision", precision.mean(), batch_size=batch_size)
        self.log("val_recall", recall.mean(), batch_size=batch_size)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
