import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_torch_coo_tensor
from torchmetrics import RetrievalPrecision, RetrievalRecall

from src.models.conv import HeteroLGConv


class HeteroLGN(pl.LightningModule):
    def __init__(
        self,
        num_users: int,
        num_recipes: int,
        embedding_dim: int = 64,
        num_layers: int = 10,
        k: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.k = k

        self.num_layers = num_layers
        self.usr_embedding = nn.Embedding(num_users, embedding_dim)
        self.rec_embedding = nn.Embedding(num_recipes, embedding_dim)

        self.alpha = 1 / (num_layers + 1)
        self.conv = HeteroLGConv()

        self.ret_recall = RetrievalRecall(k=k)
        self.ret_precision = RetrievalPrecision(k=k)

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
    
    def _eval_step(self, x: HeteroData):
        usr_out, rec_out = self(x.id_dict, x.edge_index_dict)

        scores = usr_out @ rec_out.T
        indexes = torch.arange(scores.size(0))[:, None].expand_as(scores)
        target = to_torch_coo_tensor(
            x["user", "reviews", "recipe"].edge_index,
            size=(usr_out.size(0), rec_out.size(0))
            # Current limitations with metrics require dense version :(
        ).to_dense()

        precision = self.ret_precision(
            preds=scores,
            target=target,
            indexes=indexes,
        )

        recall = self.ret_recall(
            preds=scores,
            target=target,
            indexes=indexes,
        )

        return precision, recall

    def validation_step(self, x: HeteroData, batch_idx):
        precision, recall = self._eval_step(x)
        self.log(f"val_recall_{self.k}", recall)
        self.log(f"val_precision_{self.k}", precision)

    def test_step(self, x: HeteroData, batch_idx):
        precision, recall = self._eval_step(x)
        self.log(f"test_recall_{self.k}", recall, batch_size=x["user"].num_nodes)
        self.log(f"test_precision_{self.k}", precision, batch_size=x["user"].num_nodes)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
