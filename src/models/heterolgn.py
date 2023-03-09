import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.conv import HeteroLGConv


class HeteroLGN(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_recipes: int,
        lambda_val: float = 1e-8,
        embedding_dim: int = 16,
        num_layers: int = 3,
    ):
        super().__init__()
        self.lambda_val = lambda_val

        self.num_layers = num_layers
        self.num_users = num_users
        self.num_recipes = num_recipes
        self.usr_embedding = nn.Embedding(num_users, embedding_dim)
        self.rcp_embedding = nn.Embedding(num_recipes, embedding_dim)

        nn.init.normal_(self.usr_embedding.weight, std=0.1)
        nn.init.normal_(self.rcp_embedding.weight, std=0.1)

        self.alpha = 1 / (num_layers + 1)
        self.conv = HeteroLGConv()

    def forward(
        self,
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        usr_x = self.usr_embedding.weight
        usr_out = usr_x * self.alpha
        rcp_x = self.rcp_embedding.weight
        rec_out = rcp_x * self.alpha

        for _ in range(self.num_layers):
            # Calculate new recipe embeddings
            usr_rcp_edges = edge_index_dict[("user", "reviews", "recipe")]
            rcp_x = self.conv((usr_x, rcp_x), usr_rcp_edges)
            rec_out += rcp_x * self.alpha

            # Calculate new user embeddings
            rcp_usr_edges = edge_index_dict[("recipe", "rev_reviews", "user")]
            usr_x = self.conv((rcp_x, usr_x), rcp_usr_edges)
            usr_out += usr_x * self.alpha

        return usr_out, rec_out

    def _loss(
        self,
        usr_out: torch.Tensor,
        rec_out: torch.Tensor,
        src_usr_idx: torch.Tensor,
        pos_rcp_idx: torch.Tensor,
        neg_rcp_idx: torch.Tensor,
    ):
        """BPR Loss"""
        pos_x = (rec_out[pos_rcp_idx] * usr_out[src_usr_idx]).sum(dim=-1)
        neg_x = (rec_out[neg_rcp_idx] * usr_out[src_usr_idx]).sum(dim=-1)

        log_prob = F.logsigmoid(pos_x - neg_x).mean()
        reg_loss = self.lambda_val * (
            self.usr_embedding(src_usr_idx).norm(2).pow(2)
            + self.rcp_embedding(pos_rcp_idx).norm(2).pow(2)
            + self.rcp_embedding(neg_rcp_idx).norm(2).pow(2)
        )
        return -log_prob + reg_loss
