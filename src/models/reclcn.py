import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.conv import HeteroLGConv


class RecLGN(nn.Module):
    """
    A modified version of HeteroLGN for the recipe use-case.

    1. Interactions are weighted by their review score
    2. Recipe data is concatenated with its learned embedding
    """

    def __init__(
        self,
        num_users: int,
        num_recipes: int,
        recipe_dim: int,
        lambda_val: float = 1e-5,
        embedding_dim: int = 16,
        num_layers: int = 3,
    ):
        super().__init__()
        self.lambda_val = lambda_val

        self.num_layers = num_layers
        self.num_users = num_users
        self.num_recipes = num_recipes
        self.usr_embedding = nn.Embedding(num_users, embedding_dim + recipe_dim)
        self.rcp_embedding = nn.Embedding(num_recipes, embedding_dim)

        nn.init.normal_(self.usr_embedding.weight, std=0.1)
        nn.init.normal_(self.rcp_embedding.weight, std=0.1)

        self.alpha = 1 / (num_layers + 1)
        self.conv = HeteroLGConv()

    def forward(
        self,
        x_dict: dict[tuple[str, str, str], torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
        edge_weight_dict: dict[tuple[str, str, str], torch.Tensor],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rcp_x = torch.hstack((self.rcp_embedding.weight, x_dict["recipe"]))
        rec_out = rcp_x * self.alpha

        usr_x = self.usr_embedding.weight
        usr_out = usr_x * self.alpha

        for _ in range(self.num_layers):
            # Calculate new recipe embeddings
            usr_rcp_edges = edge_index_dict[("user", "reviews", "recipe")]
            # usr_rcp_weights = edge_weight_dict[("user", "reviews", "recipe")]
            # rcp_x = self.conv((usr_x, rcp_x), usr_rcp_edges, usr_rcp_weights)
            rcp_x = self.conv((usr_x, rcp_x), usr_rcp_edges)
            rec_out += rcp_x * self.alpha

            # Calculate new user embeddings
            rcp_usr_edges = edge_index_dict[("recipe", "rev_reviews", "user")]
            # rcp_usr_weights = edge_weight_dict[("recipe", "rev_reviews", "user")]
            # usr_x = self.conv((rcp_x, usr_x), rcp_usr_edges, rcp_usr_weights)
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
        # return -log_prob
        return -log_prob + reg_loss
