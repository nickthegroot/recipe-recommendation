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
        use_weights: bool = True,
        use_recipe_data: bool = True,
    ):
        super().__init__()
        self.lambda_val = lambda_val
        self.use_weights = use_weights
        self.use_recipe_data = use_recipe_data

        self.num_layers = num_layers
        self.num_users = num_users
        self.num_recipes = num_recipes
        usr_embedding_dim = (
            embedding_dim + recipe_dim if use_recipe_data else embedding_dim
        )
        self.usr_embedding = nn.Embedding(num_users, usr_embedding_dim)
        if embedding_dim != 0:
            self.rcp_embedding = nn.Embedding(num_recipes, embedding_dim)
        else:
            self.rcp_embedding = None

        nn.init.normal_(self.usr_embedding.weight, std=0.1)
        if self.rcp_embedding is not None:
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
        if self.use_recipe_data:
            if self.rcp_embedding is None:
                rcp_x = x_dict["recipe"]
            else:
                rcp_x = torch.hstack([self.rcp_embedding.weight, x_dict["recipe"]])
        else:
            rcp_x = self.rcp_embedding.weight
        rec_out = rcp_x * self.alpha

        usr_x = self.usr_embedding.weight
        usr_out = usr_x * self.alpha

        for _ in range(self.num_layers):
            # Calculate new recipe embeddings
            usr_rcp_edges = edge_index_dict[("user", "reviews", "recipe")]
            usr_rcp_weights = None
            if self.use_weights:
                usr_rcp_weights = edge_weight_dict[("user", "reviews", "recipe")]
            rcp_x = self.conv((usr_x, rcp_x), usr_rcp_edges, usr_rcp_weights)
            rec_out += rcp_x * self.alpha

            # Calculate new user embeddings
            rcp_usr_edges = edge_index_dict[("recipe", "rev_reviews", "user")]
            rcp_usr_weights = None
            if self.use_weights:
                rcp_usr_weights = edge_weight_dict[("recipe", "rev_reviews", "user")]
            usr_x = self.conv((rcp_x, usr_x), rcp_usr_edges, rcp_usr_weights)
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
        reg_loss = self.usr_embedding(src_usr_idx).norm(2).pow(2)
        if self.rcp_embedding is not None:
            reg_loss += self.rcp_embedding(pos_rcp_idx).norm(2).pow(2)
            reg_loss += self.rcp_embedding(neg_rcp_idx).norm(2).pow(2)
        reg_loss *= self.lambda_val
        return -log_prob + reg_loss
