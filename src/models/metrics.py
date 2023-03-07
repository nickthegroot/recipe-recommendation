import torch
import torch.nn as nn
from torch_geometric.utils import to_torch_coo_tensor


def calc_retrieved_and_relevant(
    input_ids: torch.Tensor,
    top_k_scores: torch.Tensor,
    target_edges: torch.Tensor,
    num_users: int,
    num_recipes: int,
):
    relevant = to_torch_coo_tensor(target_edges, size=[num_users, num_recipes])

    batch_size, k = top_k_scores.shape
    retrived = torch.sparse_coo_tensor(
        indices=torch.vstack(
            [
                input_ids[:, None].expand_as(top_k_scores).flatten(),
                top_k_scores.flatten(),
            ]
        ),
        values=torch.ones(batch_size * k),
        size=[num_users, num_recipes],
        device=top_k_scores.device,
    )

    return (
        torch.sparse.sum(retrived * relevant, dim=1).to_dense()[input_ids],
        retrived,
        relevant,
    )


class Precision(nn.Module):
    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k

    def forward(
        self,
        input_ids: torch.Tensor,
        usr_out: torch.Tensor,
        rec_out: torch.Tensor,
        target_edges: torch.Tensor,
        num_users: int,
        num_recipes: int,
    ):
        scores = usr_out[input_ids] @ rec_out.T
        top_k_scores = scores.topk(self.k, dim=-1).indices
        retrieved_and_relevant, _, _ = calc_retrieved_and_relevant(
            input_ids,
            top_k_scores,
            target_edges,
            num_users,
            num_recipes,
        )

        # Precision = retrieved_and_relevant / retrieved
        # retrieved = k for precision@k
        return retrieved_and_relevant / self.k


class Recall(nn.Module):
    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k

    def forward(
        self,
        input_ids: torch.Tensor,
        usr_out: torch.Tensor,
        rec_out: torch.Tensor,
        target_edges: torch.Tensor,
        num_users: int,
        num_recipes: int,
    ):
        scores = usr_out[input_ids] @ rec_out.T
        top_k_scores = scores.topk(self.k, dim=-1).indices
        retrieved_and_relevant, _, relevant = calc_retrieved_and_relevant(
            input_ids,
            top_k_scores,
            target_edges,
            num_users,
            num_recipes,
        )

        # Recall = retrieved_and_relevant / relevant
        # relevant = # of relevant recipes for each user (sum relvant across each user)
        recall = (
            retrieved_and_relevant
            / torch.sparse.sum(relevant, dim=1).to_dense()[input_ids]
        )
        # Possibility that input user has no included recipes
        # Recall = 0 in this case
        return recall.nan_to_num(0)
