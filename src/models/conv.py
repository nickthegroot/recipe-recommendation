import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import degree


class HeteroLGConv(MessagePassing):
    """
    Heterogeneous version of the LightGCN Convolutional Layer

    Takes in a bipartite graph and computes the output features of the nodes
    based on a weighted sum of its neighbors' features.
    """

    def __init__(self):
        super().__init__(aggr="add")

    def forward(
        self,
        x: tuple[torch.Tensor, torch.Tensor],
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ):
        if edge_weight is None:
            edge_weight = torch.ones(
                edge_index.size(1),
                dtype=x[0].dtype,
                device=x[0].device,
            )

        in_idx, out_idx = edge_index[0], edge_index[1]
        in_deg, out_deg = degree(in_idx), degree(out_idx)
        norm_term = (in_deg[in_idx] * out_deg[out_idx]).pow(-0.5)

        new_edge_weight = edge_weight * norm_term
        return self.propagate(edge_index, x=x, edge_weight=new_edge_weight, size=None)

    def message(self, x_j: torch.Tensor, edge_weight: OptTensor):
        return edge_weight.view(-1, 1) * x_j
