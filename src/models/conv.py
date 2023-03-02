import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor
from torch_scatter import scatter_add


class HeteroLGConv(MessagePassing):
    """
    Heterogeneous version of the LightGCN Convolutional Layer

    Takes in a bipartite graph and computes the output features of the nodes
    based on a weighted sum of its neighbors' features.
    """

    def __init__(self, normalize: bool = True):
        super().__init__(aggr="add")
        self.normalize = normalize

    def forward(
        self,
        x: tuple[torch.Tensor, torch.Tensor],
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ):
        in_x, out_x = x
        if self.normalize:
            if edge_weight is None:
                edge_weight = torch.ones(
                    (edge_index.size(1),), dtype=in_x.dtype, device=in_x.device
                )

            num_in_nodes = in_x.size(0)
            num_out_nodes = out_x.size(0)

            in_idx, out_idx = edge_index[0], edge_index[1]
            in_deg = scatter_add(edge_weight, in_idx, dim=0, dim_size=num_in_nodes)
            in_deg_inv_sqrt = in_deg.pow(-0.5)
            in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("inf"), 0)

            out_deg = scatter_add(edge_weight, out_idx, dim=0, dim_size=num_out_nodes)
            out_deg_inv_sqrt = out_deg.pow(-0.5)
            out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("inf"), 0)

            edge_weight = (
                in_deg_inv_sqrt[in_idx] * edge_weight * out_deg_inv_sqrt[out_idx]
            )

        return self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

    def message(self, x_j: torch.Tensor, edge_weight: OptTensor):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
