import pytorch_lightning as pl
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.sampler import NegativeSamplingConfig

# NOTE:
# Remember to run T.ToUndirected()(data) before passing data to the model
# Also look into T.NoramlizeFeatures()(data) for normalization

# Modified version of the Unsupervised Bipartite GraphSAGE model
# @see - https://conferences.computer.org/icde/2020/pdfs/ICDE2020-5acyuqhpJ6L9P042wmjY1p/290300b677/290300b677.pdf
# @see - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/bipartite_sage_unsup.py


class RecipeGNNEncoder(nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = SAGEConv(-1, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.fc(x)
        return x


class Model(pl.LightningModule):
    def __init__(
        self,
        num_users: int,
        num_recipes: int,
        num_ingredients: int,
        hidden_channels: int,
    ):
        super().__init__()
        self.usr_embed = nn.Embedding(num_users, hidden_channels)
        self.ing_embed = nn.Embedding(num_ingredients, hidden_channels)
        self.rec_embed = nn.Embedding(num_recipes, hidden_channels)
