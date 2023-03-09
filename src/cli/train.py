import json
from datetime import datetime
from pathlib import Path

import click
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import HeteroData
from torch_geometric.utils import structured_negative_sampling, to_torch_coo_tensor
from tqdm import trange

import src.config as Config
from src.models.heterolgn import HeteroLGN
from src.models.loader import DataModule
from src.models.reclcn import RecLGN


@click.command
@click.option("--model", type=str)
@click.option("--iterations", default=2_000)
@click.option("--lr", default=1e-3)
@click.option("--k", default=10)
@click.option("--lambda-val", default=1e-8)
@click.option("--embed-dim", default=16)
@click.option("--verbosity", default=100)
@click.option("--model-dir", default=Config.MODEL_DIR)
def main(**hp):
    now = datetime.now()
    model_dir = Path(hp["model_dir"]) / now.strftime("%y%m%d-%H%M%S")
    model_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(model_dir)

    with open(model_dir / "hp.json", mode="w") as f:
        json.dump(hp, f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dm = DataModule(device)

    if hp["model"] == "HeteroLGN":
        model = HeteroLGN(
            dm.num_users,
            dm.num_recipes,
            hp["lambda_val"],
            hp["embed_dim"],
        )
    elif hp["model"] == "RecLGN":
        model = RecLGN(
            dm.num_users,
            dm.num_recipes,
            dm.recipe_dim,
            hp["lambda_val"],
            hp["embed_dim"],
        )
    else:
        raise Exception("Invalid model: expected HeteroLGN | RecLGN")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"])

    train_data = dm.train_data
    val_data = dm.val_data
    pbar = trange(hp["iterations"])

    max_precision = 0
    for i in pbar:
        loss = train_iter(model, optimizer, train_data)
        writer.add_scalar("train_loss", loss, i)

        if (i + 1) % hp['verbosity'] == 0:
            with torch.no_grad():
                precision, recall = eval_step(hp["k"], dm, model, val_data)

                if max_precision < precision.item():
                    max_precision = precision.item()
                    torch.save(model.state_dict(), model_dir / "model.pt")

                writer.add_scalar("val_precision", precision, i)
                writer.add_scalar("val_recall", recall, i)


def train_iter(model: nn.Module, optimizer, train_data: HeteroData):
    model.train()
    usr_out, rec_out = model(
        x_dict=train_data.x_dict,
        edge_index_dict=train_data.edge_index_dict,
        edge_weight_dict=train_data.edge_attr_dict,
    )
    src_usr_idx, pos_rcp_idx, neg_rcp_idx = structured_negative_sampling(
        train_data["reviews"].edge_index
    )
    loss = model._loss(
        usr_out,
        rec_out,
        src_usr_idx,
        pos_rcp_idx,
        neg_rcp_idx,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def eval_step(k: int, dm: DataModule, model: nn.Module, val_data: HeteroData):
    model.eval()
    usr_out, rec_out = model(
        x_dict=val_data.x_dict,
        edge_index_dict=val_data.edge_index_dict,
        edge_weight_dict=val_data.edge_attr_dict,
    )
    num_users = usr_out.size(0)

    recs = []
    batch_cutoffs = list(range(0, num_users, num_users // 10))
    batch_cutoffs.append(num_users)
    # Need to do this in batches, otherwise we run out of memory
    for start, end in zip(batch_cutoffs, batch_cutoffs[1:]):
        scores = usr_out[start:end] @ rec_out.T
        recs.append(scores.topk(k, dim=-1).indices)
    recs = torch.concat(recs, dim=0)
    recs = torch.sparse_coo_tensor(
        indices=torch.vstack(
            [
                torch.arange(num_users, device=recs.device)[:, None]
                .expand_as(recs)
                .flatten(),
                recs.flatten(),
            ]
        ),
        values=torch.ones(recs.numel()),
        size=[dm.num_users, dm.num_recipes],
        device=recs.device,
    )

    target = to_torch_coo_tensor(
        val_data["reviews"].edge_index,
        size=[dm.num_users, dm.num_recipes],
    )

    retrieved_and_relevant = torch.sparse.sum(recs * target, dim=1).to_dense()
    retrieved = torch.sparse.sum(recs, dim=1).to_dense()
    relevant = torch.sparse.sum(target, dim=1).to_dense()

    precision = (retrieved_and_relevant / retrieved).nan_to_num_(0).mean().item()
    recall = (retrieved_and_relevant / relevant).nan_to_num_(0).mean().item()

    return precision, recall


if __name__ == "__main__":
    main()
