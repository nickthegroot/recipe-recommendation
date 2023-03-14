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
from src.models.lightgcn import LightGCN
from src.models.loader import DataModule
from src.models.recgcn import RecGCN


def train(hp: object):
    now = datetime.now()
    model_dir = Path(hp["model_dir"]) / now.strftime("%y%m%d-%H%M%S")
    model_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(model_dir)

    with open(model_dir / "hp.json", mode="w") as f:
        json.dump(hp, f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dm = DataModule(device, hp["data_dir"])

    if hp["model"] == "LightGCN":
        model = LightGCN(
            num_users=dm.num_users,
            num_recipes=dm.num_recipes,
            lambda_val=hp["lambda_val"],
            embedding_dim=hp["embed_dim"],
        )
    elif hp["model"] == "RecGCN":
        model = RecGCN(
            num_users=dm.num_users,
            num_recipes=dm.num_recipes,
            recipe_dim=dm.recipe_dim,
            lambda_val=hp["lambda_val"],
            embedding_dim=hp["embed_dim"],
            use_weights=hp["use_weights"],
            use_recipe_data=hp["use_recipe_data"],
        )
    else:
        raise Exception("Invalid model: expected LightGCN | RecGCN")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"])

    train_data = dm.train_data
    pbar = trange(hp["max_epochs"])

    max_recall = 0
    n_decreasing = 0
    for i in pbar:
        usr_out, rec_out, loss = train_iter(model, optimizer, train_data)
        writer.add_scalar("train_loss", loss, i)

        if (i + 1) % hp["verbosity"] == 0:
            with torch.no_grad():
                precision, recall = eval_step(
                    usr_out,
                    rec_out,
                    hp["k"],
                    dm,
                    model,
                )

                writer.add_scalar("val_precision", precision, i)
                writer.add_scalar("val_recall", recall, i)

                if max_recall < recall:
                    n_decreasing = 0
                    max_recall = recall
                    torch.save(model.state_dict(), model_dir / "model.pt")
                else:
                    n_decreasing += 1
                    if n_decreasing >= hp["patience"]:
                        return


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
    return usr_out, rec_out, loss


def eval_step(
    usr_out: torch.Tensor,
    rec_out: torch.Tensor,
    k: int,
    dm: DataModule,
    model: nn.Module,
):
    model.eval()

    val_users = dm.val_data["reviews"].edge_index[0].unique()
    val_out = usr_out[val_users]
    num_val_users = val_users.size(0)

    recs = []
    batch_cutoffs = list(range(0, num_val_users, max(num_val_users // 20, 1)))
    batch_cutoffs.append(num_val_users)
    # Need to do this in batches, otherwise we run out of memory
    for start, end in zip(batch_cutoffs, batch_cutoffs[1:]):
        scores = val_out[start:end] @ rec_out.T
        recs.append(scores.topk(k, dim=-1).indices)
    recs = torch.concat(recs, dim=0)
    recs = torch.sparse_coo_tensor(
        indices=torch.vstack(
            [
                val_users[:, None].expand_as(recs).flatten(),
                recs.flatten(),
            ]
        ),
        values=torch.ones(recs.numel()),
        size=[dm.num_users, dm.num_recipes],
        device=recs.device,
    )

    val_user_idxs = dm.val_data["reviews"].edge_index[0].unique()
    target = to_torch_coo_tensor(
        dm.val_data["reviews"].edge_index,
        size=[dm.num_users, dm.num_recipes],
    )

    retrieved_and_relevant = torch.sparse.sum(recs * target, dim=1).to_dense()[
        val_user_idxs
    ]
    retrieved = torch.sparse.sum(recs, dim=1).to_dense()[val_user_idxs]
    relevant = torch.sparse.sum(target, dim=1).to_dense()[val_user_idxs]

    precision = (retrieved_and_relevant / retrieved).mean().item()
    recall = (retrieved_and_relevant / relevant).mean().item()

    return precision, recall


@click.command
@click.option("--data-dir", default=Config.PROCESSED_DATA_DIR)
@click.option("--model", type=str)
@click.option("--max-epochs", default=2_000)
@click.option("--patience", default=2)
@click.option("--lr", default=1e-3)
@click.option("--k", default=20)
@click.option("--lambda-val", default=1e-7)
@click.option("--embed-dim", default=16)
@click.option("--verbosity", default=25)
@click.option(
    "--use-weights",
    type=bool,
    default=False,
    help="Use edge weights (Only used in RecLCN model)",
)
@click.option(
    "--use-recipe-data",
    type=bool,
    default=False,
    help="Use recipe data (Only used in RecLCN model)",
)
@click.option("--model-dir", default=Config.MODEL_DIR)
def main(**hp):
    train(hp)


if __name__ == "__main__":
    main()
