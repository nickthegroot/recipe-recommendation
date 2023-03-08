import json
from datetime import datetime
from pathlib import Path

import click
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import structured_negative_sampling, to_torch_coo_tensor
from tqdm import trange

import src.config as Config
from src.models.heterolgn import HeteroLGN
from src.models.loader import DataModule


@click.command
@click.option("--iterations", default=2_000)
@click.option("--lr", default=1e-3)
@click.option("--k", default=10)
@click.option("--lambda-val", default=1e-8)
@click.option("--embed-dim", default=16)
@click.option("--model-dir", default=Config.MODEL_DIR)
def main(
    iterations: int,
    lr: float,
    k: int,
    lambda_val: float,
    embed_dim: int,
    model_dir: str,
):
    now = datetime.now()
    model_dir = Path(model_dir) / now.strftime("%y%m%d-%H%M%S")
    model_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(model_dir)

    hp = {
        "iterations": iterations,
        "lr": lr,
        "k": k,
        "lambda_val": lambda_val,
        "embed_dim": embed_dim,
    }
    with open(model_dir / "hp.json", mode="w") as f:
        json.dump(hp, f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dm = DataModule(device)

    model = HeteroLGN(
        dm.num_users,
        dm.num_recipes,
        lambda_val,
        embed_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_data = dm.train_data
    val_data = dm.val_data
    pbar = trange(iterations)

    tr_loss = []
    val_precision = []
    val_recall = []

    for i in pbar:
        model.train()
        usr_out, rec_out = model(train_data.edge_index_dict)
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

        writer.add_scalar("train_loss", loss, i)

        if i % 20 == 0:
            loss_val = loss.item()
            pbar.set_postfix(loss=loss_val)
            tr_loss.append(loss_val)

        if (i + 1) % 100 == 0:
            with torch.no_grad():
                model.eval()
                usr_out, rec_out = model(val_data.edge_index_dict)
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

                retrieved_and_relevant = torch.sparse.sum(
                    recs * target, dim=1
                ).to_dense()
                retrieved = torch.sparse.sum(recs, dim=1).to_dense()
                relevant = torch.sparse.sum(target, dim=1).to_dense()

                precision = (
                    (retrieved_and_relevant / retrieved).nan_to_num_(0).mean().item()
                )
                recall = (
                    (retrieved_and_relevant / relevant).nan_to_num_(0).mean().item()
                )

                if len(val_precision) == 0 or precision > max(val_precision):
                    torch.save(model.state_dict(), model_dir / "model.pt")

                writer.add_scalar("val_precision", precision, i)
                writer.add_scalar("val_recall", recall, i)
                val_precision.append(precision)
                val_recall.append(precision)
                pbar.write(f"\n====== EPOCH {i+1}/{iterations} ======")
                pbar.write(f"precision@{k}: {precision}")
                pbar.write(f"recall@{k}: {recall}")

    # Save all training metrics
    pd.Series(tr_loss).to_csv(model_dir / "tr_loss.csv", index=False)
    pd.DataFrame(
        {
            "precision": val_precision,
            "recall": val_recall,
        }
    ).to_csv(model_dir / "val_metrics.csv", index=False)


if __name__ == "__main__":
    main()
