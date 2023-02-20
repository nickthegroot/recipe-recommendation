from pathlib import Path

import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData


def get_data(input_dir: Path) -> HeteroData:
    data = HeteroData()

    # Add nodes for each type.
    df_rec = pd.read_parquet(
        input_dir / 'recipes.parquet',
        columns=[
            'minutes',
            'n_steps',
            'calories',
            'total_fat_pdv',
            'sugar_pdv',
            'sodium_pdv',
            'protein_pdv',
            'saturated_fat_pdv',
            'carbohydrates_pdv',
        ]
    )
    rec_id_idx = pd.Series({id: i for i, id in enumerate(df_rec.index)})
    data['recipe'].x = torch.tensor(df_rec.values)

    df_ing = pd.read_parquet(input_dir / 'ingredients.parquet')
    df_ing.set_index('ingredient_id', inplace=True)
    ing_id_idx = pd.Series({id: i for i, id in enumerate(df_ing.index)})
    data['ingredient'].x = torch.tensor(df_ing.values)

    df_usr = pd.read_parquet(input_dir / 'users.parquet')
    df_usr.set_index('user_id', inplace=True)
    usr_id_idx = pd.Series({id: i for i, id in enumerate(df_usr.index)})
    data['user'].x = torch.tensor(df_usr.values)

    # Add edges for each type.
    df_ing_edgelist = pd.read_parquet(
        input_dir / 'ingredient_edgelist.parquet'
    )
    df_ing_edgelist['recipe_id'] = (
        rec_id_idx[df_ing_edgelist['recipe_id']]
        .reset_index(drop=True)
    )
    df_ing_edgelist['ingredient_id'] = (
        ing_id_idx[df_ing_edgelist['ingredient_id']]
        .reset_index(drop=True)
    )
    data['recipe', 'uses', 'ingredient'].edge_index = torch.tensor(
        df_ing_edgelist.values
    ).T

    df_rev_edgelist = pd.read_parquet(input_dir / 'review_edgelist.parquet')
    df_rev_edgelist['user_id'] = (
        usr_id_idx[df_rev_edgelist['user_id']]
        .reset_index(drop=True)
    )
    df_rev_edgelist['recipe_id'] = (
        rec_id_idx[df_rev_edgelist['recipe_id']]
        .reset_index(drop=True)
    )
    data['user', 'reviews', 'recipe'].edge_index = torch.tensor(
        df_rev_edgelist[['user_id', 'recipe_id']].values
    ).T
    data['user', 'reviews', 'recipe'].edge_attr = torch.tensor(
        df_rev_edgelist['rating'].values
    )

    return T.ToUndirected()(data)
