import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
from pyTigerGraph import TigerGraphConnection

from .. import config as Config


def sync_dataset(
    host: str,
    secret: str,
    processed_path: Path
):
    conn = TigerGraphConnection(
        host=host,
        gsqlSecret=secret,
        graphname="RecipeGraph",
    )
    conn.getToken(secret)

    # Load data
    df_rec = pd.read_parquet(processed_path  / 'recipes.parquet')
    df_int = pd.read_parquet(processed_path / 'interactions.parquet')
    
    # Perform minor conversions due to database constraints
    df_rec.submitted = df_rec.submitted.astype(str)
    df_rec.minutes = df_rec.minutes.dt.total_seconds() / 60
    df_rec.description = df_rec.description.fillna("")
    df_rec.dropna(inplace=True) # there's one row with NULL name - safe to drop

    df_int = df_int[df_int.recipe_id.isin(df_rec.index)] # ensure all recipe IDs are in the graph
    df_int.user_id = df_int.user_id.astype(str)
    df_int.date = df_int.date.astype(str)
    df_int.rating = df_int.rating.astype(np.uint8)

    # Parse out unique information
    # Note: TigerGraph breaks when commas are used as keys
    df_ingredients = df_rec.ingredients.explode().reset_index(drop=False)
    df_ingredients['ingredients'] = df_ingredients.ingredients.str.replace(',', '')
    df_tags = df_rec.tags.explode().reset_index(drop=False)


    # -- Upsert vertices --
    upload_vertices(conn, df_int, df_rec, df_ingredients, df_tags)

    # -- Upsert edges --
    upload_edges(conn, df_int, df_rec, df_ingredients, df_tags)


    

def upload_vertices(
    conn: TigerGraphConnection,
    df_int: pd.DataFrame,
    df_rec: pd.DataFrame,
    df_ingredients: pd.DataFrame,
    df_tags: pd.DataFrame,
):
    conn.upsertVertexDataFrame(
        df_rec,
        vertexType='Recipe',
        attributes={
            'name': 'name',
            'minutes': 'minutes',
            'submitted': 'submitted',
            'description': 'description',
            'calories': 'calories',
            'total_fat_pdv': 'total_fat_pdv',
            'sugar_pdv': 'sugar_pdv',
            'sodium_pdv': 'sodium_pdv',
            'protein_pdv': 'protein_pdv',
            'saturated_fat_pdv': 'saturated_fat_pdv',
            'carbohydrates_pdv': 'carbohydrates_pdv',
        }
    )

    unique_ingredients = df_ingredients.drop_duplicates(subset=['ingredients'])
    conn.upsertVertexDataFrame(
        unique_ingredients,
        vertexType='Ingredient',
        v_id='ingredients',
        attributes={}
    )


    unique_tags = df_tags.drop_duplicates(subset=['tags'])
    conn.upsertVertexDataFrame(
        unique_tags,
        vertexType='Tag',
        v_id='tags',
        attributes={}
    )

    unique_users = df_int.drop_duplicates(subset=['user_id'])
    conn.upsertVertexDataFrame(
        unique_users,
        vertexType='User',
        v_id='user_id',
        attributes={}
    )

def upload_edges(
    conn: TigerGraphConnection,
    df_int: pd.DataFrame,
    df_ingredients: pd.DataFrame,
    df_tags: pd.DataFrame,
):

    conn.upsertEdgeDataFrame(
        df_int,
        edgeType='Review',
        sourceVertexType='Recipe',
        from_id='recipe_id',
        targetVertexType='User',
        to_id='user_id',
        attributes={
            'date': 'date',
            'rating': 'rating',
        }
    )

    conn.upsertEdgeDataFrame(
        df_ingredients,
        edgeType='Recipe_Ingredient',
        sourceVertexType='Recipe',
        from_id='id',
        targetVertexType='Ingredient',
        to_id='ingredients',
        attributes={}
    )

    conn.upsertEdgeDataFrame(
        df_tags,
        edgeType='Recipe_Tag',
        sourceVertexType='Recipe',
        from_id='id',
        targetVertexType='Tag',
        to_id='tags',
        attributes={}
    )

@click.command()
@click.argument('processed_filepath', type=click.Path(exists=True), default=Config.PROCESSED_DATA_DIR)
@click.option('--host', type=str, default=os.environ.get('TG_HOST'))
@click.option('--secret', type=str, default=os.environ.get('TG_SECRET'))
def main(
    host: str,
    secret: str,
    processed_path: str,
):
    sync_dataset(host, secret, processed_path)