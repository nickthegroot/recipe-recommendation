from pathlib import Path

import click
import pandas as pd

from src import config as Config
from src.data.clean_interactions import clean_interactions
from src.data.clean_recipes import clean_recipes


def make_dataset(
    input_path: Path,
    output_path: Path,
):
    # Load & Clean
    df_rec = pd.read_csv(input_path / 'recipes.csv', index_col='id')
    df_rec = clean_recipes(df_rec)

    df_int = pd.read_csv(input_path / 'interactions.csv')
    df_int = clean_interactions(df_int)

    # Create Cleaned Interactions (for DB sync)
    df_int.to_parquet(output_path / 'interactions.parquet')

    # Create Recipe Node List
    df_rec.to_parquet(output_path / 'recipes.parquet', index=True)

    # Create User Node List
    df_users = df_int.user_id.drop_duplicates().to_frame()
    df_users.to_parquet(output_path / 'users.parquet', index=False)

    # Create Ingredient Node List
    df_ingredients = df_rec.ingredients.explode().reset_index(drop=False)
    df_ingredients.rename(
        inplace=True,
        columns={'id': 'recipe_id', 'ingredients': 'ingredient_id'}
    )
    (
        df_ingredients[['ingredient_id']]
        .drop_duplicates()
        .to_parquet(output_path / 'ingredients.parquet', index=False)
    )

    # Create Ingredient Edge List
    ingredient_edgelist = df_ingredients[['recipe_id', 'ingredient_id']]
    ingredient_edgelist.to_parquet(
        output_path / 'ingredient_edgelist.parquet', index=False)

    # Create Review Edge List
    review_edgelist = df_int[['user_id', 'recipe_id', 'rating']]
    review_edgelist.to_parquet(
        output_path / 'review_edgelist.parquet', index=False)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True), default=Config.RAW_DATA_DIR)
@click.argument('output_filepath', type=click.Path(exists=True), default=Config.PROCESSED_DATA_DIR)
def main(input_filepath: str, output_filepath: str):
    input_path = Path(input_filepath)
    output_path = Path(output_filepath)
    make_dataset(input_path, output_path)


if __name__ == '__main__':
    main()
