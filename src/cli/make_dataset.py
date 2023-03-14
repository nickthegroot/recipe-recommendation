from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config as Config
from src.data.clean_interactions import clean_interactions
from src.data.clean_recipes import clean_recipes


def make_dataset(
    input_path: Path,
    output_path: Path,
):
    # Load & Clean
    df_rec = pd.read_csv(input_path / "RAW_recipes.csv", index_col="id")
    df_rec = clean_recipes(df_rec)

    df_int = pd.read_csv(input_path / "RAW_interactions.csv")
    df_int = clean_interactions(df_int)

    train_int, test_int = train_test_split(df_int, test_size=0.3, random_state=42)
    val_int, test_int = train_test_split(test_int, test_size=0.5, random_state=42)

    # Create Recipe Node List
    df_rec.to_parquet(output_path / "recipe_nodes.parquet", index=True)

    # Create User Node List
    df_users = df_int.user_id.drop_duplicates().to_frame().set_index("user_id")
    df_users.to_parquet(output_path / "user_nodes.parquet", index=True)

    # Create Review Edge List
    train_int.to_parquet(output_path / "train_review_edges.parquet", index=False)
    val_int.to_parquet(output_path / "val_review_edges.parquet", index=False)
    test_int.to_parquet(output_path / "test_review_edges.parquet", index=False)


@click.command()
@click.argument(
    "input_filepath", type=click.Path(exists=True), default=Config.RAW_DATA_DIR
)
@click.argument(
    "output_filepath", type=click.Path(exists=True), default=Config.PROCESSED_DATA_DIR
)
def main(input_filepath: str, output_filepath: str):
    input_path = Path(input_filepath)
    output_path = Path(output_filepath)
    make_dataset(input_path, output_path)


if __name__ == "__main__":
    main()
