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
    df_rec = pd.read_csv(input_path / "RAW_recipes.csv", index_col="id")
    df_rec = clean_recipes(df_rec)

    df_train_int = pd.read_csv(input_path / "interactions_train.csv")
    df_train_int["is_train"] = True
    df_val_int = pd.read_csv(input_path / "interactions_validation.csv")
    df_val_int["is_val"] = True
    df_test_int = pd.read_csv(input_path / "interactions_test.csv")
    df_test_int["is_test"] = True
    df_int = pd.concat([df_train_int, df_val_int, df_test_int], ignore_index=True)
    df_int = clean_interactions(df_int)
    df_int

    # Create Recipe Node List
    df_rec.to_parquet(output_path / "recipe_nodes.parquet", index=True)

    # Create User Node List
    df_users = df_int.user_id.drop_duplicates().to_frame().set_index("user_id")
    df_users.to_parquet(output_path / "user_nodes.parquet", index=True)

    # Create Review Edge List
    review_edgelist = df_int[
        ["user_id", "recipe_id", "rating", "is_train", "is_val", "is_test"]
    ]
    review_edgelist.to_parquet(output_path / "review_edges.parquet", index=False)


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
