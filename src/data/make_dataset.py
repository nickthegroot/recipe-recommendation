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
    df_rec = pd.read_csv(input_path / 'recipes.csv', index_col='id')
    print('Cleaning recipes...')
    df_rec = clean_recipes(df_rec)
    print('Done!')
    df_rec.to_parquet(output_path / 'recipes.parquet')

    df_int = pd.read_csv(input_path / 'interactions.csv')
    print('Cleaning interactions...')
    df_int = clean_interactions(df_int)
    print('Done!')
    df_int.to_parquet(output_path / 'interactions.parquet')


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True), default=Config.RAW_DATA_DIR)
@click.argument('output_filepath', type=click.Path(exists=True), default=Config.PROCESSED_DATA_DIR)
def main(input_filepath: str, output_filepath: str):
    input_path = Path(input_filepath)
    output_path = Path(output_filepath)
    make_dataset(input_path, output_path)


if __name__ == '__main__':
    main()
