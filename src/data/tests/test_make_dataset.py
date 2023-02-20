from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src import config as Config
from src.data.make_dataset import make_dataset


def test_make_dataset():
    test_path = Path(Config.TEST_DATA_DIR)
    with TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir)
        make_dataset(
            input_path=test_path,
            output_path=out_path,
        )

        assert (out_path.exists())

        assert ((out_path / 'recipes.parquet').exists())
        df = pd.read_parquet(out_path / 'recipes.parquet')
        assert len(df) == 3

        assert ((out_path / 'users.parquet').exists())
        df = pd.read_parquet(out_path / 'users.parquet')
        assert len(df) == 3

        assert ((out_path / 'ingredients.parquet').exists())
        df = pd.read_parquet(out_path / 'ingredients.parquet')
        assert len(df) == 1

        assert ((out_path / 'ingredient_edgelist.parquet').exists())
        assert ((out_path / 'review_edgelist.parquet').exists())
