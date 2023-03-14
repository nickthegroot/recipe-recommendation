from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src import config as Config
from src.cli.make_dataset import make_dataset


def test_make_dataset():
    test_path = Path(Config.TEST_DATA_DIR)
    with TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir)
        make_dataset(
            input_path=test_path,
            output_path=out_path,
        )

        assert out_path.exists()

        rec_nodes_path = out_path / "recipe_nodes.parquet"
        assert rec_nodes_path.exists()
        df = pd.read_parquet(rec_nodes_path)
        assert len(df) == 4

        user_nodes_path = out_path / "user_nodes.parquet"
        assert user_nodes_path.exists()
        df = pd.read_parquet(user_nodes_path)
        assert len(df) == 4
