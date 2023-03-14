from pathlib import Path
from tempfile import TemporaryDirectory

from src import config as Config
from src.cli.make_dataset import make_dataset
from src.cli.train import train


def test_full_pipeline():
    test_path = Path(Config.TEST_DATA_DIR)
    with TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir)
        make_dataset(
            input_path=test_path,
            output_path=out_path,
        )

        model_dir = out_path / "models"
        model_dir.mkdir()
        train(
            {
                "data_dir": str(out_path),
                "model_dir": str(model_dir),
                "gpu": False,
                "model": "LightGCN",
                "max_epochs": 1,
                "patience": 1,
                "lr": 1e-3,
                "k": 1,
                "lambda_val": 1e-7,
                "embed_dim": 16,
                "verbosity": 1,
                "use_weights": False,
                "use_recipe_data": False,
            }
        )
