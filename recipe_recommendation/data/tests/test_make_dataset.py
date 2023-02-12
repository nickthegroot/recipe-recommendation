from pathlib import Path
from tempfile import TemporaryDirectory

from ... import config as Config
from ..make_dataset import make_dataset


def test_make_dataset():
    test_path = Path(Config.TEST_DATA_DIR)
    with TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir)
        make_dataset(
            input_path=test_path,
            output_path=out_path,
        )

        assert(out_path.exists())
        assert((out_path / 'recipes.parquet').exists())
        assert((out_path / 'interactions.parquet').exists())