import logging
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# LOGGER
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

# ENVIRONMENT FILES
# find .env automagically by walking up directories until it's found, then
# load up the .env entries as environment variables
load_dotenv(find_dotenv())

# DIRECTORIES
PROJECT_DIR: str = str(Path(__file__).parents[1])

# # DATA DIRECTORIES
DATA_DIR: str = os.path.join(PROJECT_DIR, "data")
RAW_DATA_DIR: str = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR: str = os.path.join(DATA_DIR, "processed")
TEST_DATA_DIR: str = os.path.join(DATA_DIR, "test")

# # MODEL DIRECTORIES
MODEL_DIR: str = os.path.join(PROJECT_DIR, "models")
