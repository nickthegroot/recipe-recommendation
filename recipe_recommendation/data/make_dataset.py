import pandas as pd

from .clean_interactions import clean_interactions
from .clean_recipes import clean_recipes


def main():
    df_rec = pd.read_csv('../../data/raw/recipes.csv', index_col='id')
    df_rec = clean_recipes(df_rec)
    df_rec.to_parquet('../../data/processed/recipes.parquet')
    
    df_int = pd.read_csv('../../data/raw/interactions.csv')
    df_int = clean_interactions(df_int)
    df_int.to_parquet('../../data/processed/interactions.parquet')

if __name__ == '__main__':
    main()