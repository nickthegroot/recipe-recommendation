import ast

import pandas as pd


def clean_recipes(df: pd.DataFrame):
    # Parsing
    df.nutrition = df.nutrition.apply(ast.literal_eval)
    df.tags = df.tags.apply(ast.literal_eval)
    df.steps = df.steps.apply(ast.literal_eval)
    df.ingredients = df.ingredients.apply(ast.literal_eval)

    # Nutrition parsing
    nutrition = df.nutrition.apply(get_nutrition)
    df = df.join(nutrition)
    df = df.drop(['nutrition'], axis=1)

    # https://www.food.com/recipe/no-bake-granola-balls-261647
    # Misinputted as 2147483647 min, which overflows later calculations
    df.loc[261647, 'minutes'] = 25

    # Conversion to datetime
    df.submitted = pd.to_datetime(df.submitted)

    return df


def get_nutrition(x):
    # Nutrition information (calories (#), total fat (PDV), sugar (PDV) , sodium (PDV) , protein (PDV) , saturated fat (PDV) , and carbohydrates (PDV))
    [cal, fat, sugar, sodium, protein, sat_fat, carbs] = x
    return pd.Series({
        'calories': cal,
        'total_fat_pdv': fat,
        'sugar_pdv': sugar,
        'sodium_pdv': sodium,
        'protein_pdv': protein,
        'saturated_fat_pdv': sat_fat,
        'carbohydrates_pdv': carbs
    })
