from src.models.heterolgn import HeteroLGN


class RecLGN(HeteroLGN):
    """
    A modified version of HeteroLGN for the recipe use-case.

    1. Interactions are weighted by their review score
    2. Recipe data is concatenated with its learned embedding
    """
    
    # TODO
    pass