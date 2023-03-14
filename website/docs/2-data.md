# The Data

To build out our recipe recommendation system, we used an existing dataset of food.com reviews called [`food-com-recipes-and-user-interactions`](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions). After cleaning the dataset and parsing it into a graph data structure, we were left with the following nodes/edges:

```mermaid
classDiagram
    direction RL

    class User{
        <<Node>>
        () N = 226,570
    }
    class Recipe {
        <<Node>>
        () N = 231,637

        INT minutes
        INT n_steps
        FLOAT calories
        FLOAT total_fat_pdv
        FLOAT sugar_pdv
        FLOAT sodium_pdv
        FLOAT protein_pdv
        FLOAT saturated_fat_pdv
        FLOAT carbohydrates_pdv
    }

    class Review {
        <<Edge>>
        () N_TRAIN = 792,656
        () N_VAL=169,855
        () N_TEST=169,856

        INT rating
        BOOL is_train
        BOOL is_val
        BOOL is_test
    }

    User -- Review
    Review -- Recipe
```

This data was then uploaded to TigerGraph, a graph database that allows for fast and scalable storage of graph databases. Their schema system was particularly useful for this project, as it allowed us to easily define the heterogeneous structure of the graph directly inside the database.

We then downloaded the data into a Python environment and trained a variety of graph-based models.
