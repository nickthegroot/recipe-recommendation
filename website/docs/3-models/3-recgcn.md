# RecGCN

## How It Works

While LightGCN is able to capture a lot of underlying relationships, it neglects a variety of information we have available to us.

- **Recipe Features**: LightGCN only uses the user and recipe embeddings to compute the score for a user-recipe interaction. However, we have a variety of features available to us for each recipe, such as nutritional information and cooking time.
- **Review Rating**: LightGCN only normalizes the weights of each interaction by the degree of each node. However, we have access to the actual rating that each user gave to each recipe, which indicates how strong the interaction actually was.

To address these issues, we propose a new GNN architecture called RecGCN. RecGCN is a slight modification of LightGCN with two main changes. First, we scale the graph convolution layer by the review's score.

$$
  \begin{aligned}
    U_u & \gets \sum_{r \in N(u)} \frac{\text{Review}(u, r)}{\sqrt{d_u d_r}} R_r \\
    R_r & \gets \sum_{u \in N(r)} \frac{\text{Review}(u, r)}{\sqrt{d_r d_u}} U_u
  \end{aligned}
$$

Second, we define a recipe feature matrix $F$. This matrix contains a row for each recipe, and a column for each feature. We have two options for how to use this matrix.

- **Opposite Embedding**: We can substitute the recipe embedding matrix $R$ with the recipe feature matrix $F$. Due to the nature of the graph convolution layer, we also need to redefine the user embedding matrix $U$ to have the same number of columns as $F$. This will force the model to learn embeddings for the user for each recipe-defined feature.
- **Combination**: We can concatenate the recipe embedding matrix $R$ with the recipe feature matrix $F$. This will allow the model to learn embeddings for each recipe-defined feature, as well as latent embeddings for the recipe as a whole. Due to the nature of the graph convolution layer, we also need to redefine the user embedding matrix $U$ to have the same number of columns as $F$ + $R$.
