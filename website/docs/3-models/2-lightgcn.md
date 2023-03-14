# LightGCN

## How It Works

At its core, LightGCN is similar to traditional matrix factorization methods. It uses a user embedding matrix $U$ and a recipe embedding matrix $R$ to represent the latent features of users and recipes. The score for a particular user-recipe interaction is then computed as the dot product of the user and recipe embeddings.

$$
  \text{Recipe Score}(u, r) = U_u^TR_r
$$

This equation can be thought of as the sum of a user's preferences for each feature multiplied by the recipe's affinity for that feature. For example, if a latent feature corresponded with "spice", users who like spicy food would have a high value for that feature as would recipes that are spicy. Thus, the resulting score for such a user-recipe interaction would be high.

The difference between LightGCN and traditional matrix factorization methods is that LightGCN learns the embedding matrices in context of the graph. Before computing the score for a user-recipe interaction, LightGCN first passes the user and recipe embeddings through a "light graph convolution" layer. This layer uses the features of each node's neighbor(s) in a normalized weighted sum to compute the new embedding for that node.

$$
  \begin{aligned}
    U_u & \gets \sum_{r \in N(u)} \frac{1}{\sqrt{d_u d_r}} R_r \\
    R_r & \gets \sum_{u \in N(r)} \frac{1}{\sqrt{d_r d_u}} U_u
  \end{aligned}
$$

This process can then be repeated and averaged to allow for more complex modeling.

During training, we utilized Baysean Personalized Ranking (BPR) loss to optimize the each of the embedding matrices. BPR loss is a loss function that is commonly used for recommendation tasks. It is designed to maximize the score of positive/real user-recipe interactions and minimize the score of negative/fake user-recipe interactions. To help prevent overfitting, we also added a regularization term to the loss function.

$$
  \begin{aligned}
    \text{BPR Loss}       & = -\frac{1}{N} \sum_{(u, r, r') \in D} \log \sigma(U_u^TR_r - U_u^TR_{r'})             \\
    \text{Regularization} & = \lambda \cdot \left( \sum_{u} \Vert U_u \Vert^2 + \sum_{r} \Vert R_r \Vert^2 \right)
  \end{aligned}
$$
