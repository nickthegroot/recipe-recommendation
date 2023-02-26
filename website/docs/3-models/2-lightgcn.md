# LightGCN

## How It Works

LightGCN works by attempting to learn "latent features" for users and recipes. It does so by iteratively updating the latent features of each user and recipe by taking a weighted average of the latent features of their neighbors.

$$
    \mathbf{x}_i = \sum_{j \in \mathcal{N}(i)}
    \frac{1}{\sqrt{\deg(i)\deg(j)}}\mathbf{x}^{(l)}_j
$$

We train the system to produce better latent features through the following algorithm:

1. Assign a random vector to every user and recipe.
2. Calculate the new features as the average of $n$ iterations
3. Calculate the edge features as the product of the user and recipe latent features
4. Randomly sample two recipes for a random user
   - One that they **have** eaten before (the _positive_ sample)
   - One that they **have not** eaten before (the _negative_ sample)
5. Calculate the loss as the negative log probability of the difference between their edge features

## Results

:::info
WIP: TODO
:::
