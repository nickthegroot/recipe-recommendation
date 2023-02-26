# Collaborative Filtering

## How It Works

For any given user $u$:

1. Find all recipes $r$ that $u$ has interacted with.
1. Find all users $O$ that have interacted with the same recipes as $u$ ($r$).
1. Calculate the average rating for each user in $O$ ($\bar{R}_o$)
1. Calculate the similarity score for each user in $O$ using pearson correlation, where $I_u$ is the set of recipes that $u$ has interacted with:
   $$
          Sim(u, o) = \frac
          {
            \sum_{i \in I_u \cup I_o} (R_{u,i} - \bar{R_u})(R_{o,i} - \bar{R_o})
          }{
            \sqrt{
              \sum_{i \in I_u \cup I_o} (R_{u,i} - \bar{R_u})^2
              \sum_{i \in I_u \cup I_o} (R_{o,i} - \bar{R_o})^2
            }
          }
   $$
1. Calculate the rank of each recipe $r$ using the following formula, where $U_r$ is the set of users that have interacted with recipe $r$:
   $$
   Rank(r) = \sum_{o \in U_r} Sim(u, o) \cdot (R_{o, r} - \bar{R_o} + 1)
   $$

While such a model isn't exclusive to graph-based networks, using one does allow the model to perform more efficiently in finding the relevant users/items.

It's important to note that while most recommendation systems filter out recipes that users have already interacted with, we decided to keep them in the model. This is because we wanted to be able to recommend recipes that users have already interacted with, but may have not tried in awhile. This is particularly important to note when considering the results of our user research, which suggested that users generally prefer to cook recipes they've already cooked before.

## Results

:::info
WIP: TODO
:::
