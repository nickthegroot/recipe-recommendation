# Collaborative Filtering

Our first attempt at recommending recipes to users was to use a K-Means collaborative filtering model. K-Means collaborative filtering is a classical algorithm that recommends items to users under the assumption that users with similar tastes will like similar items.

## How It Works

The model works by first computing a similarity score between users. We chose to use the pearson correlation coefficient as our similarity metric, due to its ability to handle non-centered ratings (such as the 1-5 ratings we used). The model then predicts a score for each user-recipe interaction based on a weighted average of the ratings of users.

$$
  \text{Recipe Score}(u, r)
  = \mu_{u} + \frac{
    \sum_{u'} \text{Similarity}(u, u')
    \cdot (\text{Rating}(u', r) - \mu_{u'})
  }{
    \sum_{u'} \text{Similarity}(u, u')
  }
$$

Recommendations are then as simple as sorting the recipes by their predicted score and choosing the top $k$.

We implemented the algorithm in two ways for this project.

- `surprise`: An open-source Python library that implements a variety of recommendation algorithms. Their implementation of Centered K-NN closely mirrors the algorithm described above, thus making it an easy choice for our first model.
- `GSQL`: We also implemented the model inside TigerGraph using GSQL. This allowed us to run recommendations _directly inside the database_, allowing for lightning fast recommendations suitable for real-time applications.

One important distinction between our implementations and many others is that we did not filter out recipes that users have already interacted with. We decided to do this because we wanted to be able to recommend recipes that users have already interacted with, but may have not tried in awhile. This is particularly important to note when considering the results of our prior user research, which suggested that users generally prefer to cook recipes they've already cooked before.
