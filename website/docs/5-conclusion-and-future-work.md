# Conclusion

At the end of the day, the success of this project is based on whether or not it can be used to solve the problem of meal planning. To that end, we believe that our models are able to provide a good starting point for a full meal planning system. The recommendation engines we built are able to recommend recipes that a user is likely to enjoy, particularly if the rest of the system is built with the weaknesses we found in mind. We envision a system that allows users to reject recipes for their weekly meal plan, and then have the system automatically recommend new recipes to replace them. Our best model is able to recommend a recipe the user enjoys within roughly 50 iterations _even with a relatively small history_!

The best part of such a system is that those rejections can be used to improve the recommendation engine. The system can learn from the user's rejections and use that information to improve the recommendations in the future. This not only results in better recipes for the user, but solves the sparsity problem that we ran into with all of our models!

# Future Work

<iframe
    height="800"
    width="450"
    src="https://www.figma.com/embed?embed_host=share&url=https%3A%2F%2Fwww.figma.com%2Fproto%2FDHAw5qL8tDpn4SssETdUWi%2FHiFi-Prototype%3Fpage-id%3D65%253A1559%26node-id%3D65%253A2007%26viewport%3D394%252C-227%252C0.26%26scaling%3Dscale-down%26starting-point-node-id%3D65%253A2007"
    allowfullscreen
></iframe>

We plan to continue this work in a commercial setting. We believe that this project can be a core component of a full system that helps people save time and money by automating the process of meal planning and grocery shopping.

Future work includes:

- **User Interface**: Performing a user study to determine the best way to present the recommendations to the user. One early prototype of the user interface is shown above!
- **Software Development**: Developing the core app to be used by the user.
- **Alternative Recommendation Methods**: Investigate other recommendation methods, such as Facebook's Deep Learning Recommendation Model (DLRM)

As for the research side of this project, we plan to continue investigating the following:

- **RecGCN Review Weighting**: Investigate alternative methods of weighting the reviews, such scaling the weights by the user's average or normalizing the new edge weight by the sum of the reviews.
- **Recipe Features**: Investigate alternative methods of using recipe features, such as using a neural network to learn the best way to combine the features with the embeddings.
