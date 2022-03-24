# Auto

Functionality defined in auto should be all encompassing and should be defined with the prefix `auto_`.

`auto_cluster` should run many clustering algorithms with different hyperparameters and select one that optimises unsupervised clustering metrics

Likewise, `auto_dr` should run many dimensionality reductio algorithms with different hyperparameters and select one that optimises explained variance, and other reduction metrics.

TODO: `auto_vectorize` will perform a similar operation among all text/image/audio vectorizers we support
