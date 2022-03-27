from typing import Dict, Any
from typing_extensions import Literal

from joblib.memory import Memory

dict = Dict[str, Any]

DIM_REDUCTION = Literal["pca", "tsne", "umap", "ivis"]
DIM_REDUCTION_DEFAULT_ARGS: Dict[Any, Any] = {
    "pca": {"svd_solver": "auto", "random_state": 42},
    "tsne": {
        "init": "pca",
        "n_iter": 500,
        "learning_rate": 100,
        "perplexity": 30,
        "random_state": 42,
    },
    "umap": {
        "n_neighbors": 10,
        "min_dist": 0.1,
        "random_state": 42,
        "transform_seed": 42,
    },
    "ivis": {"k": 15, "model": "maaten", "n_epochs_without_progress": 2},
}


GROUPBY_MAPPING = {"text": "category", "numeric": "numeric"}


CLUSTER = Literal["kmeans", "kmedoids", "hdbscan"]  # "kmodes", "kprototypes", None]
CLUSTER_DEFAULT_ARGS: Dict[str, Dict[str, Any]] = {
    "kmeans": {
        "k": 10,
        "init": "k-means++",
        "verbose": 0,
    },
    "kmedoids": {
        "metric": "euclidean",
        "init": "k-medoids++",
        "random_state": 42,
        "method": "pam",
    },
    "hdbscan": {
        "algorithm": "best",
        "alpha": 1.0,
        "approx_min_span_tree": True,
        "gen_min_span_tree": False,
        "leaf_size": 40,
        "memory": Memory(cachedir=None),
        "metric": "euclidean",
        "min_samples": None,
        "p": None,
    },
}

NEAREST_NEIGHBOURS = Literal["cosine", "l2"]
CENTROID_DISTANCES = Literal["cosine", "l2"]


SILHOUETTE_INFO = """
Good clusters have clusters which are highly seperated and elements within which are highly cohesive. <br/>
<b>Silohuette Score</b> is a metric from <b>-1 to 1</b> that calculates the average cohesion and seperation of each element, with <b>1</b> being clustered perfectly, <b>0</b> being indifferent and <b>-1</b> being clustered the wrong way"""

RAND_INFO = """Good clusters have elements, which, when paired, belong to the same cluster label and same ground truth label. <br/>
<b>Rand Index</b> is a metric from <b>0 to 1</b> that represents the percentage of element pairs that have a matching cluster and ground truth labels with <b>1</b> matching perfect and <b>0</b> matching randomly. <br/> <i>Note: This measure is adjusted for randomness so does not equal the exact numerical percentage.</i>"""

HOMOGENEITY_INFO = """Good clusters only have elements from the same ground truth within the same cluster<br/>
<b>Homogeneity</b> is a metric from <b>0 to 1</b> that represents whether clusters contain only elements in the same ground truth with <b>1</b> being perfect and <b>0</b> being absolutely incorrect."""

COMPLETENESS_INFO = """Good clusters have all elements from the same ground truth within the same cluster <br/>
<b>Completeness</b> is a metric from <b>0 to 1</b> that represents whether clusters contain all elements in the same ground truth with <b>1</b> being perfect and <b>0</b> being absolutely incorrect."""

METRIC_DESCRIPTION = {
    "Silhouette Score": SILHOUETTE_INFO,
    "Rand Score": RAND_INFO,
    "Homogeneity": HOMOGENEITY_INFO,
    "Completeness": COMPLETENESS_INFO,
}
