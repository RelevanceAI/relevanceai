
from typing import List, Union, Dict, Any, Tuple
from typing_extensions import Literal

JSONDict = Dict[str, Any]
DR = Literal["pca", "tsne", "umap", "ivis"]
CLUSTER_NUMERIC = Literal["kmeans", "kmedoids",  None]
CLUSTER_CATEGORICAL = Literal["kmodes",  None]
CLUSTER_MIXED = Literal["kprotoypes", None]
CLUSTER = Union[CLUSTER_NUMERIC, CLUSTER_CATEGORICAL, CLUSTER_MIXED]

CLUSTER_DEFAULT_ARGS = {
    'kmeans': {
        "init": "k-means++", 
        "verbose": 1,
        "compute_labels": True,
        "max_no_improvement": 2
    },
    'kmedoids': {
        "metric": "euclidean",
        "init": "k-medoids++",
        "random_state": 42,
        "method": "pam"
    },
    'kmodes': {
        "init": "Huang", 
        "verbose": 1,
        "random_state": 42,
        "n_jobs": -1
    },
    'kprototypes': {
        "init": "Huang", 
        "verbose": 1,
        "random_state": 42,
        "n_jobs": -1
    }
}