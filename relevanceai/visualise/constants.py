
from typing import List, Sequence, Union, Dict, Any, Tuple, Optional
from typing_extensions import Literal, get_args

JSONDict = Dict[str, Any]

DIM_REDUCTION = Literal["pca", "tsne", "umap", "ivis"]
DIM_REDUCTION_DEFAULT_ARGS: Dict[Any, Any] = {
    'pca': {
        "svd_solver": "auto",
        "random_state": 42
    },
    'tsne': {
        "init": "pca",
        "n_iter": 500,
        "learning_rate": 100,
        "perplexity": 30,
        "random_state": 42,
    },
    'umap': {
        "n_neighbors": 10,
        "min_dist": 0.1,
        "random_state": 42,
        "transform_seed": 42,
    },
    'ivis': {
        "k": 15, 
        "model": "maaten", 
        "n_epochs_without_progress": 2
    }
}


CLUSTER_NUMERIC = Literal["kmeans", "kmedoids"]
CLUSTER_CATEGORICAL = Literal["kmodes"]
CLUSTER_MIXED = Literal["kprotoypes"]
CLUSTER: List = [c for c in (list(get_args(CLUSTER_NUMERIC))+ list(get_args(CLUSTER_CATEGORICAL))+ list(get_args(CLUSTER_MIXED)))]


CLUSTER_DEFAULT_ARGS: Dict[Any, Any] = {
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
    # 'kmodes': {
    #     "init": "Huang", 
    #     "verbose": 1,
    #     "random_state": 42,
    #     "n_jobs": -1
    # },
    # 'kprototypes': {
    #     "init": "Huang", 
    #     "verbose": 1,
    #     "random_state": 42,
    #     "n_jobs": -1
    # }
}
