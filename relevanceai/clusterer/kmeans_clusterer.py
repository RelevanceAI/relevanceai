"""KMeans Clustering
"""
import numpy as np
from typing import Union, List, Optional

from relevanceai.clusterer.clusterer import ClusterFlow
from relevanceai.clusterer.cluster_base import ClusterBase


class KMeansModel(ClusterBase):
    def __init__(
        self,
        k=10,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=1e-4,
        verbose=0,
        random_state=None,
        copy_x=True,
        algorithm="auto",
    ):
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.algorithm = algorithm
        self.n_clusters = k

    def _init_model(self):
        from sklearn.cluster import KMeans

        self.km = KMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            verbose=self.verbose,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            copy_x=self.copy_x,
            algorithm=self.algorithm,
        )
        return

    def fit_transform(self, vectors: Union[np.ndarray, List]):
        """
        Fit and transform transform the vectors
        """
        if not hasattr(self, "km"):
            self._init_model()
        self.km.fit(vectors)
        cluster_labels = self.km.labels_.tolist()
        # cluster_centroids = km.cluster_centers_
        return cluster_labels

    @property
    def metadata(self):
        """Editing the metadata of the function"""
        return {
            "n_clusters": self.n_clusters,
            "init": self.init,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "random_state": self.random_state,
            "copy_x": self.copy_x,
            "algorithm": self.algorithm,
        }


class KMeansClusterFlow(ClusterFlow):
    """
    KMeans Clustering Flow
    """

    def __init__(
        self,
        alias: str,
        project: str,
        api_key: str,
        k: Union[None, int] = 10,
        init: str = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        verbose: bool = True,
        random_state: Optional[int] = None,
        copy_x: bool = True,
        algorithm: str = "auto",
        cluster_field: str = "_cluster_",
    ):
        model = KMeansModel(
            k=k,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            copy_x=copy_x,
            algorithm=algorithm,
        )
        super().__init__(
            model=model,
            alias=alias,
            cluster_field=cluster_field,
            project=project,
            api_key=api_key,
        )
