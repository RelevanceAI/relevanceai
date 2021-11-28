# -*- coding: utf-8 -*-

from abc import abstractmethod
import pandas as pd
import numpy as np
import json
import warnings

from dataclasses import dataclass

from typing import List, Union, Dict, Any, Tuple, Optional
from typing_extensions import Literal

from doc_utils import DocUtils

from relevanceai.base import Base
from relevanceai.logger import LoguruLogger
from relevanceai.visualise.constants import CLUSTER, CLUSTER_DEFAULT_ARGS


class ClusterBase(LoguruLogger, DocUtils):
    def __call__(self, *args, **kwargs):
        return self.fit_transform(*args, **kwargs)

    @abstractmethod
    def fit_transform(self, vectors):
        """Return the 
        """
        raise NotImplementedError
    
    def fit_documents(
        self,
        vector_field: list,
        docs: list,
        alias: str="default",
        cluster_field: str="_clusters_"
    ):
        """
        Train clustering algorithm on documents and then store the labels
        inside the documents.
        
        Parameters
        -----------
        vector_field: list
            The vector field of the documents
        docs: list
            List of documents to run clustering on
        alias: str
            What the clusters can be called
        cluster_field: str
            What the cluster fields should be called
        """
        if len(vector_field) == 1:
            vectors = self.get_field_across_documents(vector_field[0], docs)
        else:
            raise ValueError("We currently do not support more than 1 vector field yet. This will be supported in the future.")
        cluster_labels = self.fit_transform(vectors)
        # Label the clusters
        cluster_labels = self._label_clusters(cluster_labels)
        self.set_field_across_documents(
            f"{cluster_field}.{vector_field}.{alias}", cluster_labels, docs
        )
        return docs

    def to_metadata(self):
        """You can also store the metadata of this clustering algorithm
        """
        raise NotImplementedError
    
    def _label_cluster(self, label: Union[int, str]):
        if isinstance(label, (int, float)):
            return "cluster_" + str(label)
        return str(label)

    def _label_clusters(self, labels):
        return [self._label_cluster(x) for x in labels]

class CentroidCluster(ClusterBase):
    def __call__(self, *args, **kwargs):
        return self.fit_transform(*args, **kwargs)

    @abstractmethod
    def fit_transform(self, vectors):
        raise NotImplementedError
    
    @abstractmethod
    def get_centers(self) -> Union[np.ndarray, List[list]]:
        """Get centers for the centroid-based clusters
        """
        raise NotImplementedError
    
    def get_centroid_docs(self) -> List:
        """Get the centroid documents to store.
        """
        self.centers = self.get_centers()
        if isinstance(self.centers, np.ndarray):
            self.centers = self.centers.tolist()
        return [
            {
                "_id": f"cluster_{i}",
                "centroid_vector_": self.centers[i]
            } for i in range(len(self.centers))
        ]

class DensityCluster(ClusterBase):
    def __call__(self, *args, **kwargs):
        return self.fit_transform(*args, **kwargs)

    def fit_transform(self, vectors):
        raise NotImplementedError


class KMeans(CentroidCluster):
    def __init__(
        self,
        k: Union[None, int] = 10,
        init: str = "k-means++",
        verbose: bool = True,
        compute_labels: bool = True,
        max_no_improvement: int=2
     ):
        """
        Kmeans Centroid Clustering

        Parameters
        ------------
        k: int
            The number of clusters
        init: str
            The optional parameter to be clustering
        verbose: bool
            If True, will print what is happening 
        compute_labels: bool
            If True, computes the labels of the cluster 
        max_no_improvement: int
            The maximum number of improvemnets
        """
        self.k = k
        self.init = init
        self.verbose = verbose
        self.compute_labels = compute_labels
        self.max_no_improvement = max_no_improvement

    def _init_model(self):
        from sklearn.cluster import MiniBatchKMeans
        self.km = MiniBatchKMeans(
            n_clusters=self.k, 
            init=self.init,
            verbose=self.verbose,
            compute_labels=self.compute_labels,
            max_no_improvement=self.max_no_improvement
        )
        return

    def fit_transform(
        self,
        vectors: Union[np.ndarray, List]
    ):
        """
        Fit and transform transform the vectors
        """
        if not hasattr(self, "km"):
            self._init_model()
        self.km.fit(vectors)
        cluster_labels = self.km.labels_
        # cluster_centroids = km.cluster_centers_
        return cluster_labels

    def get_centers(self) -> np.ndarray:
        """Returns a numpy array of clusters
        """
        return self.km.cluster_centers_
    
    def to_metadata(self):
        """Editing the metadata of the function
        """
        return {
            "k": self.k,
            "init": self.init,
            "verbose": self.verbose,
            "compute_labels": self.compute_labels,
            "max_no_improvement": self.max_no_improvement
        }

# class KMedoids(CentroidCluster):
#     def fit_transform(self, 
#         vectors: np.ndarray, 
#         cluster_args: Optional[Dict[Any, Any]] = CLUSTER_DEFAULT_ARGS['kmedoids'], 
#         k: Union[None, int] = 10,
#     ) -> np.ndarray:
#         try:
#             from sklearn_extra.cluster import KMedoids
#         except ModuleNotFoundError as e:
#             raise ModuleNotFoundError(
#                 f"{e}\nInstall umap\n \
#                 pip install -U relevanceai[kmedoids]"
#             )
#         self.logger.debug(f"{cluster_args}")
#         km = KMedoids(n_clusters=k, **cluster_args).fit(vectors)
#         cluster_labels = km.labels_
#         # cluster_centroids = km.cluster_centers_
#         return cluster_labels


class HDBSCAN(DensityCluster):
    def fit_transform(self, 
        vectors: np.ndarray, 
        cluster_args: Optional[Dict[Any, Any]] = CLUSTER_DEFAULT_ARGS['hdbscan'], 
        min_cluster_size: Union[None, int] = 10,
    ) -> np.ndarray:
        try:
            from hdbscan import HDBSCAN
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"{e}\nInstall hdbscan\n \
                pip install -U relevanceai[hdbscan]"
            )
        self.logger.debug(f"{cluster_args}")
        hdbscan = HDBSCAN(min_cluster_size=min_cluster_size, **cluster_args).fit(vectors)
        cluster_labels = hdbscan.labels_
  
        return cluster_labels


def _choose_k(vectors: np.ndarray):
    """ "
    Choose k clusters
    """
    # Partitioning methods
    # if check_type(self.cluster, CLUSTER_NUMERIC):
    """
    Scaled_inertia = inertia(k)/inertia(k=1) + (a * K)
    where a is penalty factor of num_clusters
    """
    warnings.warn("This method is not implemented yet k=10")
    return 10


def cluster(
    vectors: np.ndarray,
    cluster: Union[CLUSTER, ClusterBase],
    cluster_args: Union[None, dict],
    k: Union[None, int] = None,
) -> np.ndarray:
    """
    Cluster vectors
    """
    if isinstance(cluster, str):
        if cluster_args is None:
            cluster_args = CLUSTER_DEFAULT_ARGS[cluster]
        if cluster in ['kmeans', 'kmedoids']:
            if (k is None and cluster_args is None) \
                or ("n_clusters" not in cluster_args.keys()):
                k = _choose_k(vectors)
            if cluster == "kmeans":
                return KMeans(**cluster_args).fit_transform(vectors=vectors)
            elif cluster == "kmedoids":
                raise NotImplementedError
                # return KMedioids().fit_transform(vectors=vectors, cluster_args=cluster_args)
        elif cluster == "hdbscan":
            return HDBSCAN().fit_transform(vectors=vectors, cluster_args=cluster_args)
        
    elif isinstance(cluster, ClusterBase):
        return cluster().fit_transform(vectors=vectors, cluster_args=cluster_args)
    raise ValueError("Not valid cluster input.")
    