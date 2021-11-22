# -*- coding: utf-8 -*-

from abc import abstractmethod
import pandas as pd
import numpy as np
import json
import warnings

from dataclasses import dataclass

from typing import List, Union, Dict, Any, Tuple, Optional
from typing_extensions import Literal

from relevanceai.base import Base
from relevanceai.logger import LoguruLogger
from relevanceai.visualise.constants import CLUSTER, CLUSTER_DEFAULT_ARGS


class ClusterBase(LoguruLogger):
    def __call__(self, *args, **kwargs):
        return self.fit_transform(*args, **kwargs)

    @abstractmethod
    def fit_transform(self, 
            vectors: np.ndarray, 
            cluster_args: Dict[Any, Any],
    ) -> np.ndarray:
        raise NotImplementedError


class CentroidCluster(ClusterBase):
    def __call__(self, *args, **kwargs):
        return self.fit_transform(*args, **kwargs)

    @abstractmethod
    def fit_transform(self, 
            vectors: np.ndarray, 
            cluster_args: Dict[Any, Any],
            k: Union[None, int] = None
    ) -> np.ndarray:
        raise NotImplementedError


class DensityCluster(ClusterBase):
    def __call__(self, *args, **kwargs):
        return self.fit_transform(*args, **kwargs)

    @abstractmethod
    def fit_transform(self, 
            vectors: np.ndarray, 
            cluster_args: Dict[Any, Any],
            min_cluster_size: Union[None, int] = None
    ) -> np.ndarray:
        raise NotImplementedError


class KMeans(CentroidCluster):
    def fit_transform(self, 
        vectors: np.ndarray, 
        cluster_args: Optional[Dict[Any, Any]] = CLUSTER_DEFAULT_ARGS['kmeans'], 
        k: Union[None, int] = 10
    ) -> np.ndarray:
        from sklearn.cluster import MiniBatchKMeans
        self.logger.debug(f"{cluster_args}")
        km = MiniBatchKMeans(n_clusters=k, **cluster_args).fit(vectors)
        cluster_labels = km.labels_
        # cluster_centroids = km.cluster_centers_
        return cluster_labels


class KMedoids(CentroidCluster):
    def fit_transform(self, 
        vectors: np.ndarray, 
        cluster_args: Optional[Dict[Any, Any]] = CLUSTER_DEFAULT_ARGS['kmedoids'], 
        k: Union[None, int] = 10,
    ) -> np.ndarray:
        try:
            from sklearn_extra.cluster import KMedoids
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"{e}\nInstall umap\n \
                pip install -U relevanceai[kmedoids]"
            )
        self.logger.debug(f"{cluster_args}")
        km = KMedoids(n_clusters=k, **cluster_args).fit(vectors)
        cluster_labels = km.labels_
        # cluster_centroids = km.cluster_centers_
        return cluster_labels


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
                return KMeans().fit_transform(vectors=vectors, cluster_args=cluster_args)
            elif cluster == "kmedoids":
                return KMedoids().fit_transform(vectors=vectors, cluster_args=cluster_args)
        elif cluster == "hdbscan":
            return HDBSCAN().fit_transform(vectors=vectors, cluster_args=cluster_args)
        
    elif isinstance(cluster, ClusterBase):
        return cluster().fit_transform(vectors=vectors, cluster_args=cluster_args)
    