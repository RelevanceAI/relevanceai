# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import json
import warnings

from dataclasses import dataclass

from typing import List, Union, Dict, Any, Tuple, Optional
from typing_extensions import Literal

from relevanceai.base import Base
from relevanceai.visualise.constants import CLUSTER, CLUSTER_DEFAULT_ARGS


@dataclass
class Cluster(Base):
    """Clustering Visualisation Class"""

    def __init__(
        self,
        project: str,
        api_key: str,
        base_url: str,
        vectors: np.ndarray,
        cluster: CLUSTER,
        cluster_args: Union[None] = None,
        k: Union[None, int] = None,
    ):
        self.project = project
        self.api_key = api_key
        self.base_url = base_url
        super().__init__(project, api_key, base_url)

        self.vectors = vectors
        self.cluster = cluster
        self.cluster_args = cluster_args
        self.k = k
        if k is None and cluster_args is None:
            self.k = self._choose_k(vectors)
        if cluster_args is None:
            self.cluster_args = CLUSTER_DEFAULT_ARGS[cluster]
        else:
            if k is None and "n_clusters" not in cluster_args.keys():
                self.k = self._choose_k(vectors)

        if self.cluster in ["kmeans", "kmedoids", "hdbscan"]:
            self.cluster_labels, self.c_centroids = self._cluster_vectors(
                vectors=self.vectors,
                cluster=self.cluster,
                cluster_args=self.cluster_args,
            )

    def _choose_k(self, vectors: np.ndarray):
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

    def _cluster_vectors(
        self,
        vectors: np.ndarray,
        cluster: CLUSTER,
        cluster_args: Union[None, dict],
    ) -> Tuple[List[str], List[int]]:
        """
        Cluster method for numerical data
        """
        self.logger.info(f"Performing {cluster} clustering with {self.k} clusters ... ")
        if cluster == "kmeans":
            from sklearn.cluster import MiniBatchKMeans
            self.logger.debug(f"{json.dumps(cluster_args, indent=4)}")
            km = MiniBatchKMeans(n_clusters=self.k, **cluster_args).fit(vectors)
            cluster_labels = km.labels_
            cluster_centroids = km.cluster_centers_
        elif cluster == "kmedoids":
            try:
                from sklearn_extra.cluster import KMedoids
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    f"{e}\nInstall umap\n \
                    pip install -U relevanceai[kmedoids]"
                )
            self.logger.debug(f"{json.dumps(cluster_args, indent=4)}")
            km = KMedoids(n_clusters=self.k, **cluster_args).fit(vectors)
            cluster_labels = km.labels_
            cluster_centroids = km.cluster_centers_
        elif cluster == "hdbscan":
            try:
                from hdbscan import HDBSCAN
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    f"{e}\nInstall hdbscan\n \
                    pip install -U relevanceai[hdbscan]"
                )
            self.logger.debug(f"{json.dumps(cluster_args, indent=4)}")
            hdbscan = HDBSCAN(n_clusters=self.k, **cluster_args).fit(vectors)
            cluster_labels = hdbscan.labels_
            cluster_centroids = []  ## HDBSCAN does not provide centroids

        return cluster_labels, cluster_centroids

    # def _cluster_categorical(
    #     self,
    #     df: pd.DataFrame,
    #     cluster: CLUSTER,
    #     cluster_args: Union[None, dict],
    #     categorical_idx: Union[None, List[int]] = None
    # ):
    #     """
    #     Clustering categorical data types
    #     """
    #     import warnings
    #     warnings.warn(f'This method is still not yet implemented')

    #     # if cluster == "kmodes":
    #     # from kmodes.kmodes import KModes
    #     #     if categorical_idx is None:
    #     #         categorical_columns = list(df.select_dtypes('object').columns)
    #     #         categorical_idx = [df.columns.get_loc(col) for col in categorical_columns]
    #     #         self.logger.debug(f'{json.dumps(cluster_args, indent=4)}')
    #     #         km = KModes(**cluster_args).fit_predict(df.to_numpy(), categorical=categorical_idx))
    #     #         cluster_labels = km.labels_
    #     #         cluster_centroids = km.cluster_centroids_
    #     # cluster_labels = [ f'c_{c}' for c in cluster_labels ]
    #     # return cluster_labels, cluster_centroids'
    #     return NotImplementedError

    # def _cluster_mixed(
    #     self,
    #     df: pd.DataFrame,
    #     cluster: CLUSTER,
    #     cluster_args: Union[None, dict],
    #     categorical_idx: Union[None, List[int]] = None
    # ):
    #     """
    #     Clustering mixed data types
    #     """
    #     import warnings
    #     warnings.warn(f'This method is still not yet implemented')

    #     # if cluster == "kprototypes":
    #     # from kmodes.kprototypes import KPrototypes
    #     #     if categorical_idx is None:
    #     #         categorical_columns = list(df.select_dtypes('object').columns)
    #     #         categorical_idx = [df.columns.get_loc(col) for col in categorical_columns]
    #     #         self.logger.debug(f'{json.dumps(cluster_args, indent=4)}')
    #     #         kp = KPrototypes(**cluster_args).fit_predict(df.to_numpy(), categorical=categorical_idx)
    #     #         cluster_labels = kp.labels_
    #     #         cluster_centroids = kp.cluster_centroids_
    #     # cluster_labels = [ f'c_{c}' for c in cluster_labels ]
    #     # return cluster_labels, cluster_centroids
    #     return NotImplementedError
