# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import json

from dataclasses import dataclass
from pandas.core.arrays import categorical


from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn_extra.cluster import KMedoids
# from kmodes.kmodes import KModes
# from kmodes.kprototypes import KPrototypes


from typing import List, Union, Dict, Any, Tuple, Optional
from typing_extensions import Literal

from vecdb.base import Base
from vecdb.visualise.constants import *


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
        cluster_args: Union[None, JSONDict] = None,
        k: Union[None, int] = None
    ):  

        self.project = project
        self.api_key = api_key
        self.base_url = base_url
        super().__init__(project, api_key, base_url)

        self.vectors = vectors
        self.cluster = cluster
        self.cluster_args = cluster_args
        self.k = k
        if k is None:
            self.k = self._choose_k(vectors)
        if cluster_args is None:
            self.cluster_args = {'n_clusters': self.k, **CLUSTER_DEFAULT_ARGS[cluster]}
        
        self.c_labels, self.c_centroids = self._cluster_vectors(vectors, cluster, cluster_args)
    

    def _choose_k(
        self,
        vectors: np.ndarray
    ):
        """"    
        Choose k clusters
        """
        # Partitioning methods
        if isinstance(self.cluster, CLUSTER_NUMERIC):
            ### TODO: Implement scaled inertia algo to find best k
            return 10


    def _cluster_vectors(
        self,
        vectors: np.ndarray,
        cluster: CLUSTER_NUMERIC,
        cluster_args: Union[None, JSONDict],
    ) -> Tuple[List[str], List[int]]:
        """
        Cluster method for numerical data
        """
        self.logger.info(f'Performing {cluster} clustering with {self.k} clusters ... ')
        if cluster == 'kmeans':
            self.logger.debug(f'{json.dumps(cluster_args, indent=4)}')
            km = MiniBatchKMeans(**cluster_args).fit(vectors)
            c_labels = km.labels_
            cluster_centroids = km.cluster_centers_
        elif cluster == 'kmedoids':
            self.logger.debug(f'{json.dumps(cluster_args, indent=4)}')
            km = KMedoids(**cluster_args).fit(vectors)
            c_labels = km.labels_
            cluster_centroids = km.cluster_centers_
        c_labels = [ f'c_{c}' for c in c_labels ]
        return c_labels, cluster_centroids
    


    def _cluster_categorical(
        self,
        df: pd.DataFrame,
        cluster: CLUSTER_CATEGORICAL,
        cluster_args: Union[None, JSONDict],
        categorical_idx: Union[None, List[int]] = None
    ):
        """
        Clustering categorical data types
        """
        # if cluster == "kmodes":
        #     if categorical_idx is None:
        #         categorical_columns = list(df.select_dtypes('object').columns)
        #         categorical_idx = [df.columns.get_loc(col) for col in categorical_columns]
        #         self.logger.debug(f'{json.dumps(cluster_args, indent=4)}')
        #         km = KModes(**cluster_args).fit_predict(df.to_numpy(), categorical=categorical_idx))
        #         c_labels = km.labels_
        #         cluster_centroids = km.cluster_centroids_
        # c_labels = [ f'c_{c}' for c in c_labels ]
        # return c_labels, cluster_centroids
        return NotImplementedError
    
        
    def _cluster_mixed(
        self,
        df: pd.DataFrame,
        cluster: CLUSTER_MIXED,
        cluster_args: Union[None, JSONDict],
        categorical_idx: Union[None, List[int]] = None
    ):
        """
        Clustering mixed data types
        """
        # if cluster == "kprototypes":
        #     if categorical_idx is None:
        #         categorical_columns = list(df.select_dtypes('object').columns)
        #         categorical_idx = [df.columns.get_loc(col) for col in categorical_columns]
        #         self.logger.debug(f'{json.dumps(cluster_args, indent=4)}')
        #         kp = KPrototypes(**cluster_args).fit_predict(df.to_numpy(), categorical=categorical_idx)
        #         c_labels = kp.labels_
        #         cluster_centroids = kp.cluster_centroids_
        # c_labels = [ f'c_{c}' for c in c_labels ]
        # return c_labels, cluster_centroids
        return NotImplementedError