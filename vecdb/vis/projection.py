# -*- coding: utf-8 -*-
import sys
import time

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.cluster import KMeans, MiniBatchKMeans
from ivis import Ivis

import plotly.graph_objs as go

from dataclasses import dataclass

from vecdb.base import Base
from vecdb.vecdb_logging import create_logger
from api.datasets import Datasets

from typing import List, Union, Dict, Any, Literal, Callable, Tuple

JSONDict = Dict[str, Any]
DR = Literal["pca", "tsne", "umap", "ivis"]
CLUSTER = Literal["kmeans", "kmodes", None]

LOG = create_logger()


@dataclass
class Projection(Base):
    """Projection Class"""

    def __init__(
        self,
        project: str,
        api_key: str,
        base_url: str,
    ):
        self.project = project
        self.api_key = api_key
        self.base_url = base_url

    def _retrieve_documents(
        self, dataset_id: str, number_of_documents: int = 100, page_size: int = 1000
    ) -> List[JSONDict]:
        """
        Retrieve all documents from dataset
        """
        LOG.info(f'Retrieving {number_of_documents} documents from {dataset_id} ...')
        dataset = Datasets(self.project, self.api_key, self.base_url)
        resp = dataset.documents.list(
            dataset_id=dataset_id, page_size=page_size
        )  # Initial call
        _cursor = resp["cursor"]
        _page = 0
        data = []
        while _cursor:
            LOG.debug(f'Paginating {_page} page size {page_size} ...')
            resp = dataset.documents.list(
                dataset_id=dataset_id,
                page_size=page_size,
                cursor=_cursor,
                include_vector=True,
                verbose=True,
            )
            _data = resp["documents"]
            _cursor = resp["cursor"]
            if (_data is []) or (_cursor is []):
                break
            data += _data
            if number_of_documents and (len(data) >= int(number_of_documents)):
                break
            _page += 1
        return data

    @staticmethod
    def _prepare_vector_labels(
        data: List[JSONDict], label: str, vector: str
    ) -> Tuple[np.ndarray, np.ndarray, set]:
        """
        Prepare vector and labels
        """
        LOG.info(f'Preparing {label}, {vector} ...')
        vectors = np.array(
            [data[i][vector] for i in range(len(data)) if data[i].get(vector)]
        )
        labels = np.array(
            [
                data[i][label].replace(",", "")
                for i in range(len(data))
                if data[i].get(vector)
            ]
        )
        _labels = set(labels)

        return vectors, labels, _labels

    ## TODO: Separate DR into own class with default arg lut
    @staticmethod
    def _dim_reduce(
        dr: DR,
        dr_args: Union[None, JSONDict],
        vectors: np.ndarray,
        dims: Literal[2, 3] = 3,
    ) -> np.ndarray:
        """
        Dimensionality reduction
        """
        LOG.info(f'Executing {dr} from {vectors.shape[1]} to {dims} dims ...')
        if dr == "pca":
            pca = PCA(n_components=dims)
            vectors_dr = pca.fit_transform(vectors)
        elif dr == "tsne":
            pca = PCA(n_components=min(vectors.shape[1], 10))
            data_pca = pca.fit_transform(vectors)
            if dr_args is None:
                dr_args = {
                    "n_iter": 500,
                    "learning_rate": 100,
                    "perplexity": 30,
                    "random_state": 42,
                }
            tsne = TSNE(init="pca", n_components=dims, **dr_args)
            vectors_dr = tsne.fit_transform(data_pca)
        elif dr == "umap":
            if dr_args is None:
                dr_args = {
                    "n_neighbors": 15,
                    "min_dist": 0.1,
                    "random_state": 42,
                    "transform_seed": 42,
                }
            umap = UMAP(n_components=dims, **dr_args)
            vectors_dr = umap.fit_transform(vectors)
        elif dr == "ivis":
            if dr_args is None:
                dr_args = {"k": 15, 
                    "model": "maaten", 
                    "n_epochs_without_progress": 2
                    }
            vectors_dr = Ivis(embedding_dims=dims, **dr_args).fit(vectors).transform(vectors)
        return vectors_dr

    @staticmethod
    def _cluster(
        vectors: np.ndarray,
        cluster: CLUSTER,
        cluster_args: Union[None, JSONDict] = None
    ) -> Tuple[List[str], List[int]]:
        """
        Cluster method
        """
        LOG.info(f'Performing {cluster} Args: {cluster_args} ...')
        if cluster == 'kmeans':
            if cluster_args is None:
                cluster_args = {
                    "n_clusters": 20, 
                    "init": "k-means++", 
                    "verbose": 1, 
                    "algorithm": "auto"
                }
            cluster = KMeans(**cluster_args).fit(vectors)
            c_labels = cluster.labels_
            cluster_centroids = cluster.cluster_centers_
            c_labels = [ f'c_{c}' for c in c_labels ]
        
            return c_labels, cluster_centroids

    @staticmethod
    def _plot(
        embedding_df: pd.DataFrame,
        legend: str
    ) -> go.Figure:
        '''
        Generates the scatter plot 
        '''
        data = []
        groups = embedding_df.groupby(legend)
        for idx, val in groups:
            scatter = go.Scatter3d(
                name=idx,
                x=val['x'],
                y=val['y'],
                z=val['z'],
                text=[idx for _ in range(val['x'].shape[0])],
                textposition='top center',
                mode='markers',
                marker=dict(size=3, symbol='circle'),
            )
            data.append(scatter)
            
        axes = dict(title='', showgrid=True, zeroline=False, showticklabels=False)
        layout = go.Layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(xaxis=axes, yaxis=axes, zaxis=axes),
        )
        return go.Figure(data=data, layout=layout)

        

    def generate(
        self,
        dataset_id: str,
        label: str,
        vector_field: str,
        dr: DR = "ivis",
        dr_args: Union[None, JSONDict] = None,
        cluster: CLUSTER = None,
        cluster_args: Union[None, JSONDict] = None,
        legend: Literal['c_labels', 'labels'] = 'labels',
        #     point_label: List[str],
        #     hover_label: List[str],
    ):
        """
        Projection handler
        """
        self.dataset_id = dataset_id
        self.documents = self._retrieve_documents(dataset_id)

        self.documents_df = pd.DataFrame(self.documents)
        metadata_cols = [ c for c in self.documents_df.columns 
                        if '_vector_' not in c 
                        if c not in ['_id', 'insert_date_'] 
                        ]
        self.metadata_df =  self.documents_df[metadata_cols]

        vectors, labels, _labels = self._prepare_vector_labels(
            data=self.documents, label=label, vector=vector_field
        )
        vectors = MinMaxScaler().fit_transform(vectors) 
        self.vectors_dr = self._dim_reduce(dr=dr, dr_args=dr_args, vectors=vectors)
        
        data = { 'x': self.vectors_dr[:,0], 'y': self.vectors_dr[:,1], 'z': self.vectors_dr[:,2], 'labels': labels }
        self.embedding_df = pd.DataFrame(data)
        self.embedding_df['labels'] = labels

        if cluster:
            self.c_labels, self.c_centroids = self._cluster(
                    vectors=self.vectors_dr, cluster=cluster, cluster_args=cluster_args
                    )
            self.embedding_df['c_labels'] = self.c_labels

        return self._plot(embedding_df=self.embedding_df, legend=legend)
        
    