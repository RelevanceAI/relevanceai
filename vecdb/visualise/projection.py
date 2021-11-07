# -*- coding: utf-8 -*-
import sys
import time

import numpy as np
import pandas as pd
import json 

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from ivis import Ivis
from sklearn.cluster import KMeans, MiniBatchKMeans
from kmodes.kmodes import KModes

import plotly.graph_objs as go

from dataclasses import dataclass

from vecdb.base import Base
from vecdb.visualise.constants import *
from vecdb.visualise.dataset import Dataset
from vecdb.visualise.cluster import Cluster

@dataclass
class Projection(Base):
    """Projection Class"""

    def __init__(
        self,
        dataset: Dataset
    ):  
        
        self.dataset = dataset
        self.dataset_id = dataset.dataset_id
        self.vector_fields = dataset.vector_fields
        self.data = dataset.data

        self.base_args = {
            "project": self.dataset.project, 
            "api_key": self.dataset.api_key, 
            "base_url": self.dataset.base_url,
        }
        super().__init__(**self.base_args)

    
    def _prepare_vector_labels(
        self,
        data: List[JSONDict], 
        vector_label: str, 
        vector_field: str
    ) -> Tuple[np.ndarray, np.ndarray, set]:
        """
        Prepare vector and labels
        """
        self.logger.info(f'Preparing {vector_label}, {vector_field} ...')
        vectors = np.array(
            [data[i][vector_field] for i, _ in enumerate(data) if data[i].get(vector_field)]
        )
        labels = np.array(
            [
                data[i][vector_label].replace(",", "")
                for i, _ in enumerate(data)
                if data[i].get(vector_field)
            ]
        )
        _labels = set(labels)
        return vectors, labels, _labels


    ## TODO: Separate DR into own class with default arg lut
    def _dim_reduce(
        self,
        dr: DR,
        dr_args: Union[None, JSONDict],
        vectors: np.ndarray,
        dims: Literal[2, 3] = 3,
    ) -> np.ndarray:
        """
        Dimensionality reduction
        """
        self.logger.info(f'Executing {dr} from {vectors.shape[1]} to {dims} dims ...')
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
            self.logger.debug(f'{json.dumps(dr_args, indent=4)}')
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
                dr_args = {
                    "k": 15, 
                    "model": "maaten", 
                    "n_epochs_without_progress": 2
                    }
            self.logger.debug(f'{json.dumps(dr_args, indent=4)}')
            vectors_dr = Ivis(embedding_dims=dims, **dr_args).fit(vectors).transform(vectors)
        return vectors_dr


    def _generate_fig(
        self,
        embedding_df: pd.DataFrame,
        legend: str,
        point_label: bool,
        hover_label: Union[None, List[str]],
    ) -> go.Figure:
        '''
        Generates the 3D scatter plot 
        '''
        ### Layout
        # plot_title = f"{self.dataset_id}: {len(embedding_df)} points<br>{self.vector_label}: {self.vector_field}"

        axes = dict(title='', showgrid=True, zeroline=False, showticklabels=False)
        layout = go.Layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(xaxis=axes, yaxis=axes, zaxis=axes),
        )
        
        wordemb_display_mode = 'regular'
        if point_label:
            #  # Regular displays the full scatter plot with only circles
            # if wordemb_display_mode == 'regular':
            #     plot_mode = 'markers'
            # # Nearest Neighbors displays only the 200 nearest neighbors of the selected_word, in text rather than circles
            # elif wordemb_display_mode == 'neighbors':
            #     if not selected_word:
            #         return go.Figure()
            #     plot_mode = 'text'
            #     # Get the nearest neighbors indices
            #     dataset = data_dict[dataset_name].set_index('0')
            #     selected_vec = dataset.loc[selected_word]

            #     nearest_neighbours = get_nearest_neighbours(
            #                             dataset=dataset, 
            #                             selected_vec=selected_vec,
            #                             distance_measure_mode=distance_measure_mode,  
            #                             )

            #     neighbors_idx = nearest_neighbours[:100].index
            #     embedding_df =  embedding_df.loc[neighbors_idx]
            embedding_df = embedding_df[:100]

            scatter = go.Scatter3d(
                name=str(embedding_df.index),
                x=embedding_df['x'],
                y=embedding_df['y'],
                z=embedding_df['z'],
                text=embedding_df.index,
                textposition='middle center',
                showlegend=False,
                mode='text markers',
                marker=dict(size=3, color='#1854FF', symbol='circle'),
            )
            data=[scatter]
        
        else:
            
            data = []
            groups = embedding_df.groupby(legend)
            for idx, val in groups:
                scatter = go.Scatter3d(
                    name=idx,
                    x=val['x'],
                    y=val['y'],
                    z=val['z'],
                    text=[ idx for _ in range(val['x'].shape[0]) ],
                    textposition='top center',
                    mode='markers',
                    marker=dict(size=3, symbol='circle'),
                )
                data.append(scatter)

        fig = go.Figure(data=data, layout=layout)

        if not hover_label: hover_label = [self.vector_label]
        fig.update_traces(customdata=self.dataset.metadata[hover_label])
        fig.update_traces(hovertemplate='%{customdata}')

        custom_data_hover =  [f"{c}: %{{customdata[{i}]}}" for i, c in enumerate(hover_label)]
        fig.update_traces(
            hovertemplate="<br>".join([
                "X: %{x}",
                "Y: %{y}",
            ] + custom_data_hover
            )
        )
        return fig


    def plot(
        self,
        vector_label: str,
        vector_field: str,
        point_label: bool = False,
        hover_label: Union[None, List[str]] = None,
        dr: DR = "ivis",
        dr_args: Union[None, JSONDict] = None,
        cluster: CLUSTER = None,
        cluster_args: Union[None, JSONDict] = None,
        legend: Literal['c_labels', 'labels'] = 'labels',
        max_points: int = -1
    ):
        """
        Projection handler
        """
        self.vector_label = vector_label
        self.vector_field = vector_field

        if self.dataset.valid_vector_name(vector_field) and self.dataset.valid_label_name(vector_label):

            ## TODO: Implement intelligent selection of which points to show - randomly sample subselect of each cluster
            self.data = self.dataset.data[:max_points]
            
            vectors, labels, _labels = self._prepare_vector_labels(
                data=self.data, vector_label=vector_label, vector_field=vector_field
            )
            vectors = MinMaxScaler().fit_transform(vectors) 
            self.vectors_dr = self._dim_reduce(dr=dr, dr_args=dr_args, vectors=vectors)
            
            data = { 'x': self.vectors_dr[:,0], 
                    'y': self.vectors_dr[:,1], 
                    'z': self.vectors_dr[:,2], 'labels': labels }
            self.embedding_df = pd.DataFrame(data)
            self.embedding_df.index = labels

            if cluster:
                cluster = Cluster(**self.base_args,
                    vectors=vectors, cluster=cluster, cluster_args=cluster_args, k=10)
                self.c_labels = cluster.c_labels
                self.embedding_df['c_labels'] = self.c_labels

            if hover_label==None: hover_label==[vector_label]

            return self._generate_fig(embedding_df=self.embedding_df, 
                                    legend=legend,
                                    point_label=point_label, 
                                    hover_label=hover_label
                                    )

    