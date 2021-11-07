# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import json

import plotly.graph_objs as go

from dataclasses import dataclass

from relevanceai.base import Base
from relevanceai.visualise.constants import *
from relevanceai.visualise.dataset import Dataset
from relevanceai.visualise.cluster import Cluster
from relevanceai.visualise.dim_reduction import DimReduction 

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
                mode='text+markers',
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

        custom_data_hover = [f"{c}: %{{customdata[{i}]}}" for i, c in enumerate(hover_label)]
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
        point_label: bool,      ## TODO: We can change this later to show top 100 neighbours of a selected word
        hover_label: Union[None, List[str]] = None,
        dr: DIM_REDUCTION = "ivis",
        dr_args: Union[None, JSONDict] = None,
        dims: Literal[2, 3] = 3,
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
            self.docs = self.dataset.data[:max_points]
            
            dr = DimReduction(**self.base_args, data=self.docs, 
                                vector_label=vector_label, vector_field=vector_field, 
                                dr=dr, dr_args=dr_args, dims=dims
                                )
            self.vectors = dr.vectors
            self.labels = dr.labels
            self.vectors_dr = dr.vectors_dr
            points = { 'x': self.vectors_dr[:,0], 
                        'y': self.vectors_dr[:,1], 
                        'z': self.vectors_dr[:,2], 'labels': self.labels }
            self.embedding_df = pd.DataFrame(points)
            self.embedding_df.index = self.labels

            if cluster:
                cluster = Cluster(**self.base_args,
                    vectors=self.vectors, cluster=cluster, cluster_args=cluster_args)
                self.c_labels = cluster.c_labels
                self.embedding_df['c_labels'] = self.c_labels

            if hover_label==None: hover_label==[vector_label]

            return self._generate_fig(embedding_df=self.embedding_df, 
                                    legend=legend,
                                    point_label=point_label, 
                                    hover_label=hover_label
                                    )

    