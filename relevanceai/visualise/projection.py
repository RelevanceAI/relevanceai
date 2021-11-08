# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import json
import warnings

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


    def _prepare_labels(
        self,
        data: List[JSONDict],
        vector_label: str,
        vector_field: str,
    ):
        """
        Prepare labels
        """
        self.logger.info(f'Preparing {vector_label} ...')
        labels = np.array(
            [
                data[i][vector_label].replace(",", "")
                for i, _ in enumerate(data)
                if data[i].get(vector_field)
            ]
        )
        _labels = set(labels)
        return labels, _labels


    def _generate_fig(
        self,
        embedding_df: pd.DataFrame,
        legend: Union[None, str],
        vector_label: Union[None, str], 
        vector_label_char_length: int,
        colour_label: str,
        hover_label: Union[None, List[str]],
    ) -> go.Figure:
        '''
        Generates the 3D scatter plot 
        '''

        if colour_label:
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
        else:
            if vector_label:
                plot_mode ='text+markers'
                text_labels = embedding_df['labels'].apply(lambda x: x[:vector_label_char_length]+'...')
            else:
                plot_mode = 'markers'
                text_labels = None

            ## TODO: We can change this later to show top 100 neighbours of a selected word
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

            
            
            scatter = go.Scatter3d(
                name=str(embedding_df.index),
                x=embedding_df['x'],
                y=embedding_df['y'],
                z=embedding_df['z'],
                text=text_labels,
                textposition='middle center',
                showlegend=False,
                mode=plot_mode,
                marker=dict(size=3, color=RELEVANCEAI_BLUE, symbol='circle'),
            )
            data=[scatter]

        '''
        Generating figure
        '''
        plot_title = f"{self.dataset_id}: {len(embedding_df)} points<br>Vector Label: {self.vector_label}<br>Vector Field: {self.vector_field}"

        axes = dict(title='', showgrid=True, zeroline=False, showticklabels=False)
        title_axis = dict(title=plot_title, showgrid=True, zeroline=False, showticklabels=False)
        layout = go.Layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(xaxis=axes, yaxis=axes, zaxis=axes),
        )
        
        fig = go.Figure(data=data, layout=layout)

        fig.update_layout(title=plot_title)

        '''
        Updating hover label
        '''
        # if not hover_label: hover_label = [self.vector_label]
        # fig.update_traces(customdata=self.dataset.metadata[hover_label])
        # fig.update_traces(hovertemplate='%{customdata}')
        # custom_data_hover = [f"{c}: %{{customdata[{i}]}}" for i, c in enumerate(hover_label) 
        #                       if self.dataset.valid_label_name(c)]
        # fig.update_traces(
        #     hovertemplate="<br>".join([
        #         "X: %{x}",
        #         "Y: %{y}",
        #     ] + custom_data_hover
        #     )
        )
        return fig


    def plot(
        self,
        vector_label: Union[None, str],
        vector_field: str,
        colour_label: Union[None, str] = None,  
        hover_label: Union[None, List[str]] = None,
        dr: DIM_REDUCTION = "ivis",
        dr_args: Union[None, JSONDict] = None,
        dims: Literal[2, 3] = 3,
        cluster: CLUSTER = None,
        cluster_args: Union[None, JSONDict] = {"n_init" : 20},
        number_of_points_to_render: int = -1,
        vector_label_char_length: int = 10
    ):
        """
        Projection handler
        """
        self.vector_label = vector_label
        self.vector_field = vector_field

        if vector_label is None:
            warnings.warn(f'A vector label has not been specified.')

        if self.dataset.valid_vector_name(vector_field):

            ## TODO: Implement representative selection of which points to show - eg. randomly sample subselect of each cluster
            self.data = self.dataset.data[:number_of_points_to_render]
            
            dr = DimReduction(**self.base_args, data=self.data, 
                                vector_label=vector_label, vector_field=vector_field, 
                                dr=dr, dr_args=dr_args, dims=dims
                                )
            self.vectors = dr.vectors
            self.vectors_dr = dr.vectors_dr
            points = { 'x': self.vectors_dr[:,0], 
                        'y': self.vectors_dr[:,1], 
                        'z': self.vectors_dr[:,2]}
            self.embedding_df = pd.DataFrame(points)

            if vector_label and self.dataset.valid_label_name(vector_label):
                self.labels, self._labels = self._prepare_labels(data=self.data, 
                                vector_field=vector_field, vector_label=vector_label)
                self.embedding_df.index = self.labels
                self.embedding_df['labels'] = self.labels

            if colour_label and self.dataset.valid_label_name(colour_label):
                self.labels, self._labels = self._prepare_labels(data=self.data, 
                                vector_field=vector_field, vector_label=colour_label)
                self.embedding_df.index = self.labels
                self.embedding_df['labels'] = self.labels
                self.legend = 'labels'
        
            if cluster:
                cluster = Cluster(**self.base_args,
                    vectors=self.vectors, cluster=cluster, cluster_args=cluster_args)
                self.cluster_labels = cluster.cluster_labels
                self.embedding_df['cluster_labels'] = self.cluster_labels
                self.legend = 'cluster_labels'

            return self._generate_fig(embedding_df=self.embedding_df, 
                                    legend=self.legend,
                                    vector_label=vector_label, 
                                    vector_label_char_length=vector_label_char_length,
                                    colour_label= colour_label,
                                    hover_label=hover_label,
                                    )

    