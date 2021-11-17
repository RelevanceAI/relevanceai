# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import json


import plotly.graph_objs as go

from dataclasses import dataclass
from typeguard import typechecked

from relevanceai.base import Base
from relevanceai.visualise.constants import *
from relevanceai.visualise.dataset import Dataset
from relevanceai.visualise.cluster import Cluster
from relevanceai.visualise.dim_reduction import DimReduction 

from doc_utils import DocUtils

RELEVANCEAI_BLUE = '#1854FF'

@dataclass
class Projector(Base, DocUtils):
    """
        Projector class.

        Example: 
            >>> from relevanceai import Client
            >>> project = input()
            >>> api_key = input()
            >>> client = Client(project, api_key)
            >>> client.projector.plot(
                    dataset_id, vector_field, number_of_points_to_render, random_state, 
                    dr, dr_args, dims,
                    vector_label, vector_label_char_length,
                    color_label, colour_label_char_length, 
                    hover_label,
                    cluster, cluster_args,
                    )
    """

    def __init__(self, project, api_key, base_url):
        self.base_args = {
            "project": project, 
            "api_key": api_key, 
            "base_url": base_url,
        }
        super().__init__(**self.base_args)


    @typechecked
    def plot(
        self,
        dataset_id: str,
        vector_field: str,
        number_of_points_to_render: Optional[int] = 1000,
        random_state: int = 0,

        ### Dimensionality reduction args
        dr: DIM_REDUCTION = "pca",
        dr_args: Union[None, JSONDict] = DIM_REDUCTION_DEFAULT_ARGS['pca'],
        dims: Literal[2, 3] = 3,

        ### Plot rendering args
        vector_label: Union[None, str] = None,
        vector_label_char_length: Union[None, int] = 12,
        colour_label: Union[None, str] = None,  
        colour_label_char_length: Union[None, int] = 20,
        hover_label: Union[None, List[str]] = None,

        ### Cluster args
        cluster: Union[None, CLUSTER] = None,
        cluster_args: Union[None, JSONDict] = {"n_init" : 20},
        num_clusters: Union[None, int] = 10
    ):
        """
        Plot function for Embedding Projector class

        Example: 
            >>> from relevanceai import Client
            >>> project = input()
            >>> api_key = input()
            >>> client = Client(project, api_key)
            >>> client.projector.plot(
                    dataset_id, vector_field, number_of_points_to_render, random_state, 
                    dr, dr_args, dims,
                    vector_label, vector_label_char_length,
                    color_label, colour_label_char_length, 
                    hover_label,
                    cluster, cluster_args,
                    )
        """          
        ## Class args for generating figure       
        self.dataset_id = dataset_id
        self.vector_label = vector_label
        self.vector_field = vector_field
        self.random_state = random_state
        self.vector_label_char_length = vector_label_char_length
        self.colour_label = colour_label
        self.colour_label_char_length = colour_label_char_length
        self.hover_label = hover_label
        self.cluster = cluster
        self.num_clusters = num_clusters

        if (vector_label is None) and (colour_label is None):
            import warnings
            warnings.warn(f'A vector_label or colour_label has not been specified.')
        
        if number_of_points_to_render and number_of_points_to_render > 1000:
            import warnings
            warnings.warn(f'You are rendering over 1000 points, this may take some time ...')
        
        number_of_documents = number_of_points_to_render
        self.dataset = Dataset(**self.base_args, 
                                dataset_id=dataset_id, vector_field=vector_field, 
                                vector_label=vector_label, colour_label=colour_label, hover_label=hover_label,
                                number_of_documents=number_of_documents, random_state=random_state
                                )

        self.docs = self.dataset.docs
        self.detail = self.dataset.detail

        if self.dataset.valid_vector_name(vector_field):
            self.dr = DimReduction(**self.base_args, data=self.docs, 
                                vector_label=self.vector_label, vector_field=self.vector_field, 
                                dr=dr, dr_args=dr_args, dims=dims
                                )
            self.vectors = self.dr.vectors
            self.vectors_dr = self.dr.vectors_dr
            points = { 'x': self.vectors_dr[:,0], 
                        'y': self.vectors_dr[:,1], 
                        'z': self.vectors_dr[:,2], 
                        '_id': self.get_field_across_documents('_id', self.docs)}
            self.embedding_df = pd.DataFrame(points)

            if self.hover_label and all(self.dataset.valid_label_name(l) for l in self.hover_label):
                self.embedding_df = pd.concat([self.embedding_df, self.detail[self.hover_label]], axis=1)

            if self.vector_label and self.dataset.valid_label_name(self.vector_label):
                self.labels = self.get_field_across_documents(field=self.vector_label, docs=self.docs)
                self.embedding_df[self.vector_label] = self.labels
                self.embedding_df['labels'] = self.labels
                
            self.legend = None
            if self.colour_label and self.dataset.valid_label_name(self.colour_label):
                self.labels = self.get_field_across_documents(field=self.colour_label, docs=self.docs)
                self.embedding_df['labels'] = self.labels
                self.embedding_df[self.colour_label] = self.labels
                self.legend = 'labels'

            if self.cluster:
                _cluster = Cluster(**self.base_args,
                    vectors=self.vectors, cluster=cluster, cluster_args=cluster_args, k=self.num_clusters)
                self.cluster_labels = _cluster.cluster_labels
                self.embedding_df['cluster_labels'] = self.cluster_labels
                self.legend = 'cluster_labels'

            self.embedding_df.index = self.embedding_df['_id']
            return self._generate_fig(embedding_df=self.embedding_df, legend=self.legend)


    def _generate_fig(
        self,
        embedding_df: pd.DataFrame,
        legend: Union[None, str],
    ) -> go.Figure:
        '''
        Generates the 3D scatter plot 
        '''
        plot_title = f"<b>3D Embedding Projector Plot<br>Dataset Id: {self.dataset_id} - {len(embedding_df)} points<br>Vector Field: {self.vector_field}<br></b>"
        if self.colour_label:
            '''
            Generates data for colour plot
            '''
            plot_title = plot_title.replace('</b>', f"Colour Label: {self.colour_label}<br></b>")
            if self.colour_label_char_length  and not self.cluster:
                plot_title = plot_title.replace('<br></b>', f"  Char Length: {self.colour_label_char_length}<br></b>")
                colour_labels = embedding_df['labels'].apply(lambda x: x[:self.colour_label_char_length]+'...')
                embedding_df['labels'] = colour_labels
            if self.hover_label is None: self.hover_label = [ self.colour_label ] 

            data = []
            groups = embedding_df.groupby(legend)
            for idx, val in groups:
                custom_data, hovertemplate = self._generate_hover_template(df=val)
                scatter = go.Scatter3d(
                    name=idx,
                    x=val['x'],
                    y=val['y'],
                    z=val['z'],
                    text=[ idx for _ in range(val['x'].shape[0]) ],
                    textposition='top center',
                    mode='markers',
                    marker=dict(size=3, symbol='circle'),
                    customdata=custom_data,
                    hovertemplate=hovertemplate
                )
                data.append(scatter)
        else:
            '''
            Generates data for word plot
            If vector_label set, generates text_labels, otherwise shows points only
            '''
            if self.vector_label:
                plot_title = plot_title.replace('</b>', f"Vector Label: {self.vector_label}<br></b>")
                plot_mode ='text+markers'
                text_labels = embedding_df['labels']
                if self.vector_label_char_length and not self.cluster:
                    plot_title = plot_title.replace('<br></b>', f"  Char Length: {self.vector_label_char_length}<br></b>")
                    text_labels = embedding_df['labels'].apply(lambda x: x[:self.vector_label_char_length]+'...')
                if self.hover_label is None: self.hover_label = [ self.vector_label ]
             
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

            custom_data, hovertemplate = self._generate_hover_template(df=embedding_df)
            scatter = go.Scatter3d(
                x=embedding_df['x'],
                y=embedding_df['y'],
                z=embedding_df['z'],
                text=text_labels,
                textposition='middle center',
                showlegend=False,
                mode=plot_mode,
                marker=dict(size=3, color=RELEVANCEAI_BLUE, symbol='circle'),
                customdata=custom_data,
                hovertemplate=hovertemplate
            )
            data=[scatter]
            

        '''
        Generating figure
        '''
        axes = dict(title='', showgrid=True, zeroline=False, showticklabels=False)
        layout = go.Layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(xaxis=axes, yaxis=axes, zaxis=axes),
        )

        if self.cluster:
            plot_title = plot_title.replace('</b>', 
                f"<b>Cluster Method: {self.cluster}<br>Num Clusters: {self.num_clusters}</b>")
        fig = go.Figure(data=data, layout=layout)
        fig.update_layout(title={
            'text': plot_title,
            'y':0.1,
            'x':0.1,
            'xanchor': 'left',
            'yanchor': 'bottom',
            'font': {
                'size': 10
            }},
        )
        if legend and self.colour_label:
            fig.update_layout(legend={
                'title': {
                'text' : self.colour_label,
                    'font': {
                        'size': 12
                    }
                },
                'font': {
                    'size': 10
                },
                'itemwidth': 30,
                'tracegroupgap': 1
            }
        )

        return fig
    


    def _generate_hover_template(
        self,
        df: pd.DataFrame
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Generating hover template
        """
        if self.hover_label:
            hover_label  = ['_id'] + self.hover_label
            custom_data = df[ hover_label ]
            custom_data_hover = [ f"{c}: %{{customdata[{i}]}}" for i, c in enumerate(hover_label) 
                                   if self.dataset.valid_label_name(c) ]
            hovertemplate="<br>".join([
                "X: %{x}   Y: %{y}   Z: %{z}", 
            ] + custom_data_hover
            )+'<extra></extra>'
    
        else:
            custom_data = None
            hovertemplate = ''
        return custom_data, hovertemplate
