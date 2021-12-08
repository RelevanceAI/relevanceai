# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import json
import warnings

import plotly.graph_objs as go

from dataclasses import dataclass
from typeguard import typechecked

from relevanceai.api.client import BatchAPIClient
from relevanceai.base import Base
from relevanceai.vector_tools.constants import *
from relevanceai.visualise.dash_components.app import create_dash_graph


from relevanceai.vector_tools.cluster import Cluster, ClusterBase
from relevanceai.vector_tools.dim_reduction import DimReduction, DimReductionBase

from typing import List, Union, Dict, Any, Tuple, Optional
from typing_extensions import Literal

from doc_utils import DocUtils

RELEVANCEAI_BLUE = "#1854FF"
MARKER_SIZE = 5


@dataclass
class Projector(BatchAPIClient, Base, DocUtils):
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

    def __init__(self, project, api_key):
        self.project = project
        self.api_key = api_key
        super().__init__(project, api_key)

    @typechecked
    def plot(
        self,
        dataset_id: str,
        vector_field: str,
        number_of_points_to_render: int = 1000,
        # Plot rendering args
        vector_label: Union[None, str] = None,
        vector_label_char_length: Union[None, int] = 50,
        # Dimensionality reduction args
        dr: Union[DIM_REDUCTION, DimReductionBase] = "pca",
        dr_args: Union[None, Dict] = None,
        dims: Literal[2, 3] = 3,
        # Cluster args
        cluster: Union[CLUSTER, ClusterBase] = None,
        cluster_args: Union[None, Dict] = None,
        num_clusters: Union[None, int] = 10,
        marker_colour: str = RELEVANCEAI_BLUE,
        marker_size: int = MARKER_SIZE
    ):
        """
        Plot function for Embedding Projector class

        To write your own custom dimensionality reduction, you should inherit from DimReductionBase:
        from relevanceai.visualise.dim_reduction import DimReductionBase
        class CustomDimReduction(DimReductionBase):
            def fit_transform(self, vectors):
                return np.arange(512, 2)

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
        self.dataset_id = dataset_id
        self.vector_label = vector_label
        self.vector_field = vector_field
        self.vector_label_char_length = vector_label_char_length
        self.cluster = cluster
        self.num_clusters = num_clusters
        self.dr = dr
        self.dr_args = dr_args
        self.dims = dims
        self.cluster_args = cluster_args

        if (vector_label is None):
            warnings.warn(
                f"A vector_label has not been specified.")

        if number_of_points_to_render and number_of_points_to_render > 1000:
            warnings.warn(
                f"You are rendering over 1000 points, this may take some time ..."
            )

        self._is_valid_vector_name(self.vector_field)
        self._is_valid_label_name(self.vector_label)


        labels = ["_id", vector_field, vector_label]
        fields = [label for label in labels if label]
        self.docs = self.get_documents(
            dataset_id, number_of_documents=number_of_points_to_render, batch_size=1000, select_fields=fields
        )
        self._remove_empty_vector_fields(vector_field)

        return self.plot_from_docs(self.docs, self.dims, marker_size, marker_colour)

    def plot_from_docs(self, docs: List[Dict[str, Any]], dims: int, marker_size: int, marker_colour: str):

        self.vectors = np.array(
            self.get_field_across_documents(self.vector_field, docs)
        )
        self.vectors_dr = DimReduction.dim_reduce(
            vectors=self.vectors, dr=self.dr, dr_args=self.dr_args, dims=self.dims
        )
        points = {
            "x": self.vectors_dr[:, 0],
            "y": self.vectors_dr[:, 1],
            "_id": self.get_field_across_documents("_id", docs),
        }
        if dims == 3:
            points["z"] = self.vectors_dr[:, 2]

        self.embedding_df = pd.DataFrame(points)

        self.labels = self.get_field_across_documents(
            field=self.vector_label, docs=docs
        )
        self.embedding_df[self.vector_label] = self.labels

        if self.cluster:
            self.cluster_labels = Cluster.cluster(
                vectors=self.vectors,
                cluster=self.cluster,
                cluster_args=self.cluster_args,
            )
            self.embedding_df["cluster_labels"] = self.cluster_labels

        self.embedding_df.index = self.embedding_df["_id"]
        plot_data, layout =  self._generate_fig(
            embedding_df=self.embedding_df, marker_size = marker_size, marker_colour = marker_colour
        )

        create_dash_graph(plot_data, layout, docs, self.vector_label, self.vector_field)
        return

    def _generate_fig(
        self,
        embedding_df: pd.DataFrame,
        marker_size: int,
        marker_colour: str
    ) -> go.Figure:
        """
        """
    
        plot_title = f"<b>{self.dims}D Embedding Projector Plot<br>Dataset Id: {self.dataset_id} - {len(embedding_df)} points<br>Vector Field: {self.vector_field}<br></b>"
        self.hover_label = ["_id", self.vector_label]

        
        if self.vector_label:
            data = []
            custom_data, hovertemplate = self._generate_hover_template(
                df=embedding_df, dims=self.dims
            )
            data.append(self._generate_plot_info(embedding_df, self.dims, custom_data, hovertemplate, marker_size))

    
        if self.cluster:
            data = []
            groups = embedding_df.groupby("cluster_labels")
            for idx, val in groups:
                custom_data, hovertemplate = self._generate_hover_template(
                    df=val, dims=self.dims
                )
                data.append(self._generate_plot_info(val, self.dims, custom_data, hovertemplate, marker_size))

        """
        Generating figure
        """
        axes = {
            "title": "",
            "showgrid": True,
            "zeroline": False,
            "showticklabels": False,
        }
        layout = go.Layout(
            margin={"l": 0, "r": 0, "b": 0, "t": 0},
            scene={"xaxis": axes, "yaxis": axes, "zaxis": axes},
        )

        return data, layout

    def _generate_plot_info(self, embedding_df, dims, custom_data, hovertemplate, marker_size):

        scatter_args = (
            {
                "showlegend": False,
                "mode": "markers",
                "marker": {"size": marker_size, "symbol": "circle"},
                "customdata": custom_data,
                "hovertemplate": hovertemplate,
            }
        )

        custom_data, hovertemplate = self._generate_hover_template(
                df=embedding_df, dims=dims
            )

        if dims == 3:
            scatter = go.Scatter3d(
                x=embedding_df["x"],
                y=embedding_df["y"],
                z=embedding_df["z"],
                **scatter_args,
            )

        else:
            scatter = go.Scatter(
                x=embedding_df["x"], y=embedding_df["y"], **scatter_args
            )

        return scatter


    def _generate_hover_template(
        self, df: pd.DataFrame, dims: int
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Generating hover template
        """
        self.hover_label = list(sorted(set(self.hover_label)))
        custom_data = df[self.hover_label]
        custom_data_hover = [
            f"{c}: %{{customdata[{i}]}}"
            for i, c in enumerate(self.hover_label)
            if self._is_valid_label_name(c)
        ]

        if dims == 2:
            coord_info = "X: %{x}   Y: %{y}"
        else:
            coord_info = "X: %{x}   Y: %{y}   Z: %{z}"

        hovertemplate = (
            "<br>".join(
                [
                    coord_info,
                ]
                + custom_data_hover
            )
            + "<extra></extra>"
        )

        return custom_data, hovertemplate

    def _is_valid_vector_name(self, vector_name: str) -> bool:
        """
        Check vector field name is valid
        """
        vector_fields = self.get_vector_fields(self.dataset_id)
        schema = self.datasets.schema(self.dataset_id)
        if vector_name in schema.keys():
            if vector_name in vector_fields:
                return True
            else:
                raise ValueError(f"{vector_name} is not a valid vector name")
        else:
            raise ValueError(
                f"{vector_name} is not in the {self.dataset_id} schema")

    def _is_valid_label_name(self, label_name: str) -> bool:
        """
        Check vector label name is valid. Checks that it is either numeric or text
        """
        schema = self.datasets.schema(self.dataset_id)
        if label_name == "_id":
            return True
        if label_name in list(schema.keys()):
            if schema[label_name] in ["numeric", "text"]:
                return True
            else:
                raise ValueError(f"{label_name} is not a valid label name")
        else:
            raise ValueError(
                f"{label_name} is not in the {self.dataset_id} schema")

    def _remove_empty_vector_fields(self, vector_field: str) -> List[Dict]:
        """
        Remove documents with empty vector fields
        """
        self.docs = [d for d in self.docs if d.get(vector_field)]
        return self.docs
