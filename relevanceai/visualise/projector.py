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
        dims: Literal[2, 3] = 2,
        dr_args: Union[None, Dict] = None,
        # Cluster args
        cluster: Union[CLUSTER, ClusterBase] = None,
        num_clusters: Union[None, int] = 10,
        cluster_args: Union[None, Dict] = None,
        # Decoration args
        hover_label: list = [],
        show_image: bool = False,
        marker_size: int = 5
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
        # Check vector field
        self._is_valid_vector_name(dataset_id, vector_field)

        # Check vector label field
        if vector_label is None:
            self.logger.warning("A vector_label has not been specified.")
        else:
            self._is_valid_label_name(dataset_id, vector_label)

        # Check hover label field
        [self._is_valid_label_name(dataset_id, label) for label in hover_label];

        docs = self.get_documents(
            dataset_id, number_of_documents=number_of_points_to_render, batch_size=1000, select_fields=["_id", vector_field, vector_label] + hover_label
        )
        docs = self._remove_empty_vector_fields(docs, vector_field)

        return self.plot_from_docs(docs, vector_field=vector_field, vector_label=vector_label,
                                   vector_label_char_length=vector_label_char_length, dr=dr,
                                   dims=dims, dr_args=dr_args, cluster=cluster,
                                   num_clusters=num_clusters, cluster_args=cluster_args, hover_label=hover_label, show_image=show_image, marker_size=marker_size)

    def plot_from_docs(
        self,
        docs: List[Dict[str, Any]],
        vector_field: str,
        # Plot rendering args
        vector_label: Union[None, str] = None,
        vector_label_char_length: Union[None, int] = 50,
        # Dimensionality reduction args
        dr: Union[DIM_REDUCTION, DimReductionBase] = "pca",
        dims: Literal[2, 3] = 3,
        dr_args: Union[None, Dict] = None,
        # Cluster args
        cluster: Union[CLUSTER, ClusterBase] = None,
        num_clusters: Union[None, int] = 10,
        cluster_args: Union[None, Dict] = None,
        # Decoration args
        hover_label: list = [],
        show_image: bool = False,
        marker_size: int = 5):

        # Dimension reduce vectors
        vectors = np.array(
            self.get_field_across_documents(vector_field, docs)
        )
        vectors_dr = DimReduction.dim_reduce(
            vectors=vectors, dr=dr, dr_args=dr_args, dims=dims
        )
        points = {
            "x": vectors_dr[:, 0],
            "y": vectors_dr[:, 1],
            "_id": self.get_field_across_documents("_id", docs),
        }
        if dims == 3:
            points["z"] = vectors_dr[:, 2]

        embedding_df = pd.DataFrame(points)

        # Prepare vector labels
        labels = self.get_field_across_documents(
            field=vector_label, docs=docs
        )
        if show_image is False:
            labels = [i[:vector_label_char_length] + '...' for i in labels]
        embedding_df[vector_label] = labels

        # Cluster vectors
        if cluster:
            cluster_labels = Cluster.cluster(
                vectors=vectors,
                cluster=cluster,
                cluster_args=cluster_args,
                k=num_clusters
            )
            embedding_df["cluster_labels"] = cluster_labels

        embedding_df.index = embedding_df["_id"]

        # Set hover labels
        hover_label = ["_id", vector_label] + hover_label

        plot_data, layout = self._generate_fig(
            embedding_df=embedding_df, hover_label=hover_label, dims=dims, marker_size=marker_size, cluster=cluster
        )

        create_dash_graph(plot_data=plot_data, layout=layout, show_image=show_image,
                          docs=docs, vector_label=vector_label, vector_field=vector_field)
        return

    def _generate_fig(
        self,
        embedding_df: pd.DataFrame,
        hover_label: str,
        dims: int,
        marker_size: int,
        cluster: bool
    ) -> go.Figure:
        """
        """

        if cluster:
            data = []
            groups = embedding_df.groupby("cluster_labels")
            for idx, val in groups:
                data.append(self._generate_plot_info(
                    embedding_df=val, hover_label=hover_label, dims=dims, marker_size=marker_size))

        else:
            data = []
            data.append(self._generate_plot_info(
                embedding_df=embedding_df, hover_label=hover_label,  dims=dims, marker_size=marker_size))

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

    def _generate_plot_info(self, embedding_df, hover_label, dims, marker_size):

        custom_data, hovertemplate = self._generate_hover_template(
            df=embedding_df, dims=dims, hover_label=hover_label
        )

        scatter_args = (
            {
                "showlegend": False,
                "mode": "markers",
                "marker": {"size": marker_size, "symbol": "circle"},
                "customdata": custom_data,
                "hovertemplate": hovertemplate,
            }
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
        self, df: pd.DataFrame, dims: int, hover_label: list
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Generating hover template
        """
        custom_data = df[hover_label]
        custom_data_hover = [
            f"{c}: %{{customdata[{i}]}}"
            for i, c in enumerate(hover_label)
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

    def _is_valid_vector_name(self, dataset_id, vector_name: str) -> bool:
        """
        Check vector field name is valid
        """
        vector_fields = self.get_vector_fields(dataset_id)
        schema = self.datasets.schema(dataset_id)
        if vector_name in schema.keys():
            if vector_name in vector_fields:
                return True
            else:
                raise ValueError(f"{vector_name} is not a valid vector name")
        else:
            raise ValueError(
                f"{vector_name} is not in the {dataset_id} schema")

    def _is_valid_label_name(self, dataset_id, label_name: str) -> bool:
        """
        Check vector label name is valid. Checks that it is either numeric or text
        """
        schema = self.datasets.schema(dataset_id)
        if label_name == "_id":
            return True
        if label_name in list(schema.keys()):
            if schema[label_name] in ["numeric", "text"]:
                return True
            else:
                raise ValueError(f"{label_name} is not a valid label name")
        else:
            raise ValueError(
                f"{label_name} is not in the {dataset_id} schema")

    def _remove_empty_vector_fields(self, docs, vector_field: str) -> List[Dict]:
        """
        Remove documents with empty vector fields
        """
        return [d for d in docs if d.get(vector_field)]
