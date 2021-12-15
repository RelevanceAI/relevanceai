# -*- coding: utf-8 -*-

from doc_utils import DocUtils
from typing_extensions import Literal
from typing import List, Union, Dict, Any, Tuple, Optional
from relevanceai.vector_tools.dim_reduction import DimReduction, DimReductionBase
from relevanceai.vector_tools.cluster import Cluster, ClusterBase
from relevanceai.visualise.dash_components.app import create_dash_graph
from relevanceai.vector_tools.constants import *
from relevanceai.base import _Base
from relevanceai.api.client import BatchAPIClient
from typeguard import typechecked
from dataclasses import dataclass
import plotly.graph_objs as go
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None


RELEVANCEAI_BLUE = "#1854FF"


@dataclass
class Projector(BatchAPIClient, _Base, DocUtils):
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
                vector_label, label_char_length,
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
        # Dimensionality reduction args
        dr: Union[DIM_REDUCTION, DimReductionBase] = "pca",
        dims: Literal[2, 3] = 3,
        dr_args: Union[None, Dict] = None,
        # Cluster args
        cluster: Union[CLUSTER, ClusterBase] = None,
        num_clusters: Union[None, int] = 10,
        cluster_args: Union[None, Dict] = None,
        cluster_on_dr: bool = False,
        # Decoration args
        hover_label: list = [],
        show_image: bool = False,
        label_char_length: int = 50,
        marker_size: int = 5,
        interactive: bool = False,
    ):
        """
        Dimension reduce vectors and plot with functionality to visualise different clusters and nearest neighbours

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
                    vector_label, label_char_length,
                    color_label, colour_label_char_length,
                    hover_label,
                    cluster, cluster_args,
                    )

        Parameters
        ----------
        dataset_id : string
            Unique name of dataset
        vector_field : list
            Vector field to plot
        number_of_points_to_render: int
            Number of vector fields to plot
        vector_label: string
            Field to use as label to describe vector on plot
        dr: string
            Method of dimension reduction for vectors
        dims: int
            Number of dimensions to reduce to
        dr_args: dict
            Additional arguments for dimension reduction
        cluster: string
            Method of clustering for vectors
        num_clusters: string
            Number of clusters to create
        cluster_args: dict
            Additional arguments for clustering
        cluster_on_dr: int
            Whether to cluster on the dimension reduced or original vectors
        hover_label: list
            Additional labels to include as plot labels
        show_image: bool
            Whether vector labels are image urls
        label_char_length: int
            Maximum length of text for each hover label
        marker_size: int
            Marker size of the plot
        interactive: bool
            Whether to include interactive features including nearest neighbours

        """

        # Check vector field
        self._is_valid_vector_name(dataset_id, vector_field)

        # Check vector label field
        if vector_label is None:
            self.logger.warning("A vector_label has not been specified.")
        else:
            self._is_valid_label_name(dataset_id, vector_label)

        # Check hover label field
        [self._is_valid_label_name(dataset_id, label) for label in hover_label]

        docs = self.get_documents(
            dataset_id,
            number_of_documents=number_of_points_to_render,
            batch_size=1000,
            select_fields=["_id", vector_field, vector_label] + hover_label,
        )
        docs = self._remove_empty_vector_fields(docs, vector_field)

        return self.plot_from_docs(
            docs,
            vector_field=vector_field,
            vector_label=vector_label,
            label_char_length=label_char_length,
            dr=dr,
            dims=dims,
            dr_args=dr_args,
            cluster=cluster,
            num_clusters=num_clusters,
            cluster_args=cluster_args,
            cluster_on_dr=cluster_on_dr,
            hover_label=hover_label,
            show_image=show_image,
            marker_size=marker_size,
            dataset_name=dataset_id,
            interactive=interactive,
        )

    def plot_from_docs(
        self,
        docs: List[Dict],
        vector_field: str,
        # Plot rendering args
        vector_label: Union[None, str] = None,
        # Dimensionality reduction args
        dr: Union[DIM_REDUCTION, DimReductionBase] = "pca",
        dims: Literal[2, 3] = 3,
        dr_args: Union[None, Dict] = None,
        # Cluster args
        cluster: Union[CLUSTER, ClusterBase] = None,
        num_clusters: Union[None, int] = 10,
        cluster_args: Union[None, Dict] = None,
        cluster_on_dr: bool = False,
        # Decoration args
        hover_label: list = [],
        show_image: bool = False,
        label_char_length: int = 50,
        marker_size: int = 5,
        dataset_name: Union[None, str] = None,
        interactive: bool = False,
    ):

        # Adjust vector label
        if show_image is False:
            self.set_field_across_documents(
                vector_label,
                [i[vector_label][:label_char_length] + "..." for i in docs],
                docs,
            )

        # Dimension reduce vectors
        vectors = np.array(self.get_field_across_documents(vector_field, docs))
        vectors_dr = DimReduction.dim_reduce(
            vectors=vectors, dr=dr, dr_args=dr_args, dims=dims
        )
        points = {"x": vectors_dr[:, 0], "y": vectors_dr[:, 1]}
        if dims == 3:
            points["z"] = vectors_dr[:, 2]

        embedding_df = pd.DataFrame(points)
        embedding_df = pd.concat([embedding_df, pd.DataFrame(docs)], axis=1)

        # Set hover labels
        hover_label = ["_id", vector_label] + hover_label

        # Cluster vectors
        if cluster:
            if cluster_on_dr:
                cluster_vec = vectors_dr
            else:
                cluster_vec = vectors

            cluster_labels = Cluster.cluster(
                vectors=cluster_vec,
                cluster=cluster,
                cluster_args=cluster_args,
                k=num_clusters,
            )
            embedding_df["cluster_labels"] = cluster_labels
            hover_label = hover_label + ["cluster_labels"]

        embedding_df.index = embedding_df["_id"]

        # Generate plot title
        plot_title = self._generate_plot_title(
            dims,
            dataset_name,
            len(embedding_df),
            vector_field,
            vector_label,
            label_char_length,
        )

        plot_data = self._generate_plot_data(
            embedding_df=embedding_df,
            hover_label=hover_label,
            dims=dims,
            marker_size=marker_size,
            cluster=cluster,
            label_char_length=label_char_length,
        )

        layout = self._generate_layout(plot_title=plot_title)

        create_dash_graph(
            plot_data=plot_data,
            layout=layout,
            show_image=show_image,
            docs=docs,
            vector_label=vector_label,
            vector_field=vector_field,
            interactive=interactive,
        )
        return

    def _generate_plot_data(
        self,
        embedding_df: pd.DataFrame,
        hover_label: List[Optional[str]],
        dims: int,
        marker_size: int,
        cluster: Union[
            Literal["kmeans"],
            Literal["kmedoids"],
            Literal["hdbscan"],
            ClusterBase,
            None,
        ],
        label_char_length: int,
    ):
        """ """

        if cluster:
            data = []
            groups = embedding_df.groupby("cluster_labels")
            for idx, val in groups:
                data.append(
                    self._generate_plot_info(
                        embedding_df=val,
                        hover_label=hover_label,
                        dims=dims,
                        marker_size=marker_size,
                        label_char_length=label_char_length,
                    )
                )

        else:
            data = []
            data.append(
                self._generate_plot_info(
                    embedding_df=embedding_df,
                    hover_label=hover_label,
                    dims=dims,
                    marker_size=marker_size,
                    label_char_length=label_char_length,
                )
            )

        return data

    def _generate_layout(self, plot_title):

        axes_3d = {
            "title": "",
            "backgroundcolor": "#ffffff",
            "showgrid": False,
            "showticklabels": False,
        }

        axes_2d = {"title": "", "visible": False, "showticklabels": False}

        layout = go.Layout(
            margin={"l": 0, "r": 0, "b": 0, "t": 0},
            scene={"xaxis": axes_3d, "yaxis": axes_3d, "zaxis": axes_3d},
            title={
                "text": plot_title,
                "y": 0.1,
                "x": 0.1,
                "xanchor": "left",
                "yanchor": "bottom",
                "font": {"size": 10},
            },
            plot_bgcolor="#FFF",
            xaxis=axes_2d,
            yaxis=axes_2d,
        )

        return layout

    def _generate_plot_info(
        self, embedding_df, hover_label, dims, marker_size, label_char_length
    ):

        custom_data, hovertemplate = self._generate_hover_template(
            df=embedding_df,
            dims=dims,
            hover_label=hover_label,
            label_char_length=label_char_length,
        )

        scatter_args = {
            "x": embedding_df["x"],
            "y": embedding_df["y"],
            "showlegend": False,
            "mode": "markers",
            "marker": {"size": marker_size, "symbol": "circle", "opacity": 0.75},
            "customdata": custom_data,
            "hovertemplate": hovertemplate,
        }

        if dims == 2:
            scatter = go.Scatter(**scatter_args)

        else:
            scatter_args["z"] = embedding_df["z"]
            scatter = go.Scatter3d(**scatter_args)

        return scatter

    def _generate_hover_template(
        self, df: pd.DataFrame, dims: int, hover_label: list, label_char_length: int
    ):
        """
        Generating hover template
        """
        custom_data = df[hover_label]
        custom_data = custom_data.loc[:, ~custom_data.columns.duplicated()]

        for label in hover_label:
            try:
                if label != "_id":
                    custom_data.loc[:, label] = [
                        i[:label_char_length] + "..." for i in custom_data.loc[:, label]
                    ]
            except:
                pass

        custom_data_hover = [
            f"{c}: %{{customdata[{i}]}}" for i, c in enumerate(hover_label)
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

    def _generate_plot_title(
        self,
        dims,
        dataset_name,
        number_of_points,
        vector_field,
        vector_label,
        label_char_length,
    ):
        title = "</b>"
        title += f"{dims}D Embedding Projector Plot<br>"
        if dataset_name:
            title += f"Dataset Name: {dataset_name}<br>"
        title += f"Points: {number_of_points} points<br>"
        title += f"Vector Field: {vector_field}<br>"
        title += f"Vector Label: {vector_label}  Char Length: {label_char_length}<br>"
        title += "</b>"
        return title

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
            raise ValueError(f"{vector_name} is not in the {dataset_id} schema")

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
            raise ValueError(f"{label_name} is not in the {dataset_id} schema")

    def _remove_empty_vector_fields(self, docs, vector_field: str) -> List[Dict]:
        """
        Remove documents with empty vector fields
        """
        return [d for d in docs if d.get(vector_field)]
