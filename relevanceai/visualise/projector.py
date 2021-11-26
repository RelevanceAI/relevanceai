# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import json
import warnings

import plotly.graph_objs as go

from dataclasses import dataclass
from typeguard import typechecked

from relevanceai.api.client import APIClient
from relevanceai.base import Base
from relevanceai.visualise.constants import *

from relevanceai.visualise.cluster import cluster, ClusterBase
from relevanceai.visualise.dim_reduction import dim_reduce, DimReductionBase

from doc_utils import DocUtils

RELEVANCEAI_BLUE = "#1854FF"
MARKER_SIZE = 5


@dataclass
class Projector(APIClient, Base, DocUtils):
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
        # Dimensionality reduction args
        dr: Union[DIM_REDUCTION, DimReductionBase] = "pca",
        dr_args: Union[None, Dict] = None,
        # TODO: Add support for 2
        dims: Literal[2, 3] = 3,
        # Plot rendering args
        vector_label: Union[None, str] = None,
        vector_label_char_length: Union[None, int] = 50,
        colour_label: Union[None, str] = None,
        colour_label_char_length: Union[None, int] = 20,
        hover_label: List[str] = [],
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
        self.number_of_points_to_render = number_of_points_to_render
        self.random_state = random_state
        self.vector_label_char_length = vector_label_char_length
        self.colour_label = colour_label
        self.colour_label_char_length = colour_label_char_length
        self.hover_label = hover_label
        self.cluster = cluster
        self.num_clusters = num_clusters
        self.dr = dr
        self.dr_args = dr_args
        self.dims = dims
        self.cluster_args = cluster_args

        if (vector_label is None) and (colour_label is None):
            warnings.warn(f"A vector_label or colour_label has not been specified.")

        if number_of_points_to_render and number_of_points_to_render > 1000:
            warnings.warn(
                f"You are rendering over 1000 points, this may take some time ..."
            )

        number_of_documents = number_of_points_to_render
        self.vector_fields = self._get_vector_fields()

        labels = ["_id", vector_field, vector_label, colour_label]
        if hover_label:
            labels += hover_label
        fields = [label for label in labels if label]
        self.docs = self._retrieve_documents(
            dataset_id, fields, number_of_documents, page_size=1000
        )
        self._remove_empty_vector_fields(vector_field)

        return self.plot_from_docs(self.docs, self.dims, marker_size, marker_colour)

    def plot_from_docs(self, docs: List[Dict[str, Any]], dims: int, marker_size: int, marker_colour: str, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        if self._is_valid_vector_name(self.vector_field):

            self.vectors = np.array(
                self.get_field_across_documents(self.vector_field, docs)
            )
            self.vectors_dr = dim_reduce(
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
            if self.hover_label and all(
                self._is_valid_label_name(l) for l in self.hover_label
            ):
                self.embedding_df = pd.concat(
                    [self.embedding_df, pd.DataFrame(docs)], axis=1
                )

            if self.vector_label and self._is_valid_label_name(self.vector_label):
                self.labels = self.get_field_across_documents(
                    field=self.vector_label, docs=docs
                )
                self.embedding_df[self.vector_label] = self.labels
                self.embedding_df["labels"] = self.labels

            self.legend = None
            if self.colour_label and self._is_valid_label_name(self.colour_label):
                self.labels = self.get_field_across_documents(
                    field=self.colour_label, docs=docs
                )
                self.embedding_df["labels"] = self.labels
                self.embedding_df[self.colour_label] = self.labels
                self.legend = "labels"

            # TODO: refactor Cluster
            if self.cluster:
                # _cluster = Cluster(
                #     **self.base_args,
                #     vectors=self.vectors,
                #     cluster=self.cluster,
                #     cluster_args=self.cluster_args,
                #     k=self.num_clusters,
                # )
                # self.cluster_labels = _cluster.cluster_labels
                self.cluster_labels = cluster(
                    vectors=self.vectors,
                    cluster=self.cluster,
                    cluster_args=self.cluster_args,
                )
                self.embedding_df["cluster_labels"] = self.cluster_labels
                self.legend = "cluster_labels"

            self.embedding_df.index = self.embedding_df["_id"]
            return self._generate_fig(
                embedding_df=self.embedding_df, legend=self.legend, marker_size = marker_size, marker_colour = marker_colour
            )

    def _get_vector_fields(self) -> List[str]:
        """
        Returns list of valid vector fields from dataset schema
        """
        self.schema = self.datasets.schema(self.dataset_id)
        self.vector_dim = self.schema[self.vector_field]["vector"]
        return [k for k in self.schema.keys() if k.endswith("_vector_")]

    def _is_valid_vector_name(self, vector_name: str) -> bool:
        """
        Check vector field name is valid
        """
        if vector_name in self.schema.keys():
            if vector_name in self.vector_fields:
                return True
            else:
                raise ValueError(f"{vector_name} is not a valid vector name")
        else:
            raise ValueError(f"{vector_name} is not in the {self.dataset_id} schema")

    def _is_valid_label_name(self, label_name: str) -> bool:
        """
        Check vector label name is valid. Checks that it is either numeric or text
        """
        if label_name == "_id":
            return True
        if label_name in list(self.schema.keys()):
            if self.schema[label_name] in ["numeric", "text"]:
                return True
            else:
                raise ValueError(f"{label_name} is not a valid label name")
        else:
            raise ValueError(f"{label_name} is not in the {self.dataset_id} schema")

    def _remove_empty_vector_fields(self, vector_field: str) -> List[Dict]:
        """
        Remove documents with empty vector fields
        """
        self.docs = [d for d in self.docs if d.get(vector_field)]
        return self.docs

    def _retrieve_documents(
        self,
        dataset_id: str,
        fields: List[str],
        number_of_documents: Optional[int] = 1000,
        page_size: int = 1000,
        filters=[],
    ) -> List[Dict]:
        """
        Retrieve all documents from dataset
        """
        if number_of_documents:
            if page_size > number_of_documents or self.random_state != 0:
                page_size = number_of_documents  # type: ignore
        else:
            number_of_documents = 999999999999999

        is_random = True if self.random_state != 0 else False
        resp = self.datasets.documents.get_where(
            dataset_id=dataset_id,
            select_fields=fields,
            include_vector=True,
            page_size=page_size,
            is_random=is_random,
            random_state=self.random_state,
            filters=filters,
        )
        data = resp["documents"]

        if (
            (number_of_documents > page_size)
            and (is_random == False)
            and (self.random_state == 0)
        ):
            _cursor = resp["cursor"]
            _page = 0
            while resp:
                self.logger.debug(f"Paginating {_page} page size {page_size} ...")
                resp = self.datasets.documents.get_where(
                    dataset_id=dataset_id,
                    select_fields=fields,
                    page_size=page_size,
                    cursor=_cursor,
                    include_vector=True,
                    filters=filters,
                )
                _data = resp["documents"]
                _cursor = resp["cursor"]
                if (_data == []) or (_cursor == []):
                    break
                data += _data
                if number_of_documents and (len(data) >= int(number_of_documents)):
                    break
                _page += 1
            data = data[:number_of_documents]

        self.docs = data
        return self.docs

    def _generate_fig(
        self,
        embedding_df: pd.DataFrame,
        marker_size: int,
        marker_colour: str,
        legend: Union[None, str],
    ) -> go.Figure:
        """
        Generates the Scatter plot
        """
        plot_title = f"<b>{self.dims}D Embedding Projector Plot<br>Dataset Id: {self.dataset_id} - {len(embedding_df)} points<br>Vector Field: {self.vector_field}<br></b>"
        self.hover_label = ["_id"] + self.hover_label
        text_labels = None
        plot_mode = "markers"

        """
        Generates data for word plot
        If vector_label set, generates text_labels, otherwise shows points only
        """
        if self.vector_label:
            plot_title = plot_title.replace(
                "</b>", f"Vector Label: {self.vector_label}<br></b>"
            )
            plot_mode = "text+markers"
            text_labels = embedding_df["labels"]
            if self.vector_label_char_length and not self.cluster:
                plot_title = plot_title.replace(
                    "<br></b>",
                    f"  Char Length: {self.vector_label_char_length}<br></b>",
                )
                text_labels = embedding_df["labels"].apply(
                    lambda x: x[: self.vector_label_char_length] + "..."
                )

            self.hover_label.insert(1, self.vector_label)

            # self.hover_label = [self.vector_label] + self.hover_label
            # self.hover_label = list(set(self.hover_label))

            # TODO: We can change this later to show top 100 neighbours of a selected word
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

        custom_data, hovertemplate = self._generate_hover_template(
            df=embedding_df, dims=self.dims
        )

        vector_label_scatter_args = {
            "text": text_labels,
            "textposition": "middle center",
            "showlegend": False,
            "mode": plot_mode,
            "marker": {
                "size": marker_size,
                "color": marker_colour,
                "symbol": "circle",
            },
            "customdata": custom_data,
            "hovertemplate": hovertemplate,
        }

        if self.dims == 3:
            scatter = go.Scatter3d(
                x=embedding_df["x"],
                y=embedding_df["y"],
                z=embedding_df["z"],
                **vector_label_scatter_args,
            )

        else:
            scatter = go.Scatter(
                x=embedding_df["x"], y=embedding_df["y"], **vector_label_scatter_args
            )

        data = [scatter]

        """
        Generates data for colour plot if selected
        """
        if self.colour_label:
            plot_title = plot_title.replace(
                "</b>", f"Colour Label: {self.colour_label}<br></b>"
            )
            if self.colour_label_char_length and not self.cluster:
                plot_title = plot_title.replace(
                    "<br></b>",
                    f"  Char Length: {self.colour_label_char_length}<br></b>",
                )
                colour_labels = embedding_df["labels"].apply(
                    lambda x: x[: self.colour_label_char_length] + "..."
                )
                embedding_df["labels"] = colour_labels

            self.hover_label.insert(1, self.colour_label)

            data = []
            groups = embedding_df.groupby(legend)
            for idx, val in groups:
                # if self.vector_label:
                #     plot_mode = "text+markers"
                #     text_labels = val["labels"]
                #     if self.vector_label_char_length and not self.cluster:
                #         plot_title = plot_title.replace(
                #             "<br></b>",
                #             f"  Char Length: {self.vector_label_char_length}<br></b>",
                #         )
                #         text_labels = val["labels"].apply(
                #             lambda x: x[: self.vector_label_char_length] + "..."
                #         )

                #     self.hover_label = [self.vector_label] + self.hover_label
                #     self.hover_label = list(set(self.hover_label))

                custom_data, hovertemplate = self._generate_hover_template(
                    df=val, dims=self.dims
                )

                colour_label_scatter_args = (
                    {  # text:[ idx for _ in range(val["x"].shape[0]) ],
                        "text": text_labels,
                        "textposition": "top center",
                        "showlegend": False,
                        "mode": plot_mode,
                        "marker": {"size": marker_size, "symbol": "circle"},
                        "customdata": custom_data,
                        "hovertemplate": hovertemplate,
                    }
                )

                if self.dims == 3:
                    scatter = go.Scatter3d(
                        name=idx,
                        x=val["x"],
                        y=val["y"],
                        z=val["z"],
                        **colour_label_scatter_args,
                    )

                else:

                    scatter = go.Scatter(
                        name=idx, x=val["x"], y=val["y"], **colour_label_scatter_args
                    )

                data.append(scatter)

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

        if self.cluster:
            plot_title = plot_title.replace(
                "</b>",
                f"<b>Cluster Method: {self.cluster}<br>Num Clusters: {self.num_clusters}</b>",
            )
        fig = go.Figure(data=data, layout=layout)
        fig.update_layout(
            title={
                "text": plot_title,
                "y": 0.1,
                "x": 0.1,
                "xanchor": "left",
                "yanchor": "bottom",
                "font": {"size": 10},
            },
        )
        if legend and self.colour_label:
            fig.update_layout(
                legend={
                    "title": {"text": self.colour_label, "font": {"size": 12}},
                    "font": {"size": 10},
                    "itemwidth": 30,
                    "tracegroupgap": 1,
                }
            )
        return fig

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
