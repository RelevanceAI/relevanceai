# -*- coding: utf-8 -*-
"""
Pandas like dataset API
"""
import warnings
import itertools
from itertools import chain
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Optional
from tqdm.auto import tqdm
from relevanceai.package_utils.analytics_funcs import track
from relevanceai.dataset.crud.dataset_write import Write
from relevanceai.unstructured_data.text.base_text_processing import MLStripper
from relevanceai.package_utils.logger import FileLogger
from relevanceai.package_utils.version_decorators import introduced_in_version, beta
from relevanceai.vector_tools.local_nearest_neighbours import (
    NearestNeighbours,
    NEAREST_NEIGHBOURS,
)

# TODO: Separate out operations into different files - cluster/search/dr
class DimensionalityReduction(Write):
    def _run_dr_algorithm(
        self,
        algorithm: str,
        vector_fields: list,
        documents: list,
        alias: str,
        n_components: int = 3,
    ):
        original_name = algorithm
        # Make sure that the letter case does not matter
        algorithm = algorithm.upper()
        if algorithm == "PCA":
            from relevanceai.workflows.dim_reduction_ops.dim_reduction import PCA

            model = PCA()
        elif algorithm == "TSNE":
            from relevanceai.workflows.dim_reduction_ops.dim_reduction import TSNE

            model = TSNE()
        elif algorithm == "UMAP":
            from relevanceai.workflows.dim_reduction_ops.dim_reduction import UMAP

            model = UMAP()
        elif algorithm == "IVIS":
            from relevanceai.workflows.dim_reduction_ops.dim_reduction import Ivis

            model = Ivis()
        else:
            raise ValueError(
                f'"{original_name}" is not a supported '
                "dimensionality reduction algorithm. "
                "Currently, the supported algorithms are: "
                "PCA, TSNE, UMAP, and IVIS"
            )

        print(f"Run {algorithm}...")
        # Returns a list of documents with dr vector
        return model.fit_transform_documents(
            vector_field=vector_fields[0],
            documents=documents,
            alias=alias,
            dims=n_components,
        )

    @track
    def auto_reduce_dimensions(
        self,
        alias: str,
        vector_fields: list,
        filters: Optional[list] = None,
        number_of_documents: Optional[int] = None,
    ):
        """
        Run dimensionality reduction quickly on a dataset on a small number of documents.
        This is useful if you want to quickly see a projection of your dataset.

        .. warning::
            This function is currently in beta and is likely to change in the future.
            We recommend not using this in any production systems.

        Parameters
        ----------
        vector_fields: list
            The vector fields to run dimensionality reduction on
        number_of_documents: int
            The number of documents to get
        algorithm: str
            The algorithm to run. The only supported algorithm is `pca` at this
            current point in time.
        n_components: int
            The number of components

        Example
        ----------

        .. code-block::

            from relevanceai import Client
            client = Client()
            df = client.Dataset("sample")
            df.auto_reduce_dimensions(
                "pca-3",
                ["sample_vector_"],
            )

        """
        if len(vector_fields) > 1:
            raise ValueError("We only support 1 vector field at the moment.")

        dr_args = alias.split("-")

        if len(dr_args) != 2:
            raise ValueError("""Your DR alias should be in the form of `pca-3`.""")

        algorithm = dr_args[0]
        n_components = int(dr_args[1])

        print("Getting documents...")
        if filters is None:
            filters = []
        filters += [
            {
                "field": vf,
                "filter_type": "exists",
                "condition": ">=",
                "condition_value": " ",
            }
            for vf in vector_fields
        ]

        if number_of_documents is None:
            number_of_documents = self.get_number_of_documents(self.dataset_id, filters)

        documents = self.get_documents(
            select_fields=vector_fields,
            filters=filters,
            number_of_documents=number_of_documents,
        )

        dr_documents = self._run_dr_algorithm(
            algorithm=algorithm,
            vector_fields=vector_fields,
            documents=documents,
            alias=alias,
            n_components=n_components,
        )

        results = self.update_documents(self.dataset_id, dr_documents)

        if n_components == 3:
            projector_url = f"https://cloud.relevance.ai/dataset/{self.dataset_id}/deploy/recent/projector"
            print(f"You can now view your projector at {projector_url}")

        return results

    @track
    def reduce_dimensions(
        self,
        vector_fields: list,
        alias: str,
        number_of_documents: int = 1000,
        algorithm: str = "pca",
        n_components: int = 3,
        filters: Optional[list] = None,
    ):
        """
        Run dimensionality reduction quickly on a dataset on a small number of documents.
        This is useful if you want to quickly see a projection of your dataset.

        .. warning::
            This function is currently in beta and is likely to change in the future.
            We recommend not using this in any production systems.

        Parameters
        ----------
        vector_fields: list
            The vector fields to run dimensionality reduction on
        number_of_documents: int
            The number of documents to get
        algorithm: str
            The algorithm to run. The only supported algorithm is `pca` at this
            current point in time.
        n_components: int
            The number of components

        Example
        ----------

        .. code-block::

            from relevanceai import Client
            client = Client()
            df = client.Dataset("sample")
            df.auto_reduce_dimensions(
                alias="pca-3",
                ["sample_vector_"],
                number_of_documents=1000
            )

        """
        filters = [] if filters is None else filters

        if len(vector_fields) > 1:
            raise ValueError("We only support 1 vector field at the moment.")

        print("Getting documents...")
        filters += [
            {
                "field": vf,
                "filter_type": "exists",
                "condition": ">=",
                "condition_value": " ",
            }
            for vf in vector_fields
        ]
        documents = self.get_documents(
            select_fields=vector_fields,
            filters=filters,
            number_of_documents=number_of_documents,
        )

        dr_documents = self._run_dr_algorithm(
            algorithm=algorithm,
            vector_fields=vector_fields,
            documents=documents,
            alias=alias,
            n_components=n_components,
        )

        results = self.update_documents(self.dataset_id, dr_documents)

        return results
