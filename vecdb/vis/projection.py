# -*- coding: utf-8 -*-
import sys
import time

import numpy as np

from sklearn.decomposition import PCA
from ivis import Ivis

from dataclasses import dataclass

# from ivis import Ivis

from vecdb.base import Base
from vecdb.vecdb_logging import create_logger
from api.datasets import Datasets

from typing import List, Union, Dict, Any, Literal, Callable

JSONDict = Dict[str, Any]
DR_METHODS = Literal["pca", "tsne", "umap", "umap_fast", "pacamp", "ivis"]

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
    ):
        """
        Retrieve all documents from dataset
        """

        dataset = Datasets(self.project, self.api_key, self.base_url)
        resp = dataset.documents.list(
            dataset_id=dataset_id, page_size=page_size
        )  # Initial call
        _cursor = resp["cursor"]
        data = []
        while _cursor:
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
        return data

    @staticmethod
    def _prepare_vector_labels(data: List[JSONDict], label: str, vector: str):
        """
        Prepare vector and labels
        """
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

    @staticmethod
    def _dim_reduce(
        dr_method: DR_METHODS,
        vectors: np.ndarray,
        dims: Literal[2, 3] = 3,
        k: int = 15,
    ) -> np.ndarray:
        """
        Dimensionality reduction
        """
        if dr_method == "pca":
            pca = PCA(n_components=dims)
            vectors_dr = pca.fit_transform(vectors)
        elif dr_method == "ivis":
            vectors_dr = Ivis(embedding_dims=dims, k=k)
        return vectors_dr

    def projection(
        self,
        dataset_id: str,
        label: str,
        vector_field: str,
        dr_method: DR_METHODS = "ivis",
    ):
        self.dataset_id = dataset_id
        self.documents = self._retrieve_documents(dataset_id)

        vectors, labels, _labels = self._prepare_vector_labels(
            data=self.documents, label=label, vector=vector_field
        )

        self.vectors_dr = self._dim_reduce(dr_method="pca", vectors=vectors)

    # def dr(
    #     self,
    #     vector_field: str,
    #     point_label: List[str],
    #     hover_label: List[str],
    #     colour_label: str,
    #     dr_method: Literal['ivis'] = 'ivis',
    #     sample: List = None,
    # ):
    #     """
    #     Dim reduction method
    #     """
    #     pass
