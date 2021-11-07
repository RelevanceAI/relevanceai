# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import json

from dataclasses import dataclass


from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from ivis import Ivis

from typing import List, Union, Dict, Any, Tuple, Optional
from typing_extensions import Literal

from relevanceai.base import Base
from relevanceai.visualise.constants import DIM_REDUCTION, DIM_REDUCTION_DEFAULT_ARGS
from relevanceai.visualise.dataset import JSONDict

@dataclass
class DimReduction(Base):
    """Dim Reduction Class"""

    def __init__(
        self,
        project: str,
        api_key: str,
        base_url: str,
        data: List[JSONDict],
        vector_label: str,
        vector_field: str,
        dr: DIM_REDUCTION,
        dr_args: Union[None, JSONDict] = None,
        dims: Literal[2, 3] = 3,
    ):  

        self.project = project
        self.api_key = api_key
        self.base_url = base_url
        super().__init__(project, api_key, base_url)

        self.data = data
        self.vector_label = vector_label
        self.vector_field = vector_field

        self.dr = dr
        self.dr_args = dr_args
        self.dims = dims

        if dr_args is None:
            self.dr_args = {**DIM_REDUCTION_DEFAULT_ARGS[dr]}

        self.vectors, self.labels, _labels = self._prepare_vector_labels(
                data=self.data, vector_label=self.vector_label, vector_field=self.vector_field
            )
        
        self.vectors_dr = self._dim_reduce(vectors=self.vectors, 
                                            dr=self.dr, dr_args=self.dr_args, 
                                            dims=self.dims)
    
    
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
            [data[i][vector_field] 
            for i, _ in enumerate(data) 
            if data[i].get(vector_field)]
        )
        vectors = MinMaxScaler().fit_transform(vectors) 
        labels = np.array(
            [
                data[i][vector_label].replace(",", "")
                for i, _ in enumerate(data)
                if data[i].get(vector_field)
            ]
        )
        _labels = set(labels)
        return vectors, labels, _labels


    def _dim_reduce(
        self,
        dr: DIM_REDUCTION,
        dr_args: Union[None, JSONDict],
        vectors: np.ndarray,
        dims: Literal[2, 3] ,
    ) -> np.ndarray:
        """
        Dimensionality reduction
        """
        self.logger.info(f'Executing {dr} from {vectors.shape[1]} to {dims} dims ...')
        if dr == "pca":
            self.logger.debug(f'{json.dumps(dr_args, indent=4)}')
            pca = PCA(n_components=min(vectors.shape[1], dims), **dr_args)
            vectors_dr = pca.fit_transform(vectors)
        elif dr == "tsne":
            pca = PCA(n_components=min(vectors.shape[1], 10))
            data_pca = pca.fit_transform(vectors)
            self.logger.debug(f'{json.dumps(dr_args, indent=4)}')
            tsne = TSNE(n_components=dims, **dr_args)
            vectors_dr = tsne.fit_transform(data_pca)
        elif dr == "umap":
            self.logger.debug(f'{json.dumps(dr_args, indent=4)}')
            umap = UMAP(n_components=dims, **dr_args)
            vectors_dr = umap.fit_transform(vectors)
        elif dr == "ivis":
            self.logger.debug(f'{json.dumps(dr_args, indent=4)}')
            vectors_dr = Ivis(embedding_dims=dims, **dr_args).fit(vectors).transform(vectors)
        return vectors_dr
