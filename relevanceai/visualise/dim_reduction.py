# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import json

from dataclasses import dataclass

from doc_utils.doc_utils import DocUtils

from typing import List, Union, Dict, Any, Tuple, Optional
from typing_extensions import Literal

from relevanceai.base import Base
from relevanceai.visualise.constants import DIM_REDUCTION, DIM_REDUCTION_DEFAULT_ARGS
from relevanceai.visualise.dataset import JSONDict

@dataclass
class DimReduction(Base, DocUtils):
    """Dim Reduction Class"""

    def __init__(
        self,
        project: str,
        api_key: str,
        base_url: str,
        data: List[JSONDict],
        vector_label: Union[None, str],
        vector_field: str,
        dr: DIM_REDUCTION,
        dr_args: Optional[Dict[Any, Any]] = None,
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
            self.dr_args = DIM_REDUCTION_DEFAULT_ARGS[dr]

        self.vectors  = self._prepare_vectors(data=self.data, vector_field=self.vector_field)
        self.vectors_dr = self._dim_reduce(vectors=self.vectors, 
                                            dr=self.dr, dr_args=self.dr_args, 
                                            dims=self.dims)
    
    
    def _prepare_vectors(
        self,
        data: List[JSONDict], 
        vector_field: str
    ) -> np.ndarray:
        """
        Prepare vectors
        """
        self.logger.info(f'Preparing {vector_field} ...')
        vectors = self.get_field_across_documents(field=vector_field, docs=data)
        from sklearn.preprocessing import MinMaxScaler
        vectors = MinMaxScaler().fit_transform(vectors) 
        return vectors


    def _dim_reduce(
        self,
        dr: DIM_REDUCTION,
        dr_args: Union[None, JSONDict],
        vectors: np.ndarray,
        dims: Literal[2, 3],
    ) -> np.ndarray:
        """
        Dimensionality reduction
        """
        self.logger.info(f'Executing {dr} from {vectors.shape[1]} to {dims} dims ...')
        if dr == "pca":
            from sklearn.decomposition import PCA
            self.logger.debug(f'{json.dumps(dr_args, indent=4)}')
            pca = PCA(n_components=min(dims, vectors.shape[1]), **dr_args)
            vectors_dr = pca.fit_transform(vectors)
        elif dr == "tsne":
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE
            pca = PCA(n_components=min(10, vectors.shape[1]))
            data_pca = pca.fit_transform(vectors)
            self.logger.debug(f'{json.dumps(dr_args, indent=4)}')
            tsne = TSNE(n_components=dims, **dr_args)
            vectors_dr = tsne.fit_transform(data_pca)
        elif dr == "umap":
            try:
                from umap import UMAP
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(f'{e}\nInstall umap\n \
                    pip install -U relevanceai[umap]')
            self.logger.debug(f'{json.dumps(dr_args, indent=4)}')
            umap = UMAP(n_components=dims, **dr_args)
            vectors_dr = umap.fit_transform(vectors)
        elif dr == "ivis":
            try:
                from ivis import Ivis
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(f'{e}\nInstall ivis\n \
                    CPU: pip install -U relevanceai[ivis-cpu]\n \
                    GPU: pip install -U relevanceai[ivis-gpu]')
            self.logger.debug(f'{json.dumps(dr_args, indent=4)}')
            ivis = Ivis(embedding_dims=dims, **dr_args)
            if ivis.batch_size > vectors.shape[0]: ivis.batch_size = vectors.shape[0]
            vectors_dr = ivis.fit(vectors).transform(vectors)
        return vectors_dr
