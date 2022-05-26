import numpy as np
from relevanceai.operations.dr.base import DimReductionBase
from typing import Optional, Dict, Any
from relevanceai.operations.cluster.constants import (
    DIM_REDUCTION,
    DIM_REDUCTION_DEFAULT_ARGS,
)


class PCA(DimReductionBase):
    def fit(self, vectors: np.ndarray, dims: int = 3, *args, **kw):
        from sklearn.decomposition import PCA as SKLEARN_PCA

        pca = SKLEARN_PCA(n_components=min(dims, vectors.shape[1]))
        return pca.fit(vectors)

    def fit_transform(
        self,
        vectors: np.ndarray,
        dr_args: Optional[Dict[Any, Any]] = DIM_REDUCTION_DEFAULT_ARGS["pca"],
        dims: int = 3,
    ) -> np.ndarray:
        from sklearn.decomposition import PCA as SKLEARN_PCA

        self.logger.debug(f"{dr_args}")
        vector_length = len(vectors[0])
        pca = SKLEARN_PCA(n_components=min(dims, vector_length), **dr_args)
        return pca.fit_transform(vectors)
