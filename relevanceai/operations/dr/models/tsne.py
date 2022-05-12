import numpy as np
from relevanceai.operations.dr.base import DimReductionBase
from typing import Optional, Dict, Any
from relevanceai.operations.cluster.constants import (
    DIM_REDUCTION,
    DIM_REDUCTION_DEFAULT_ARGS,
)


class TSNE(DimReductionBase):
    def fit_transform(
        self,
        vectors: np.ndarray,
        dr_args: Optional[Dict[Any, Any]] = DIM_REDUCTION_DEFAULT_ARGS["tsne"],
        dims: int = 3,
    ) -> np.ndarray:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        pca = PCA(n_components=min(10, vectors.shape[1]))
        data_pca = pca.fit_transform(vectors)
        self.logger.debug(f"{dr_args}")
        tsne = TSNE(n_components=dims, **dr_args)
        return tsne.fit_transform(data_pca)
