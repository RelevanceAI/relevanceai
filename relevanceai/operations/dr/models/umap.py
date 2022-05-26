import numpy as np
from relevanceai.operations.dr.base import DimReductionBase
from typing import Optional, Dict, Any
from relevanceai.operations.cluster.constants import (
    DIM_REDUCTION,
    DIM_REDUCTION_DEFAULT_ARGS,
)


class UMAP(DimReductionBase):
    def fit_transform(
        self,
        vectors: np.ndarray,
        dr_args: Optional[Dict[Any, Any]] = DIM_REDUCTION_DEFAULT_ARGS["umap"],
        dims: int = 3,
    ) -> np.ndarray:
        try:
            from umap import UMAP
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"{e}\nInstall umap\n \
                pip install -U relevanceai[umap]"
            )
        self.logger.debug(f"{dr_args}")
        umap = UMAP(n_components=dims, **dr_args)
        return umap.fit_transform(vectors)
