import numpy as np
from relevanceai.operations.dr.base import DimReductionBase
from typing import Optional, Dict, Any
from relevanceai.operations.cluster.constants import (
    DIM_REDUCTION,
    DIM_REDUCTION_DEFAULT_ARGS,
)


class Ivis(DimReductionBase):
    def fit_transform(
        self,
        vectors: np.ndarray,
        dr_args: Optional[Dict[Any, Any]] = DIM_REDUCTION_DEFAULT_ARGS["ivis"],
        dims: int = 3,
    ) -> np.ndarray:
        try:
            from ivis import Ivis
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"{e}\nInstall ivis\n \
                CPU: pip install -U ivis-cpu\n \
                GPU: pip install -U ivis-gpu"
            )
        self.logger.debug(f"{dr_args}")
        ivis = Ivis(embedding_dims=dims, **dr_args)
        if ivis.batch_size > vectors.shape[0]:
            ivis.batch_size = vectors.shape[0]
        vectors_dr = ivis.fit(vectors).transform(vectors)
        return vectors_dr
