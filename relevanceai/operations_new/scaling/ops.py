from typing import Optional

from relevanceai.operations_new.apibase import OperationAPIBase
from relevanceai.operations_new.scaling.base import ScalerBase


class ScaleOps(ScalerBase, OperationAPIBase):
    """
    API related Functionality for Operation
    """

    def __init__(self, alias, model, model_kwargs: Optional[dict] = None, **kwargs):
        if model_kwargs is None:
            model_kwargs = {}

        super().__init__(
            model=model,
            alias=alias,
            model_kwargs=model_kwargs,
            **kwargs,
        )
        self.model = self._get_model(
            model=model,
            alias=alias,
            model_kwargs=model_kwargs,
        )
