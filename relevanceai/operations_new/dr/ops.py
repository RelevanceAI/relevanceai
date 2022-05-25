from typing import Optional

from relevanceai.operations_new.apibase import OperationAPIBase
from relevanceai.operations_new.dr.base import DimReductionBase


class DimReductionOps(DimReductionBase, OperationAPIBase):
    """
    API related Functionality for Operation
    """

    def __init__(
        self,
        alias,
        model,
        n_components: int,
        model_kwargs: Optional[dict] = None,
        **kwargs
    ):
        if model_kwargs is None:
            model_kwargs = {}

        super().__init__(
            model=model,
            n_components=n_components,
            model_kwargs=model_kwargs,
            alias=alias,
            **kwargs,
        )
        self.model = self._get_model(
            model=model,
            n_components=n_components,
            alias=alias,
            **model_kwargs,
        )
