from relevanceai.operations_new.apibase import OperationsAPIBase
from relevanceai.operations_new.dr.base import DimReductionBase
from typing import Optional


class DimReductionOps(DimReductionBase, OperationsAPIBase):
    """
    API related Functionality for Operation
    """

    def __init__(
        self,
        alias,
        model,
        n_components: int,
        model_kwargs: Optional[dict] = None,
        *args,
        **kwargs
    ):
        if model_kwargs is None:
            model_kwargs = {}
        super().__init__(
            model=model, n_components=n_components, model_kwargs=model_kwargs, **kwargs
        )
        self.model = self._get_model(
            model=model, n_components=n_components, alias=alias, **model_kwargs
        )
        for k, v in kwargs.items():
            setattr(self, k, v)
