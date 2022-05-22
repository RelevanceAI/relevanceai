from relevanceai.operations_new.apibase import OperationsAPIBase
from relevanceai.operations_new.dr.base import DimReductionBase


class DimReductionOps(DimReductionBase, OperationsAPIBase):
    """
    API related Functionality for Operation
    """

    def __init__(self, model, *args, **kwargs):
        super().__init__(model=model, *args, **kwargs)
        self.model = self._get_model(model)
        for k, v in kwargs.items():
            setattr(self, k, v)
