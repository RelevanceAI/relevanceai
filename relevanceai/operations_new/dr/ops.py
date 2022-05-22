from relevanceai.operations_new.apibase import OperationsAPIBase
from relevanceai.operations_new.dr.base import DimReductionBase


class DimReductionOps(DimReductionBase, OperationsAPIBase):
    """
    API related Functionality for Operation
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        super(OperationsAPIBase).__init__(*args, **kwargs)
