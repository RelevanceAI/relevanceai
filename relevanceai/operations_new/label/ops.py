"""
Labelling with API-related functions
"""
from relevanceai.operations_new.label.transform import LabelTransform
from relevanceai.operations_new.ops_base import OperationAPIBase


class LabelOps(LabelTransform, OperationAPIBase):  # type: ignore
    """
    Label Operations
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
