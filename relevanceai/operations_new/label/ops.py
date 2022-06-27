"""
Labelling with API-related functions
"""
from relevanceai.operations_new.label.base import LabelBase
from relevanceai.operations_new.ops_api_base import OperationAPIBase


class LabelOps(LabelBase, OperationAPIBase):  # type: ignore
    """
    Label Operations
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
