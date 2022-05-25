"""
Labelling with API-related functions
"""
from relevanceai.operations_new.label.base import LabelBase
from relevanceai.operations_new.apibase import OperationAPIBase


class LabelOps(LabelBase, OperationAPIBase):  # type: ignore
    """
    Label Operations
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
