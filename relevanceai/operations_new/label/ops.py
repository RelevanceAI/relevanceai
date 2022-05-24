"""
Labelling with API-related functions
"""
from relevanceai.dataset import Dataset
from relevanceai.operations_new.label.base import LabelBase
from relevanceai.operations_new.apibase import OperationAPIBase
from relevanceai.operations_new.context import Upload


class LabelOps(OperationAPIBase, LabelBase):  # type: ignore
    """
    Label Operations
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
