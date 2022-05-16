"""
Labelling with API-related functions
"""
from relevanceai.operations_new.base import BaseOps
from relevanceai.operations_new.label.base import LabelBase

class LabelOps(BaseOps, LabelBase):
    """
    Label Operations
    """
    def label(self):
        raise NotImplementedError
