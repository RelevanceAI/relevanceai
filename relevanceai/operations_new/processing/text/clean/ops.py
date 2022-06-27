"""
Clean HTML
"""
from relevanceai.operations_new.ops_base import OperationAPIBase
from relevanceai.operations_new.processing.text.clean.transform import (
    CleanTextTransform,
)


class CleanTextOps(CleanTextTransform, OperationAPIBase):
    """
    Clean text operations
    """

    @property
    def name(self):
        return "clean-text"
