"""
Clean HTML
"""
from relevanceai.operations_new.ops_api_base import OperationAPIBase
from relevanceai.operations_new.processing.text.clean.base import (
    CleanTextBase,
)


class CleanTextOps(CleanTextBase, OperationAPIBase):
    """
    Clean text operations
    """

    @property
    def name(self):
        return "clean-text"
