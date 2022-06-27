"""
Clean HTML
"""
from relevanceai.operations_new.ops_base import OperationAPIBase
from relevanceai.operations_new.processing.text.html_clean.base import (
    CleanTextBase,
)


class CleanTextOps(CleanTextBase, OperationAPIBase):
    """
    Clean text operations
    """
