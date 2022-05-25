"""
Clean HTML
"""
from relevanceai.operations_new.apibase import OperationAPIBase
from relevanceai.operations_new.processing.text.html_clean.base import (
    CleanTextBase,
)


class CleanTextOps(CleanTextBase, OperationAPIBase):
    """
    Clean text operations
    """
