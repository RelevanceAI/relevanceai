"""
Clean HTML
"""
from relevanceai.operations_new.apibase import OperationsAPIBase
from relevanceai.operations_new.processing.text.html_clean.base import (
    CleanTextBase,
)


class CleanTextOps(CleanTextBase, OperationsAPIBase):
    """
    Clean text operations
    """
