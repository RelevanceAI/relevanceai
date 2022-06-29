"""
Clean HTML
"""
from relevanceai.operations_new.ops_base import OperationAPIBase
from relevanceai.operations_new.processing.text.html_clean.transform import (
    CleanTextTransform,
)


class CleanTextOps(CleanTextTransform, OperationAPIBase):
    """
    Clean text operations
    """
