"""Counting Text operations
"""
from relevanceai.operations_new.processing.text.count.transform import (
    CountTextTransform,
)
from relevanceai.operations_new.ops_base import OperationAPIBase


class CountTextOps(CountTextTransform, OperationAPIBase):
    @property
    def name(self):
        return "count-text"
