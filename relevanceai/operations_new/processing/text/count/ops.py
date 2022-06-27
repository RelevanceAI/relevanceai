"""Counting Text operations
"""
from relevanceai.operations_new.processing.text.count.base import CountTextBase
from relevanceai.operations_new.ops_base import OperationAPIBase


class CountTextOps(CountTextBase, OperationAPIBase):
    @property
    def name(self):
        return "count-text"
