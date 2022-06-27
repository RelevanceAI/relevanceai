"""Counting Text operations
"""
from relevanceai.operations_new.processing.text.count.base import CountTextBase
from relevanceai.operations_new.apibase import OperationAPIBase


class CountTextOps(CountTextBase, OperationAPIBase):
    @property
    def name(self):
        return "count-text"
