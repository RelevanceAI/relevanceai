"""
Sentence Splitting Operator
Split Text
"""
from relevanceai.operations_new.ops_base import OperationAPIBase
from relevanceai.operations_new.processing.text.ner.transform import ExtractNER


class ExtractNEROps(OperationAPIBase, ExtractNER):
    pass
