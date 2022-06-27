"""
Sentence Splitting Operator
Split Text
"""
from relevanceai.operations_new.ops_base import OperationAPIBase
from relevanceai.operations_new.processing.text.sentence_splitting.base import (
    SentenceSplittingBase,
)


class SentenceSplitterOps(OperationAPIBase, SentenceSplittingBase):
    pass
