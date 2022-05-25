"""
Sentence Splitting Operator
Split Text
"""
from relevanceai.operations_new.apibase import OperationAPIBase
from relevanceai.operations_new.processing.text.sentence_splitting.base import (
    SentenceSplittingBase,
)


class SentenceSplitterOps(OperationAPIBase, SentenceSplittingBase):
    pass
