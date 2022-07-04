"""
Sentence Splitting Operator
Split Text
"""
from relevanceai.operations_new.ops_base import OperationAPIBase
from relevanceai.operations_new.processing.text.sentence_splitting.transform import (
    SentenceSplittingTransform,
)


class SentenceSplitterOps(OperationAPIBase, SentenceSplittingTransform):
    pass
