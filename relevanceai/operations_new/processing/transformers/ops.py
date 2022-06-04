"""Transformers
"""
from relevanceai.operations_new.processing.transformers.base import (
    TransformersPipelineBase,
)
from relevanceai.operations_new.apibase import OperationAPIBase


class TransformersPipelineOps(TransformersPipelineBase, OperationAPIBase):
    pass
