"""Transformers
"""
from relevanceai.operations_new.processing.transformers.base import (
    TransformersPipelineBase,
)
from relevanceai.operations_new.ops_api_base import OperationAPIBase


class TransformersPipelineOps(TransformersPipelineBase, OperationAPIBase):
    pass
