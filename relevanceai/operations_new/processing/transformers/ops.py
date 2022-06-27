"""Transformers
"""
from relevanceai.operations_new.processing.transformers.transform import (
    TransformersPipelineTransform,
)
from relevanceai.operations_new.ops_base import OperationAPIBase


class TransformersPipelineOps(TransformersPipelineTransform, OperationAPIBase):
    pass
