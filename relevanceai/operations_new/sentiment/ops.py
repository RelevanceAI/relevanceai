"""
All operations related to sentiment
"""
from relevanceai.operations_new.sentiment.transform import SentimentTransform
from relevanceai.operations_new.ops_base import OperationAPIBase


class SentimentOps(SentimentTransform, OperationAPIBase):
    pass
