"""
All operations related to sentiment
"""
from relevanceai.operations_new.sentiment.transform import SentimentTransform
from relevanceai.operations_new.ops_base import OperationAPIBase


class SentimentOps(SentimentTransform, OperationAPIBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
