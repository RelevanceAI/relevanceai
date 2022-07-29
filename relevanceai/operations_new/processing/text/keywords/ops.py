"""
    Run operations to get the keyphrases of this document
"""

from relevanceai.operations_new.processing.text.keywords.transform import (
    KeyWordTransform,
)
from relevanceai.operations_new.ops_base import OperationAPIBase


class KeyWordOps(OperationAPIBase, KeyWordTransform):
    def __init__(
        self,
        fields: list,
        model_name: str = "all-mpnet-base-v2",
        lower_bound: int = 0,
        upper_bound: int = 3,
        output_fields: list = None,
        stop_words: list = None,
        max_keywords: int = 1,
        use_maxsum: bool = False,
        nr_candidates: int = 20,
        **kwargs
    ):
        self.fields = fields
        self.model_name = model_name
        self.output_fields = output_fields
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.stop_words = stop_words
        self.max_keywords = max_keywords
        self.use_maxsum = use_maxsum
        self.nr_candidates = nr_candidates
        super().__init__(**kwargs)
