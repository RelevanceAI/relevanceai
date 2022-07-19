"""Counting Text operations
"""
from relevanceai.operations_new.processing.text.count.transform import (
    CountTextTransform,
)
from relevanceai.operations_new.ops_base import OperationAPIBase


class CountTextOps(CountTextTransform, OperationAPIBase):
    def __init__(
        self,
        credentials,
        text_fields: list,
        include_char_count: bool = True,
        include_word_count: bool = True,
        include_sentence_count: bool = False,
        output_fields: list = None,
        **kwargs
    ):
        self.text_fields = text_fields
        self.include_char_count = include_char_count
        self.include_word_count = include_word_count
        self.include_sentence_count = include_sentence_count
        self.output_fields = output_fields
        self.credentials = credentials
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def name(self):
        return "count-text"
