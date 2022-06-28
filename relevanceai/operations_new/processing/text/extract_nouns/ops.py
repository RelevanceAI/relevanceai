"""Extract noun operations
"""

from relevanceai.operations_new.apibase import OperationAPIBase
from relevanceai.operations_new.processing.text.extract_nouns.transform import (
    ExtractNounsBase,
)


class ExtractNounsOps(OperationAPIBase, ExtractNounsBase):
    def __init__(
        self,
        text_fields: list,
        model_name: str,
        output_fields: list,
        cutoff_probability: float,
        stopwords: list = None,
        **kwargs
    ):
        self.text_fields = text_fields
        self.model_name = model_name
        self.output_fields = output_fields
        self.cutoff_probability = cutoff_probability
        self.tagger = SequenceTagger.load(model_name)
        from relevanceai.constants import STOPWORDS

        if stopwords is not None:
            self.stopwords = STOPWORDS + stopwords
        else:
            self.stopwords = STOPWORDS

        for k, v in kwargs:
            setattr(self, k, v)

        super().__init__(**kwargs)
