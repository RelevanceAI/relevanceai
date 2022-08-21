"""Extract noun operations
"""
import string
from relevanceai.constants.errors import MissingPackageError
from relevanceai.operations_new.ops_base import OperationAPIBase
from relevanceai.operations_new.processing.text.extract_nouns.transform import (
    ExtractNounsTransform,
)


class ExtractNounsOps(ExtractNounsTransform, OperationAPIBase):
    def __init__(
        self,
        fields: list,
        model_name: str,
        output_fields: list,
        cutoff_probability: float,
        stopwords: list = None,
        **kwargs
    ):
        try:
            from flair.models import SequenceTagger
        except ModuleNotFoundError:
            raise MissingPackageError("flair")
        self.fields = fields
        self.model_name = model_name
        self.output_fields = output_fields
        self.cutoff_probability = cutoff_probability
        self.tagger = SequenceTagger.load(model_name)
        from relevanceai.constants import STOPWORDS

        PUNCTUATION = [x for x in string.punctuation]
        if stopwords is not None:
            self.stopwords = STOPWORDS + PUNCTUATION + stopwords
        else:
            self.stopwords = STOPWORDS + PUNCTUATION

        for k, v in kwargs.items():
            setattr(self, k, v)

        # super().__init__(**kwargs)

    def get_operation_metadata(self):
        return {
            "fields": self.fields,
            "model_name": self.model_name,
            "output_fields": self.output_fields,
            "cutoff_probability": self.cutoff_probability,
            "stopwords": self.stopwords,
        }
