"""
Labelling with API-related functions
"""
from re import I
from relevanceai.operations_new.text_tagging.transform import TextTagTransform
from relevanceai.operations_new.ops_base import OperationAPIBase


class TextTagOps(TextTagTransform, OperationAPIBase):  # type: ignore
    """
    Label Operations
    """

    def __init__(
        self, credentials, fields, labels, output_fields, model_id=None, **kwargs
    ):
        self.credentials = credentials
        self.fields = fields
        self.text_field = fields[0]
        self.labels = labels
        if len(fields) > 1:
            raise ValueError("cannot support more than 1 field.")
        if model_id is None:
            self.model_id = "cross-encoder/nli-deberta-v3-large"
        else:
            self.model_id = model_id
        if output_fields is None:
            self.output_fields = [self._generate_output_field(f) for f in fields]
        else:
            self.output_fields = output_fields
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.model_id = model_id
