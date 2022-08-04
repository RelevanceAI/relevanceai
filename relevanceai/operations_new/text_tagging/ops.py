"""
Labelling with API-related functions
"""
from relevanceai.operations_new.text_tagging.transform import TextTagTransform
from relevanceai.operations_new.ops_base import OperationAPIBase


class TextTagOps(TextTagTransform, OperationAPIBase):  # type: ignore
    """
    Label Operations
    """

    def __init__(
        self,
        credentials,
        fields,
        output_fields,
        model_id="cross-encoder/nli-deberta-v3-large",
        **kwargs
    ):
        self.credentials = credentials
        self.fields = fields
        if model_id is None:
            self.model_id = ("cross-encoder/nli-deberta-v3-large",)
        if output_fields is None:
            self.output_fields = [self._generate_output_field(f) for f in fields]
        else:
            self.output_fields = output_fields
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.model_id = model_id
