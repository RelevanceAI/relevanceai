from relevanceai.operations_new.processing.text.translate.transform import (
    TranslateTransform,
)
from relevanceai.operations_new.ops_base import OperationAPIBase


class TranslateOps(TranslateTransform, OperationAPIBase):
    def __init__(
        self, credentials, fields: list, model_id: str = None, *args, **kwargs
    ):
        self.fields = fields
        self.model_id = model_id
        self.credentials = credentials
        super().__init__(fields, model_id)

    @property
    def name(self):
        return "translate"
