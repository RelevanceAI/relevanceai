from relevanceai.operations_new.processing.text.translate.transform import (
    TranslateTransform,
)
from relevanceai.operations_new.ops_base import OperationAPIBase


class TranslateOps(TranslateTransform, OperationAPIBase):
    def __init__(self, fields: list, model_id: str = None, *args, **kwargs):
        self.fields = fields
        self.model_id = model_id
        super().__init__(fields, model_id, *args, **kwargs)

    @property
    def name(self):
        return "translate"
