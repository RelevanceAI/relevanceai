from typing import Any, List

from relevanceai.operations_new.vectorize.base import VectorizeBase


class VectorizeOps(VectorizeBase):
    def __init__(self, fields: List[str], models: List[Any]):

        self.fields = fields
        self.models = [self._get_model(model) for model in models]
