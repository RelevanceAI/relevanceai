from typing import Any, List

from relevanceai.operations_new.vectorize.ops import VectorizeOps
from relevanceai.operations_new.vectorize.text.base import VectorizeTextBase


class VectorizeTextOps(VectorizeOps, VectorizeTextBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
