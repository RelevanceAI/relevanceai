from typing import Any, List

from relevanceai.operations_new.vectorize.ops import VectorizeOps
from relevanceai.operations_new.vectorize.text.transform import VectorizeTextTransform


class VectorizeTextOps(VectorizeOps, VectorizeTextTransform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
