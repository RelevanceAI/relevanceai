from typing import Any, List

from relevanceai.operations_new.vectorize.ops import VectorizeOps
from relevanceai.operations_new.vectorize.image.base import VectorizeImageBase


class VectorizeImageOps(VectorizeOps, VectorizeImageBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
