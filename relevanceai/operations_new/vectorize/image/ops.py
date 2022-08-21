from typing import Any, List

from relevanceai.operations_new.vectorize.ops import VectorizeOps
from relevanceai.operations_new.vectorize.image.transform import VectorizeImageTransform


class VectorizeImageOps(VectorizeOps, VectorizeImageTransform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
