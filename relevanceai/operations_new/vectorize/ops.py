from relevanceai.operations_new.ops_base import OperationAPIBase
from relevanceai.operations_new.vectorize.transform import VectorizeTransform


class VectorizeOps(VectorizeTransform, OperationAPIBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
