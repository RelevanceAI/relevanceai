from tkinter import N
from relevanceai.operations_new.apibase import OperationAPIBase
from relevanceai.operations_new.vectorize.base import VectorizeBase


class VectorizeOps(VectorizeBase, OperationAPIBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
