from relevanceai.operations_new.cluster.models.sklearn.base import SklearnModelBase
from sklearn.cluster import OPTICS
from typing import Optional


class OpticsModel(SklearnModelBase):
    def __init__(self, model_kwargs: Optional[dict] = None):
        if model_kwargs is None:
            model_kwargs = {}
        self.model = OPTICS(**model_kwargs)

    @property
    def name(self):
        return "optics"

    @property
    def cluster_centers_(self):
        return None

    @property
    def alias(self):
        return "optics"
