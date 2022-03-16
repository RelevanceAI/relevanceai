from typing import List, Any

from sklearn.preprocessing import (
    MinMaxScaler,
    MaxAbsScaler,
    Normalizer,
    RobustScaler,
    StandardScaler,
)

from relevanceai.dataset.crud.dataset_write import Write


class Scale(Write):
    def scale(self, fields: List[Any], scaler: Any):
        documents = self.get_all_documents(fields=fields)
