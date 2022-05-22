from abc import ABC, abstractmethod
from typing import Any, Dict, List

from relevanceai.utils import DocUtils


class ModelBase(ABC, DocUtils):
    model_name: str

    def fit(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def fit_predict(self, *args, **kwargs) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def predict_documents(self, *args, **kwargs) -> List[Dict[str, Any]]:
        raise NotImplementedError
