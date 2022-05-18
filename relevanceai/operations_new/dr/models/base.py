from abc import abstractmethod
from typing import Any

from relevanceai.utils import DocUtils


class DimReductionModelBase(DocUtils):
    model_name: str

    def vector_name(self, field):
        return f"{field}_{self.model_name}_vector_"

    @abstractmethod
    def fit(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def fit_predict(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    @abstractmethod
    def fit_transform(self, *args, **kwargs) -> Any:
        raise NotImplementedError
