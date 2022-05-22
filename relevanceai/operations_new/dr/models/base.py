from abc import ABC, abstractmethod

from typing import Any, Union

from relevanceai.utils import DocUtils


class DimReductionModelBase(ABC, DocUtils):
    model_name: str
    alias: Union[str, None]

    def vector_name(self, field):
        if isinstance(self.alias, str):
            return f"{self.alias}_vector_"
        else:

            return f"{self.model_name}_{field}"

    def fit(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def fit_transform(self, *args, **kwargs) -> Any:
        raise NotImplementedError
