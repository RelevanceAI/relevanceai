from abc import abstractmethod

from typing import Any, Union

from relevanceai.utils import DocUtils


class DimReductionModelBase(DocUtils):
    model_name: str
    alias: Union[str, None]

    def vector_name(self, field):
        if isinstance(self.alias, str):
            return self.alias
        else:
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
