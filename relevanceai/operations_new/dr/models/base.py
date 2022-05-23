from abc import ABC, abstractmethod

from typing import Any, List, Union

from relevanceai.utils import DocUtils


class DimReductionModelBase(ABC, DocUtils):
    model_name: str
    alias: Union[str, None]

    def vector_name(self, fields: List[str]) -> str:
        if isinstance(self.alias, str):
            return f"{self.alias}_vector_"
        else:
            if len(fields) > 1:
                return f"{self.model_name}_{fields[0]}_vector_"
            else:
                return f"concat_{self.model_name}_vector_"

    def fit(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def fit_transform(self, *args, **kwargs) -> Any:
        raise NotImplementedError
