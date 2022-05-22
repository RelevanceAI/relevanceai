"""
Base Operations Class
"""
from abc import ABC, abstractmethod

from typing import Any, Dict, List

from relevanceai.utils import DocUtils


class OperationBase(ABC, DocUtils):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.run(*args, **kwargs)

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def run(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """The function is an abstract method that raises a NotImplementedError if it is not implemented"""
        raise NotImplementedError
