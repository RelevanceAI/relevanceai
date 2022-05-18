"""
Base Operations Class
"""


from typing import Any, Dict, List
from abc import abstractmethod

from relevanceai.utils import DocUtils


class OperationBase(DocUtils):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """This function is called when an instance of the class is called

        Parameters
        ----------
         : Any
            param func: The function to be decorated.
         : Any
            param func: The function to be decorated.

        """
        self.run(*args, **kwargs)

    @abstractmethod
    def run(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """The function is an abstract method that raises a NotImplementedError if it is not implemented"""
        raise NotImplementedError
