"""
Base Operations Class
"""


from typing import Any
from abc import abstractmethod

from relevanceai.utils import DocUtils
from relevanceai._api import APIClient


class OperationsBase(APIClient, DocUtils):
    """Contains API-related functions"""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

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
    def run(self, *args, **kwargs):
        """The function is an abstract method that raises a NotImplementedError if it is not implemented"""
        raise NotImplementedError
