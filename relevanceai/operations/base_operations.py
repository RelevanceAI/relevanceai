"""
Base class for operations.
"""
from typing import Any, List
from relevanceai.client.helpers import Credentials


class BaseOps:
    """
    Base class for operations
    """

    @classmethod
    def init(self, **kwargs):
        return self(**kwargs)
