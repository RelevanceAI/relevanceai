from typing import Optional
from relevanceai.client.helpers import Credentials

from relevanceai.utils.base import _Base


class AggregateClient(_Base):
    """Aggregate service"""

    def __init__(self, credentials: Credentials):
        super().__init__(credentials)
