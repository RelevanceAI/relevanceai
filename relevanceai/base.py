import sys
import time

from relevanceai.config import CONFIG
from relevanceai.transport import Transport
from relevanceai.logger import LoguruLogger


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


class _Base(Transport, LoguruLogger):
    """Base class for all relevanceai client utilities"""

    def __init__(self, project: str, api_key: str):
        self.project = project
        self.api_key = api_key
        self.config = CONFIG
        # Initialize logger
        super().__init__()
