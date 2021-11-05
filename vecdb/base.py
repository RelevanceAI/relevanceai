import os
import sys

from loguru import logger as loguru_logger

from .config import CONFIG
from .transport import Transport


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


class Base(Transport):
    """Base class for all VecDB utilities"""

    def __init__(self, project: str, api_key: str, base_url: str):
        self.project = project
        self.api_key = api_key
        self.base_url = base_url
        self.config = CONFIG

    # Add Logging
    @property
    def logger(self):
        logging_level = self.config.get_option("logging.logging_level")
        log_to_file = str2bool(self.config.get_option("logging.log_to_file"))
        logger = loguru_logger
        logger.remove()
        logger.add(sys.stdout, level=logging_level)
        if log_to_file:
            logger.add(f"vecdb.log", level=logging_level, rotation="100 MB")
        return logger
