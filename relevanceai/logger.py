from os import error
import sys
from typing import Callable
from loguru import logger as loguru_logger
from abc import abstractmethod
from relevanceai.config import CONFIG


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


class AbstractLogger:
    """Base Logging Instance"""

    info: Callable
    error: Callable
    success: Callable
    debug: Callable
    warning: Callable
    critical: Callable
    warn: Callable
    # @property
    # @abstractmethod
    # def logger(self):
    #     raise NotImplementedError


class LoguruLogger(AbstractLogger):
    """Using verbose loguru as base logger for now"""

    # Add Logging
    def __init__(self):
        self._init_logger()

    @property
    def logger(self):
        # if hasattr(self, "_logger"):
        #     return self._logger
        self._init_logger()
        return self._logger

    def _init_logger(self):
        self.config = CONFIG
        logging_level = self.config.get_option("logging.logging_level")
        log_to_file = str2bool(self.config.get_option("logging.log_to_file"))
        logger = loguru_logger
        logger.remove()
        logger.add(sys.stdout, level=logging_level)
        if log_to_file:
            logger.add(f"relevanceai.log", level=logging_level, rotation="100 MB")
        self._logger = logger
