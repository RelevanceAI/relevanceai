from os import error
import sys

from typing import Callable
from loguru import logger as loguru_logger
from abc import abstractmethod
from relevanceai.config import CONFIG


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


class AbstractLogger:
    """_Base Logging Instance"""

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
        self._init_logger()
        return self._logger

    def _init_logger(self):
        self.config = CONFIG
        logging_level = self.config.get_option("logging.logging_level")
        log_to_file = str2bool(self.config.get_option("logging.log_to_file"))
        log_file_name = self.config.get_option("logging.log_file_name") + ".log"
        enable_logging = str2bool(self.config.get_option("logging.enable_logging"))

        logger = loguru_logger
        logger.remove()
        if enable_logging:
            logger.add(sys.stdout, level=logging_level)
            if log_to_file:
                logger.add(log_file_name, level=logging_level, rotation="100 MB")
        self._logger = logger


class FileLogger:
    """Log system output to a file if it gets messy."""

    def __init__(self, fn: str = "logs"):
        self.fn = fn
        self._original_output = sys.stdout

    def __enter__(self, fn: str = "logs"):
        sys.stdout = open(self.fn, "w")

    def __exit__(self, *args, **kw):
        sys.stdout.close()
        sys.stdout = self._original_output
        if self._if_not_empty():
            print(f"Logs have been saved to {self.fn}")

    def _if_not_empty(self):
        log_file = open(self.fn, "r")
        for line in log_file:
            log_file.close()
            if line != "\n":
                return True
            else:
                return False
