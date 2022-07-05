import os
import sys

from typing import Callable
from loguru import logger as loguru_logger
from abc import abstractmethod

# from relevanceai.constants import CONFIG
from relevanceai.utils.config_mixin import ConfigMixin

# suppressing warnings and errors
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull


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


class LoguruLogger(AbstractLogger, ConfigMixin):
    """Using verbose loguru as base logger for now"""

    # Add Logging

    def __init__(self, **kwargs):
        self._init_logger()

    @property
    def logger(self):
        self._init_logger()
        return self._logger

    def _init_logger(self):
        logging_level = self.config.get_option("logging.logging_level")
        log_to_file = str2bool(self.config.get_option("logging.log_to_file"))
        log_file_name = self.config.get_option("logging.log_file_name") + ".log"
        enable_logging = str2bool(self.config.get_option("logging.enable_logging"))

        logger = loguru_logger
        logger.remove()
        if enable_logging:
            logger.add(sys.stdout, level=logging_level, format="{message}")
            if log_to_file:
                logger.add(log_file_name, level=logging_level, rotation="100 MB")
        self._logger = logger


class FileLogger:
    """Log system output to a file if it gets messy."""

    def __init__(self, fn: str = "logs.txt", verbose: bool = False, log_to_file=True):
        self.fn = fn
        if verbose:
            print(f"Logging to {self.fn}")
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self.verbose = verbose
        # set this as a parameter in case users are in debugging mode and don't want to open a log
        # file when experimenting
        self.log_to_file = log_to_file

    def __enter__(self, fn: str = "logs"):
        if not os.path.exists(self.fn):
            self._existed = False
            self._initial_length = 0
            if self.log_to_file:
                sys.stdout = open(self.fn, "w")
                sys.stderr = open(self.fn, "w")
        else:
            self._existed = True
            with open(self.fn, "rb") as f:
                # must include "b" because of occasional test failure,
                # probably due to the use of emojis.
                self._initial_length = len(f.readlines())
            if self.log_to_file:
                sys.stdout = open(self.fn, "a")
                sys.stderr = open(self.fn, "a")

    def __exit__(self, *args, **kw):
        sys.stderr.close()
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        # explicitly spell out the four cases of whether the file existed
        # beforehand and whether new lines were added
        if self.log_to_file:
            if self._existed and self._lines_added():
                if self.verbose:
                    print(f"Log {self.fn} has been updated")
            elif not self._existed and self._lines_added():
                if self.verbose:
                    print(
                        f"📌 Your logs have been saved to {self.fn}. If you are debugging, you can turn file logging off by setting `log_to_file=False`.📌"
                    )
            elif self._existed and not self._lines_added():
                # Do nothing if the file already existed and no lines were added
                pass
            elif not self._existed and not self._lines_added():
                # If the file did not already exist and no lines were added, just
                # delete the file.
                os.remove(self.fn)

    def _lines_added(self):
        with open(self.fn, "rb") as f:
            final_length = len(f.readlines())

        if final_length > self._initial_length:
            return True
        else:
            return False

    def _if_not_empty(self):
        with open(self.fn, "rb") as f:
            lines = f.readlines()
        if len(lines) > 1:
            return True
        return False

    def log(self, text):
        with open(self.fn, "wb") as f:
            f.write(text)


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)
