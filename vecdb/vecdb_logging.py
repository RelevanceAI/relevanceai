import time

from loguru import logger


def create_logger(*args, **kwargs):
    """Use loguru's default logger for now."""
    return logger


class Profiler:
    """The profiler is intended to track time in a function"""

    def __init__(self, log, logging_level, log_to_file, log_to_console, logging_info):
        self.log = log
        if self.log is True:
            self.logger = create_logger(logging_level, log_to_file, log_to_console)
            self.info = logging_info

    def __enter__(self):
        self.t1 = time.time()
        return self

    def __exit__(self, *args):
        time_diff = time.time() - self.t1
        if self.log is True:
            self.logger.info(f"{self.info} ran in {time_diff} seconds")
        return args
