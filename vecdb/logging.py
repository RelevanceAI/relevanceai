import logging
import time

def create_logger(logging_level, log_to_file, log_to_console):
    logger = logging.getProfiler()
    logger.setLevel(logging_level)

    formatter = logging.Formatter('%(asctime)s | %(message)s')

    if (logger.hasHandlers()):
        logger.handlers.clear()

    if log_to_file == True:
        file_handler = logging.FileHandler('vecdb.log')
        file_handler.setLevel(logging_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if log_to_console == True:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging_level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger

class Profiler:
    """The profiler is intended to track time in a function
    """
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
            self.logger.info(f'{self.info} ran in {time_diff} seconds')
        return args
