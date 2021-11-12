import sys
from loguru import logger as loguru_logger
from abc import abstractmethod

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

class AbstractLogger:
    """Base Logging Instance
    """
    # @property
    # @abstractmethod
    # def logger(self):
    #     raise NotImplementedError

class LoguruLogger(AbstractLogger):
    # Add Logging
    def __init__(self):
        self._init_logger()

    @property
    def logger(self):
        return self._logger

    def _init_logger(self):
        logging_level = self.config.get_option("logging.logging_level")
        log_to_file = str2bool(self.config.get_option("logging.log_to_file"))
        logger = loguru_logger
        logger.remove()
        logger.add(sys.stdout, level=logging_level)
        if log_to_file:
            logger.add(f"relevanceai.log", level=logging_level, rotation="100 MB")
        self._logger = logger 
