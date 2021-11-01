import sys
import time

from loguru import logger

from .config import CONFIG
from .transport import Transport


class Base(Transport):
    """Base class for all VecDB utilities
    """
    config = CONFIG

    #Add Logging
    logger = logger
    logger.remove()
    logger.add(sys.stdout, level= CONFIG.logging_level)
    if CONFIG.log_to_file:
        logger.add("vecdb_{time}.log", level=CONFIG.logging_level, rotation="100 MB")

    def __init__(self, 
        project: str, api_key: str, 
        base_url: str):
        self.project = project
        self.api_key = api_key
        self.base_url = base_url
