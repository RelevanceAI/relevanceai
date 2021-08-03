import logging

class BaseConfig:
    def __getitem__(self, item):
        return getattr(self, item)
    
    def __setitem__(self, item, value):
        setattr(self, item, value)

class TransportConfig(BaseConfig):
    number_of_retries: int = 1
    seconds_between_retries: int = 2

    #Set Logging Rules
    log: bool = False
    log_to_file: bool = True
    log_to_console: bool = True
    logging_level: int = logging.INFO

class Config(TransportConfig):
    """All the configs - which are to be inhertied
    """

CONFIG = Config()
