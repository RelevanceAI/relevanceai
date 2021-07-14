class BaseConfig:
    def __getitem__(self, item):
        return getattr(self, item)
    
    def __setitem__(self, item, value):
        setattr(self, item, value)

class TransportConfig(BaseConfig):
    number_of_retries: int = 3
    seconds_between_retries: int = 2

class Config(TransportConfig):
    """All the configs - which are to be inhertied
    """

CONFIG = Config()
