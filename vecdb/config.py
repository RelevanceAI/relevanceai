<<<<<<< Updated upstream
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
=======
"""Config
"""
class Config:
    number_of_retries = 3
    seconds_between_retries = 2
    seconds_between_status = 10
    def __getitem__(self, item):
        return getattr(self, item)
    
    def __setitem__(self, item, value):
        setattr(self, item, value)
>>>>>>> Stashed changes
