import time
from .transport import Transport
from .config import Config

class Base(Transport):
    """Base class for all VecDB utilities
    """
    config: Config = Config()
