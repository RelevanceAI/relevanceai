from relevanceai.config import CONFIG
from relevanceai.transport import Transport
from relevanceai.logger import LoguruLogger


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


class _Base(Transport, LoguruLogger):
    """Base class for all relevanceai client utilities"""

    def __init__(self, project: str, api_key: str):
        self.project = project
        self.api_key = api_key
        self.config = CONFIG
        # Initialize logger
        super().__init__()

    ### Configurations

    @property
    def base_url(self):
        return CONFIG.get_field("api.base_url", CONFIG.config)

    @base_url.setter
    def base_url(self, value: str):
        if value.endswith("/"):
            value = value[:-1]
        CONFIG.set_option("api.base_url", value)

    @property
    def base_ingest_url(self):
        return CONFIG.get_field("api.base_ingest_url", CONFIG.config)

    @base_ingest_url.setter
    def base_ingest_url(self, value: str):
        if value.endswith("/"):
            value = value[:-1]
        CONFIG.set_option("api.base_ingest_url", value)
