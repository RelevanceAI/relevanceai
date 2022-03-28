from relevanceai.client.helpers import Credentials
from relevanceai.constants import CONFIG
from relevanceai.utils.transport import Transport
from relevanceai.utils.logger import LoguruLogger


def str2bool(v: str):
    return v.lower() in ("yes", "true", "t", "1")


class _Base(Transport, LoguruLogger):
    """_Base class for all relevanceai client utilities"""

    def __init__(
        self,
        credentials: Credentials,
        **kwargs,
    ):
        self.project = credentials.project
        self.api_key = credentials.api_key
        self.firebase_uid = credentials.firebase_uid

        self.config = CONFIG
        # Initialize logger
        super().__init__(**kwargs)

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
    def mixpanel_write_key(self):
        return CONFIG.get_field("mixpanel.write_key", CONFIG.config)

    @property
    def base_ingest_url(self):
        return CONFIG.get_field("api.base_ingest_url", CONFIG.config)

    @base_ingest_url.setter
    def base_ingest_url(self, value: str):
        if value.endswith("/"):
            value = value[:-1]
        CONFIG.set_option("api.base_ingest_url", value)

    @property
    def region(self):
        if hasattr(self, "_region"):
            return self._region
        return CONFIG["api.region"]

    @region.setter
    def region(self, region_value):
        CONFIG["api.region"] = region_value
        self._region = region_value
