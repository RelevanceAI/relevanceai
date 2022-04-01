from relevanceai.client.helpers import Credentials
from relevanceai.utils.transport import Transport
from relevanceai.utils.logger import LoguruLogger
from relevanceai.utils.creds_mixin import CredentialsMixin


def str2bool(v: str):
    return v.lower() in ("yes", "true", "t", "1")


class _Base(Transport, LoguruLogger, CredentialsMixin):
    """_Base class for all relevanceai client utilities"""

    def __init__(
        self,
        credentials: Credentials,
        **kwargs,
    ):
        self.project = credentials.project
        self.api_key = credentials.api_key
        self.firebase_uid = credentials.firebase_uid
        self.credentials = credentials
        self._set_variables()
        # Initialize logger
        super().__init__(**kwargs)

    ### Configurations
    @property
    def mixpanel_write_key(self):
        return self.config.get_field("mixpanel.write_key", self.config.config)

    @property
    def base_ingest_url(self):
        return self.config.get_field("api.base_ingest_url", self.config.config)

    @base_ingest_url.setter
    def base_ingest_url(self, value: str):
        if value.endswith("/"):
            value = value[:-1]
        self.config.set_option("api.base_ingest_url", value)

    @property
    def region(self):
        if hasattr(self, "_region"):
            return self._region
        return self.config["api.region"]

    @region.setter
    def region(self, region_value):
        self.config["api.region"] = region_value
        self._region = region_value
