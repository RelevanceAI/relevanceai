from relevanceai.package_utils.config import CONFIG
from relevanceai.package_utils.transport import Transport
from relevanceai.package_utils.logger import LoguruLogger


def str2bool(v: str):
    return v.lower() in ("yes", "true", "t", "1")


class _Base(Transport, LoguruLogger):
    """_Base class for all relevanceai client utilities"""

    def __init__(self, project: str, api_key: str, firebase_uid: str):
        self.project = project
        self.api_key = api_key
        self.firebase_uid = firebase_uid

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

    def _region_to_url(self, region: str):
        # to match our logic in dashboard
        # add print statement to double-check region support now
        if region == "old-australia-east":
            url = "https://gateway-api-aueast.relevance.ai/latest"
        else:
            url = f"https://api.{region}.relevance.ai/latest"
        return url
