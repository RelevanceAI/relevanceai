"""
Base class for operations
"""
from relevanceai.client.helpers import region_to_url, Credentials

class BaseOps:
    """
    Base class for operations
    """
    def __init__(self, credentials: Credentials, **kwargs):
        self.credentials = credentials
        self.project = credentials.project
        self.api_key = credentials.api_key
        self.firebase_uid = credentials.firebase_uid
        self.credentials = credentials
        self._set_variables()
        # Initialize logger
        super().__init__(**kwargs)

    def _set_variables(self):
        (
            self.project,
            self.api_key,
            self.region,
            self.firebase_uid,
        ) = self.credentials.split_token()

        self.base_url = region_to_url(self.region)
        self.base_ingest_url = region_to_url(self.region)

        try:
            self._set_mixpanel_write_key()
        except Exception as e:
            pass

