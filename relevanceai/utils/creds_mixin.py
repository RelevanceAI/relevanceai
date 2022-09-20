"""
Credential-related helpers
"""
from relevanceai.client.helpers import region_to_url


class CredentialsMixin:
    def _set_variables(self):
        (
            self.project,
            self.api_key,
            self.region,
            self.session_token,
        ) = self.credentials.split_token()

        self.base_url = region_to_url(self.region)
        self.base_ingest_url = region_to_url(self.region)

        try:
            self._set_mixpanel_write_key()
        except Exception as e:
            pass
