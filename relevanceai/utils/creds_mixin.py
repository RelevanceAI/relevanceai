# mypy: ignore-errors

"""
Credential-related helpers
"""
from relevanceai.client.helpers import Credentials, region_to_url


class CredentialsMixin:
    project: str
    api_key: str
    region: str
    firebase_uid: str

    def _set_mixpanel_write_key(self, *args, **kwargs):
        raise NotImplementedError

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
