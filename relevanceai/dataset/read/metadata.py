from relevanceai.client.helpers import Credentials
from relevanceai._api import APIClient


class Metadata(APIClient):
    """Metadata object"""

    def __init__(self, metadata: dict, credentials: Credentials, dataset_id: str):
        self._metadata = metadata
        self.dataset_id = dataset_id
        super().__init__(credentials)

    def __repr__(self):
        return str(self.to_dict())

    def __contains__(self, m):
        return m in self.to_dict().keys()

    def insert(self, metadata: dict, verbose: bool = False):
        """Insert metadata"""
        results = self.datasets.post_metadata(self.dataset_id, metadata)
        if results == {}:
            return
        else:
            return results

    def upsert(self, metadata: dict, verbose: bool = False):
        """Upsert metadata."""
        original_metadata: dict = self.datasets.metadata(self.dataset_id)
        original_metadata.update(metadata)

        # TODO: how do you fire and forget this
        return self.insert(metadata)

    def __getitem__(self, key):
        return self.get_field(key, self._metadata)

    def __setitem__(self, key, value):
        self.set_field(key, self._metadata, value)
        self.upsert(self._metadata)

    def to_dict(self):
        return self._metadata

    def clear(self):
        self.datasets.post_metadata(dataset_id=self.dataset_id, metadata={})
