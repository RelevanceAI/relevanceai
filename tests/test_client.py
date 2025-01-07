import pytest
from relevanceai import RelevanceAI

class TestRelevanceAI:
    def test_client_init(self):
        client = RelevanceAI(api_key="test_key", region="test_region", project="test_project")
        assert client.api_key == "test_key"
        assert client.region == "test_region"
        assert client.project == "test_project"
