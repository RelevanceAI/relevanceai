import pytest
from unittest.mock import patch
from relevanceai import RelevanceAI


class TestRelevanceAI:
    @patch("relevanceai.RelevanceAI")
    def test_init(self, MockRelevanceAI):
        mock_client = MockRelevanceAI.return_value
        mock_client.api_key = "test"
        mock_client.region = "us-east-1"
        mock_client.project = "default"
        
        # Instantiate the client (this will use the mock)
        client = RelevanceAI(api_key="test")
        
        MockRelevanceAI.assert_called_once_with(api_key="test")
        assert client.api_key == "test"
        assert client.region == "us-east-1"
        assert client.project == "default"
