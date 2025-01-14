import pytest
from unittest.mock import MagicMock
from relevanceai.resources.tasks import Tasks
from relevanceai.types.knowledge import Metadata
from relevanceai.types.task import TaskMetadata

class TestTasks:
    @pytest.fixture
    def mock_client(self):
        """Fixture to create a mock RelevanceAI client."""
        return MagicMock()

    @pytest.fixture
    def tasks(self, mock_client):
        """Fixture to create a Tasks instance."""
        return Tasks(client=mock_client)

    def test_get_metadata_without_metadata(self, tasks):
        """Test getting metadata when no metadata exists in response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"metadata": None}
        tasks._get = MagicMock(return_value=mock_response)

        result = tasks.get_metadata("conversation-123")

        tasks._get.assert_called_once_with(
            "knowledge/sets/conversation-123/get_metadata"
        )
        assert isinstance(result, dict)
        assert result == {}

    def test_get_metadata_empty_response(self, tasks):
        """Test getting metadata with empty response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        tasks._get = MagicMock(return_value=mock_response)

        result = tasks.get_metadata("conversation-123")

        tasks._get.assert_called_once_with(
            "knowledge/sets/conversation-123/get_metadata"
        )
        assert isinstance(result, dict)
        assert result == {}

    def test_delete_task_success(self, tasks):
        """Test successful task deletion."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        tasks._post = MagicMock(return_value=mock_response)

        result = tasks.delete_task("conversation-123")

        tasks._post.assert_called_once_with(
            path="knowledge/sets/delete",
            body={"knowledge_set": ["conversation-123"]}
        )
        assert result is True

    def test_delete_task_failure(self, tasks):
        """Test failed task deletion."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        tasks._post = MagicMock(return_value=mock_response)

        result = tasks.delete_task("conversation-123")

        tasks._post.assert_called_once_with(
            path="knowledge/sets/delete",
            body={"knowledge_set": ["conversation-123"]}
        )
        assert result is False