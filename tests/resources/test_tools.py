import pytest
import json
import uuid
from unittest.mock import MagicMock, patch
from relevanceai.resources.tools import ToolsManager
from relevanceai.resources.tool import Tool


class TestToolsManager:
    @pytest.fixture
    def mock_client(self):
        """Fixture to create a mock RelevanceAI client."""
        client = MagicMock()
        client.project = "test-project"
        client.region = "test-region"
        return client

    @pytest.fixture
    def tools_manager(self, mock_client):
        """Fixture to create a ToolsManager instance."""
        return ToolsManager(client=mock_client)

    def test_list_tools(self, tools_manager):
        """Test listing tools with successful response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "studio_id": "tool-1",
                    "title": "Tool 1",
                    "_id": "id-1",
                    "project": "test-project"
                },
                {
                    "studio_id": "tool-2",
                    "title": "Tool 2",
                    "_id": "id-2",
                    "project": "test-project"
                }
            ]
        }
        tools_manager._client.get = MagicMock(return_value=mock_response)

        result = tools_manager.list_tools(max_results=100)

        expected_params = {
            "filters": json.dumps([{
                "filter_type": "exact_match",
                "field": "project",
                "condition_value": "test-project",
                "condition": "=="
            }]),
            "page_size": 100
        }
        tools_manager._client.get.assert_called_once_with("studios/list", params=expected_params)
        assert len(result) == 2
        assert all(isinstance(tool, Tool) for tool in result)
        assert [tool.metadata.studio_id for tool in result] == ["tool-1", "tool-2"]

    def test_retrieve_tool(self, tools_manager):
        """Test retrieving a specific tool."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "studio": {
                "studio_id": "test-tool",
                "title": "Test Tool",
                "_id": "test-id",
                "project": "test-project"
            }
        }
        tools_manager._get = MagicMock(return_value=mock_response)

        result = tools_manager.retrieve_tool("test-tool")

        tools_manager._get.assert_called_once_with("studios/test-tool/get")
        assert isinstance(result, Tool)
        assert result.metadata.studio_id == "test-tool"
        assert result.metadata.title == "Test Tool"

    @patch('uuid.uuid4')
    def test_create_tool(self, mock_uuid, tools_manager):
        """Test creating a new tool."""
        mock_uuid.return_value = "new-tool-id"
        
        # Mock the bulk_update response
        mock_update_response = MagicMock()
        mock_update_response.status_code = 200
        tools_manager._post = MagicMock(return_value=mock_update_response)

        # Mock the retrieve_tool response
        mock_tool = Tool(
            client=tools_manager._client,
            studio_id="new-tool-id",
            title="New Tool",
            description="Test Description",
            _id="new-id",
            project="test-project"
        )
        tools_manager.retrieve_tool = MagicMock(return_value=mock_tool)

        result = tools_manager.create_tool(
            title="New Tool",
            description="Test Description",
            public=False
        )

        expected_body = {
            "updates": [{
                "title": "New Tool",
                "public": False,
                "project": "test-project",
                "description": "Test Description",
                "version": "latest",
                "params_schema": {
                    "properties": {},
                    "required": [],
                    "type": "object"
                },
                "output_schema": {},
                "transformations": {
                    "steps": []
                },
                "studio_id": "new-tool-id"
            }],
            "partial_update": True
        }

        tools_manager._post.assert_called_once_with("studios/bulk_update", body=expected_body)
        tools_manager.retrieve_tool.assert_called_once_with("new-tool-id")
        assert isinstance(result, Tool)
        assert result.metadata.studio_id == "new-tool-id"
        assert result.metadata.title == "New Tool"

    def test_clone_tool_success(self, tools_manager):
        """Test successful tool cloning."""
        # Mock the clone response
        mock_clone_response = MagicMock()
        mock_clone_response.json.return_value = {"studio_id": "cloned-tool-id"}
        tools_manager._post = MagicMock(return_value=mock_clone_response)

        # Mock the retrieve_tool response
        mock_tool = Tool(
            client=tools_manager._client,
            studio_id="cloned-tool-id",
            title="Cloned Tool",
            _id="cloned-id",
            project="test-project"
        )
        tools_manager.retrieve_tool = MagicMock(return_value=mock_tool)

        result = tools_manager.clone_tool("original-tool-id")

        expected_body = {
            "studio_id": "original-tool-id",
            "project": "test-project",
            "region": "test-region"
        }
        tools_manager._post.assert_called_once_with("/studios/clone", body=expected_body)
        tools_manager.retrieve_tool.assert_called_once_with("cloned-tool-id")
        assert isinstance(result, Tool)
        assert result.metadata.studio_id == "cloned-tool-id"

    def test_clone_tool_failure(self, tools_manager):
        """Test failed tool cloning."""
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        tools_manager._post = MagicMock(return_value=mock_response)

        result = tools_manager.clone_tool("original-tool-id")

        expected_body = {
            "studio_id": "original-tool-id",
            "project": "test-project",
            "region": "test-region"
        }
        tools_manager._post.assert_called_once_with("/studios/clone", body=expected_body)
        assert result is False

    def test_delete_tool(self, tools_manager):
        """Test successful tool deletion."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        tools_manager._post = MagicMock(return_value=mock_response)

        result = tools_manager.delete_tool("test-tool")

        tools_manager._post.assert_called_once_with(
            "studios/bulk_delete",
            body={"ids": ["test-tool"]}
        )
        assert result is True

    def test_delete_tool_failure(self, tools_manager):
        """Test failed tool deletion."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        tools_manager._post = MagicMock(return_value=mock_response)

        result = tools_manager.delete_tool("test-tool")

        tools_manager._post.assert_called_once_with(
            "studios/bulk_delete",
            body={"ids": ["test-tool"]}
        )
        assert result is False