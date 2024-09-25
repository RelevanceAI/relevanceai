import pytest
from unittest.mock import Mock, patch
from ..types.tool import Tool, ToolOutput
from ..resources.tools import Tools

@pytest.fixture
def tools():
    client_mock = Mock()
    return Tools(client_mock)

def test_list_tools_success(tools):
    response_mock = Mock()
    response_mock.json.return_value = {
        "results": [
            {"id": "tool1", "name": "Tool 1", "metadata": {"source_studio_id": "studio1"}},
            {"id": "tool2", "name": "Tool 2", "metadata": {}}
        ]
    }
    tools._client.get.return_value = response_mock
    
    tools_list = tools.list_tools()
    assert len(tools_list) == 2
    assert isinstance(tools_list[0], Tool)
    assert tools_list[0].studio_id == "studio1"
    assert tools_list[1].studio_id is None  # No source_studio_id in metadata

def test_retrieve_tool_success(tools):
    response_mock = Mock()
    response_mock.json.return_value = {
        "studio": {"id": "tool1", "name": "Tool 1"}
    }
    with patch.object(tools, "_get", return_value=response_mock):
        tool = tools.retrieve_tool("tool1")
        assert isinstance(tool, Tool)
        assert tool.id == "tool1"
        assert tool.name == "Tool 1"

def test_trigger_tool_success(tools):
    response_mock = Mock()
    response_mock.json.return_value = {
        "output": "Output data", 
        "status": "success"
    }
    with patch.object(tools, "_post", return_value=response_mock):
        tool_output = tools.trigger_tool("tool1", params={"param1": "value"})
        assert isinstance(tool_output, ToolOutput)
        assert tool_output.output == "Output data"
        assert tool_output.status == "success"

def test_get_params_as_json_string_success(tools):
    response_mock = Mock()
    response_mock.json.return_value = {
        "studio": {"params_schema": {"properties": {"param1": {"type": "string"}}}}
    }
    with patch.object(tools, "_get", return_value=response_mock):
        params_json = tools._get_params_as_json_string("tool1")
        assert '"param1": {' in params_json

def test_get_steps_as_json_string_success(tools):
    response_mock = Mock()
    response_mock.json.return_value = {
        "studio": {"transformations": {"steps": [{"step1": "transformation1"}]}}
    }
    with patch.object(tools, "_get", return_value=response_mock):
        steps_json = tools._get_steps_as_json_string("tool1")
        assert '"step1": "transformation1"' in steps_json

def test_delete_tool_success(tools):
    response_mock = Mock()
    response_mock.status_code = 200
    with patch.object(tools, "_post", return_value=response_mock):
        result = tools.delete_tool("tool1")
        assert result is True

def test_delete_tool_failed(tools):
    response_mock = Mock()
    response_mock.status_code = 400  # Simulate failed deletion
    with patch.object(tools, "_post", return_value=response_mock):
        result = tools.delete_tool("tool1")
        assert result is False
