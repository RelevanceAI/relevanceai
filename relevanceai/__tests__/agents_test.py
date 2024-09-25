import pytest
from unittest.mock import Mock, patch
from ..resources.agents import Agents
from ..types.agent import Agent

@pytest.fixture
def agents():
    return Agents(Mock())

def test_list_agents_success(agents):
    response_mock = Mock()
    response_mock.json.return_value = {"results": [{"id": 1, "name": "agent1"}, {"id": 2, "name": "agent2"}]}
    with patch.object(agents, "_post", return_value=response_mock):
        agents_list = agents.list_agents()
        assert len(agents_list) == 2
        assert isinstance(agents_list[0], Agent)
        assert agents_list[0].id == 1
        assert agents_list[0].name == "agent1"

def test_list_agents_no_agents(agents):
    response_mock = Mock()
    response_mock.json.return_value = {"results": []}
    with patch.object(agents, "_post", return_value=response_mock):
        agents_list = agents.list_agents()
        assert len(agents_list) == 0

def test_list_agents_failed_response(agents):
    response_mock = Mock()
    response_mock.json.return_value = {"error": "Failed to retrieve agents"}
    with patch.object(agents, "_post", return_value=response_mock):
        with pytest.raises(Exception):
            agents.list_agents()

def test_retrieve_agent_success(agents):
    response_mock = Mock()
    response_mock.json.return_value = {
        "agent": {"id": 1, "name": "agent1"}
    }
    with patch.object(agents, "_get", return_value=response_mock):
        agent = agents.retrieve_agent("1")
        assert isinstance(agent, Agent)
        assert agent.id == 1
        assert agent.name == "agent1"

def test_retrieve_agent_not_found(agents):
    response_mock = Mock()
    response_mock.json.return_value = {
        "agent": None
    }
    with patch.object(agents, "_get", return_value=response_mock):
        with pytest.raises(Exception, match="Agent not found"):
            agents.retrieve_agent("999")  # Simulating an agent ID that doesn't exist

def test_retrieve_agent_failed_response(agents):
    response_mock = Mock()
    response_mock.json.return_value = {"error": "Failed to retrieve agent"}
    with patch.object(agents, "_get", return_value=response_mock):
        with pytest.raises(Exception, match="Failed to retrieve agent"):
            agents.retrieve_agent("1")

def test_delete_agent_success(agents):
    response_mock = Mock()
    response_mock.status_code = 200
    with patch.object(agents, "_post", return_value=response_mock):
        result = agents.delete_agent("1")
        assert result is True

def test_delete_agent_not_found(agents):
    response_mock = Mock()
    response_mock.status_code = 404  # Simulating agent not found
    with patch.object(agents, "_post", return_value=response_mock):
        result = agents.delete_agent("999")  # Non-existent agent ID
        assert result is False

def test_delete_agent_failed_response(agents):
    response_mock = Mock()
    response_mock.status_code = 500  # Simulating server error
    with patch.object(agents, "_post", return_value=response_mock):
        result = agents.delete_agent("1")
        assert result is False
